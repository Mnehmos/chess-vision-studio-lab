"""S11: targeted precision study — is 4k non-inferior to 8k once the limiting width gets its seeds?

Same frozen 9,495 positions, same student compute, 4k vs 8k. The only design change: each
(depth, width) pair is its own arm carrying its own seed budget, so extra seeds land where
variance actually limits (H32: 400 seeds; H1/H4/H16: 40 each). Zero new teacher labels.

    python tools/s11/s11_run.py freeze
    python tools/s11/s11_run.py run
    python tools/s11/s11_run.py analyse
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.data import read_jsonl
from cvslab.funnel.campaign import paired_effect
from cvslab.hashing import hash_obj
from cvslab.schemas import Ablation, Dataset, Run, RunStatus, TERMINAL_RUN_STATUSES, TrainingRecipe
from cvslab.service import LabService
from cvslab.store import Store
from cvslab.targets import TargetSpec

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools"
S11 = TOOLS / "s11"
S10_STATE = TOOLS / "s10" / "s10-state.json"
S9_STATE = TOOLS / "s9" / "s9-state.json"
S8_STATE = TOOLS / "s8" / "s8-state.json"
S10_RESULT = TOOLS / "s10" / "s10-result.json"
STATE = S11 / "s11-state.json"
PREREG = S11 / "s11-preregistration.json"
PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"
SEED_ALLOCATION = {1: 40, 4: 40, 16: 40, 32: 400}
DEPTHS = (8_000, 4_000)
BATCH, MAX_UPDATES, DELTA = 256, 760, 0.001
LABEL = {8_000: "8K", 4_000: "4K"}
DECLARATION = ("S11 precision arm: same positions, same student compute, same teacher depth as its "
               "partner; arms differ only in width and in how many seeds were spent")
SPECS = {depth: TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                           producer=PRODUCER, budget={"nodeBudget": depth}, value_path=("scoreCpStm",),
                           pov="stm", k=256.0, lam=1.0) for depth in DEPTHS}


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] saved {list(updates)}", flush=True)


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def step_freeze() -> int:
    if "experiment" in state():
        print("[cached] freeze")
        return 0
    service = svc()
    s10 = json.loads(S10_STATE.read_text(encoding="utf-8"))
    s9 = json.loads(S9_STATE.read_text(encoding="utf-8"))
    s8 = json.loads(S8_STATE.read_text(encoding="utf-8"))
    ids = None
    dataset_2k: Dataset = service.store.get_as(s9["datasets"]["2k"], Dataset)
    split = next(entry for entry in dataset_2k.splits if entry.name == "train")
    ids = [row["record_id"] for row in read_jsonl(service.store.abs(split.path))]
    if len(ids) != 9_495:
        print(f"STOP: {len(ids)} ids, not 9,495")
        return 3
    datasets = {"8K": service.store.get_as(s10["datasets"]["8K"], Dataset),
                "4K": service.store.get_as(s10["datasets"]["4K"], Dataset)}
    for label, dataset in datasets.items():
        entry = next(e for e in dataset.splits if e.name == "train")
        stored = [row["record_id"] for row in read_jsonl(service.store.abs(entry.path))]
        if stored != ids:
            print(f"STOP: the {label} dataset is not the same ids in the same order")
            return 3

    # a failed attempt can leave ablations that already claim the id it reserved; skip such an id
    # rather than letting the stray-member guard refuse an otherwise valid experiment
    while True:
        candidate = service.store.next_id("X")
        if not [a for a in service.store.list("A", verify=True, kind=Ablation)
                if a.experiment_id == candidate]:
            xid = candidate
            break
    attempt = len([a for a in service.store.list("A", verify=True, kind=Ablation)
                   if a.baseline_name.startswith("S11_")]) + 1
    control: TrainingRecipe = service.store.get_as("T0004", TrainingRecipe)
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    recipes = {depth: service.create_training_recipe(
        name=f"s11-{depth}", params={**base, "EPOCHS": 1, "BATCH": BATCH, "MAX_UPDATES": MAX_UPDATES},
        target_spec=SPECS[depth], description=f"S11 precision arm at {depth} nodes") for depth in DEPTHS}

    arms = []
    for depth, label in LABEL.items():
        # each (depth, width) pair is its own arm, so a width can carry its own seed budget
        baseline = service.register_baseline(
            name=f"S11_{label}_H16_A{attempt}", dataset_id=datasets[label].id, training_recipe_id=recipes[depth].id,
            eval_protocol_id="E0004", model_config={"INPUT": "RAW", "H": 16},
            supervision_divergence=DECLARATION, experiment_id=xid,
            notes=f"S11 {label} H16: 40 seeds")
        ablation_ids = {16: baseline.id}
        for width in (1, 4, 32):
            ablation_ids[width] = service.create_ablation(
                baseline_id=baseline.id, overrides={"H": width}, supervision_divergence=DECLARATION,
                experiment_id=xid, notes=f"S11 {label} H{width}").id
        split = next(entry for entry in datasets[label].splits if entry.name == "train")
        for width in (1, 4, 16, 32):
            arms.append({"arm_id": f"{label}-H{width}", "scale": f"h{width}-seeds",
                         "scale_nodes": SEED_ALLOCATION[width], "node_budget": depth,
                         "prefix_size": len(ids),
                         "realized_nodes": s10["realized"][label],
                         "dataset_id": datasets[label].id,
                         "dataset_manifest_hash": datasets[label].manifest_hash,
                         "record_ids_hash": split.record_ids_hash, "training_recipe_id": recipes[depth].id,
                         "recipe_hash": recipes[depth].recipe_hash,
                         "train_target_spec_hash": SPECS[depth].spec_hash(),
                         "ablations": [ablation_ids[width]],
                         "seeds": list(range(SEED_ALLOCATION[width]))})

    prereg = hash_obj(json.loads(PREREG.read_text(encoding="utf-8")))
    experiment = service.create_experiment(
        name="S11 targeted precision: 4k vs 8k", preregistration_hash=prereg, reference_unit_nodes=37_285_491,
        scales={f"h{width}-seeds": SEED_ALLOCATION[width] for width in (1, 4, 16, 32)},
        node_budgets=list(DEPTHS), arms=arms, eval_protocol_id="E0004",
        widths=list(SEED_ALLOCATION), seeds=list(range(max(SEED_ALLOCATION.values()))),
        source_id=s8["verify"]["universe"]["source"],
        normalization_id=s8["verify"]["universe"]["normalization"],
        candidate_universe_hash=s8["verify"]["universe"]["universe_hash"], order_seed=20260920,
        order_hash=s8["verify"]["order_hash"],
        analysis={"design": "targeted precision: per-(depth,width) arms with their own seed budgets",
                  "targeted_seeds": {f"h{w}": SEED_ALLOCATION[w] for w in (1, 4, 16, 32)},
                  "primary": "d = test_loss(4k) - test_loss(8k) per width, paired over that width's seeds",
                  "margin": {"delta": DELTA},
                  "bounds": "separate one-sided 95% lower and upper bounds",
                  "decision": ["4K_NONINFERIOR if upper < +0.001 at all widths",
                               "4K_TAXED if lower > +0.001 at every width", "INCONCLUSIVE otherwise"],
                  "determinism_check": "seeds 0..19 must reproduce X0004 per-seed values exactly"},
        experiment_id=xid, notes="8 arms (2 depths x 4 widths) x 1040 targeted cells; zero new labels")
    expected = service.expected_membership(experiment)
    if len(expected) != 1040:
        print(f"STOP: expected membership is {len(expected)}, not 1040")
        return 3
    save(xid=xid, ids=len(ids), experiment={"id": experiment.id, "arms": len(experiment.arms),
                                            "cells": len(expected)})
    print(f"experiment {experiment.id}: {len(experiment.arms)} arms, {len(expected)} cells", flush=True)
    return 0


def step_run() -> int:
    st = state()
    service = svc()
    queued = st.get("queued")
    if not queued:
        experiment = service.store.get(st["experiment"]["id"])
        queued = {}
        for arm in experiment.arms:
            for ablation_id in arm.ablations:
                queued[ablation_id] = [run.id for run in service.queue_runs(ablation_id, seeds=arm.seeds)]
        save(queued=queued)
        print(f"queued {sum(len(v) for v in queued.values())} runs", flush=True)
    counts: dict = {}
    done = 0
    for run_ids in queued.values():
        for run_id in run_ids:
            run: Run = service.store.get_as(run_id, Run)
            if run.status in TERMINAL_RUN_STATUSES:
                counts[run.status.value] = counts.get(run.status.value, 0) + 1
                done += 1
                continue
            finished = service.execute_run(run_id)
            counts[finished.status.value] = counts.get(finished.status.value, 0) + 1
            done += 1
            if finished.status.value != "COMPLETED":
                print(f"  {run_id} {finished.status.value}: "
                      f"{[c.detail for c in finished.integrity if c.status == 'fail'][:2]}", flush=True)
            elif done % 200 == 0:
                print(f"    ... {done} runs terminal", flush=True)
    save(run_statuses=counts)
    print("run statuses:", counts, flush=True)
    return 0


def step_analyse() -> int:
    st = state()
    service = svc()
    experiment = service.store.get(st["experiment"]["id"])
    arm_of = {ablation_id: arm.arm_id for arm in experiment.arms for ablation_id in arm.ablations}
    seed_of = {arm.arm_id: list(arm.seeds) for arm in experiment.arms}
    values: dict = {}
    invalid: list = []
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != experiment.id:
            continue
        key = arm_of.get(run.ablation_id)
        if key is None:
            continue
        if run.status != RunStatus.COMPLETED:
            invalid.append(run.id)
            continue
        metric = next((m for m in run.metrics if m.name == "test_loss"), None)
        if metric is None:
            invalid.append(run.id)
            continue
        values.setdefault(key, {})[run.seed] = metric.value

    def cell(width: int) -> dict:
        a = {seed: value for seed, value in values.get(f"4K-H{width}", {}).items()}
        b = {seed: value for seed, value in values.get(f"8K-H{width}", {}).items()}
        seeds = tuple(seed_of[f"4K-H{width}"])
        two = paired_effect(a, b, seeds=seeds)
        one = paired_effect(a, b, seeds=seeds, one_sided=True)
        stderr = (two["ci_high"] - two["width_estimate"]) / two["critical"]
        return {"n": two["n"], "d": two["width_estimate"], "critical_two_sided": two["critical"],
                "critical_one_sided": one["critical"], "stderr": stderr,
                "one_sided_bounds": [one["ci_low"], one["ci_high"]],
                "two_sided_ci": [two["ci_low"], two["ci_high"]]}

    cells = {str(width): cell(width) for width in (1, 4, 16, 32)}
    upper = {width: cells[str(width)]["one_sided_bounds"][1] for width in (1, 4, 16, 32)}
    lower = {width: cells[str(width)]["one_sided_bounds"][0] for width in (1, 4, 16, 32)}
    if all(upper[width] < DELTA for width in upper):
        decision = "4K_NONINFERIOR"
    elif all(lower[width] > DELTA for width in lower):
        decision = "4K_TAXED"
    else:
        decision = "INCONCLUSIVE"
    equivalence = {width: (cells[str(width)]["two_sided_ci"][0] > -DELTA
                           and cells[str(width)]["two_sided_ci"][1] < DELTA) for width in upper}

    # determinism across experiments: seeds 0..19 must reproduce X0004 exactly
    s10_state = json.loads(S10_STATE.read_text(encoding="utf-8"))
    mismatches = []
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != "X0004" or run.status != RunStatus.COMPLETED:
            continue
        if run.effective_config["H"] not in (1, 4, 16, 32):
            continue
        metric = next((m for m in run.metrics if m.name == "test_loss"), None)
        if metric is None:
            continue
        label = "4K" if run.dataset_id == s10_state["datasets"]["4K"] else None
        if label != "4K" or run.seed >= 20:
            continue
        mine = values.get(f"4K-H{run.effective_config['H']}", {}).get(run.seed)
        if mine is not None and abs(mine - metric.value) > 1e-12:
            mismatches.append({"run": run.id, "seed": run.seed, "H": run.effective_config["H"],
                               "x0004": metric.value, "s11": mine})

    result = {"experiment": experiment.id, "preregistration_hash": experiment.preregistration_hash,
              "cells": {"expected": experiment.expected_run_count, "invalid_runs": invalid},
              "targeted_seeds": {f"h{w}": SEED_ALLOCATION[w] for w in (1, 4, 16, 32)},
              "primary_4k_minus_8k": cells, "decision": decision, "margin_delta": DELTA,
              "point_estimates_vs_margin": {str(width): ("above" if cells[str(width)]["d"] > DELTA else "below")
                                            for width in upper},
              "equivalence_by_width": {str(width): equivalence[width] for width in upper},
              "equivalence_all_widths": all(equivalence.values()),
              "determinism_vs_x0004": {"mismatches": mismatches, "seeds_compared": 20 * 4,
                                       "note": "seed 0..19 of each 4k width must equal X0004's stored value exactly"},
              "mean_test_loss": {arm: statistics.mean(values[arm].values()) for arm in sorted(values)}}
    (S11 / "s11-result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"experiment": experiment.id, "decision": decision, "margin_delta": DELTA,
               "targeted_seeds": result["targeted_seeds"],
               "primary_4k_minus_8k": {width: {"d": round(cells[str(width)]["d"], 6),
                                               "one_sided_bounds": [round(v, 6) for v in cells[str(width)]["one_sided_bounds"]],
                                               "two_sided_ci": [round(v, 6) for v in cells[str(width)]["two_sided_ci"]],
                                               "n": cells[str(width)]["n"]} for width in upper},
               "max_upper_bound": max(upper.values()),
               "equivalence_all_widths": result["equivalence_all_widths"],
               "determinism_mismatches": len(mismatches),
               "mean_test_loss": {k: round(v, 6) for k, v in result["mean_test_loss"].items()},
               "evidence": {"result": "tools/s11/s11-result.json",
                            "preregistration": "tools/s11/s11-preregistration.json",
                            "power_planning": "tools/s11/s11-power-planning.json",
                            "state": "tools/s11/s11-state.json"}}
    (S11 / "s11-result-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({"decision": decision, "primary": summary["primary_4k_minus_8k"],
                      "max_upper_bound": summary["max_upper_bound"],
                      "equivalence_all_widths": summary["equivalence_all_widths"],
                      "determinism_mismatches": summary["determinism_mismatches"]}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["freeze", "run", "analyse"])
    args = parser.parse_args()
    S11.mkdir(parents=True, exist_ok=True)
    return {"freeze": step_freeze, "run": step_run, "analyse": step_analyse}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
