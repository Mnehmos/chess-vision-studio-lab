"""S10: is 4k the economic sweet spot between 8k and 2k?

Three arms over the SAME frozen 9,495 positions, identical student compute, zero new labels:
8k (quality reference), 4k (the candidate point), 2k (the measured tax).

    python tools/s10/s10_run.py freeze
    python tools/s10/s10_run.py run
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
from cvslab.schemas import Dataset, Run, RunStatus, TERMINAL_RUN_STATUSES, TrainingRecipe
from cvslab.service import LabService
from cvslab.store import Store
from cvslab.targets import TargetSpec

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools"
S10 = TOOLS / "s10"
S9_STATE = TOOLS / "s9" / "s9-state.json"
S8_STATE = TOOLS / "s8" / "s8-state.json"
S8_LABELS = pathlib.Path(r"F:\Github\_parity_tmp\s8")
STATE = S10 / "s10-state.json"
PREREG = S10 / "s10-preregistration.json"
PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"
WIDTHS = (1, 4, 16, 32)
SEEDS = tuple(range(20))
BATCH, MAX_UPDATES, DELTA = 256, 760, 0.001
DEPTHS = (8_000, 4_000, 2_000)
LABEL = {8_000: "8K", 4_000: "4K", 2_000: "2K"}
DECLARATION = ("S10 arm: same positions and same student compute as its paired arms, differing only in the "
               "teacher depth that produced the targets")
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


def frozen_ids(service: LabService) -> list[str]:
    """The one id list this family of experiments shares, read from S9's frozen 2k dataset."""
    s9 = json.loads(S9_STATE.read_text(encoding="utf-8"))
    dataset: Dataset = service.store.get_as(s9["datasets"]["2k"], Dataset)
    split = next(entry for entry in dataset.splits if entry.name == "train")
    ids = [row["record_id"] for row in read_jsonl(service.store.abs(split.path))]
    if len(ids) != 9_495:
        raise SystemExit(f"STOP: the frozen dataset holds {len(ids)} ids, not 9,495")
    return ids


def step_freeze() -> int:
    if "experiment" in state():
        print("[cached] freeze")
        return 0
    service = svc()
    s9 = json.loads(S9_STATE.read_text(encoding="utf-8"))
    s8 = json.loads(S8_STATE.read_text(encoding="utf-8"))
    ids = frozen_ids(service)
    normalization = s8["verify"]["universe"]["normalization"]

    datasets = {"8K": service.store.get_as(s9["datasets"]["8k"], Dataset),
                "2K": service.store.get_as(s9["datasets"]["2k"], Dataset)}
    for label, dataset in datasets.items():
        split = next(entry for entry in dataset.splits if entry.name == "train")
        stored = [row["record_id"] for row in read_jsonl(service.store.abs(split.path))]
        if stored != ids:
            print(f"STOP: the {label} dataset does not hold exactly these ids in this order")
            return 3
    context = [row for depth in DEPTHS
               for row in read_jsonl(S8_LABELS / f"labels-{depth}.jsonl")]
    by_depth = {depth: {row["record_id"]: int(row["budget_nodes"]) for row in context
                        if row["row"]["budget"]["nodeBudget"] == depth} for depth in DEPTHS}
    for depth in DEPTHS:
        missing = [rid for rid in ids if rid not in by_depth[depth]]
        if missing:
            print(f"STOP: the {depth} stream is missing {len(missing)} of the frozen positions")
            return 3
    realized = {label: sum(by_depth[depth][rid] for rid in ids) for depth, label in LABEL.items()}

    dataset_4k = service.freeze_dataset(normalization, name="s10-same-9495-4k", record_ids=ids,
                                        required_labels=["search_shallow_cp"], target_spec=SPECS[4_000],
                                        fractions=(1.0, 0.0, 0.0),
                                        campaign={"purpose": "S10 sweet-spot arm", "arm": "4k",
                                                  "node_budget": 4_000, "shared_record_ids_hash": hash_obj(ids),
                                                  "preregistration": "tools/s10/s10-preregistration.json"})
    split_4k = next(entry for entry in dataset_4k.splits if entry.name == "train")
    if [row["record_id"] for row in read_jsonl(service.store.abs(split_4k.path))] != ids:
        print("STOP: the new 4k dataset does not hold exactly the same ids in the same order")
        return 3
    datasets["4K"] = dataset_4k
    print(f"realized teacher nodes over the same {len(ids)} positions: "
          + ", ".join(f"{label} {realized[label]:,}" for label in ("8K", "4K", "2K"))
          + f"; 8k/4k = {realized['8K'] / realized['4K']:.2f}x, 8k/2k = {realized['8K'] / realized['2K']:.2f}x",
          flush=True)

    xid = service.store.next_id("X")
    control: TrainingRecipe = service.store.get_as("T0004", TrainingRecipe)
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    recipes = {depth: service.create_training_recipe(
        name=f"s10-{depth}", params={**base, "EPOCHS": 1, "BATCH": BATCH, "MAX_UPDATES": MAX_UPDATES},
        target_spec=SPECS[depth], description=f"S10 fixed-update regime at {depth} nodes") for depth in DEPTHS}
    ablations: dict = {}
    for depth, label in LABEL.items():
        baseline = service.register_baseline(
            name=f"S10_SAME_{label}", dataset_id=datasets[label].id, training_recipe_id=recipes[depth].id,
            eval_protocol_id="E0004", model_config={"INPUT": "RAW", "H": 16},
            supervision_divergence=DECLARATION, experiment_id=xid,
            notes=f"SAME-9495-{label}: {len(ids)} positions, identical student compute")
        by_width = {16: baseline.id}
        for width in WIDTHS:
            if width == 16:
                continue
            by_width[width] = service.create_ablation(baseline_id=baseline.id, overrides={"H": width},
                                                      supervision_divergence=DECLARATION, experiment_id=xid,
                                                      notes=f"SAME-9495-{label} H={width}").id
        ablations[label] = [by_width[width] for width in WIDTHS]

    prereg = hash_obj(json.loads(PREREG.read_text(encoding="utf-8")))
    (S10 / "s10-preregistration.hash").write_text(prereg + "\n", encoding="utf-8")
    arms = []
    for depth, label in LABEL.items():
        split = next(entry for entry in datasets[label].splits if entry.name == "train")
        arms.append({"arm_id": f"SAME-{label}", "scale": "same-9495", "scale_nodes": 1, "node_budget": depth,
                     "prefix_size": len(ids), "realized_nodes": realized[label],
                     "dataset_id": datasets[label].id,
                     "dataset_manifest_hash": datasets[label].manifest_hash,
                     "record_ids_hash": split.record_ids_hash, "training_recipe_id": recipes[depth].id,
                     "recipe_hash": recipes[depth].recipe_hash,
                     "train_target_spec_hash": SPECS[depth].spec_hash(), "ablations": ablations[label]})
    experiment = service.create_experiment(
        name="S10 is 4k the economic sweet spot", preregistration_hash=prereg, reference_unit_nodes=37_285_491,
        scales={"same-9495": 1}, node_budgets=list(DEPTHS), arms=arms, eval_protocol_id="E0004",
        widths=list(WIDTHS), seeds=list(SEEDS), source_id=s8["verify"]["universe"]["source"],
        normalization_id=normalization, candidate_universe_hash=s8["verify"]["universe"]["universe_hash"],
        order_seed=20260920, order_hash=s8["verify"]["order_hash"],
        analysis={"design": "three teacher depths over one frozen position set, identical student compute",
                  "primary": "d = test_loss(4k) - test_loss(8k), paired over twenty seeds, per width",
                  "margin": {"delta": DELTA},
                  "bounds": "separate one-sided 95% lower and upper bounds (t=1.729, df=19)",
                  "decision": ["4K_NONINFERIOR if upper < +0.001 at all widths",
                               "4K_TAXED if lower > +0.001 at every width", "INCONCLUSIVE otherwise"],
                  "secondary": ["2k-4k", "2k-8k", "realized cost ratio at each depth"]},
        experiment_id=xid, notes="3 depths x 4 widths x 20 seeds = 240 cells; zero new teacher labels")
    expected = service.expected_membership(experiment)
    if len(expected) != 240:
        print(f"STOP: expected membership is {len(expected)}, not 240")
        return 3
    save(xid=xid, ids=len(ids), datasets={label: dataset.id for label, dataset in datasets.items()},
         realized=realized, experiment={"id": experiment.id, "arms": len(experiment.arms),
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
                queued[ablation_id] = [run.id for run in service.queue_runs(ablation_id, seeds=SEEDS)]
        save(queued=queued)
        print(f"queued {sum(len(v) for v in queued.values())} runs", flush=True)
    counts: dict = {}
    for run_ids in queued.values():
        for run_id in run_ids:
            run: Run = service.store.get_as(run_id, Run)
            if run.status in TERMINAL_RUN_STATUSES:
                counts[run.status.value] = counts.get(run.status.value, 0) + 1
                continue
            finished = service.execute_run(run_id)
            counts[finished.status.value] = counts.get(finished.status.value, 0) + 1
            if finished.status.value != "COMPLETED":
                print(f"  {run_id} {finished.status.value}: "
                      f"{[c.detail for c in finished.integrity if c.status == 'fail'][:2]}", flush=True)
    save(run_statuses=counts)
    print("run statuses:", counts, flush=True)
    return 0


def step_analyse() -> int:
    st = state()
    service = svc()
    experiment = service.store.get(st["experiment"]["id"])
    arm_of = {ablation_id: arm.arm_id for arm in experiment.arms for ablation_id in arm.ablations}
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
        values.setdefault(key, {}).setdefault(str(run.effective_config["H"]), {})[run.seed] = metric.value

    def contrast(a: str, b: str, width: int, *, one_sided: bool) -> dict:
        return paired_effect(values.get(a, {}).get(str(width), {}), values.get(b, {}).get(str(width), {}),
                             seeds=SEEDS, one_sided=one_sided)

    def cells(a: str, b: str) -> dict:
        return {str(width): {"d": contrast(a, b, width, one_sided=True)["width_estimate"],
                             "one_sided_low": contrast(a, b, width, one_sided=True)["ci_low"],
                             "one_sided_high": contrast(a, b, width, one_sided=True)["ci_high"],
                             "two_sided": [contrast(a, b, width, one_sided=False)["ci_low"],
                                           contrast(a, b, width, one_sided=False)["ci_high"]],
                             "n": 20} for width in WIDTHS}

    primary = cells("SAME-4K", "SAME-8K")
    upper = {width: primary[str(width)]["one_sided_high"] for width in WIDTHS}
    lower = {width: primary[str(width)]["one_sided_low"] for width in WIDTHS}
    if all(upper[width] < DELTA for width in WIDTHS):
        decision = "4K_NONINFERIOR"
    elif all(lower[width] > DELTA for width in WIDTHS):
        decision = "4K_TAXED"
    else:
        decision = "INCONCLUSIVE"
    equivalence = {width: (primary[str(width)]["two_sided"][0] > -DELTA
                           and primary[str(width)]["two_sided"][1] < DELTA) for width in WIDTHS}

    result = {
        "experiment": experiment.id, "preregistration_hash": experiment.preregistration_hash,
        "cells": {"expected": experiment.expected_run_count, "invalid_runs": invalid},
        "design": "three teacher depths over one frozen 9,495-position set, identical student compute "
                  "(760 updates x 256), one exam",
        "primary_4k_minus_8k": primary, "decision": decision, "margin_delta": DELTA,
        "point_estimates_vs_margin": {str(width): ("above" if primary[str(width)]["d"] > DELTA else "below")
                                      for width in WIDTHS},
        "secondary_equivalence_4k": equivalence, "equivalence_all_widths": all(equivalence.values()),
        "three_point_curve": {"2k-4k": cells("SAME-2K", "SAME-4K"), "2k-8k": cells("SAME-2K", "SAME-8K")},
        "mean_test_loss": {arm: statistics.mean(v for widths in values.get(arm, {}).values()
                                                for v in widths.values()) for arm in sorted(values)},
        "realized_teacher_nodes": st["realized"],
    }
    (S10 / "s10-result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"experiment": experiment.id, "decision": decision, "margin_delta": DELTA,
               "primary_4k_minus_8k": {width: {"d": round(primary[str(width)]["d"], 6),
                                               "one_sided_bounds": [round(primary[str(width)]["one_sided_low"], 6),
                                                                    round(primary[str(width)]["one_sided_high"], 6)],
                                               "two_sided_ci": [round(v, 6) for v in primary[str(width)]["two_sided"]]}
                                       for width in WIDTHS},
               "point_estimates_vs_margin": result["point_estimates_vs_margin"],
               "equivalence_all_widths": result["equivalence_all_widths"],
               "three_point_curve": {name: {str(width): {"d": round(c[str(width)]["d"], 6),
                                                         "one_sided_bounds": [round(c[str(width)]["one_sided_low"], 6),
                                                                              round(c[str(width)]["one_sided_high"], 6)]}
                                            for width in WIDTHS}
                                     for name, c in result["three_point_curve"].items()},
               "mean_test_loss": {k: round(v, 6) for k, v in result["mean_test_loss"].items()},
               "realized_teacher_nodes": st["realized"], "invalid_runs": invalid,
               "evidence": {"result": "tools/s10/s10-result.json",
                            "preregistration": "tools/s10/s10-preregistration.json",
                            "state": "tools/s10/s10-state.json"}}
    (S10 / "s10-result-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({"decision": decision, "primary": summary["primary_4k_minus_8k"],
                      "estimates_vs_margin": summary["point_estimates_vs_margin"],
                      "equivalence_all_widths": summary["equivalence_all_widths"],
                      "mean_test_loss": summary["mean_test_loss"],
                      "realized": summary["realized_teacher_nodes"]}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["freeze", "run", "analyse"])
    args = parser.parse_args()
    S10.mkdir(parents=True, exist_ok=True)
    return {"freeze": step_freeze, "run": step_run, "analyse": step_analyse}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
