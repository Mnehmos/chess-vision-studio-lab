"""S11 successor: the clean confirmatory experiment (X0007) under the repaired contract.

880 cells: 20 fresh paired seeds for H1/H4/H16, 380 for H32, across 4k and 8k. The seed
allocation lives solely in arm.seeds; the scale axis carries the real stratum ("same-9495",
both depths at one 2x-scale teacher budget); width-specific arms partition the
(scale, depth) cells. Must reproduce X0006 deterministically — any per-seed mismatch is a stop.
X0006 stays preserved as the historical malformed attempt, explicitly superseded.

    python tools/s11/s11_successor.py freeze
    python tools/s11/s11_successor.py run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.data import read_jsonl
from cvslab.funnel.campaign import paired_effect, t_critical
from cvslab.hashing import hash_obj
from cvslab.schemas import Ablation, Dataset, Run, RunStatus, TERMINAL_RUN_STATUSES, TrainingRecipe
from cvslab.service import LabService
from cvslab.store import Store
from cvslab.targets import TargetSpec

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools"
S11 = TOOLS / "s11"
S10_STATE = TOOLS / "s10" / "s10-state.json"
S8_STATE = TOOLS / "s8" / "s8-state.json"
STATE = S11 / "s11-successor-state.json"
PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"
FRESH = {1: range(20, 40), 4: range(20, 40), 16: range(20, 40), 32: range(20, 400)}
DEPTHS = (8_000, 4_000)
BATCH, MAX_UPDATES = 256, 760
DECLARATION = ("S11 successor confirmatory arm: same positions, same student compute, one teacher depth; "
               "only fresh seeds 20+")
SPECS = {depth: TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                           producer=PRODUCER, budget={"nodeBudget": depth}, value_path=("scoreCpStm",),
                           pov="stm", k=256.0, lam=1.0) for depth in DEPTHS}
EXPECTED_CELLS = 2 * (3 * 20 + 380)


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
    s8 = json.loads(S8_STATE.read_text(encoding="utf-8"))
    dataset_2k: Dataset = service.store.get_as(json.loads((TOOLS / "s9" / "s9-state.json")
                                                          .read_text(encoding="utf-8"))["datasets"]["2k"], Dataset)
    split = next(entry for entry in dataset_2k.splits if entry.name == "train")
    ids = [row["record_id"] for row in read_jsonl(service.store.abs(split.path))]
    if len(ids) != 9_495:
        print(f"STOP: {len(ids)} ids, not 9,495")
        return 3
    datasets = {"8K": service.store.get_as(s10["datasets"]["8K"], Dataset),
                "4K": service.store.get_as(s10["datasets"]["4K"], Dataset)}
    for label, dataset in datasets.items():
        entry = next(e for e in dataset.splits if e.name == "train")
        if [row["record_id"] for row in read_jsonl(service.store.abs(entry.path))] != ids:
            print(f"STOP: the {label} dataset is not the same ids in the same order")
            return 3

    while True:                      # a refused attempt poisons its id; take a clean one
        candidate = service.store.next_id("X")
        if not [a for a in service.store.list("A", verify=True, kind=Ablation)
                if a.experiment_id == candidate]:
            xid = candidate
            break
    control: TrainingRecipe = service.store.get_as("T0004", TrainingRecipe)
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    recipes = {depth: service.create_training_recipe(
        name=f"s11s-{depth}", params={**base, "EPOCHS": 1, "BATCH": BATCH, "MAX_UPDATES": MAX_UPDATES},
        target_spec=SPECS[depth], description=f"S11 successor confirmatory arm at {depth} nodes")
        for depth in DEPTHS}

    arms = []
    for label, depth in (("8K", 8_000), ("4K", 4_000)):
        baseline = service.register_baseline(
            name=f"S11S_{label}_H16", dataset_id=datasets[label].id, training_recipe_id=recipes[depth].id,
            eval_protocol_id="E0004", model_config={"INPUT": "RAW", "H": 16},
            supervision_divergence=DECLARATION, experiment_id=xid,
            notes=f"S11S {label} H16: fresh seeds 20-39")
        ablation_ids = {16: baseline.id}
        for width in (1, 4, 32):
            ablation_ids[width] = service.create_ablation(
                baseline_id=baseline.id, overrides={"H": width}, supervision_divergence=DECLARATION,
                experiment_id=xid, notes=f"S11S {label} H{width}").id
        entry = next(e for e in datasets[label].splits if e.name == "train")
        for width in (1, 4, 16, 32):
            # PARTITIONING: four width-specific arms share the (same-9495, depth) cell; the seed
            # allocation lives here and only here — never in the scale axis
            arms.append({"arm_id": f"{label}-H{width}", "scale": "same-9495", "scale_nodes": 1,
                         "node_budget": depth, "prefix_size": len(ids),
                         "realized_nodes": s10["realized"][label], "dataset_id": datasets[label].id,
                         "dataset_manifest_hash": datasets[label].manifest_hash,
                         "record_ids_hash": entry.record_ids_hash,
                         "training_recipe_id": recipes[depth].id, "recipe_hash": recipes[depth].recipe_hash,
                         "train_target_spec_hash": SPECS[depth].spec_hash(),
                         "ablations": [ablation_ids[width]], "seeds": list(FRESH[width])})

    experiment = service.create_experiment(
        name="S11 successor: confirmatory 4k vs 8k on fresh seeds", preregistration_hash=(S11 /
                                                                                          "s11-preregistration.hash").read_text(encoding="utf-8").strip(),
        reference_unit_nodes=37_285_491, scales={"same-9495": 1}, node_budgets=list(DEPTHS), arms=arms,
        eval_protocol_id="E0004", widths=list(FRESH), seeds=list(range(20, 400)),
        source_id=s8["verify"]["universe"]["source"], normalization_id=s8["verify"]["universe"]["normalization"],
        candidate_universe_hash=s8["verify"]["universe"]["universe_hash"], order_seed=20260920,
        order_hash=s8["verify"]["order_hash"],
        analysis={"design": "confirmatory: fresh seeds only (20-39 H1/H4/H16, 20-399 H32), partitioned arms",
                  "supersedes": "X0006 (preserved as the historical attempt whose scale axis encoded seed allocation)",
                  "primary": "d = test_loss(4k) - test_loss(8k) per width over that width's fresh seeds",
                  "margin": {"delta": 0.001}, "bounds": "separate one-sided 95% (exact Student-t per df)",
                  "decision": ["4K_NONINFERIOR if upper < +0.001 at all widths",
                               "4K_TAXED if lower > +0.001 at every width", "INCONCLUSIVE otherwise"],
                  "determinism": "must reproduce X0006 per-seed values exactly; any mismatch is a stop"},
        experiment_id=xid, notes=f"{EXPECTED_CELLS} fresh-seed confirmatory cells; zero new labels")
    expected = service.expected_membership(experiment)
    if len(expected) != EXPECTED_CELLS:
        print(f"STOP: expected membership is {len(expected)}, not {EXPECTED_CELLS}")
        return 3
    save(xid=xid, experiment={"id": experiment.id, "arms": len(experiment.arms), "cells": len(expected)})
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
            elif done % 100 == 0:
                print(f"    ... {done} runs terminal", flush=True)
    save(run_statuses=counts)
    print("run statuses:", counts, flush=True)
    return 0


def step_verify() -> int:
    """Determinism: every successor cell must equal X0006's value for the same (depth, width, seed)."""
    st = state()
    service = svc()
    successor = service.store.get(st["experiment"]["id"])
    arm_of = {ablation_id: arm.arm_id for arm in successor.arms for ablation_id in arm.ablations}
    mine: dict = {}
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != successor.id:
            continue
        metric = next((m for m in run.metrics if m.name == "test_loss"), None)
        if metric is not None:
            mine[(arm_of[run.ablation_id], run.seed)] = metric.value
    x0006_arm_of = {ablation_id: arm.arm_id for arm in service.store.get("X0006").arms
                    for ablation_id in arm.ablations}
    mismatches, compared = [], 0
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != "X0006":
            continue
        metric = next((m for m in run.metrics if m.name == "test_loss"), None)
        if metric is None:
            continue
        key = (x0006_arm_of[run.ablation_id], run.seed)
        if key in mine:
            compared += 1
            if abs(mine[key] - metric.value) > 1e-12:
                mismatches.append({"cell": key, "x0006": metric.value, "successor": mine[key]})
    save(determinism={"compared": compared, "mismatches": mismatches[:20], "mismatch_count": len(mismatches)})
    print(f"determinism vs X0006: {compared} cells compared, {len(mismatches)} mismatches", flush=True)
    return 3 if mismatches else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["freeze", "run", "verify"])
    args = parser.parse_args()
    return {"freeze": step_freeze, "run": step_run, "verify": step_verify}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
