"""S9: same positions, cheaper labels. Freeze two datasets over one id list, then run X0003.

    python tools/s9/s9_run.py freeze       # ids, datasets, recipes, ablations, X0003
    python tools/s9/s9_run.py run          # queue and execute the 160 cells
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.data import read_jsonl
from cvslab.hashing import hash_obj
from cvslab.schemas import Dataset, Run, TERMINAL_RUN_STATUSES, TrainingRecipe
from cvslab.service import LabService
from cvslab.store import Store
from cvslab.targets import TargetSpec

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools"
S9 = TOOLS / "s9"
S8_STATE = TOOLS / "s8" / "s8-state.json"
STATE = S9 / "s9-state.json"
PREREG = S9 / "s9-preregistration.json"
PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"
WIDTHS = (1, 4, 16, 32)
SEEDS = tuple(range(20))
BATCH = 256
MAX_UPDATES = 760
DELTA = 0.001
DECLARATION = ("S9 arm: same positions and same student compute as its paired arm, differing only in the "
               "teacher depth that produced the targets")
SPECS = {depth: TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                           producer=PRODUCER, budget={"nodeBudget": depth}, value_path=("scoreCpStm",),
                           pov="stm", k=256.0, lam=1.0) for depth in (8_000, 2_000)}


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] saved {list(updates)}", flush=True)


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def step_freeze() -> int:
    st = state()
    if "experiment" in st:
        print("[cached] freeze")
        return 0
    service = svc()
    s8 = json.loads(S8_STATE.read_text(encoding="utf-8"))
    arm8k = next(arm for arm in s8["arms"] if arm["arm_id"] == "8k")
    order = s8["order"]
    ids = order[: arm8k["prefix_size"]]
    if len(ids) != 9_495:
        print(f"STOP: expected 9,495 ids, the S8 8k arm has {len(ids)}")
        return 3

    # the S8 8k dataset must reproduce exactly over this id list, or we freeze a fresh pair
    source_dataset: Dataset = service.store.get_as(arm8k["dataset_id"], Dataset)
    split = next(entry for entry in source_dataset.splits if entry.name == "train")
    stored_ids = [row["record_id"] for row in read_jsonl(service.store.abs(split.path))]
    reproduces = (stored_ids == ids and split.count == len(ids)
                  and split.record_ids_hash == arm8k["record_ids_hash"])
    if not reproduces:
        print("STOP: the S8 8k dataset does not contain exactly these ids in this order")
        return 3

    normalization = s8["verify"]["universe"]["normalization"]
    dataset_8k = source_dataset
    dataset_2k = service.freeze_dataset(normalization, name="s9-same-9495-2k", record_ids=ids,
                                        required_labels=["search_shallow_cp"], target_spec=SPECS[2_000],
                                        fractions=(1.0, 0.0, 0.0),
                                        campaign={"purpose": "S9 same positions, cheaper labels",
                                                  "arm": "2k", "node_budget": 2_000,
                                                  "shared_record_ids_hash": hash_obj(ids),
                                                  "preregistration": "tools/s9/s9-preregistration.json"})
    split_2k = next(entry for entry in dataset_2k.splits if entry.name == "train")
    ids_2k = [row["record_id"] for row in read_jsonl(service.store.abs(split_2k.path))]
    if ids_2k != ids or split_2k.count != len(ids):
        print("STOP: the 2k dataset does not contain exactly the same ids in the same order")
        return 3
    print(f"SAME-9495: 8k = {dataset_8k.id} (reused, verified), 2k = {dataset_2k.id} (new); "
          f"{len(ids)} ids, identical order", flush=True)

    # the 2k stream was walked along the same order, so its first 9,495 rows are these positions
    stream_2k = read_jsonl(pathlib.Path(r"F:\Github\_parity_tmp\s8") / "labels-2000.jsonl")
    by_id_2k = {row["record_id"]: int(row["budget_nodes"]) for row in stream_2k}
    missing = [rid for rid in ids if rid not in by_id_2k]
    if missing:
        print(f"STOP: the 2k stream is missing {len(missing)} of the dataset's positions")
        return 3
    # the stream file is ordered by worker chunk, so realized cost is summed over the ids
    realized_2k = sum(by_id_2k[rid] for rid in ids)
    print(f"realized teacher nodes over the same {len(ids)} positions: 8k {arm8k['realized_nodes']:,} vs "
          f"2k {realized_2k:,} (ratio {arm8k['realized_nodes'] / realized_2k:.2f}x)", flush=True)

    xid = service.store.next_id("X")
    control: TrainingRecipe = service.store.get_as("T0004", TrainingRecipe)
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    recipes = {}
    for depth in (8_000, 2_000):
        recipes[depth] = service.create_training_recipe(
            name=f"s9-{depth}", params={**base, "EPOCHS": 1, "BATCH": BATCH, "MAX_UPDATES": MAX_UPDATES},
            target_spec=SPECS[depth],
            description=f"S9 fixed-update regime at {depth} nodes")
    ablations: dict = {}
    for depth, dataset, label in ((8_000, dataset_8k, "8K"), (2_000, dataset_2k, "2K")):
        baseline = service.register_baseline(
            name=f"S9_SAME_{label}", dataset_id=dataset.id, training_recipe_id=recipes[depth].id,
            eval_protocol_id="E0004", model_config={"INPUT": "RAW", "H": 16},
            supervision_divergence=DECLARATION, experiment_id=xid,
            notes=f"SAME-9495-{label}: {len(ids)} positions, identical student compute")
        ids_by_width = {16: baseline.id}
        for width in WIDTHS:
            if width == 16:
                continue
            ids_by_width[width] = service.create_ablation(baseline_id=baseline.id, overrides={"H": width},
                                                          supervision_divergence=DECLARATION,
                                                          experiment_id=xid,
                                                          notes=f"SAME-9495-{label} H={width}").id
        ablations[label] = [ids_by_width[width] for width in WIDTHS]

    prereg = hash_obj(json.loads(PREREG.read_text(encoding="utf-8")))
    (S9 / "s9-preregistration.hash").write_text(prereg + "\n", encoding="utf-8")
    experiment = service.create_experiment(
        name="S9 same positions, cheaper labels", preregistration_hash=prereg,
        reference_unit_nodes=37_285_491,
        scales={"same-9495": 1},
        node_budgets=[8_000, 2_000],
        arms=[{"arm_id": f"SAME-{label}", "scale": "same-9495", "scale_nodes": 1,
               "node_budget": depth, "prefix_size": len(ids),
               "realized_nodes": arm8k["realized_nodes"] if depth == 8_000 else realized_2k,
               "dataset_id": dataset.id, "dataset_manifest_hash": dataset.manifest_hash,
               "record_ids_hash": split.record_ids_hash if depth == 8_000 else split_2k.record_ids_hash,
               "training_recipe_id": recipes[depth].id, "recipe_hash": recipes[depth].recipe_hash,
               "train_target_spec_hash": SPECS[depth].spec_hash(), "ablations": ablations[label]}
              for depth, dataset, label, split in ((8_000, dataset_8k, "8K", split),
                                                    (2_000, dataset_2k, "2K", split_2k))],
        eval_protocol_id="E0004", widths=list(WIDTHS), seeds=list(SEEDS),
        source_id=s8["verify"]["universe"]["source"],
        normalization_id=normalization,
        candidate_universe_hash=s8["verify"]["universe"]["universe_hash"], order_seed=20260920,
        order_hash=s8["verify"]["order_hash"],
        analysis={"design": "same positions, same count, same student compute, same exam; only the teacher depth differs",
                  "primary": "d = test_loss(2k) - test_loss(8k), paired over twenty seeds, per width",
                  "margin": {"delta": DELTA, "direction": "2k is worse if d > 0"},
                  "interval": "one-sided 95% bounds (t=1.729, df=19)",
                  "decision": ["2K_NONINFERIOR if upper < +0.001 at all widths",
                               "FLOOR_CROSSED if lower > +0.001 at every width",
                               "INCONCLUSIVE otherwise"],
                  "secondary_equivalence": "two-sided 95% CI inside [-0.001, +0.001] (t=2.093, df=19)"},
        experiment_id=xid, notes="2 depths x 4 widths x 20 seeds = 160 cells; zero new teacher labels")
    expected = service.expected_membership(experiment)
    if len(expected) != 160:
        print(f"STOP: expected membership is {len(expected)}, not 160")
        return 3
    save(xid=xid, shared_ids_hash=hash_obj(ids), ids=len(ids),
         datasets={"8k": dataset_8k.id, "2k": dataset_2k.id},
         experiment={"id": experiment.id, "arms": len(experiment.arms), "cells": len(expected)},
         realized_nodes={"8k": arm8k["realized_nodes"], "2k": realized_2k,
                         "ratio": round(arm8k["realized_nodes"] / realized_2k, 3)})
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["freeze", "run"])
    args = parser.parse_args()
    S9.mkdir(parents=True, exist_ok=True)
    return {"freeze": step_freeze, "run": step_run}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
