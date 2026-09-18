"""S17 driver: freeze the RAW scaling ladder and its undertraining control, then execute.

    python tools/s17/s17_run.py preregister   # ladder + the one scaling rule + analysis plan
    python tools/s17/s17_run.py experiment    # freeze the lattice (primary + control arms)
    python tools/s17/s17_run.py run           # execute every cell
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s17_common import (BATCH, CONTROL_FACTOR, CONTROL_UPDATES, CONTROL_WIDTHS, CVS_SPEC_HASH,
                        DATASET, EXAM_PROTOCOL, LAB, LADDER, NODE_BUDGET, PRIMARY_UPDATES,
                        SCALE_NAME, SCALE_NODES, S17, ladder_table, rng_contract, save,
                        seeds_for, state)

sys.path.insert(0, str(LAB))

from cvslab.hashing import hash_obj
from cvslab.schemas import Ablation, Experiment, Run, RunStatus
from cvslab.service import LabService
from cvslab.store import LabError, Store
from cvslab.targets import TargetSpec

PREREG = S17 / "s17-preregistration.json"
DECLARATION = ("S17 rung: identical teacher, positions, rows, exam, optimizer, initialization and "
               "sampling contract as every other rung; only the width and (for the control arm) the "
               "single predeclared update multiplier change")


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def write_artifact(path: pathlib.Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return hash_obj(payload)


def step_preregister() -> int:
    if "preregistration" in state():
        print("[cached] preregister")
        return 0
    table = ladder_table()
    prereg = {
        "id": "s17-scaling-ladder",
        "version": 1,
        "status": "materialized before the lattice is frozen and before any S17 outcome exists",
        "issue": "Mnehmos/chess-vision-studio-lab#17 (the axis S14, S15 and S16 all pointed at)",
        "question": "how much learned capacity does the raw 768-input evaluator need before more "
                    "parameters stop buying useful capability — and is the turn-up that S14 saw at "
                    "fixed student compute a capacity effect or an undertraining effect?",
        "contract": {"teacher": f"CVS-4k over {DATASET} (the S13-S16 contract unchanged; TargetSpec "
                                f"{CVS_SPEC_HASH[:19]}… asserted against X0008's sealed 4k arm)",
                     "exam": f"{EXAM_PROTOCOL} CVS-DEEP 400k over the 200 held-out identities",
                     "input": "RAW-768 for every rung",
                     "primary_regime": f"MAX_UPDATES {PRIMARY_UPDATES} x BATCH {BATCH} = "
                                       f"{PRIMARY_UPDATES * BATCH} presentations at EVERY width "
                                       "(the frozen S13-S16 student-compute contract)",
                     "control_regime": f"MAX_UPDATES {CONTROL_UPDATES} = {CONTROL_FACTOR}x the "
                                       f"primary updates, applied to H={list(CONTROL_WIDTHS)} only",
                     "random_streams": rng_contract()},
        "ladder": table,
        "control_rule": {
            "rule": f"one constant multiplier, {CONTROL_FACTOR}x, identical for all three control "
                    f"widths; no per-width tuning, no tuning after outcomes",
            "anchor": "H=128 is the smallest control width, so the report can see whether extra "
                      "compute merely rescues the largest models or moves the curve more generally",
            "why_not_proportional_to_parameters": "updates proportional to parameter count would "
                                                  "multiply H=512's run time by ~128 (days); a "
                                                  "constant factor is the largest predeclared "
                                                  "multiplier that keeps the control arm inside a "
                                                  "few hours",
            "limitation_stated_up_front": "if 4x does not rescue a width, the evidence "
                                          "distinguishes 'capacity saturated' from 'compute helped "
                                          "less than 4x', NOT from 'no amount of compute helps'",
        },
        "analysis": {
            "primary_curve": "test_loss against exact parameter count and against presentations per "
                             "parameter, with seed uncertainty at every rung",
            "thresholds": "first useful rung (best loss below the H=1 rung with bounds excluding "
                          "zero), the knee (the rung with the largest marginal gain), the "
                          "diminishing-return region (marginal gains whose bounds include zero), "
                          "and every non-monotonic step",
            "marginal_gains": "per rung, the paired-by-seed loss change vs the previous rung, per "
                              "added parameter, with exact one-sided bounds",
            "undertraining_test": "for H=128/256/512: control vs primary at the same width, paired "
                                  "by seed; a control rung that recovers the knee's loss indicates "
                                  "undertraining, one that does not indicates capacity saturation "
                                  "(within the 4x envelope)",
            "slices": "phase and material slices at the knee and at H=512, computed from the exam "
                      "FENs",
            "cost": "train wall time, serialized bytes, inference throughput per width",
            "no_ranking": "no pooled score; the curve and its marginal gains are the result",
        },
        "acceptance": {"every_width_same_contract": True,
                       "exact_counts_generated_by_lab": "cvslab.families.parameter_shapes",
                       "marginal_gain_reported": True,
                       "non_monotonic_rungs_kept_visible": True,
                       "gen10_is_comparison_not_truth": "H=256 (~197K parameters) approximates the "
                                                        "historical scale as a comparison point only"},
        "guardrails": ["no new labels", "no teacher change", "no per-width tuning",
                       "no optimizer change", "no geometry or auxiliary supervision in this study",
                       "negative and non-monotonic results preserved"],
        "stop_conditions": ["the D0036 TargetSpec hash differs from X0008's sealed 4k arm",
                            "a rung cannot train or evaluate", "membership fails"],
    }
    digest = write_artifact(PREREG, prereg)
    (S17 / "s17-preregistration.hash").write_text(digest + "\n", encoding="utf-8")
    save(preregistration={"file": "tools/s17/s17-preregistration.json", "hash": digest})
    print(f"preregistration {digest}", flush=True)
    return 0


def _unique_name(service: LabService, base: str) -> str:
    taken = {a.baseline_name for a in service.store.list("A", verify=True, kind=Ablation)
             if a.baseline_name}
    if base not in taken:
        return base
    index = 1
    while f"{base}_R{index}" in taken:
        index += 1
    return f"{base}_R{index}"


def step_experiment() -> int:
    if "experiment" in state():
        print("[cached] experiment")
        return 0
    service = svc()
    spec = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                      producer="36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838",
                      budget={"nodeBudget": 4_000}, value_path=("scoreCpStm",), pov="stm",
                      k=256.0, lam=1.0)
    if spec.spec_hash() != CVS_SPEC_HASH:
        raise LabError("the reconstructed CVS-4k spec does not hash to the sealed value")
    dataset = service.store.get(DATASET)
    split = next(entry for entry in dataset.splits if entry.name == "train")
    control = service.store.get("T0004")
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    # one recipe per regime: MAX_UPDATES is recipe-fixed, and the control arm is one rule
    recipes = {regime: service.create_training_recipe(
        name=f"s17-{regime}", params={**base, "EPOCHS": 1, "BATCH": BATCH, "MAX_UPDATES": updates},
        target_spec=spec, description=f"S17 {regime}: RAW ladder, {updates} updates x {BATCH}")
        for regime, updates in (("primary", PRIMARY_UPDATES), ("control", CONTROL_UPDATES))}
    while True:
        xid = service.store.next_id("X")
        if not [a for a in service.store.list("A", verify=True, kind=Ablation)
                if a.experiment_id == xid]:
            break
    # Arms are partitioned by SEED POLICY: per-arm seeds apply to every ablation in the arm, so
    # the small-width rungs keep the S13-S16 20-seed list while the large rungs use the
    # predeclared 10-seed policy. All three arms share one (scale, node_budget) cell.
    arm_specs = (
        ("primary-small", "primary", tuple(h for h in LADDER if h <= 32), seeds_for(1)),
        ("primary-large", "primary", tuple(h for h in LADDER if h > 32), seeds_for(64)),
        ("control-large", "control", CONTROL_WIDTHS, seeds_for(128)),
    )
    arms_payload, conditions = [], {}
    for arm_id, regime, widths, arm_seeds in arm_specs:
        ablation_ids = {}
        for hidden in widths:
            updates = PRIMARY_UPDATES if regime == "primary" else CONTROL_UPDATES
            baseline = service.register_baseline(
                name=_unique_name(service, f"S17_{arm_id.upper().replace('-', '_')}_H{hidden}"),
                dataset_id=DATASET, training_recipe_id=recipes[regime].id,
                eval_protocol_id=EXAM_PROTOCOL, model_config={"INPUT": "RAW", "H": hidden},
                supervision_divergence=DECLARATION, experiment_id=xid,
                notes=f"S17 {arm_id} H={hidden} ({updates} updates)")
            ablation_ids[hidden] = baseline.id
        conditions[arm_id] = {str(k): v for k, v in ablation_ids.items()}
        arms_payload.append({
            "arm_id": f"S17-{arm_id}", "scale": SCALE_NAME, "scale_nodes": SCALE_NODES,
            "node_budget": NODE_BUDGET, "prefix_size": split.count, "realized_nodes": SCALE_NODES,
            "dataset_id": DATASET, "dataset_manifest_hash": dataset.manifest_hash,
            "training_recipe_id": recipes[regime].id,
            "recipe_hash": recipes[regime].recipe_hash,
            "train_target_spec_hash": spec.spec_hash(),
            "record_ids_hash": split.record_ids_hash,
            "ablations": [ablation_ids[hidden] for hidden in widths],
            "seeds": list(arm_seeds)})
    experiment = service.create_experiment(
        name="S17 raw scaling ladder: 771 to 394,241 parameters at fixed student compute",
        preregistration_hash=(S17 / "s17-preregistration.hash").read_text(encoding="utf-8").strip(),
        reference_unit_nodes=37_285_491, scales={SCALE_NAME: SCALE_NODES},
        node_budgets=[NODE_BUDGET], arms=arms_payload, eval_protocol_id=EXAM_PROTOCOL,
        widths=list(LADDER), seeds=list(range(20)),
        source_id="S0012", normalization_id="N0014",
        candidate_universe_hash=json.loads((LAB / "tools" / "s12" / "s12-state.json")
                                           .read_text(encoding="utf-8"))["universe"]["universe_hash"],
        order_seed=20260920,
        order_hash=json.loads((LAB / "tools" / "s12" / "s12-state.json")
                              .read_text(encoding="utf-8"))["universe"]["order_hash"],
        analysis={"design": "RAW-768 ladder H=1..512 at the frozen S13-S16 contract, plus a "
                            "4x-updates control at H=128/256/512",
                  "primary_curve": "test_loss vs exact parameter count and vs presentations",
                  "undertraining_test": "control vs primary at the same width, paired by seed",
                  "marginal_gains": "per rung, per added parameter, with exact bounds"},
        experiment_id=xid, notes="10 primary rungs at 194,560 presentations + 3 control rungs at "
                                 "778,240 presentations; no other variable changes")
    expected = service.expected_membership(experiment)
    total = sum(len(arm_seeds) for _a, _r, widths, arm_seeds in arm_specs for _h in widths)
    if len(expected) != total:
        raise LabError(f"membership {len(expected)} != {total}")
    save(xid=xid, experiment={"id": experiment.id, "arms": len(experiment.arms),
                              "cells": len(expected)}, conditions=conditions,
         arm_seeds={arm_id: list(arm_seeds) for arm_id, _r, _w, arm_seeds in arm_specs})
    print(f"experiment {experiment.id}: {len(experiment.arms)} arms, {len(expected)} cells", flush=True)
    return 0


def step_run() -> int:
    st = state()
    service = svc()
    queued = st.get("queued")
    if not queued:
        experiment = service.store.get_as(st["xid"], Experiment)
        queued = {}
        for arm_id, by_hidden in st["conditions"].items():
            seeds = st["arm_seeds"][arm_id]
            for hidden, ablation_id in by_hidden.items():
                queued[ablation_id] = [run.id for run in service.queue_runs(ablation_id, seeds=seeds)]
        save(queued=queued)
        print(f"queued {sum(len(v) for v in queued.values())} runs", flush=True)
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != st["xid"] or run.status.value != "RUNNING":
            continue
        run.status = RunStatus.INVALID
        run.error = "orphaned RUNNING: the executing driver died; retried by a later driver"
        service.store.update_run(run)
        print(f"  orphaned {run.id} marked INVALID (dead driver)", flush=True)
    by_cell: dict = {}
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id == st["xid"]:
            by_cell.setdefault((run.ablation_id, run.seed), []).append(run)
    for (ablation_id, seed), runs in sorted(by_cell.items()):
        if len(runs) == 1 and runs[0].status.value == "INVALID":
            service.queue_runs(ablation_id, seeds=[seed])
            print(f"  retry {ablation_id} seed {seed} (crash: {runs[0].error})", flush=True)
    counts: dict[str, int] = {}
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != st["xid"]:
            continue
        if run.status.value in ("COMPLETED", "FAILED", "INVALID"):
            counts[run.status.value] = counts.get(run.status.value, 0) + 1
            continue
        finished = service.execute_run(run.id)
        counts[finished.status.value] = counts.get(finished.status.value, 0) + 1
        if finished.status.value != "COMPLETED":
            print(f"  {run.id} {finished.status.value}: {finished.error}", flush=True)
    save(run_statuses=counts)
    print("run statuses:", counts, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["preregister", "experiment", "run"])
    args = parser.parse_args()
    return {"preregister": step_preregister, "experiment": step_experiment, "run": step_run}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
