"""S15: auxiliary geometry supervision — the deployed evaluator stays RAW-768 -> score.

    python tools/s15/s15_run.py manifest      # freeze the auxiliary target manifest + hash
    python tools/s15/s15_run.py preregister   # freeze the preregistration (weight + lattices + trigger)
    python tools/s15/s15_run.py experiment    # freeze the primary lattice (2 arms x 4 widths x 20 seeds)
    python tools/s15/s15_run.py run           # execute every queued cell
    python tools/s15/s15_run.py ablation      # iff the preregistered trigger fired: freeze + run it

Teacher and exam are S13/S14's CVS contract unchanged; the only difference between arms is whether
the RAW trunk is additionally trained to predict named deterministic geometry facts.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s15_common import (ABLATION_MODES, ABLATION_WIDTH, BATCH, CVS_SPEC_HASH, DATASET, EXAM_PROTOCOL,
                        LAB, MAX_UPDATES, NODE_BUDGET, SCALE_NAME, SCALE_NODES, SEEDS, S15, STATE,
                        WIDTHS, aux_manifest, freeze_hash, save, state, width_ladder)

sys.path.insert(0, str(LAB))

from cvslab.hashing import hash_obj, read_jsonl
from cvslab.schemas import Ablation, Experiment, Run, RunStatus
from cvslab.service import LabService
from cvslab.store import LabError, Store
from cvslab.targets import TargetSpec

MANIFEST = S15 / "s15-aux-manifest.json"
PREREG = S15 / "s15-preregistration.json"
DECLARATION = ("S15 arm: identical RAW-768 inputs, positions, score labels, split identities, "
               "optimizer and update count as its paired arm; the only difference is training-only "
               "auxiliary geometry supervision, discarded at inference")
CVS_PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def write_artifact(path: pathlib.Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return hash_obj(payload)


def step_manifest() -> int:
    if MANIFEST.is_file():
        print("[cached] manifest")
        return 0
    digest = write_artifact(MANIFEST, aux_manifest())
    (S15 / "s15-aux-manifest.hash").write_text(digest + "\n", encoding="utf-8")
    print(f"auxiliary manifest: 21 families, 21 value targets + 84 bucket logits, {digest}", flush=True)
    return 0


def step_preregister() -> int:
    if "preregistration" in state():
        print("[cached] preregister")
        return 0
    calibration = json.loads((S15 / "s15-aux-weight-calibration.json").read_text(encoding="utf-8"))
    prereg = {
        "id": "s15-auxiliary-geometry",
        "version": 1,
        "status": "materialized before the lattice is frozen and before any outcome is inspected",
        "issue": "Mnehmos/chess-vision-studio-lab#42 (slice of #18)",
        "questions": ["does explicit semantic supervision help a RAW evaluator learn a better static "
                      "representation even when no semantic features are supplied at inference?",
                      "if it helps, does it reduce the learned capacity needed for a given quality?",
                      "which named geometry families provide useful training signal?"],
        "contract": {"teacher": f"CVS-4k over {DATASET} (S13/S14 unchanged; TargetSpec "
                                f"{CVS_SPEC_HASH[:19]}… asserted against X0008's sealed 4k arm)",
                     "exam": f"{EXAM_PROTOCOL} CVS-DEEP 400k over the 200 held-out identities",
                     "student": f"RAW-768 -> H clipped-ReLU -> score; MAX_UPDATES {MAX_UPDATES} x "
                                f"BATCH {BATCH} = {MAX_UPDATES * BATCH} presentations for every cell; "
                                "cold start; T0004-derived optimizer/LR/init/K/LAMBDA"},
        "arms": {"RAW-SCORE": {"AUX": "none", "role": "control: scalar score supervision only"},
                 "RAW-SCORE+AUX-GEO": {"AUX": "geo", "AUX_WEIGHT": calibration["chosen_aux_weight"],
                                       "role": "treatment: the same trunk and score head, plus "
                                               "training-only heads predicting the auxiliary targets"}},
        "auxiliary_targets": {"manifest": "tools/s15/s15-aux-manifest.json",
                              "hash": (S15 / "s15-aux-manifest.hash").read_text(encoding="utf-8").strip(),
                              "families": 21, "value_targets": 21, "bucket_logits": 84,
                              "never_depends_on": ["teacher search", "the target score", "outcome",
                                                   "best move", "evaluation split membership"]},
        "loss": {"score": "MSE(sigmoid(pred/K), target) — unchanged",
                 "auxiliary": "mean(value MSE over families) + mean(4-class cross-entropy over "
                              "families), each averaged so no family dominates by column count",
                 "total": "score + AUX_WEIGHT * auxiliary",
                 "weight": calibration["chosen_aux_weight"],
                 "weight_provenance": {"artifact": "tools/s15/s15-aux-weight-calibration.json",
                                       "hash": calibration and (S15 / "s15-aux-weight-calibration.hash")
                                       .read_text(encoding="utf-8").strip(),
                                       "rule": "25% of the score loss's trunk gradient at "
                                               "initialisation, measured on TRAINING rows only"},
                 "schedule": "fixed for the whole run; no per-seed or per-width tuning"},
        "capacity": {"widths": list(WIDTHS), "primary_cells": 2 * len(WIDTHS) * len(SEEDS),
                     "ladder": width_ladder(),
                     "accounting": "total = persistent inference parameters + training-only "
                                   "auxiliary parameters; both reported, neither hidden"},
        "primary_analysis": {
            "contrast": "per width, paired by seed: RAW-SCORE+AUX-GEO - RAW-SCORE on E0004 test_loss, "
                        "separate one-sided 95% exact-t bounds at that width's df",
            "metrics": ["test_loss", "cp_mae", "sign_agreement (all recorded per run)",
                        "train/validation/test gap", "persistent inference parameter count",
                        "training-only auxiliary parameter count", "train wall time",
                        "inference throughput after auxiliary heads are disabled",
                        "auxiliary prediction quality by named family"],
            "no_pooling": "widths reported separately; no global winner",
            "caveat": "better auxiliary prediction is not evidence of better chess judgment unless "
                      "the score exam also improves"},
        "family_ablation": {"trigger": "run iff the H=1 primary contrast's one-sided UPPER bound is "
                                       "below zero (auxiliary strictly better at the smallest width). "
                                       "The rule is frozen here, before outcomes",
                            "matrix": {"modes": list(ABLATION_MODES), "width": ABLATION_WIDTH,
                                       "seeds": len(SEEDS), "cells": len(ABLATION_MODES) * len(SEEDS)},
                            "reading": "per family group, (full - removed) change in score loss at "
                                       "the sentinel width; negative findings preserved",
                            "lattice": "a SECOND experiment, frozen only if the trigger fires",
                            "not_run_is_a_result": "if the trigger does not fire, no ablation runs "
                                                   "and the absence is recorded as the rule's outcome"},
        "fairness": ["identical RAW-768 inputs in both arms", "identical score labels, positions and "
                     "split identities", "identical persistent score topology and optimizer",
                     "identical update count", "no inference-time inputs added to the treatment arm",
                     "training-only parameters and extra training FLOPs reported",
                     "no per-arm LR/optimizer tuning", "no teacher-authority change"],
        "guardrails": ["no search or game-strength claims", "no GEO/HYBRID inputs at inference",
                       "no active learning", "no combinatorial ablations",
                       "geometry is target-side supervision only"],
        "stop_conditions": ["the D0036 TargetSpec hash differs from X0008's sealed 4k arm",
                            "the auxiliary targets do not align with the encoded training rows",
                            "the geometry registry hash differs from the manifest's pin",
                            "membership fails"],
    }
    digest = write_artifact(PREREG, prereg)
    (S15 / "s15-preregistration.hash").write_text(digest + "\n", encoding="utf-8")
    save(preregistration={"file": "tools/s15/s15-preregistration.json", "hash": digest})
    print(f"preregistration {digest}", flush=True)
    return 0


def _spec_and_recipe(service: LabService, name: str, mode: str, weight: float):
    """One recipe per (AUX mode, weight): the switch registry makes those recipe-fixed, so an arm
    cannot override them — the recipe IS the arm's supervision contract."""
    spec = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                      producer=CVS_PRODUCER, budget={"nodeBudget": 4_000}, value_path=("scoreCpStm",),
                      pov="stm", k=256.0, lam=1.0)
    if spec.spec_hash() != CVS_SPEC_HASH:
        raise LabError("the reconstructed CVS-4k spec does not hash to the sealed value")
    control = service.store.get("T0004")
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    recipe = service.create_training_recipe(
        name=f"{name}-{mode}", params={**base, "EPOCHS": 1, "BATCH": BATCH,
                                       "MAX_UPDATES": MAX_UPDATES, "AUX": mode,
                                       "AUX_WEIGHT": float(weight)},
        target_spec=spec, description=f"S15 {name} {mode}: RAW-768 -> score, auxiliary weight {weight}")
    return spec, recipe


def _unique_name(service: LabService, base: str) -> str:
    taken = {a.baseline_name for a in service.store.list("A", verify=True, kind=Ablation)
             if a.baseline_name}
    if base not in taken:
        return base
    index = 1
    while f"{base}_R{index}" in taken:
        index += 1
    return f"{base}_R{index}"


def _next_xid(service: LabService) -> str:
    while True:
        candidate = service.store.next_id("X")
        if not [a for a in service.store.list("A", verify=True, kind=Ablation)
                if a.experiment_id == candidate]:
            return candidate


def _freeze(service: LabService, *, name: str, conditions: list[tuple[str, int, str, float]],
            notes: str) -> tuple[str, dict]:
    """conditions: (label, H, AUX mode, AUX_WEIGHT) — one ablation per condition."""
    dataset = service.store.get(DATASET)
    split = next(entry for entry in dataset.splits if entry.name == "train")
    xid = _next_xid(service)
    slug = name.lower().replace(" ", "-")
    recipes = {}
    for mode, weight in sorted({(mode, weight) for _l, _h, mode, weight in conditions}):
        spec, recipe = _spec_and_recipe(service, slug, mode, weight)
        recipes[(mode, weight)] = (spec, recipe)
    per_arm: dict[str, list[str]] = {}
    for label, hidden, mode, weight in conditions:
        spec, recipe = recipes[(mode, weight)]
        config = {"INPUT": "RAW", "H": hidden}
        baseline = service.register_baseline(
            name=_unique_name(service, f"S15_{label.upper().replace(chr(45), chr(95))}_H{hidden}"),
            dataset_id=DATASET,
            training_recipe_id=recipe.id, eval_protocol_id=EXAM_PROTOCOL, model_config=config,
            supervision_divergence=DECLARATION, experiment_id=xid,
            notes=f"S15 {label} H={hidden} AUX={mode} weight={weight}")
        per_arm.setdefault(label, []).append(baseline.id)
    arms_payload = []
    for label, ablation_ids in per_arm.items():
        mode = next(mode for lab, _h, mode, _w in conditions if lab == label)
        weight = next(weight for lab, _h, _m, weight in conditions if lab == label)
        spec, recipe = recipes[(mode, weight)]
        arms_payload.append({
            "arm_id": f"S15-{label}", "scale": SCALE_NAME, "scale_nodes": SCALE_NODES,
            "node_budget": NODE_BUDGET, "prefix_size": split.count, "realized_nodes": SCALE_NODES,
            "dataset_id": DATASET, "dataset_manifest_hash": dataset.manifest_hash,
            "training_recipe_id": recipe.id, "recipe_hash": recipe.recipe_hash,
            "train_target_spec_hash": spec.spec_hash(), "record_ids_hash": split.record_ids_hash,
            "ablations": ablation_ids})
    experiment = service.create_experiment(
        name=name, preregistration_hash=(S15 / "s15-preregistration.hash").read_text(
            encoding="utf-8").strip(),
        reference_unit_nodes=37_285_491, scales={SCALE_NAME: SCALE_NODES},
        node_budgets=[NODE_BUDGET], arms=arms_payload, eval_protocol_id=EXAM_PROTOCOL,
        widths=sorted({hidden for _l, hidden, _m, _w in conditions}), seeds=list(SEEDS),
        source_id="S0012", normalization_id="N0014",
        candidate_universe_hash=json.loads((LAB / "tools" / "s12" / "s12-state.json")
                                           .read_text(encoding="utf-8"))["universe"]["universe_hash"],
        order_seed=20260920,
        order_hash=json.loads((LAB / "tools" / "s12" / "s12-state.json")
                              .read_text(encoding="utf-8"))["universe"]["order_hash"],
        analysis={"design": notes,
                  "primary": "per width, paired by seed: AUX-GEO - RAW-SCORE on E0004 test_loss",
                  "accounting": "training-only auxiliary parameters reported separately"},
        experiment_id=xid, notes=notes)
    expected = service.expected_membership(experiment)
    if len(expected) != len(conditions) * len(SEEDS):
        raise LabError(f"membership {len(expected)} != {len(conditions) * len(SEEDS)}")
    return xid, {"id": experiment.id, "arms": len(experiment.arms), "cells": len(expected),
                 "conditions": [{"label": l, "H": h, "AUX": m, "weight": w} for l, h, m, w in conditions]}


def step_experiment() -> int:
    if "experiment" in state():
        print("[cached] experiment")
        return 0
    service = svc()
    weight = float(state()["calibration"]["aux_weight"])
    conditions = ([(f"RAW-SCORE", h, "none", 0.0) for h in WIDTHS]
                  + [(f"AUX-GEO", h, "geo", weight) for h in WIDTHS])
    xid, record = _freeze(service, name="S15 auxiliary geometry supervision",
                          conditions=conditions,
                          notes="2 arms x 4 widths x 20 seeds; the deployed evaluator is RAW-768 -> "
                                "score in both arms")
    save(xid=xid, experiment=record)
    print(f"experiment {record['id']}: {record['arms']} arms, {record['cells']} cells", flush=True)
    return 0


def _execute(service: LabService, xid: str, expected: int) -> int:
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != xid or run.status.value != "RUNNING":
            continue
        run.status = RunStatus.INVALID
        run.error = "orphaned RUNNING: the executing driver died; retried by a later driver"
        service.store.update_run(run)
        print(f"  orphaned {run.id} marked INVALID (dead driver)", flush=True)
    by_cell: dict = {}
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id == xid:
            by_cell.setdefault((run.ablation_id, run.seed), []).append(run)
    for (ablation_id, seed), runs in sorted(by_cell.items()):
        if len(runs) == 1 and runs[0].status.value == "INVALID":
            retry = service.queue_runs(ablation_id, seeds=[seed])[0]
            print(f"  retry {ablation_id} seed {seed} (crash: {runs[0].error})", flush=True)
    counts: dict[str, int] = {}
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != xid:
            continue
        if run.status.value in ("COMPLETED", "FAILED", "INVALID"):
            counts[run.status.value] = counts.get(run.status.value, 0) + 1
            continue
        finished = service.execute_run(run.id)
        counts[finished.status.value] = counts.get(finished.status.value, 0) + 1
        if finished.status.value != "COMPLETED":
            print(f"  {run.id} {finished.status.value}: {finished.error}", flush=True)
    print("run statuses:", counts, flush=True)
    return 0


def step_run() -> int:
    st = state()
    service = svc()
    queued = st.get("queued")
    if not queued:
        experiment = service.store.get_as(st["xid"], Experiment)
        queued = {ablation_id: [run.id for run in service.queue_runs(ablation_id, seeds=SEEDS)]
                  for arm in experiment.arms for ablation_id in arm.ablations}
        save(queued=queued)
        print(f"queued {sum(len(v) for v in queued.values())} runs", flush=True)
    return _execute(service, st["xid"], st["experiment"]["cells"])


def step_ablation() -> int:
    """Freeze and run the family-ablation matrix IFF the preregistered trigger fired."""
    st = state()
    service = svc()
    if "ablation" in st:
        print("[cached] ablation")
        return _execute(service, st["ablation"]["xid"], st["ablation"]["cells"])
    result_path = S15 / "s15-result.json"
    if not result_path.is_file():
        raise LabError("run the primary analysis first: the trigger reads its H=1 contrast")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    contrast = result["per_width"][str(ABLATION_WIDTH)]["contrast"]
    fired = contrast["one_sided_bounds"][1] < 0
    record = {"trigger": "H=1 one-sided upper bound < 0 (auxiliary strictly better)",
             "h1_contrast": contrast, "fired": fired}
    if not fired:
        record["outcome"] = "not run: the trigger did not fire; this absence is the rule's outcome"
        save(ablation=record)
        print(f"family ablation NOT triggered (H=1 upper bound {contrast['one_sided_bounds'][1]:+.6f})",
              flush=True)
        return 0
    weight = float(st["calibration"]["aux_weight"])
    conditions = [(f"AUX-{mode}", ABLATION_WIDTH, mode, weight) for mode in ABLATION_MODES]
    xid, frozen = _freeze(service, name="S15 family ablation at H=1", conditions=conditions,
                          notes="one named family group removed per arm, at the sentinel width H=1")
    queued = {ablation_id: [run.id for run in service.queue_runs(ablation_id, seeds=SEEDS)]
              for arm in service.store.get_as(frozen["id"], Experiment).arms
              for ablation_id in arm.ablations}
    record.update({"xid": xid, "experiment": frozen, "cells": frozen["cells"],
                   "queued": sum(len(v) for v in queued.values())})
    save(ablation=record)
    print(f"family ablation triggered: {frozen['id']} with {frozen['cells']} cells", flush=True)
    return _execute(service, xid, frozen["cells"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["manifest", "preregister", "experiment", "run", "ablation"])
    args = parser.parse_args()
    return {"manifest": step_manifest, "preregister": step_preregister,
            "experiment": step_experiment, "run": step_run, "ablation": step_ablation}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
