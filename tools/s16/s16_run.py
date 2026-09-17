"""S16 driver: freeze the minimal teacher x representation matrix, then execute it.

    python tools/s16/s16_run.py preregister   # matrix + selection rules + analysis plan
    python tools/s16/s16_run.py experiment    # freeze the lattice (4 arms x 2 capacities x 20 seeds)
    python tools/s16/s16_run.py run           # execute every cell
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s16_common import (BATCH, CAPACITIES, EXAM_CVS, EXAM_SF, LAB, MAX_UPDATES, REPRESENTATIONS,
                        S16, SEEDS, SELECTION_RULES, TEACHERS, cell_manifest, economics, save, state)

sys.path.insert(0, str(LAB))

from cvslab.hashing import hash_obj
from cvslab.schemas import Ablation, Experiment, Run, RunStatus
from cvslab.service import LabService
from cvslab.store import LabError, Store
from cvslab.targets import TargetSpec

PREREG = S16 / "s16-preregistration.json"
DECLARATION = ("S16 cell: identical positions, rows, split identities, optimizer, update count and "
               "dual exams as every other cell; only the declared factors (teacher authority, "
               "representation family, capacity) differ")


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def write_artifact(path: pathlib.Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return hash_obj(payload)


def cvsspec() -> TargetSpec:
    return TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                      producer="36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838",
                      budget={"nodeBudget": 4_000}, value_path=("scoreCpStm",), pov="stm",
                      k=256.0, lam=1.0)


def sf_spec() -> TargetSpec:
    return TargetSpec(family="oracle_cp", authority="oracle.stockfish.18.cold",
                      producer="c86215fa1977d53b82ed854540a4c7b025be4cd042276c85ba3de53fb9118911",
                      budget={"nodes": 32_000}, value_path=("scoreCpStm",), pov="stm",
                      k=256.0, lam=1.0)


def step_preregister() -> int:
    if "preregistration" in state():
        print("[cached] preregister")
        return 0
    manifest = cell_manifest()
    prereg = {
        "id": "s16-teacher-representation",
        "version": 1,
        "status": "materialized before the lattice is frozen and before any S16 outcome exists",
        "issue": "Mnehmos/chess-vision-studio-lab#43 (factorial capstone of the static phase; #18/#20)",
        "questions": ["does explicit semantic representation reduce the need for a stronger/more "
                      "expensive teacher?",
                      "does Stockfish supervision help RAW and HYBRID students equally, or is the "
                      "benefit representation-dependent?",
                      "are teacher quality and explicit structure substitutes, complements, or "
                      "largely independent resources?"],
        "frozen_inputs": {"S13": "two pinned teacher authorities + the dual exam contract "
                                 "(X0010, cold: E0006/D0043)",
                          "S14": "RAW/GEO/HYBRID identities and parameter accounting (X0012)",
                          "S15": "training-only auxiliary supervision (X0014; not carried here)"},
        "selection_rules": SELECTION_RULES,
        "matrix": manifest,
        "analysis": {
            "per_exam": "every quantity is reported separately for CVS-DEEP (E0004) and SF-DEEP "
                        "(E0006/D0043); the two are never pooled",
            "teacher_effect": "within each representation and capacity: SF - CVS, paired by seed",
            "representation_effect": "within each teacher and capacity: HYBRID - RAW, paired by seed",
            "interaction": "difference-in-differences per capacity and exam: "
                           "(SF - CVS | HYBRID) - (SF - CVS | RAW), paired by seed, exact one-sided "
                           "95% bounds at that contrast's df",
            "hypotheses": {"substitutes": "interaction > 0 for SF-better (semantics shrink the "
                                          "teacher benefit)",
                           "complements": "interaction < 0 (the teacher helps HYBRID more)",
                           "independent": "interval contains 0 with each main effect intact",
                           "authority_specific": "the sign differs between the two exams"},
            "no_ranking": "no pooled overall score and no universal winner",
            "economics": "per-cell teacher nodes/ms, student presentations, persistent parameters, "
                          "training-only parameters, plus measured inference throughput"},
        "guardrails": ["no new teacher-budget tuning", "no new geometry vocabulary",
                       "no per-cell optimizer tuning", "no game or search claims",
                       "no pooled score across exams", "no cell selection after outcomes",
                       "no factorial change after partial results",
                       "null, heterogeneous or inconvenient interactions are preserved"],
        "stop_conditions": ["the two teacher datasets do not share their record ids",
                            "a lattice cell cannot train or evaluate",
                            "the dual-exam pass cannot reproduce the in-lab CVS metrics",
                            "membership fails"],
    }
    digest = write_artifact(PREREG, prereg)
    (S16 / "s16-preregistration.hash").write_text(digest + "\n", encoding="utf-8")
    save(preregistration={"file": "tools/s16/s16-preregistration.json", "hash": digest})
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
    manifest = cell_manifest()
    # fairness precondition: both teacher datasets carry the same rows in the same order
    from cvslab.hashing import read_jsonl
    order = {}
    for teacher, info in TEACHERS.items():
        dataset = service.store.get(info["dataset"])
        split = next(entry for entry in dataset.splits if entry.name == "train")
        order[teacher] = split.record_ids_hash
    if len(set(order.values())) != 1:
        raise LabError(f"teacher datasets do not share their record ids: {order}")
    cvs, sf = cvsspec(), sf_spec()
    for spec, info in ((cvs, TEACHERS["CVS-4k"]), (sf, TEACHERS["SF-32k"])):
        if spec.spec_hash() != info["spec_hash"]:
            raise LabError(f"reconstructed spec {spec.spec_hash()[:19]}… != frozen {info['spec_hash'][:19]}…")
    control = service.store.get("T0004")
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    recipes = {teacher: service.create_training_recipe(
        name=f"s16-{teacher.lower()}",
        params={**base, "EPOCHS": 1, "BATCH": BATCH, "MAX_UPDATES": MAX_UPDATES},
        target_spec=spec, description=f"S16 {teacher} teacher contract (frozen by S13)")
        for teacher, spec in (("CVS-4k", cvs), ("SF-32k", sf))}
    while True:
        xid = service.store.next_id("X")
        if not [a for a in service.store.list("A", verify=True, kind=Ablation)
                if a.experiment_id == xid]:
            break
    conditions = {}
    arms_payload = []
    for teacher, info in TEACHERS.items():
        dataset = service.store.get(info["dataset"])
        split = next(entry for entry in dataset.splits if entry.name == "train")
        for representation in REPRESENTATIONS:
            ablation_ids = {}
            for hidden in CAPACITIES:
                baseline = service.register_baseline(
                    name=_unique_name(service, f"S16_{teacher.upper().replace('-', '_')}_{representation}_H{hidden}"),
                    dataset_id=info["dataset"], training_recipe_id=recipes[teacher].id,
                    eval_protocol_id=EXAM_CVS["protocol"],
                    model_config={"INPUT": representation, "H": hidden},
                    supervision_divergence=DECLARATION, experiment_id=xid,
                    notes=f"S16 {teacher} x {representation} H={hidden}")
                ablation_ids[hidden] = baseline.id
            conditions[f"{teacher}|{representation}"] = ablation_ids
            arms_payload.append({
                "arm_id": f"S16-{teacher}-{representation}", "scale": info["scale"],
                "scale_nodes": info["scale_nodes"], "node_budget": info["node_budget"],
                "prefix_size": split.count, "realized_nodes": info["scale_nodes"],
                "dataset_id": info["dataset"], "dataset_manifest_hash": dataset.manifest_hash,
                "training_recipe_id": recipes[teacher].id,
                "recipe_hash": recipes[teacher].recipe_hash,
                "train_target_spec_hash": info["spec_hash"],
                "record_ids_hash": split.record_ids_hash,
                "ablations": [ablation_ids[hidden] for hidden in CAPACITIES]})
    experiment = service.create_experiment(
        name="S16 teacher x representation: the minimal factorial matrix",
        preregistration_hash=(S16 / "s16-preregistration.hash").read_text(encoding="utf-8").strip(),
        reference_unit_nodes=37_285_491,
        scales={info["scale"]: info["scale_nodes"] for info in TEACHERS.values()},
        node_budgets=[info["node_budget"] for info in TEACHERS.values()],
        declared_cells=manifest["declared_cells"], arms=arms_payload,
        eval_protocol_id=EXAM_CVS["protocol"], widths=list(CAPACITIES), seeds=list(SEEDS),
        source_id="S0012", normalization_id="N0014",
        candidate_universe_hash=json.loads((LAB / "tools" / "s12" / "s12-state.json")
                                           .read_text(encoding="utf-8"))["universe"]["universe_hash"],
        order_seed=20260920,
        order_hash=json.loads((LAB / "tools" / "s12" / "s12-state.json")
                              .read_text(encoding="utf-8"))["universe"]["order_hash"],
        analysis={"design": "2 teacher authorities x 2 representation families x 2 capacities x 20 "
                            "seeds; identical rows, optimizer, update count and exam pair",
                  "teacher_effect": "per representation and capacity: SF - CVS on each exam",
                  "representation_effect": "per teacher and capacity: HYBRID - RAW on each exam",
                  "interaction": "difference-in-differences (SF-CVS | HYBRID) - (SF-CVS | RAW)",
                  "exams": "CVS-DEEP (E0004) and SF-DEEP (E0006/D0043), never pooled"},
        experiment_id=xid, notes="minimal factorial selected from S13/S14 evidence by predeclared "
                                 "rules; S15's auxiliary arm excluded (outside these capacities)")
    expected = service.expected_membership(experiment)
    total = len(arms_payload) * len(CAPACITIES) * len(SEEDS)
    if len(expected) != total:
        raise LabError(f"membership {len(expected)} != {total}")
    save(xid=xid, experiment={"id": experiment.id, "arms": len(experiment.arms),
                              "cells": len(expected)}, conditions={
        key: {str(hidden): ablation for hidden, ablation in ids.items()}
        for key, ids in conditions.items()}, economics=economics())
    print(f"experiment {experiment.id}: {len(experiment.arms)} arms, {len(expected)} cells", flush=True)
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
