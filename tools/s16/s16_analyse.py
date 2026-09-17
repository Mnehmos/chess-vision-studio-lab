"""S16 analysis: main effects and the difference-in-differences interaction, per exam.

    python tools/s16/s16_analyse.py dual     # score every stored model on BOTH exams (validated)
    python tools/s16/s16_analyse.py report   # main effects + interactions + economics
    python tools/s16/s16_analyse.py seal     # seal the factorial with the result

The dual-exam pass is the S13 discipline: the lab's Run carries one exam (CVS-DEEP), so the SF-DEEP
instrument is applied to the stored weights here and believed only if the recomputed CVS numbers
reproduce every run's recorded metric exactly.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s16_common import CAPACITIES, EXAM_CVS, EXAM_SF, LAB, REPRESENTATIONS, SEEDS, S16, TEACHERS, save, state

sys.path.insert(0, str(LAB))

import numpy as np

import cvslab.data as data
import cvslab.nnue as nnue
from cvslab.funnel.campaign import paired_effect
from cvslab.hashing import hash_obj, sha256_file
from cvslab.schemas import Experiment, Run, RunStatus
from cvslab.service import LabService, protocol_target_spec
from cvslab.store import LabError, Store

DUAL = S16 / "s16-dual-exam.json"
RESULT = S16 / "s16-result.json"
SUMMARY = S16 / "s16-result-summary.json"
EXAMS = ("CVS-DEEP", "SF-DEEP")


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def _exam_splits(service: LabService) -> dict:
    """Each exam encoded under EVERY representation the matrix uses (a HYBRID model needs the
    same 810-column encoding at evaluation as it had at training)."""
    out = {}
    for name, protocol_id in (("CVS-DEEP", EXAM_CVS["protocol"]), ("SF-DEEP", EXAM_SF["protocol"])):
        protocol = service.store.get(protocol_id)
        dataset = service.store.get(protocol.dataset_id)
        spec = protocol_target_spec(protocol)
        records = data.load_split(service.store, dataset, protocol.split)
        splits = {kind: nnue.encode_records(records, target_spec=spec, input_kind=kind)[0]
                  for kind in ("RAW", "HYBRID")}
        out[name] = {"protocol": protocol, "dataset": dataset, "spec": spec, "splits": splits,
                     "positions": len(splits["RAW"].cp),
                     "mate_sentinels": int((abs(splits["RAW"].cp) >= 100_000).sum())}
    return out


def collect(service: LabService):
    experiment: Experiment = service.store.get_as(state()["xid"], Experiment)
    arm_of = {a: arm.arm_id.replace("S16-", "") for arm in experiment.arms for a in arm.ablations}
    by_cell: dict = {}
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id == experiment.id:
            by_cell.setdefault((run.ablation_id, run.seed), []).append(run)
    values: dict = {}
    models: dict = {}
    for cell, runs in by_cell.items():
        completed = [r for r in runs if r.status == RunStatus.COMPLETED]
        if not completed:
            raise LabError(f"cell {cell} has no COMPLETED run")
        run = completed[0]
        arm_label = arm_of[run.ablation_id]
        teacher = next(name for name in TEACHERS if arm_label.startswith(name + "-"))
        representation = arm_label[len(teacher) + 1:]
        hidden = int(run.effective_config["H"])
        values.setdefault((teacher, representation, hidden), {})[run.seed] = {
            "test_loss": next(m.value for m in run.metrics if m.name == "test_loss"),
            "cp_mae": next(m.value for m in run.metrics if m.name == "cp_mae"),
            "sign": next(m.value for m in run.metrics if m.name == "sign_agreement")}
        models[(teacher, representation, hidden, run.seed)] = {
            "path": f"artifacts/{run.id}/model.json",
            "wall_seconds": run.compute.wall_seconds if run.compute else None,
            "bytes": service.store.abs(f"artifacts/{run.id}/model.json").stat().st_size}
    for teacher in TEACHERS:
        for representation in REPRESENTATIONS:
            for hidden in CAPACITIES:
                if len(values[(teacher, representation, hidden)]) != len(SEEDS):
                    raise LabError(f"{teacher}/{representation}/H{hidden}: "
                                   f"{len(values[(teacher, representation, hidden)])} seeds")
    return experiment, values, models


def step_dual() -> int:
    service = svc()
    splits = _exam_splits(service)
    experiment, values, models = collect(service)
    scores: dict = {}
    mismatches = []
    for key, entry in models.items():
        teacher, representation, hidden, seed = key
        payload = json.loads(service.store.abs(entry["path"]).read_bytes())
        model = nnue.load_serialized(payload)
        cell = {"persistent_parameters": int(sum(np.asarray(payload[name]).size for name in
                                                 ("w1", "b1", "w2", "b2")))}
        input_kind = str(payload.get("input", "RAW"))
        for exam in EXAMS:
            split = splits[exam]["splits"][input_kind]
            per_position = nnue.evaluate(model, split, {"K": 256.0, "LAMBDA": 1.0})
            cell[exam] = float(np.mean(per_position["test_loss"]))
        recorded = values[(teacher, representation, hidden)][seed]["test_loss"]
        if abs(cell["CVS-DEEP"] - recorded) > 1e-12:
            mismatches.append({"cell": f"{teacher}/{representation}/H{hidden}", "seed": seed,
                               "recorded": recorded, "recomputed": cell["CVS-DEEP"]})
        scores[f"{teacher}|{representation}|{hidden}|{seed}"] = cell
    if mismatches:
        raise LabError(f"the dual pass does not reproduce {len(mismatches)} recorded CVS metrics: "
                       f"{mismatches[:3]}")
    artifact = {"id": "s16-dual-exam-scores", "experiment": experiment.id,
                "validation": f"{len(scores)} cells: every recomputed CVS-DEEP loss equals the run's "
                              "recorded metric exactly",
                "exams": {name: {"protocol": exam["protocol"].id, "dataset": exam["dataset"].id,
                                 "spec_hash": exam["spec"].spec_hash(),
                                 "positions": exam["positions"],
                                 "mate_sentinels": exam["mate_sentinels"]}
                          for name, exam in splits.items()},
                "scores": scores}
    digest = hash_obj(artifact)
    DUAL.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (S16 / "s16-dual-exam.hash").write_text(digest + "\n", encoding="utf-8")
    print(f"dual pass: {len(scores)} cells, CVS metrics reproduced exactly; {digest}", flush=True)
    return 0


def _series(scores: dict, teacher: str, representation: str, hidden: int, exam: str) -> dict:
    return {seed: scores[f"{teacher}|{representation}|{hidden}|{seed}"][exam] for seed in SEEDS}


def _paired(a: dict, b: dict, label: str) -> dict:
    two = paired_effect(a, b, seeds=SEEDS)
    one = paired_effect(a, b, seeds=SEEDS, one_sided=True)
    return {"label": label, "d": two["width_estimate"], "n": two["n"],
            "one_sided_bounds": [one["ci_low"], one["ci_high"]],
            "t_one_sided": one["critical"],
            "verdict": ("positive" if one["ci_low"] > 0 else
                        "negative" if one["ci_high"] < 0 else "unresolved"),
            "mean_a": statistics.mean(a.values()), "mean_b": statistics.mean(b.values())}


def step_report() -> int:
    service = svc()
    experiment, values, models = collect(service)
    scores = json.loads(DUAL.read_text(encoding="utf-8"))["scores"]
    per_capacity = {}
    for hidden in CAPACITIES:
        entry = {"exam": {}}
        for exam in EXAMS:
            teacher_effects, representation_effects = {}, {}
            for representation in REPRESENTATIONS:
                teacher_effects[representation] = _paired(
                    _series(scores, "SF-32k", representation, hidden, exam),
                    _series(scores, "CVS-4k", representation, hidden, exam),
                    f"teacher SF-CVS | {representation} | H{hidden} | {exam}")
            for teacher in TEACHERS:
                representation_effects[teacher] = _paired(
                    _series(scores, teacher, "HYBRID", hidden, exam),
                    _series(scores, teacher, "RAW", hidden, exam),
                    f"representation HYBRID-RAW | {teacher} | H{hidden} | {exam}")
            # difference-in-differences, paired by seed, computed both ways (they are identical)
            did_a = {seed: (scores[f"SF-32k|HYBRID|{hidden}|{seed}"][exam]
                            - scores[f"CVS-4k|HYBRID|{hidden}|{seed}"][exam]) for seed in SEEDS}
            did_b = {seed: (scores[f"SF-32k|RAW|{hidden}|{seed}"][exam]
                            - scores[f"CVS-4k|RAW|{hidden}|{seed}"][exam]) for seed in SEEDS}
            interaction = _paired(did_a, did_b, f"(SF-CVS | HYBRID) - (SF-CVS | RAW) | H{hidden} | {exam}")
            interaction["interaction"] = interaction["d"]
            symmetric = _paired(
                {seed: (scores[f"SF-32k|HYBRID|{hidden}|{seed}"][exam]
                        - scores[f"SF-32k|RAW|{hidden}|{seed}"][exam]) for seed in SEEDS},
                {seed: (scores[f"CVS-4k|HYBRID|{hidden}|{seed}"][exam]
                        - scores[f"CVS-4k|RAW|{hidden}|{seed}"][exam]) for seed in SEEDS},
                "same interaction, reconstructed")
            entry["exam"][exam] = {"means": {f"{teacher}/{representation}": statistics.mean(
                _series(scores, teacher, representation, hidden, exam).values())
                for teacher in TEACHERS for representation in REPRESENTATIONS},
                "teacher_effect": teacher_effects,
                "representation_effect": representation_effects,
                "interaction": interaction,
                "interaction_symmetric_check": symmetric["d"]}
        entry["cell_metrics"] = {f"{teacher}/{representation}": {
            "test_loss_cvs_exam": statistics.mean(v["test_loss"] for v in
                                                  values[(teacher, representation, hidden)].values()),
            "cp_mae": statistics.mean(v["cp_mae"] for v in values[(teacher, representation, hidden)].values()),
            "sign_agreement": statistics.mean(v["sign"] for v in
                                              values[(teacher, representation, hidden)].values()),
            "wall_seconds": round(statistics.mean(
                models[(teacher, representation, hidden, seed)]["wall_seconds"] or 0.0 for seed in SEEDS), 2),
            "model_bytes": models[(teacher, representation, hidden, 0)]["bytes"]}
            for teacher in TEACHERS for representation in REPRESENTATIONS}
        per_capacity[str(hidden)] = entry

    interactions = [per_capacity[str(h)]["exam"][exam]["interaction"] for h in CAPACITIES
                    for exam in EXAMS]
    resolved = [i for i in interactions if i["verdict"] != "unresolved"]
    negative = [i for i in interactions if i["interaction"] < 0]
    positive = [i for i in interactions if i["interaction"] > 0]
    if len(resolved) == len(interactions) and all(i["interaction"] < 0 for i in interactions):
        pattern = "COMPLEMENTS"
    elif len(resolved) == len(interactions) and all(i["interaction"] > 0 for i in interactions):
        pattern = "SUBSTITUTES"
    elif len(negative) == len(interactions) and resolved:
        pattern = "NEGATIVE_INTERACTION_PARTLY_RESOLVED"
    elif not resolved:
        pattern = "INDEPENDENT_OR_UNRESOLVED"
    else:
        pattern = "HETEROGENEOUS"
    counts = {"intervals": len(interactions), "resolved": len(resolved),
              "negative_point_estimates": len(negative), "positive_point_estimates": len(positive)}
    result = {
        "experiment": experiment.id,
        "preregistration_hash": experiment.preregistration_hash,
        "questions": ["does explicit semantic representation reduce the need for a stronger teacher?",
                      "does Stockfish supervision help RAW and HYBRID equally?",
                      "are teacher quality and explicit structure substitutes, complements, or "
                      "independent?"],
        "matrix": {"teachers": list(TEACHERS), "representations": list(REPRESENTATIONS),
                   "capacities": list(CAPACITIES), "seeds": list(SEEDS),
                   "cells": len(models), "exams": list(EXAMS),
                   "shared_rows": "D0036 and D0044 carry the same 18,953 record ids in the same "
                                  "order (record_ids_hash sha256:7d5a8a78…)"},
        "per_capacity": per_capacity,
        "observed_interaction_pattern": pattern,
        "observed_interaction_counts": counts,
        "pattern_note": "the label counts resolved intervals and the sign of every point estimate "
                        "separately; unresolved intervals are reported, never forced into a claim "
                        "of significance",
        "economics": state()["economics"],
        "evidence": {"result": "tools/s16/s16-result.json",
                     "dual_exam": "tools/s16/s16-dual-exam.json",
                     "preregistration": "tools/s16/s16-preregistration.json",
                     "state": "tools/s16/s16-state.json"},
        "non_claims": ["no pooled score across the two exams", "no game or search claims",
                       "static-evaluator results only", "one population, one optimizer, 20 seeds"],
    }
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary = {"experiment": experiment.id, "observed_interaction_pattern": pattern,
               "per_capacity": {str(h): {exam: {
                   "means": {k: round(v, 6) for k, v in per_capacity[str(h)]["exam"][exam]["means"].items()},
                   "teacher_effect": {rep: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                            for kk, vv in per_capacity[str(h)]["exam"][exam]["teacher_effect"][rep].items()
                                            if kk in ("d", "one_sided_bounds", "verdict")}
                                      for rep in REPRESENTATIONS},
                   "representation_effect": {t: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                                 for kk, vv in per_capacity[str(h)]["exam"][exam]["representation_effect"][t].items()
                                                 if kk in ("d", "one_sided_bounds", "verdict")}
                                             for t in TEACHERS},
                   "interaction": {"interaction": round(per_capacity[str(h)]["exam"][exam]["interaction"]["interaction"], 6),
                                   "one_sided_bounds": [round(x, 6) for x in per_capacity[str(h)]["exam"][exam]["interaction"]["one_sided_bounds"]],
                                   "verdict": per_capacity[str(h)]["exam"][exam]["interaction"]["verdict"]}}
                   for exam in EXAMS} for h in CAPACITIES},
               "evidence": result["evidence"]}
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"experiment {experiment.id}: interaction pattern {pattern}", flush=True)
    for hidden in CAPACITIES:
        for exam in EXAMS:
            row = per_capacity[str(hidden)]["exam"][exam]
            print(f"  H{hidden} {exam}: teacher SF-CVS | RAW "
                  f"{row['teacher_effect']['RAW']['d']:+.6f} {row['teacher_effect']['RAW']['verdict']}"
                  f" | HYBRID {row['teacher_effect']['HYBRID']['d']:+.6f} "
                  f"{row['teacher_effect']['HYBRID']['verdict']}", flush=True)
            print(f"      repr HYBRID-RAW | CVS {row['representation_effect']['CVS-4k']['d']:+.6f} "
                  f"{row['representation_effect']['CVS-4k']['verdict']} | SF "
                  f"{row['representation_effect']['SF-32k']['d']:+.6f} "
                  f"{row['representation_effect']['SF-32k']['verdict']}", flush=True)
            print(f"      interaction {row['interaction']['interaction']:+.6f} "
                  f"[{row['interaction']['one_sided_bounds'][0]:+.6f}, "
                  f"{row['interaction']['one_sided_bounds'][1]:+.6f}] "
                  f"{row['interaction']['verdict']}", flush=True)
    return 0


def step_seal() -> int:
    service = svc()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    experiment = service.seal_experiment(state()["xid"], result)
    print(f"{experiment.id} sealed: {json.dumps(experiment.result.get('membership'), default=str)}",
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["dual", "report", "seal"])
    args = parser.parse_args()
    return {"dual": step_dual, "report": step_report, "seal": step_seal}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
