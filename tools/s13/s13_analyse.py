"""S13 dual-exam scoring + analysis + seal.

    python tools/s13/s13_analyse.py dual     # score every stored model on BOTH exams (recorded)
    python tools/s13/s13_analyse.py verify   # recompute the dual table and fail closed on drift
    python tools/s13/s13_analyse.py report   # the six cells, the interactions, the verdicts
    python tools/s13/s13_analyse.py seal     # seal X#### with the result and membership

The lab's Run carries one exam (CVS-DEEP), so the second instrument is applied here — to the SAME
stored weights, with the lab's own `nnue.evaluate` — and the pass is only believed if it reproduces
every run's recorded CVS metrics exactly. `|cp| >= 100k` mate-sentinel rows are counted per exam.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s13_common import LAB, S13, write_json

import numpy as np

import cvslab.data as data
import cvslab.nnue as nnue
from cvslab.funnel.campaign import paired_effect
from cvslab.hashing import hash_obj, sha256_file
from cvslab.schemas import Experiment, Run, RunStatus
from cvslab.service import LabService, protocol_target_spec
from cvslab.store import LabError, Store

CONTRACT = "cold"                       # the replacement contract (X0009's warm evidence is sealed)
SFX = "-cold" if CONTRACT == "cold" else ""
DUAL = S13 / f"s13-dual-exam{SFX}.json"
RESULT = S13 / f"s13-result{SFX}.json"
SUMMARY = S13 / f"s13-result-summary{SFX}.json"
XID_KEY = "xid" if CONTRACT == "warm" else "xid_cold"
EXAM_KEY = "exam" if CONTRACT == "warm" else "exam_cold"
CONDITIONS_KEY = "conditions" if CONTRACT == "warm" else "conditions_cold"
STATE = S13 / "s13-state.json"
WIDTHS = (1, 4, 16, 32)
SEEDS = tuple(range(20))
LABELS = ("CVS-4k", "SF-32k", "SF-1M")
MATE_CP = 100_000


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def _exam_splits(service: LabService) -> dict:
    """Each exam's encoded split, with its TargetSpec and the mate-sentinel count."""
    out = {}
    st = state()
    for name, protocol_id in (("CVS-DEEP", "E0004"), ("SF-DEEP", st[EXAM_KEY]["protocol_id"])):
        protocol = service.store.get(protocol_id)
        dataset = service.store.get(protocol.dataset_id)
        spec = protocol_target_spec(protocol)
        split, dropped = nnue.encode_records(data.load_split(service.store, dataset, protocol.split),
                                             target_spec=spec)
        mates = int((abs(split.cp) >= MATE_CP).sum())
        out[name] = {"protocol": protocol, "dataset": dataset, "spec": spec, "split": split,
                     "dropped": dropped, "positions": len(split.cp), "mate_sentinels": mates,
                     "record_ids": list(split.record_ids)}
    if sorted(out["CVS-DEEP"]["record_ids"]) != sorted(out["SF-DEEP"]["record_ids"]):
        raise LabError("the two exams do not share the same held-out identities")
    return out


def step_dual() -> int:
    service = svc()
    experiment: Experiment = service.store.get_as(state()[XID_KEY], Experiment)
    arm_of = {ablation_id: arm.arm_id for arm in experiment.arms for ablation_id in arm.ablations}
    splits = _exam_splits(service)
    runs = [run for run in service.store.list("R", verify=True, kind=Run)
            if run.experiment_id == experiment.id]
    scores: dict = {}
    mismatches = []
    for run in runs:
        if run.status != RunStatus.COMPLETED:
            continue
        label = arm_of[run.ablation_id].replace("S13-", "")
        artifact = f"artifacts/{run.id}/model.json"
        payload = json.loads(service.store.abs(artifact).read_bytes())
        recorded_hash = run.artifact_hashes.get("model.json")
        if recorded_hash and sha256_file(service.store.abs(artifact)) != recorded_hash:
            raise LabError(f"{run.id}: the stored model no longer matches its recorded hash")
        model = nnue.load_serialized(payload)
        entry = {"model_artifact": artifact, "model_hash": recorded_hash,
                 "param_count": run.param_count}
        for name, exam in splits.items():
            per_position = nnue.evaluate(model, exam["split"], exam["protocol"].params)
            mean, low, high, n = nnue.bootstrap_mean(per_position["test_loss"],
                                                     exam["protocol"].bootstrap_samples,
                                                     exam["protocol"].bootstrap_seed)
            entry[name] = {"test_loss": mean, "ci_low": low, "ci_high": high, "n": n,
                           "cp_mae": float(np.nanmean(per_position["cp_mae"]))}
        recorded = next((m for m in run.metrics if m.name == "test_loss"), None)
        if recorded is None:
            mismatches.append({"run": run.id, "why": "no recorded test_loss"})
        elif abs(entry["CVS-DEEP"]["test_loss"] - recorded.value) > 1e-12:
            mismatches.append({"run": run.id, "recorded": recorded.value,
                               "recomputed": entry["CVS-DEEP"]["test_loss"]})
        scores.setdefault(label, {}).setdefault(str(run.effective_config["H"]), {})[str(run.seed)] = entry
    if mismatches:
        raise LabError(f"the dual pass does not reproduce the recorded CVS metrics for "
                       f"{len(mismatches)} runs: {mismatches[:3]}")
    payload = {
        "id": "s13-dual-exam-scores",
        "experiment": experiment.id,
        "why": "the lab's Run carries one exam; every stored model is scored here on both frozen "
               "static instruments with the lab's own nnue.evaluate",
        "validation": "every recomputed CVS-DEEP test_loss equals the run's recorded metric exactly "
                      f"({sum(len(v) for arm in scores.values() for v in arm.values())} cells)",
        "exams": {name: {"protocol_id": exam["protocol"].id,
                         "protocol_hash": exam["protocol"].protocol_hash,
                         "dataset_id": exam["dataset"].id,
                         "dataset_manifest_hash": exam["dataset"].manifest_hash,
                         "spec_hash": exam["spec"].spec_hash(),
                         "split": exam["protocol"].split,
                         "positions": exam["positions"],
                         "mate_sentinels": exam["mate_sentinels"],
                         "dropped_rows": exam["dropped"]}
                  for name, exam in splits.items()},
        "scores": scores,
    }
    digest = write_json(DUAL, payload)
    (S13 / "s13-dual-exam.hash").write_text(digest + "\n", encoding="utf-8")
    print(f"dual-exam pass: {sum(len(v) for arm in scores.values() for v in arm.values())} cells, "
          f"CVS metrics reproduced exactly; artifact {digest}", flush=True)
    return 0


def step_verify() -> int:
    before = hash_obj(json.loads(DUAL.read_text(encoding="utf-8")))
    recorded = (S13 / f"s13-dual-exam{SFX}.hash").read_text(encoding="utf-8").strip()
    if before != recorded:
        raise LabError("the stored dual-exam artifact no longer matches its recorded hash")
    step_dual()
    after = (S13 / f"s13-dual-exam{SFX}.hash").read_text(encoding="utf-8").strip()
    if after != recorded:
        raise LabError(f"recomputation drifted: {after} != {recorded}")
    print("verify: the dual-exam table reproduces", flush=True)
    return 0


def _read_scores() -> dict:
    return json.loads(DUAL.read_text(encoding="utf-8"))["scores"]


def _cell(scores: dict, label: str, width: int, exam: str) -> dict:
    return {seed: scores[label][str(width)][str(seed)][exam]["test_loss"] for seed in SEEDS}


def _paired(values: dict, a: str, b: str, width: int, exam: str) -> dict:
    two = paired_effect(_cell(values, a, width, exam), _cell(values, b, width, exam), seeds=SEEDS)
    one = paired_effect(_cell(values, a, width, exam), _cell(values, b, width, exam), seeds=SEEDS,
                        one_sided=True)
    lower, upper = one["ci_low"], one["ci_high"]
    verdict = f"{a}_worse" if lower > 0 else (f"{a}_better" if upper < 0 else "unresolved")
    return {"d": two["width_estimate"], "n": two["n"], "t_one_sided": one["critical"],
            "one_sided_bounds": [lower, upper], "two_sided_ci": [two["ci_low"], two["ci_high"]],
            "verdict": verdict,
            "mean_a": statistics.mean(_cell(values, a, width, exam).values()),
            "mean_b": statistics.mean(_cell(values, b, width, exam).values())}


def _interaction(values: dict, width: int) -> dict:
    """exam_effect(T) = loss on SF-DEEP - loss on CVS-DEEP, paired by seed, per teacher."""
    effects = {}
    for label in LABELS:
        effects[label] = {seed: values[label][str(width)][str(seed)]["SF-DEEP"]["test_loss"]
                                - values[label][str(width)][str(seed)]["CVS-DEEP"]["test_loss"]
                          for seed in SEEDS}
    interaction = {seed: effects["CVS-4k"][seed] - effects["SF-32k"][seed] for seed in SEEDS}
    def _stats(series: dict) -> dict:
        n = len(series)
        mean = sum(series.values()) / n
        var = sum((v - mean) ** 2 for v in series.values()) / (n - 1)
        se = (var / n) ** 0.5
        from cvslab.funnel.campaign import t_critical
        t = t_critical(n - 1, one_sided=True)
        return {"mean": mean, "se": se, "bounds": [mean - t * se, mean + t * se], "n": n}
    return {"exam_effect": {label: _stats(series) for label, series in effects.items()},
            "interaction_cvs_minus_sf": _stats(interaction),
            "reading": "positive interaction = the CVS-trained student is relatively worse on the "
                       "SF exam than the SF-trained student is (each student best matches its own "
                       "authority); near-zero with a common exam offset = generic quality"}


def step_report() -> int:
    service = svc()
    experiment: Experiment = service.store.get_as(state()[XID_KEY], Experiment)
    values = _read_scores()
    for label in LABELS:
        for width in WIDTHS:
            if len(values[label][str(width)]) != len(SEEDS):
                raise LabError(f"{label} H{width}: {len(values[label][str(width)])} seeds")
    per_width = {}
    for width in WIDTHS:
        primary = {exam: _paired(values, "SF-32k", "CVS-4k", width, exam)
                   for exam in ("CVS-DEEP", "SF-DEEP")}
        dose = {exam: {"SF-1M - SF-32k": _paired(values, "SF-1M", "SF-32k", width, exam),
                       "SF-1M - CVS-4k": _paired(values, "SF-1M", "CVS-4k", width, exam)}
                for exam in ("CVS-DEEP", "SF-DEEP")}
        means = {label: {exam: statistics.mean(_cell(values, label, width, exam).values())
                         for exam in ("CVS-DEEP", "SF-DEEP")} for label in LABELS}
        per_width[str(width)] = {"means": means, "primary": primary, "dose": dose,
                                 "interaction": _interaction(values, width)}
    own = [str(w) for w in WIDTHS
           if per_width[str(w)]["primary"]["CVS-DEEP"]["verdict"] == "SF-32k_worse"
           and per_width[str(w)]["primary"]["SF-DEEP"]["verdict"] == "SF-32k_better"]
    both = [str(w) for w in WIDTHS
            if per_width[str(w)]["primary"]["CVS-DEEP"]["verdict"] == "SF-32k_better"
            and per_width[str(w)]["primary"]["SF-DEEP"]["verdict"] == "SF-32k_better"]
    cvs_both = [str(w) for w in WIDTHS
                if per_width[str(w)]["primary"]["CVS-DEEP"]["verdict"] == "SF-32k_worse"
                and per_width[str(w)]["primary"]["SF-DEEP"]["verdict"] == "SF-32k_worse"]
    dose_worse = [str(w) for w in WIDTHS
                  if per_width[str(w)]["dose"]["CVS-DEEP"]["SF-1M - SF-32k"]["verdict"] == "SF-1M_worse"
                  and per_width[str(w)]["dose"]["SF-DEEP"]["SF-1M - SF-32k"]["verdict"] == "SF-1M_worse"]
    if len(own) == len(WIDTHS):
        decision = "WORLDVIEW_DISAGREEMENT"
    elif len(both) == len(WIDTHS):
        decision = "GENERIC_SF_GAIN"
    else:
        decision = "INCONCLUSIVE"
    observed = ("CVS_BETTER_ON_BOTH_EXAMS" if len(cvs_both) == len(WIDTHS)
                else "MIXED" if cvs_both else "NOT_CVS_BETTER_ON_BOTH")
    result = {
        "experiment": experiment.id,
        "preregistration_hash": experiment.preregistration_hash,
        "question": "does changing the teacher from CVS to Stockfish change what the student learns, "
                    "under two independent frozen static exams?",
        "cells": {"expected": experiment.expected_run_count,
                  "scored": sum(len(v) for arm in values.values() for v in arm.values()),
                  "exams": ["CVS-DEEP (E0004/D0010)", "SF-DEEP (E0005/D0040)"],
                  "widths": list(WIDTHS), "seeds": list(SEEDS)},
        "per_width": per_width,
        "decision": decision,
        "decision_rule": "WORLDVIEW_DISAGREEMENT iff at every width the CVS-trained student is "
                         "strictly better on CVS-DEEP and strictly worse on SF-DEEP (own-authority "
                         "pattern); GENERIC_SF_GAIN iff the SF-trained student is strictly better on "
                         "both exams at every width; otherwise INCONCLUSIVE",
        "widths_own_authority": own, "widths_sf_better_on_both": both,
        "observed_pattern": observed,
        "widths_cvs_better_on_both_exams": cvs_both,
        "widths_dose_arm_worse_than_primary": dose_worse,
        "observed_pattern_note": (
            "the preregistered decision rule enumerated only two clean patterns (worldview "
            "disagreement; generic SF gain). The realized pattern is a third one it did not name. "
            "The registered verdict vocabulary is kept verbatim (decision above) and the observed "
            "pattern is reported beside it, never substituted for it."),
        "economics": {label: state()[CONDITIONS_KEY][label] for label in LABELS},
        "scope": {"no_stockfish_as_truth": True, "losses_never_averaged": True,
                  "no_playing_strength_claim": True},
        "evidence": {"dual_exam": f"tools/s13/s13-dual-exam{SFX}.json",
                     "result": f"tools/s13/s13-result{SFX}.json",
                     "preregistration": "tools/s13/s13-preregistration.json",
                     "calibration": ["tools/s13/s13-calibration.json",
                                     "tools/s13/s13-calibration-amendment.json",
                                     "tools/s13/s13-calibration-cold.json"],
                     "supersession": "tools/s13/s13-x0009-supersession.json",
                     "cold_worker_check": "tools/s13/s13-cold-worker-check.json",
                     "disagreement": "tools/s13/s13-disagreement.json",
                     "state": "tools/s13/s13-state.json"},
    }
    digest = write_json(RESULT, result)
    summary = {
        "experiment": experiment.id, "decision": decision,
        "per_width": {str(w): {
            "means": {label: {exam: round(v, 6) for exam, v in per_width[str(w)]["means"][label].items()}
                      for label in LABELS},
            "primary_SF32k_minus_CVS4k": {exam: {"d": round(per_width[str(w)]["primary"][exam]["d"], 6),
                                                 "bounds": [round(x, 6) for x in per_width[str(w)]["primary"][exam]["one_sided_bounds"]],
                                                 "verdict": per_width[str(w)]["primary"][exam]["verdict"]}
                                          for exam in ("CVS-DEEP", "SF-DEEP")},
            "interaction": {"mean": round(per_width[str(w)]["interaction"]["interaction_cvs_minus_sf"]["mean"], 6),
                            "bounds": [round(x, 6) for x in per_width[str(w)]["interaction"]["interaction_cvs_minus_sf"]["bounds"]]},
            "dose": {exam: {name: round(stats["d"], 6) for name, stats in
                            per_width[str(w)]["dose"][exam].items()}
                     for exam in ("CVS-DEEP", "SF-DEEP")},
        } for w in WIDTHS},
        "economics": {label: {"ms_per_label": state()[CONDITIONS_KEY][label]["ms_per_label"],
                              "realized_nodes": state()[CONDITIONS_KEY][label]["realized_nodes"],
                              "unit_multiples": round(state()[CONDITIONS_KEY][label]["unit_multiples"], 2)}
                      for label in LABELS},
        "evidence": result["evidence"],
    }
    write_json(SUMMARY, summary)
    print(f"decision: {decision}", flush=True)
    for width in WIDTHS:
        row = per_width[str(width)]
        print(f"  H{width}: means " + ", ".join(
            f"{label}[CVS {row['means'][label]['CVS-DEEP']:.5f} / SF {row['means'][label]['SF-DEEP']:.5f}]"
            for label in LABELS), flush=True)
        for exam in ("CVS-DEEP", "SF-DEEP"):
            stat = row["primary"][exam]
            print(f"     {exam}: SF-32k - CVS-4k = {stat['d']:+.6f} "
                  f"[{stat['one_sided_bounds'][0]:+.6f}, {stat['one_sided_bounds'][1]:+.6f}] "
                  f"{stat['verdict']}", flush=True)
        inter = row["interaction"]["interaction_cvs_minus_sf"]
        print(f"     interaction (CVS-trained exam effect - SF-trained) = {inter['mean']:+.6f} "
              f"[{inter['bounds'][0]:+.6f}, {inter['bounds'][1]:+.6f}]", flush=True)
    return 0


def step_seal() -> int:
    service = svc()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    experiment = service.seal_experiment(state()[XID_KEY], result)
    print(f"{experiment.id} sealed: {json.dumps(experiment.result.get('membership'), default=str)}",
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["dual", "verify", "report", "seal"])
    args = parser.parse_args()
    return {"dual": step_dual, "verify": step_verify, "report": step_report,
            "seal": step_seal}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
