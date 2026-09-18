"""S17 analysis: the scaling curve, its marginal gains, and the undertraining test.

    python tools/s17/s17_analyse.py report   # curve vs parameters and vs presentations + control
    python tools/s17/s17_analyse.py seal     # seal the study with the result

All numbers come from the runs' recorded exam metrics and the stored model artifacts. The
undertraining test is the control rungs against their same-width primary rungs, paired by seed:
a control rung that recovers the knee's loss indicates the primary turn-up was undersupplied
training compute; one that does not indicates capacity saturation *within the 4x envelope*.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s17_common import (BATCH, CONTROL_FACTOR, CONTROL_UPDATES, CONTROL_WIDTHS, DATASET,
                        EXAM_PROTOCOL, LAB, LADDER, PRIMARY_UPDATES, S17, ladder_table, save,
                        seeds_for, state)

sys.path.insert(0, str(LAB))

import numpy as np

import cvslab.data as data
import cvslab.nnue as nnue
from cvslab.funnel.campaign import paired_effect
from cvslab.funnel.providers import board_geometry
from cvslab.schemas import Experiment, Run, RunStatus
from cvslab.service import LabService, protocol_target_spec
from cvslab.store import LabError, Store

RESULT = S17 / "s17-result.json"
SUMMARY = S17 / "s17-result-summary.json"


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def collect(service: LabService):
    experiment: Experiment = service.store.get_as(state()["xid"], Experiment)
    conditions = state()["conditions"]
    where = {}
    for arm_id, by_hidden in conditions.items():
        for hidden, ablation_id in by_hidden.items():
            regime = "control" if arm_id.startswith("control") else "primary"
            where[ablation_id] = (regime, int(hidden))
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
        regime, hidden = where[run.ablation_id]
        values.setdefault((regime, hidden), {})[run.seed] = {
            "test_loss": next(m.value for m in run.metrics if m.name == "test_loss"),
            "cp_mae": next(m.value for m in run.metrics if m.name == "cp_mae"),
            "sign": next(m.value for m in run.metrics if m.name == "sign_agreement")}
        models[(regime, hidden, run.seed)] = {
            "path": f"artifacts/{run.id}/model.json",
            "wall_seconds": run.compute.wall_seconds if run.compute else None,
            "bytes": service.store.abs(f"artifacts/{run.id}/model.json").stat().st_size}
    for regime, widths in (("primary", LADDER), ("control", CONTROL_WIDTHS)):
        for hidden in widths:
            if len(values[(regime, hidden)]) != len(seeds_for(hidden)):
                raise LabError(f"{regime} H{hidden}: {len(values[(regime, hidden)])} seeds")
    return experiment, values, models


def _paired(a: dict, b: dict, label: str, metric: str = "test_loss") -> dict:
    """Paired by seed; entries may be metric dicts (the collector's form) or plain floats."""
    if a and isinstance(next(iter(a.values())), dict):
        a = {seed: entry[metric] for seed, entry in a.items()}
    if b and isinstance(next(iter(b.values())), dict):
        b = {seed: entry[metric] for seed, entry in b.items()}
    seeds = tuple(sorted(set(a) & set(b)))
    two = paired_effect(a, b, seeds=seeds)
    one = paired_effect(a, b, seeds=seeds, one_sided=True)
    return {"label": label, "d": two["width_estimate"], "n": two["n"],
            "one_sided_bounds": [one["ci_low"], one["ci_high"]],
            "verdict": ("worse" if one["ci_low"] > 0 else
                        "better" if one["ci_high"] < 0 else "unresolved"),
            "mean_a": statistics.mean(a.values()), "mean_b": statistics.mean(b.values())}


def step_report() -> int:
    service = svc()
    experiment, values, models = collect(service)
    table = ladder_table()
    params = {row["H"]: row["parameters"] for row in table["rows"]}
    rungs = []
    for index, hidden in enumerate(LADDER):
        entry = {"H": hidden, "parameters": params[hidden],
                 "presentations": PRIMARY_UPDATES * BATCH,
                 "presentations_per_parameter": PRIMARY_UPDATES * BATCH / params[hidden],
                 "mean_test_loss": statistics.mean(v["test_loss"] for v in values[("primary", hidden)].values()),
                 "cp_mae": statistics.mean(v["cp_mae"] for v in values[("primary", hidden)].values()),
                 "sign_agreement": statistics.mean(v["sign"] for v in values[("primary", hidden)].values()),
                 "wall_seconds": round(statistics.mean(
                     models[("primary", hidden, seed)]["wall_seconds"] or 0.0
                     for seed in values[("primary", hidden)]), 2),
                 "model_bytes": models[("primary", hidden, 0)]["bytes"],
                 "seeds": len(values[("primary", hidden)])}
        if index:
            previous = LADDER[index - 1]
            step = _paired(values[("primary", hidden)], values[("primary", previous)],
                           f"H{hidden} vs H{previous}")
            entry["step_vs_previous"] = {**step,
                                         "per_added_parameter": step["d"] / (params[hidden] - params[previous]),
                                         "better_or_worse": ("worse" if step["d"] > 0 else "better")}
        rungs.append(entry)
    best = min(rungs, key=lambda r: r["mean_test_loss"])
    knee = min([r for r in rungs if "step_vs_previous" in r],
               key=lambda r: r["step_vs_previous"]["d"])
    flat = [r["H"] for r in rungs
            if "step_vs_previous" in r and r["step_vs_previous"]["verdict"] == "unresolved"
            and r["step_vs_previous"]["d"] > 0]
    non_monotonic = [r["H"] for r in rungs
                     if "step_vs_previous" in r and r["step_vs_previous"]["d"] < 0
                     and r["step_vs_previous"]["verdict"] == "better"]
    control = []
    for hidden in CONTROL_WIDTHS:
        entry = {"H": hidden, "parameters": params[hidden],
                 "presentations": CONTROL_UPDATES * BATCH,
                 "presentations_per_parameter": CONTROL_UPDATES * BATCH / params[hidden],
                 "mean_test_loss": statistics.mean(v["test_loss"] for v in values[("control", hidden)].values()),
                 "primary_mean": statistics.mean(v["test_loss"] for v in values[("primary", hidden)].values()),
                 "wall_seconds": round(statistics.mean(
                     models[("control", hidden, seed)]["wall_seconds"] or 0.0
                     for seed in values[("control", hidden)]), 2),
                 "seeds": len(values[("control", hidden)])}
        entry["control_minus_primary"] = _paired(
            values[("control", hidden)], values[("primary", hidden)],
            f"control x{CONTROL_FACTOR} vs primary | H{hidden}")
        entry["control_vs_knee"] = {k: round(v, 6) if isinstance(v, float) else v for k, v in
                                    _paired(values[("control", hidden)],
                                            values[("primary", knee["H"])],
                                            f"control H{hidden} vs primary knee H{knee['H']}").items()
                                    if k in ("d", "verdict", "mean_a", "mean_b")}
        control.append(entry)
    rescued = [row["H"] for row in control
               if row["control_minus_primary"]["verdict"] == "better"]
    still_worse = [row["H"] for row in control
                   if row["control_minus_primary"]["verdict"] == "worse"]
    result = {
        "experiment": experiment.id,
        "preregistration_hash": experiment.preregistration_hash,
        "question": "how much capacity does the RAW evaluator need before more parameters stop "
                    "buying capability, and is the turn-up at fixed student compute a capacity "
                    "effect or an undertraining effect?",
        "contract": {"teacher": f"CVS-4k over {DATASET}", "exam": f"{EXAM_PROTOCOL}",
                     "primary": f"{PRIMARY_UPDATES} updates x {BATCH} = "
                                f"{PRIMARY_UPDATES * BATCH} presentations at every width",
                     "control": f"{CONTROL_UPDATES} updates ({CONTROL_FACTOR}x) at "
                                f"H={list(CONTROL_WIDTHS)}"},
        "primary_curve": rungs,
        "curve_summary": {"best_rung": best["H"], "best_loss": best["mean_test_loss"],
                          "knee_rung": knee["H"],
                          "knee_marginal_gain": knee["step_vs_previous"]["d"],
                          "flat_or_worse_rungs": flat, "non_monotonic_better_steps": non_monotonic},
        "undertraining_control": control,
        "undertraining_verdict": {
            "rescued_widths": rescued, "still_worse_widths": still_worse,
            "reading": ("control rungs that recovered the primary loss at the same width indicate "
                        "the primary turn-up was undersupplied training compute; rungs that did "
                        "not indicate capacity saturation WITHIN the 4x envelope")},
        "slices": _slices(service, models, values),
        "evidence": {"result": "tools/s17/s17-result.json",
                     "preregistration": "tools/s17/s17-preregistration.json",
                     "state": "tools/s17/s17-state.json"},
        "non_claims": ["no search, game or strength claims", "one population, one teacher, one exam",
                       "the control envelope is 4x, not 'unlimited compute'",
                       "H=256 approximates the historical ~200K-parameter scale as a comparison "
                       "point only, not inherited truth"],
    }
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary = {"experiment": experiment.id, "curve_summary": result["curve_summary"],
               "primary_curve": [{"H": r["H"], "parameters": r["parameters"],
                                  "mean_test_loss": round(r["mean_test_loss"], 6),
                                  "step": (round(r["step_vs_previous"]["d"], 6)
                                           if "step_vs_previous" in r else None),
                                  "step_verdict": (r["step_vs_previous"]["verdict"]
                                                   if "step_vs_previous" in r else None)}
                                 for r in rungs],
               "undertraining_control": [{"H": c["H"], "primary": round(c["primary_mean"], 6),
                                          "control_x4": round(c["mean_test_loss"], 6),
                                          "control_minus_primary": round(c["control_minus_primary"]["d"], 6),
                                          "verdict": c["control_minus_primary"]["verdict"]}
                                         for c in control],
               "undertraining_verdict": result["undertraining_verdict"],
               "evidence": result["evidence"]}
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"experiment {experiment.id}", flush=True)
    for r in rungs:
        step = (f" step {r['step_vs_previous']['d']:+.6f} {r['step_vs_previous']['verdict']}"
                if "step_vs_previous" in r else "")
        print(f"  H{r['H']:>4} ({r['parameters']:>7} p): {r['mean_test_loss']:.6f}{step}", flush=True)
    print("  control:", flush=True)
    for c in control:
        print(f"  H{c['H']:>4}: primary {c['primary_mean']:.6f} -> x{CONTROL_FACTOR} "
              f"{c['mean_test_loss']:.6f} ({c['control_minus_primary']['d']:+.6f} "
              f"{c['control_minus_primary']['verdict']})", flush=True)
    print("  verdict:", json.dumps(result["undertraining_verdict"]), flush=True)
    return 0


def _slices(service: LabService, models, values) -> dict:
    """Phase/material slices at the knee and at H=512 (deterministic, from the exam FENs)."""
    protocol = service.store.get(EXAM_PROTOCOL)
    dataset = service.store.get(protocol.dataset_id)
    spec = protocol_target_spec(protocol)
    records = data.load_split(service.store, dataset, protocol.split)
    split, _dropped = nnue.encode_records(records, target_spec=spec)
    out = {}
    for hidden in (32, 512):
        payload = json.loads(service.store.abs(models[("primary", hidden, 0)]["path"]).read_bytes())
        model = nnue.load_serialized(payload)
        losses = np.asarray(nnue.evaluate(model, split, {"K": 256.0, "LAMBDA": 1.0})["test_loss"])
        phases: dict = {}
        for record, loss in zip(records, losses):
            geometry, _ = board_geometry(record["fen"])
            phases.setdefault(geometry["phase"], []).append(float(loss))
        out[f"H{hidden}"] = {phase: round(float(np.mean(values_)), 6) for phase, values_ in sorted(phases.items())}
    return out


def step_seal() -> int:
    service = svc()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    experiment = service.seal_experiment(state()["xid"], result)
    print(f"{experiment.id} sealed: {json.dumps(experiment.result.get('membership'), default=str)}",
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["report", "seal"])
    args = parser.parse_args()
    return {"report": step_report, "seal": step_seal}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
