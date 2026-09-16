"""S9 analysis: is 2k supervision non-inferior to 8k on the same positions?

Primary: d = test_loss(2k) - test_loss(8k), paired over twenty seeds, per width; one-sided 95%
bounds (t = 1.729, df = 19) against the preregistered margin +0.001. Secondary: the two-sided
95% CI inside [-0.001, +0.001] (equivalence). Also the realized teacher-cost ratio and the
teacher-disagreement profile on the shared positions (analysis only, never selection).
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.data import read_jsonl
from cvslab.funnel.campaign import paired_effect
from cvslab.schemas import Dataset, Run, RunStatus, TERMINAL_RUN_STATUSES
from cvslab.service import LabService
from cvslab.store import Store

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
S9 = LAB / "tools" / "s9"
STATE = S9 / "s9-state.json"
RESULT = S9 / "s9-result.json"
SUMMARY = S9 / "s9-result-summary.json"
S8_LABELS = pathlib.Path(r"F:\Github\_parity_tmp\s8")
WIDTHS = (1, 4, 16, 32)
SEEDS = tuple(range(20))
DELTA = 0.001


def main() -> int:
    st = json.loads(STATE.read_text(encoding="utf-8"))
    service = LabService(Store(str(LAB / "labstore")))
    experiment = service.store.get(st["experiment"]["id"])

    arm_of: dict = {}
    for arm in experiment.arms:
        for ablation_id in arm.ablations:
            arm_of[ablation_id] = arm.arm_id                      # "SAME-8K" / "SAME-2K"

    values: dict = {}
    compute: dict = {}
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
        compute[f"{key}/H{run.effective_config['H']}"] = {
            "rows": run.compute.accepted_examples - 200, "examples_seen": run.compute.train_examples_seen,
            "updates": int(run.effective_config.get("MAX_UPDATES", 0) or 0),
            "cpu_seconds": run.compute.cpu_seconds, "wall_seconds": run.compute.wall_seconds}

    def cell(width: int) -> dict:
        two = paired_effect(values.get("SAME-2K", {}).get(str(width), {}),
                            values.get("SAME-8K", {}).get(str(width), {}), seeds=SEEDS)
        one = paired_effect(values.get("SAME-2K", {}).get(str(width), {}),
                            values.get("SAME-8K", {}).get(str(width), {}), seeds=SEEDS, one_sided=True)
        return {"n": two["n"], "d": two["width_estimate"], "se": (two["ci_high"] - two["width_estimate"]) / two["critical"],
                "two_sided_ci": [two["ci_low"], two["ci_high"]], "two_sided_t": two["critical"],
                "one_sided_ci": [one["ci_low"], one["ci_high"]], "one_sided_t": one["critical"],
                "differences": two["differences"], "per_seed": {seed: values["SAME-2K"][str(width)][seed]
                                                                for seed in SEEDS}}

    cells = {str(width): cell(width) for width in WIDTHS}
    upper = {width: cells[str(width)]["one_sided_ci"][1] for width in WIDTHS}
    lower = {width: cells[str(width)]["one_sided_ci"][0] for width in WIDTHS}
    if all(upper[width] < DELTA for width in WIDTHS):
        decision = "2K_NONINFERIOR"
    elif all(lower[width] > DELTA for width in WIDTHS):
        decision = "FLOOR_CROSSED"
    else:
        decision = "INCONCLUSIVE"
    equivalence = {width: (cells[str(width)]["two_sided_ci"][0] > -DELTA
                           and cells[str(width)]["two_sided_ci"][1] < DELTA) for width in WIDTHS}

    # teacher disagreement on the same positions (analysis only)
    # teacher disagreement on THE FROZEN POSITIONS ONLY (analysis only): X0003's 9,495 ids read
    # from its own dataset split. Intersecting the two label streams would have used the whole 2k
    # stream (38,989 positions) and silently widened the population to 9,770.
    dataset_2k: Dataset = service.store.get_as(st["datasets"]["2k"], Dataset)
    split_2k = next(entry for entry in dataset_2k.splits if entry.name == "train")
    frozen_ids = [row["record_id"] for row in read_jsonl(service.store.abs(split_2k.path))]
    if len(frozen_ids) != 9_495:
        print(f"STOP: the frozen 2k dataset holds {len(frozen_ids)} rows, not 9,495")
        return 3
    stream_2k = {row["record_id"]: row for row in read_jsonl(S8_LABELS / "labels-2000.jsonl")}
    stream_8k = {row["record_id"]: row for row in read_jsonl(S8_LABELS / "labels-8000.jsonl")}
    shared = [rid for rid in frozen_ids if rid in stream_2k and rid in stream_8k]
    if len(shared) != len(frozen_ids):
        print(f"STOP: {len(frozen_ids) - len(shared)} frozen positions lack a 2k or 8k observation")
        return 3
    diffs, signs, moves, mate_rows = [], 0, 0, 0
    for rid in shared:
        a = stream_2k[rid]["row"]["value"].get("scoreCpStm")
        b = stream_8k[rid]["row"]["value"].get("scoreCpStm")
        if a is None or b is None:
            continue
        if abs(a) > 10_000 or abs(b) > 10_000:
            mate_rows += 1          # the engine's mate sentinel, not a centipawn value
            continue
        diffs.append(abs(a - b))
        if (a > 0) != (b > 0):
            signs += 1
        if stream_2k[rid]["row"]["value"].get("bestMove") != stream_8k[rid]["row"]["value"].get("bestMove"):
            moves += 1
    diffs.sort()
    disagreement = {
        "shared_positions": len(shared), "compared_positions": len(diffs),
        "mate_sentinel_rows_excluded": mate_rows,
        "abs_cp": {"mean": round(statistics.mean(diffs), 1), "median": diffs[len(diffs) // 2],
                   "p90": diffs[int(len(diffs) * 0.9)], "p99": diffs[int(len(diffs) * 0.99)],
                   "max": diffs[-1]},
        "rate_above_cp": {str(threshold): round(sum(1 for value in diffs if value > threshold) / len(diffs), 4)
                          for threshold in (0, 10, 25, 50, 100)},
        "sign_disagreement_rate": round(signs / len(diffs), 4),
        "move_disagreement_rate": round(moves / len(diffs), 4),
        "note": "the same positions carry both targets, so this is a direct teacher comparison; analysis only, "
                "never a selection mechanism",
    }

    realized = st["realized_nodes"]
    result = {
        "experiment": experiment.id, "preregistration_hash": experiment.preregistration_hash,
        "cells": {"expected": experiment.expected_run_count, "invalid_runs": invalid},
        "design": "same 9,495 positions, same order, same student compute (760 updates x 256), same exam; "
                  "only the teacher depth differs",
        "primary": {"quantity": "test_loss(2k) - test_loss(8k), positive = 2k worse",
                    "margin_delta": DELTA,
                    "interval": "separate one-sided 95% bounds (t=1.729, df=19) — a lower and an upper bound computed independently, not a single interval",
                    "cells": cells, "decision": decision,
                    "upper_bounds": upper, "lower_bounds": lower},
        "secondary_equivalence": {"rule": "two-sided 95% CI inside [-0.001, +0.001]",
                                  "per_width": equivalence,
                                  "all_widths_equivalent": all(equivalence.values())},
        "mean_test_loss": {arm: statistics.mean(v for widths in values.get(arm, {}).values()
                                                for v in widths.values()) for arm in values},
        "realized_teacher_nodes": realized,
        "teacher_disagreement_on_shared_positions": disagreement,
        "student_compute": compute,
    }
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "experiment": experiment.id, "decision": decision,
        "margin_delta": DELTA,
        "bounds_note": "separate one-sided 95% lower and upper bounds (t=1.729, df=19); the pair is not a single 95% interval",
        "point_estimates_vs_margin": {str(width): ("above" if cells[str(width)]["d"] > DELTA else "below")
                                      for width in WIDTHS},
        "primary": {width: {"d": round(cells[str(width)]["d"], 6),
                            "one_sided_ci": [round(cells[str(width)]["one_sided_ci"][0], 6),
                                             round(cells[str(width)]["one_sided_ci"][1], 6)],
                            "two_sided_ci": [round(cells[str(width)]["two_sided_ci"][0], 6),
                                             round(cells[str(width)]["two_sided_ci"][1], 6)],
                            "n": cells[str(width)]["n"]} for width in WIDTHS},
        "equivalence_all_widths": all(equivalence.values()),
        "equivalence_by_width": {str(width): equivalence[width] for width in WIDTHS},
        "realized": realized,
        "teacher_disagreement": disagreement,
        "mean_test_loss": {k: round(v, 6) for k, v in result["mean_test_loss"].items()},
        "invalid_runs": invalid,
        "evidence": {"result": "tools/s9/s9-result.json", "preregistration": "tools/s9/s9-preregistration.json",
                     "state": "tools/s9/s9-state.json"},
    }
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision,
                      "primary": summary["primary"],
                      "equivalence_all_widths": summary["equivalence_all_widths"],
                      "realized": realized,
                      "mean_test_loss": summary["mean_test_loss"],
                      "disagreement": disagreement}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
