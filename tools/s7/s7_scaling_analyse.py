"""S7-S analysis: the frontier at each scale, the scaling curve at each depth, the interaction.

Reads only sealed evidence: the frozen experiment's arms, the 320 terminal runs' test_loss, and
each run's own compute record. Reports per width, never pooled; the exploratory 2k pilot arm is
not part of this study.

    python tools/s7/s7_scaling_analyse.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.funnel.campaign import paired_effect
from cvslab.schemas import Run, TERMINAL_RUN_STATUSES
from cvslab.service import LabService
from cvslab.store import Store

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools" / "s7"
STATE = TOOLS / "s7-scaling-state.json"
RESULT = TOOLS / "s7-scaling-result.json"
SUMMARY = TOOLS / "s7-scaling-result-summary.json"
SCALES = ("2x", "5x", "10x", "20x")
DEPTHS = ("400k", "160k", "64k", "16k")
WIDTHS = (1, 4, 16, 32)
SEEDS = (0, 1, 2, 3, 4)


def main() -> int:
    st = json.loads(STATE.read_text(encoding="utf-8"))
    service = LabService(Store(str(LAB / "labstore")))
    experiment_id = st["experiment"]["id"]
    experiment = service.store.get(experiment_id)

    arm_of: dict = {}
    for arm in experiment.arms:
        for ablation_id in arm.ablations:
            arm_of[ablation_id] = arm.arm_id

    values: dict = {}       # (arm_id, width) -> {seed: test_loss}
    compute: dict = {}      # (arm_id, width) -> student-side compute
    invalid: list = []
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != experiment_id:
            continue
        arm_id = arm_of.get(run.ablation_id)
        if arm_id is None:
            continue
        width = str(run.effective_config["H"])
        if run.status != TERMINAL_RUN_STATUSES and run.status.value not in ("COMPLETED", "INVALID"):
            continue
        if run.status.value == "INVALID":
            invalid.append(run.id)
            continue
        metric = next((m for m in run.metrics if m.name == "test_loss"), None)
        if metric is None:
            invalid.append(run.id)
            continue
        values.setdefault((arm_id, width), {})[run.seed] = metric.value
        key = f"{arm_id}/H{width}"
        compute[key] = {"rows": run.compute.accepted_examples - 200,
                        "examples_seen": run.compute.train_examples_seen,
                        "cpu_seconds": run.compute.cpu_seconds, "wall_seconds": run.compute.wall_seconds}

    def effect(arm_a: str, arm_b: str, width: int) -> dict:
        return paired_effect(values.get((arm_a, str(width)), {}), values.get((arm_b, str(width)), {}),
                             seeds=SEEDS)

    def block(pairs: dict) -> dict:
        return {name: {str(width): {"a": a, "b": b, **effect(a, b, width)} for width in WIDTHS}
                for name, (a, b) in pairs.items()}

    frontier = block({f"scale {scale}": (f"{scale}-16k", f"{scale}-400k") for scale in SCALES})
    adjacent = {}
    for scale in SCALES:
        adjacent[f"{scale}: 160k-400k"] = (f"{scale}-160k", f"{scale}-400k")
        adjacent[f"{scale}: 64k-160k"] = (f"{scale}-64k", f"{scale}-160k")
        adjacent[f"{scale}: 16k-64k"] = (f"{scale}-16k", f"{scale}-64k")
    adjacent = block(adjacent)
    scaling = block({f"{depth}": (f"20x-{depth}", f"2x-{depth}") for depth in DEPTHS})
    steps = block({f"{depth}: 5x-2x": (f"5x-{depth}", f"2x-{depth}") for depth in DEPTHS})
    steps.update(block({f"{depth}: 10x-5x": (f"10x-{depth}", f"5x-{depth}") for depth in DEPTHS}))
    steps.update(block({f"{depth}: 20x-10x": (f"20x-{depth}", f"10x-{depth}") for depth in DEPTHS}))

    # where does each depth's curve turn, and how does the depth profile move from 2x to 20x?
    profile = {scale: {depth: {str(width): (sum(values[(f"{scale}-{depth}", str(width))].values()) /
                                            len(values[(f"{scale}-{depth}", str(width))]))
                                 if (f"{scale}-{depth}", str(width)) in values else None
                                 for width in WIDTHS} for depth in DEPTHS} for scale in SCALES}
    best_depth = {scale: {str(width): min((depth for depth in DEPTHS
                                           if profile[scale][depth][str(width)] is not None),
                                          key=lambda depth: profile[scale][depth][str(width)],
                                          default=None)
                          for width in WIDTHS} for scale in SCALES}

    result = {
        "experiment": experiment_id,
        "preregistration_hash": experiment.preregistration_hash,
        "unit": "paired seed difference over five seeds, 95% CI (Student t, df=4, t=2.776); negative favours the first term",
        "primary_endpoint": "test_loss on E0004/D0010",
        "cells": {"expected": experiment.expected_run_count, "recorded": len(values) * len(SEEDS),
                  "invalid_runs": invalid},
        "frontier_16k_minus_400k": frontier,
        "adjacent_depths": adjacent,
        "scaling_20x_minus_2x": scaling,
        "scaling_steps": steps,
        "mean_test_loss": profile,
        "best_depth_by_scale": best_depth,
        "student_compute": compute,
        "realized_teacher_nodes": {arm.arm_id: {"node_budget": arm.node_budget,
                                                "scale_nodes": arm.scale_nodes,
                                                "prefix_size": arm.prefix_size,
                                                "realized_nodes": arm.realized_nodes}
                                   for arm in experiment.arms},
    }
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    def verdict(cells: dict) -> str:
        if all(cells[str(width)]["ci_high"] < 0 for width in WIDTHS):
            return "WIDE_SHALLOW_BETTER_AT_EVERY_WIDTH"
        if all(cells[str(width)]["ci_low"] > 0 for width in WIDTHS):
            return "NARROW_DEEP_BETTER_AT_EVERY_WIDTH"
        return "UNRESOLVED_ON_THIS_SCALE"

    summary = {
        "experiment": experiment_id,
        "cells_expected": experiment.expected_run_count,
        "invalid_runs": invalid,
        "frontier": {scale: verdict(frontier[f"scale {scale}"]) for scale in SCALES},
        "frontier_effects": {scale: {str(width): {"effect": round(frontier[f"scale {scale}"][str(width)]["width_estimate"], 6),
                                                  "ci": [round(frontier[f"scale {scale}"][str(width)]["ci_low"], 6),
                                                         round(frontier[f"scale {scale}"][str(width)]["ci_high"], 6)]}
                                     for width in WIDTHS} for scale in SCALES},
        "scaling_20x_minus_2x": {depth: {str(width): {"effect": round(scaling[depth][str(width)]["width_estimate"], 6),
                                                       "ci": [round(scaling[depth][str(width)]["ci_low"], 6),
                                                              round(scaling[depth][str(width)]["ci_high"], 6)]}
                                          for width in WIDTHS} for depth in DEPTHS},
        "best_depth_by_scale": best_depth,
        "student_compute_note": ("broader arms receive more optimizer work under the fixed-epoch recipe; "
                                 "the claim is about supervision scaling, not about teacher compute alone"),
        "evidence": {"result": "tools/s7/s7-scaling-result.json",
                     "preregistration": "tools/s7/s7-scaling-preregistration.json",
                     "state": "tools/s7/s7-scaling-state.json"},
    }
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"frontier": summary["frontier"], "best_depth_by_scale": best_depth}, indent=1))
    print("written", RESULT, "and", SUMMARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
