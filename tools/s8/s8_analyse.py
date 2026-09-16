"""S8 analysis: the lower frontier under matched student compute, and the regime comparison.

Primary branch: fixed updates (equal student optimizer work per arm) — 2k − 16k per width, with
the preregistered decision rule. Descriptive: the adjacent depth steps, the same contrasts under
the fixed-epoch regime, and the between-regime comparison of the depth effect.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.funnel.campaign import paired_effect
from cvslab.schemas import Run, RunStatus, TERMINAL_RUN_STATUSES
from cvslab.service import LabService
from cvslab.store import Store

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
S8 = LAB / "tools" / "s8"
STATE = S8 / "s8-state.json"
RESULT = S8 / "s8-result.json"
SUMMARY = S8 / "s8-result-summary.json"
WIDTHS = (1, 4, 16, 32)
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("16k", "8k", "4k", "2k")
REGIMES = ("updates", "epochs")


def main() -> int:
    st = json.loads(STATE.read_text(encoding="utf-8"))
    service = LabService(Store(str(LAB / "labstore")))
    experiment = service.store.get(st["experiment"]["id"])

    arm_of: dict = {}
    for arm in experiment.arms:
        for ablation_id in arm.ablations:
            arm_of[ablation_id] = arm.arm_id          # e.g. "2k-updates"

    values: dict = {}
    compute: dict = {}
    invalid: list = []
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != experiment.id:
            continue
        key = arm_of.get(run.ablation_id)
        if key is None:
            continue
        if run.status == RunStatus.COMPLETED:
            metric = next((m for m in run.metrics if m.name == "test_loss"), None)
            if metric is None:
                invalid.append(run.id)
                continue
            values.setdefault(key, {}).setdefault(str(run.effective_config["H"]), {})[run.seed] = metric.value
            compute[f"{key}/H{run.effective_config['H']}"] = {
                "rows": run.compute.accepted_examples - 200,
                "updates": int(run.effective_config.get("MAX_UPDATES", 0) or 0) or None,
                "epochs": int(run.effective_config["EPOCHS"]),
                "examples_seen": run.compute.train_examples_seen,
                "cpu_seconds": run.compute.cpu_seconds, "wall_seconds": run.compute.wall_seconds}
        else:
            invalid.append(run.id)

    def effect(a: str, b: str, width: int) -> dict:
        return paired_effect(values.get(a, {}).get(str(width), {}), values.get(b, {}).get(str(width), {}),
                             seeds=SEEDS)

    def block(pairs: dict) -> dict:
        return {name: {str(width): {"a": a, "b": b, **effect(a, b, width)} for width in WIDTHS}
                for name, (a, b) in pairs.items()}

    def regime_block(regime: str) -> dict:
        steps = {f"{depth}-16k": (f"{depth}-{regime}", f"16k-{regime}") for depth in ("8k", "4k", "2k")}
        steps.update({"8k-16k": ("8k-" + regime, "16k-" + regime),
                      "4k-8k": ("4k-" + regime, "8k-" + regime),
                      "2k-4k": ("2k-" + regime, "4k-" + regime),
                      "2k-16k": ("2k-" + regime, "16k-" + regime)})
        return block(steps)

    primary = block({"2k-16k": ("2k-updates", "16k-updates"),
                     "8k-16k": ("8k-updates", "16k-updates"),
                     "4k-8k": ("4k-updates", "8k-updates"),
                     "2k-4k": ("2k-updates", "4k-updates")})
    epochs_branch = regime_block("epochs")

    def decide(cells: dict) -> str:
        if all(cells[str(width)]["ci_high"] < 0 for width in WIDTHS):
            return "LOWER_FRONTIER_EXTENDS"
        if all(cells[str(width)]["ci_low"] > 0 for width in WIDTHS):
            return "TWO_K_TOO_SHALLOW"
        return "INCONCLUSIVE"

    decision = decide(primary["2k-16k"])
    mean_level = {arm: statistics.mean(v for widths in values.get(arm, {}).values() for v in widths.values())
                  for arm in values}

    between_regimes = {str(width): {
        "matched_student_updates": primary["2k-16k"][str(width)]["width_estimate"],
        "fixed_epochs": epochs_branch["2k-16k"][str(width)]["width_estimate"],
        "difference": primary["2k-16k"][str(width)]["width_estimate"]
                      - epochs_branch["2k-16k"][str(width)]["width_estimate"]}
        for width in WIDTHS}

    result = {
        "experiment": experiment.id, "preregistration_hash": experiment.preregistration_hash,
        "cells": {"expected": experiment.expected_run_count, "invalid_runs": invalid},
        "unit": "paired seed difference over five seeds, 95% CI (t=2.776); negative favours the first term",
        "primary_branch": "fixed_updates (equal student optimizer work per arm)",
        "primary_2k_minus_16k": primary["2k-16k"],
        "adjacent_matched": {name: primary[name] for name in ("8k-16k", "4k-8k", "2k-4k")},
        "fixed_epochs_branch": epochs_branch,
        "between_regimes_2k_minus_16k": between_regimes,
        "decision": decision,
        "mean_test_loss_by_arm": mean_level,
        "student_compute": compute,
        "arms": [{"arm_id": arm.arm_id, "node_budget": arm.node_budget, "prefix_size": arm.prefix_size,
                  "realized_nodes": arm.realized_nodes} for arm in experiment.arms],
    }
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "experiment": experiment.id, "decision": decision,
        "primary_2k_minus_16k": {str(width): {"effect": round(primary["2k-16k"][str(width)]["width_estimate"], 6),
                                              "ci": [round(primary["2k-16k"][str(width)]["ci_low"], 6),
                                                     round(primary["2k-16k"][str(width)]["ci_high"], 6)]}
                                 for width in WIDTHS},
        "adjacent_matched": {name: {str(width): {"effect": round(primary[name][str(width)]["width_estimate"], 6),
                                                 "ci": [round(primary[name][str(width)]["ci_low"], 6),
                                                        round(primary[name][str(width)]["ci_high"], 6)]}
                                    for width in WIDTHS} for name in ("8k-16k", "4k-8k", "2k-4k")},
        "fixed_epochs_2k_minus_16k": {str(width): {"effect": round(epochs_branch["2k-16k"][str(width)]["width_estimate"], 6),
                                                   "ci": [round(epochs_branch["2k-16k"][str(width)]["ci_low"], 6),
                                                          round(epochs_branch["2k-16k"][str(width)]["ci_high"], 6)]}
                                      for width in WIDTHS},
        "between_regimes": between_regimes,
        "mean_test_loss_by_arm": mean_level,
        "invalid_runs": invalid,
        "evidence": {"result": "tools/s8/s8-result.json", "preregistration": "tools/s8/s8-preregistration.json",
                     "state": "tools/s8/s8-state.json"},
    }
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision,
                      "primary_2k_minus_16k": summary["primary_2k_minus_16k"],
                      "adjacent_matched": summary["adjacent_matched"],
                      "fixed_epochs_2k_minus_16k": summary["fixed_epochs_2k_minus_16k"],
                      "between_regimes": {w: round(v["difference"], 6) for w, v in between_regimes.items()},
                      "mean_by_arm": {k: round(v, 5) for k, v in mean_level.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
