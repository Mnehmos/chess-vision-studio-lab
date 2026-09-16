"""S11 confirmatory analysis: fresh seeds as the confirmatory test, full sets as precision estimates.

S11 chose its sample sizes from S10 seeds 0-19 and then included those same observations in its
intervals, which makes the all-seed interval a data-dependent precision estimate. The fix uses the
data the experiment already contains:

    confirmatory (PRIMARY):   seeds 20-39 for H1/H4/H16, seeds 20-399 for H32
    precision (SECONDARY):    every seed the experiment ran (0-39 / 0-399)

Both are computed with exact Student-t criticals at each width's own degrees of freedom
(cvslab.funnel.campaign.t_critical), so n=40 and n=400 no longer fall back to the normal values.

    python tools/s11/s11_confirmatory.py --experiment X0006
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.funnel.campaign import paired_effect, t_critical
from cvslab.schemas import Run, RunStatus, TERMINAL_RUN_STATUSES
from cvslab.service import LabService
from cvslab.store import Store

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
S11 = LAB / "tools" / "s11"
WIDTHS = (1, 4, 16, 32)
FRESH = {1: range(20, 40), 4: range(20, 40), 16: range(20, 40), 32: range(20, 400)}
DELTA = 0.001


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="X0006")
    parser.add_argument("--out", default="s11-confirmatory.json")
    args = parser.parse_args()
    service = LabService(Store(str(LAB / "labstore")))
    experiment = service.store.get(args.experiment)
    arm_of = {ablation_id: arm.arm_id for arm in experiment.arms for ablation_id in arm.ablations}

    values: dict = {}          # (arm, width) -> {seed: test_loss}
    invalid: list = []
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != experiment.id:
            continue
        arm_id = arm_of.get(run.ablation_id)
        if arm_id is None:
            continue
        if run.status != RunStatus.COMPLETED:
            invalid.append(run.id)
            continue
        metric = next((m for m in run.metrics if m.name == "test_loss"), None)
        if metric is None:
            invalid.append(run.id)
            continue
        depth = "4K" if arm_id.startswith("4K") else "8K"
        width = int(arm_id.split("H")[-1])
        values.setdefault((depth, width), {})[run.seed] = metric.value

    def contrast(width: int, seeds) -> dict:
        seeds = tuple(sorted(seeds))
        a = {seed: values.get(("4K", width), {}).get(seed) for seed in seeds}
        b = {seed: values.get(("8K", width), {}).get(seed) for seed in seeds}
        missing = [seed for seed in seeds if a[seed] is None or b[seed] is None]
        if missing:
            raise SystemExit(f"STOP: H{width} lacks runs for seeds {missing[:5]}")
        one = paired_effect(a, b, seeds=seeds, one_sided=True)
        two = paired_effect(a, b, seeds=seeds, one_sided=False)
        return {"n": one["n"], "d": one["width_estimate"],
                "one_sided_low": one["ci_low"], "one_sided_high": one["ci_high"],
                "two_sided_ci": [two["ci_low"], two["ci_high"]],
                "critical_one_sided": t_critical(one["n"] - 1, one_sided=True),
                "critical_two_sided": t_critical(two["n"] - 1),
                "seeds": list(seeds)}

    confirmatory = {str(width): contrast(width, FRESH[width]) for width in WIDTHS}
    precision = {str(width): contrast(width, values.get(("4K", width), {}).keys()) for width in WIDTHS}
    decisions = {}
    for label, cells, keys in (("confirmatory_fresh_seeds", confirmatory, FRESH),
                               ("precision_all_seeds", precision, None)):
        upper = {width: cells[str(width)]["one_sided_high"] for width in WIDTHS}
        lower = {width: cells[str(width)]["one_sided_low"] for width in WIDTHS}
        if all(upper[width] < DELTA for width in WIDTHS):
            verdict = "4K_NONINFERIOR"
        elif all(lower[width] > DELTA for width in WIDTHS):
            verdict = "4K_TAXED"
        else:
            verdict = "INCONCLUSIVE"
        decisions[label] = {"verdict": verdict,
                            "widths_inside_margin": [width for width in WIDTHS if upper[width] < DELTA],
                            "widths_outside_margin": [width for width in WIDTHS if upper[width] >= DELTA],
                            "max_upper_bound": max(upper.values())}

    report = {"experiment": experiment.id, "preregistration_hash": experiment.preregistration_hash,
              "margin_delta": DELTA,
              "confirmatory_rule": "fresh seeds only: 20-39 for H1/H4/H16, 20-399 for H32",
              "invalid_runs": invalid,
              "confirmatory_fresh_seeds": confirmatory,
              "precision_all_seeds": precision,
              "decisions": decisions}
    (S11 / args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"experiment": experiment.id, "margin_delta": DELTA,
               "decisions": decisions,
               "confirmatory_fresh_seeds": {width: {"n": cells["n"], "d": round(cells["d"], 6),
                                                    "one_sided_bounds": [round(cells["one_sided_low"], 6),
                                                                         round(cells["one_sided_high"], 6)],
                                                    "t_one_sided": round(cells["critical_one_sided"], 4)}
                                            for width, cells in confirmatory.items()},
               "precision_all_seeds": {width: {"n": cells["n"], "d": round(cells["d"], 6),
                                               "one_sided_bounds": [round(cells["one_sided_low"], 6),
                                                                    round(cells["one_sided_high"], 6)],
                                               "t_one_sided": round(cells["critical_one_sided"], 4)}
                                       for width, cells in precision.items()}}
    (S11 / args.out.replace(".json", "-summary.json")).write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
