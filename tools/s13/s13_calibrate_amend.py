"""S13 calibration amendment: the convergence rule failed, so the design controls on cost and dose.

Attempt 1 (`tools/s13/s13-calibration.json`, hash `sha256:478aea89…`) applied the preregistered
rule "cheapest rung whose labels agree with the 2M reference at >= 0.99 sign, >= 0.99 best move and
<= 15 mean |dcp|". NO rung qualified, not even 4M (best-move agreement 0.898, mean |dcp| 14.6), and
the tool refused to choose a budget. That is a result, not an obstacle: **Stockfish labels are not
converged at any affordable budget**, so "the converged SF teacher" cannot be the design's control.

Two facts then decide the training budgets (both measured before any student existed):

* **Equal labeling spend is unreachable.** CVS-4k costs 7.5 ms/label (measured on the same 64
  positions, realized 3,939 nodes); the cheapest calibrated SF rung (32k nodes) costs 50 ms — 6.7x.
  A cost-matched SF budget would sit near ~5k nodes, below the calibrated ladder and far outside the
  region where SF's sign is stable.
* **Deeper is not converged, it is just more.** Sign agreement vs 2M: 0.983 (32k) -> 1.000 (128k)
  -> 0.983 (512k) -> 1.000 (1M); best-move agreement climbs slowly: 0.746 -> 0.797 -> 0.814 ->
  0.848 -> 0.898 (4M).

Amended rule (frozen before any training run):

* **SF-32k** is the PRIMARY Stockfish arm: the cheapest calibrated rung, 6.7x CVS-4k's labeling cost.
  Because it is MORE expensive than the CVS arm, any CVS win cannot be a spending artifact and any
  SF win must be reported at that price.
* **SF-1M** is the DOSE arm: 173x CVS-4k's labeling cost, 51 min of labeling for the 18,953 rows.
  It bounds "is this authority or budget?" — if the 32k -> 1M step does not move the student, the
  effect is authority-shaped, not budget-shaped.
* **SF-DEEP exam** = 4,000,000 nodes/label over the 200 held-out identities (measured 5.0 s/label,
  mean depth 35.1, deterministic: 8/8 repeats byte-identical in cp and best move).

All three SF arms use the same 18,953 position identities as the CVS arm (D0036's prefix), the same
student compute, the same seeds, and differ only in teacher authority and label budget.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from s13_common import S13, identity_canonical, write_json

CVS_MEASUREMENT = {"budget": {"nodeBudget": 4000}, "positions": 64,
                   "mean_wall_ms": 7.5, "median_wall_ms": 6.5, "p10_wall_ms": 5.0, "p90_wall_ms": 7.8,
                   "realized_nodes_mean": 3939, "realized_nodes_min": 78,
                   "under_run_rows": 1,
                   "method": "AnalyzeTransport + AnalyzeSearchProvider over the same 64 calibration "
                             "positions, wall clock per label including the JSON-lines round trip"}
AMENDED = {"primary_arm": {"label": "SF-32k", "nodes": 32000},
           "dose_arm": {"label": "SF-1M", "nodes": 1_000_000},
           "exam": {"label": "SF-DEEP", "nodes": 4_000_000},
           "rule": "no rung is converged; equal spend is unreachable; control on the cheapest "
                   "calibrated rung plus one dose rung, and report the economic ladder in full"}
COST_MULTIPLES = {"SF-32k": 50 / 7.5, "SF-1M": 1299 / 7.5, "SF-DEEP": 5014 / 7.5}


def main() -> int:
    calibration = json.loads((S13 / "s13-calibration.json").read_text(encoding="utf-8"))
    amendment = {
        "id": "s13-calibration-amendment",
        "supersedes_decision_not_measurements": "tools/s13/s13-calibration.json (unchanged; its "
                                                "comparison table is reused verbatim)",
        "attempt_1": {"rule": calibration["rule"], "outcome": "no candidate qualified",
                      "artifact": "tools/s13/s13-calibration.json",
                      "hash": (S13 / "s13-calibration.hash").read_text(encoding="utf-8").strip()},
        "why_the_rule_failed": "Stockfish labels are not converged at any affordable budget: even "
                               "4M-vs-2M best-move agreement is 0.898",
        "cvs_cost_measurement": CVS_MEASUREMENT,
        "cost_multiples_vs_cvs_4k": COST_MULTIPLES,
        "amended_rule": AMENDED,
        "teacher": identity_canonical(),
        "measured_table": {k: {kk: vv for kk, vv in v.items()}
                           for k, v in calibration["comparisons"].items()},
        "determinism": calibration["determinism"],
        "guardrail": "chosen before any student exists; no downstream outcome was inspected",
    }
    digest = write_json(S13 / "s13-calibration-amendment.json", amendment)
    (S13 / "s13-calibration-amendment.hash").write_text(digest + "\n", encoding="utf-8")
    print(f"amendment written: {digest}")
    print(json.dumps({"primary": AMENDED["primary_arm"], "dose": AMENDED["dose_arm"],
                      "exam": AMENDED["exam"], "cost_multiples": COST_MULTIPLES}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
