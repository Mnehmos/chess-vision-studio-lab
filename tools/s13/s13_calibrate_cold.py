"""S13 cold calibration: re-run the cost/stability ladder under the cold-search contract.

The replacement experiment's Stockfish labels are cold per label (`ucinewgame` + `Clear Hash` +
`isready` before every search). This re-measures the ladder under that contract, on the same frozen
64-position calibration set, against the same 2M reference, and applies the SAME rules as written:

* if any rung reaches the original convergence bar (sign >= 0.99, best move >= 0.99, mean |dcp| <=
  15), the primary condition is the cheapest such rung;
* otherwise the primary condition is the cheapest rung (the amendment's branch), with SF-1M as the
  dose arm and 4M as the SF-DEEP exam budget.

Nothing here inspects a student, a loss, or any downstream metric — cost and stability only.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from s13_common import OUT, S13, buy, identity_canonical, order, record_fens, write_json

CALIBRATION_N = 64
DETERMINISM_N = 8
CANDIDATES = (32_000, 128_000, 512_000, 1_000_000, 4_000_000)
REFERENCE = 2_000_000
SIGN_MIN, MOVE_MIN, MEAN_ABS_CP_MAX = 0.99, 0.99, 15.0
DOSE_BUDGET = 1_000_000
EXAM_BUDGET = 4_000_000


def _cp(row: dict):
    cp = row["value"]["scoreCpStm"]
    return None if cp is None or abs(cp) >= 100_000 else cp


def _compare(rows: list[dict], reference: list[dict]) -> dict:
    pairs = [(a, b) for a, b in zip(rows, reference) if _cp(a) is not None and _cp(b) is not None]
    mate_rows = sum(1 for row in rows if _cp(row) is None)
    diffs = [abs(_cp(a) - _cp(b)) for a, b in pairs]
    n = max(len(pairs), 1)
    return {"positions": len(rows), "non_mate_pairs": len(pairs), "mate_rows": mate_rows,
            "sign_agreement": sum(1 for a, b in pairs
                                  if (_cp(a) > 0) == (_cp(b) > 0)) / n,
            "best_move_agreement": sum(1 for a, b in pairs
                                       if a["value"]["bestMove"] == b["value"]["bestMove"]) / n,
            "mean_abs_cp": (sum(diffs) / len(diffs)) if diffs else None,
            "median_abs_cp": sorted(diffs)[len(diffs) // 2] if diffs else None,
            "mean_depth": sum(row["value"]["depth"] for row in rows) / len(rows),
            "mean_nodes": sum(row["budget_nodes"] for row in rows) / len(rows),
            "mean_wall_ms": sum(row["wall_ms"] for row in rows) / len(rows)}


def _qualifies(comparison: dict) -> bool:
    return (comparison["sign_agreement"] >= SIGN_MIN
            and comparison["best_move_agreement"] >= MOVE_MIN
            and comparison["mean_abs_cp"] is not None
            and comparison["mean_abs_cp"] <= MEAN_ABS_CP_MAX)


def _determinism(items, budget: int) -> dict:
    first = buy(items, budget, workers=4, log=False, cold=True)
    second = buy(items, budget, workers=12, log=False, cold=True)
    same = sum(1 for a, b in zip(first, second)
               if (a["value"]["scoreCpStm"], a["value"]["bestMove"], a["budget_nodes"])
               == (b["value"]["scoreCpStm"], b["value"]["bestMove"], b["budget_nodes"]))
    return {"positions": len(items), "identical": same, "workers_compared": [4, 12],
            "deterministic": same == len(items)}


def main() -> int:
    started = time.time()
    ids = order()[:CALIBRATION_N]
    fens = record_fens(ids)
    items = [(rid, fens[rid]) for rid in ids]
    identity = identity_canonical()
    print(f"cold calibration: {len(items)} positions, identity {identity['binarySha256'][:16]}…",
          flush=True)
    reference = buy(items, REFERENCE, workers=8, cold=True)
    table = {}
    for budget in CANDIDATES:
        rows = buy(items, budget, workers=8, cold=True)
        table[str(budget)] = _compare(rows, reference)
        print(f"{budget:>9}: sign {table[str(budget)]['sign_agreement']:.4f} move "
              f"{table[str(budget)]['best_move_agreement']:.4f} mean|dcp| "
              f"{table[str(budget)]['mean_abs_cp']:.1f} depth {table[str(budget)]['mean_depth']:.1f} "
              f"wall {table[str(budget)]['mean_wall_ms']:.0f}ms", flush=True)

    converged = [budget for budget in CANDIDATES if _qualifies(table[str(budget)])]
    if converged:
        primary, branch = min(converged), "original rule: cheapest converged rung"
    else:
        primary, branch = min(CANDIDATES), "amendment branch: no rung converged, cheapest rung"
    tick = max(1, len(items) // DETERMINISM_N)
    determinism = {str(budget): _determinism(items[::tick], budget)
                   for budget in sorted({primary, DOSE_BUDGET, EXAM_BUDGET})}
    artifact = {
        "id": "s13-calibration-cold",
        "purpose": "choose the Stockfish TRAINING budget under the COLD contract on cost and "
                   "stability; choose the SF-DEEP exam setting; inspect no student outcome",
        "contract": "cold per label: ucinewgame + setoption name Clear Hash + isready before every "
                    "search (readyok is the sync point)",
        "supersedes_measurements_in": "tools/s13/s13-calibration.json (warm contract)",
        "teacher": {"exe": identity["binary"], "binarySha256": identity["binarySha256"],
                    "uciName": identity["name"], "nets": identity["nets"],
                    "options": identity["options"]},
        "calibration_set": {"ids": ids, "count": len(items),
                            "source": "first 64 ids of the frozen S12 universe order"},
        "rule": {"candidates": list(CANDIDATES), "reference_nodes": REFERENCE,
                 "qualifies": {"sign_agreement_min": SIGN_MIN, "best_move_agreement_min": MOVE_MIN,
                               "mean_abs_cp_max": MEAN_ABS_CP_MAX},
                 "branch_taken": branch,
                 "mate_rows": "|cp| >= 100000 excluded from cp statistics and counted"},
        "comparisons": table,
        "chosen_training_budget": primary,
        "dose_budget": DOSE_BUDGET,
        "exam_budget": EXAM_BUDGET,
        "determinism_across_worker_counts": determinism,
        "wall_seconds": round(time.time() - started, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    digest = write_json(S13 / "s13-calibration-cold.json", artifact)
    (S13 / "s13-calibration-cold.hash").write_text(digest + "\n", encoding="utf-8")
    print(f"branch: {branch}; primary {primary}; dose {DOSE_BUDGET}; exam {EXAM_BUDGET}; {digest}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
