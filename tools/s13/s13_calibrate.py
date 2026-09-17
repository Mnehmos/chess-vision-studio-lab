"""S13 calibration: choose the Stockfish training budget on COST and STABILITY only.

Frozen rule (preregistered before any student outcome exists):

  Candidates are ``go nodes`` budgets {32k, 128k, 512k}; the reference is 2M nodes. On a frozen
  64-position calibration set (the first 64 ids of the S12 universe order), a candidate QUALIFIES
  when, over non-mate rows:
    * sign agreement with the reference  >= 0.99, and
    * best-move agreement with the reference >= 0.99, and
    * mean |cp - reference_cp| <= 15.
  The chosen training budget is the CHEAPEST qualifying candidate. If none qualifies, the
  calibration escalates to 1M and 4M and the artifact records that; if still none, calibration
  FAILS and no training happens.

Determinism is checked by re-labelling 8 positions at each budget: a fixed node count with
Threads=1 must reproduce byte-identical labels, otherwise the teacher is not reproducible enough
to freeze.

Nothing here inspects a student, a loss, or any downstream metric — cost and stability only.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from s13_common import (MATE_CP_LIMIT, OUT, S13, buy, identity_canonical, order, record_fens,
                        write_json)

CALIBRATION_N = 64
DETERMINISM_N = 8
CANDIDATES = (32_000, 128_000, 512_000)
ESCALATION = (1_000_000, 4_000_000)
REFERENCE = 2_000_000
SIGN_MIN = 0.99
MOVE_MIN = 0.99
MEAN_ABS_CP_MAX = 15.0


def _cp(row: dict):
    cp = row["value"]["scoreCpStm"]
    return None if cp is None or abs(cp) >= MATE_CP_LIMIT else cp


def _compare(rows: list[dict], reference: list[dict]) -> dict:
    pairs = [(a, b) for a, b in zip(rows, reference) if _cp(a) is not None and _cp(b) is not None]
    mate_rows = sum(1 for row in rows if _cp(row) is None)
    diffs = [abs(_cp(a) - _cp(b)) for a, b in pairs]
    sign_same = sum(1 for a, b in pairs if (_cp(a) > 0) == (_cp(b) > 0) or (_cp(a) == 0 and _cp(b) == 0))
    move_same = sum(1 for a, b in pairs if a["value"]["bestMove"] == b["value"]["bestMove"])
    exact = sum(1 for a, b in pairs if _cp(a) == _cp(b))
    n = max(len(pairs), 1)
    return {"positions": len(rows), "non_mate_pairs": len(pairs), "mate_rows": mate_rows,
            "sign_agreement": sign_same / n, "best_move_agreement": move_same / n,
            "exact_cp_agreement": exact / n,
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
    first = buy(items, budget, workers=4, log=False)
    second = buy(items, budget, workers=4, log=False)
    same_cp = sum(1 for a, b in zip(first, second) if a["value"]["scoreCpStm"] == b["value"]["scoreCpStm"])
    same_move = sum(1 for a, b in zip(first, second)
                    if a["value"]["bestMove"] == b["value"]["bestMove"])
    return {"positions": len(items), "identical_cp": same_cp, "identical_bestmove": same_move,
            "deterministic": same_cp == len(items) and same_move == len(items)}


def main() -> int:
    started = time.time()
    ids = order()[:CALIBRATION_N]
    fens = record_fens(ids)
    items = [(rid, fens[rid]) for rid in ids]
    print(f"calibration set: {len(items)} frozen positions; identity:", flush=True)
    identity = identity_canonical()
    print(json.dumps(identity, indent=1, sort_keys=True), flush=True)

    reference = buy(items, REFERENCE, workers=8)
    print(f"reference at {REFERENCE} nodes: mean depth "
          f"{sum(r['value']['depth'] for r in reference) / len(reference):.1f}", flush=True)
    table = {}
    for budget in CANDIDATES + ESCALATION:
        rows = buy(items, budget, workers=8)
        comparison = _compare(rows, reference)
        table[str(budget)] = comparison
        print(f"{budget:>9}: sign {comparison['sign_agreement']:.4f} move "
              f"{comparison['best_move_agreement']:.4f} mean|dcp| {comparison['mean_abs_cp']} "
              f"depth {comparison['mean_depth']:.1f}", flush=True)

    chosen = None
    for budget in CANDIDATES + ESCALATION:
        if _qualifies(table[str(budget)]):
            chosen = budget
            break
    tick = max(1, len(items) // DETERMINISM_N)
    determinism = {str(budget): _determinism(items[::tick], budget)
                   for budget in ([chosen] if chosen else [])}
    exam_budget = 4_000_000
    determinism[str(exam_budget)] = _determinism(items[::tick], exam_budget)

    artifact = {
        "id": "s13-calibration",
        "purpose": "choose the Stockfish TRAINING budget on cost and stability; choose the SF-DEEP "
                   "exam setting; inspect no student outcome",
        "teacher": {"exe": identity["binary"], "binarySha256": identity["binarySha256"],
                    "uciName": identity["name"], "nets": identity["nets"], "options": identity["options"]},
        "calibration_set": {"ids": ids, "count": len(items),
                            "source": "first 64 ids of the frozen S12 universe order"},
        "rule": {"candidates": list(CANDIDATES), "escalation": list(ESCALATION),
                 "reference_nodes": REFERENCE,
                 "qualifies": {"sign_agreement_min": SIGN_MIN, "best_move_agreement_min": MOVE_MIN,
                               "mean_abs_cp_max": MEAN_ABS_CP_MAX},
                 "choice": "cheapest qualifying candidate; fail if none",
                 "mate_rows": f"|cp| >= {MATE_CP_LIMIT} (mate sentinel) excluded from cp statistics"},
        "comparisons": table,
        "chosen_training_budget": chosen,
        "exam_budget": exam_budget,
        "determinism": determinism,
        "economics": {str(budget): {"mean_wall_ms": table[str(budget)]["mean_wall_ms"],
                                    "mean_nodes": table[str(budget)]["mean_nodes"],
                                    "nodes_per_second": (table[str(budget)]["mean_nodes"]
                                                         / max(table[str(budget)]["mean_wall_ms"] / 1000, 1e-9))}
                      for budget in CANDIDATES + ESCALATION},
        "wall_seconds": round(time.time() - started, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    digest = write_json(S13 / "s13-calibration.json", artifact)
    (S13 / "s13-calibration.hash").write_text(digest + "\n", encoding="utf-8")
    print(f"chosen training budget: {chosen} nodes; exam budget {exam_budget}; artifact {digest}",
          flush=True)
    if chosen is None:
        print("CALIBRATION FAILED: no candidate qualified; no training will run", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
