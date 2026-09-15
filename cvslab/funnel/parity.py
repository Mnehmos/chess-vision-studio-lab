"""Fixture parity harness: lab providers vs the frozen legacy funnel (issue #8, plan §8).

Consumes a legacy funnel run directory (`positions.jsonl`, `tier0.jsonl`,
`tier1.jsonl`) and replays the same positions through the lab providers,
comparing:

* **tier0** — exact JSON equality of `status`, `deterministic_geometry`,
  `bounded_tactical_proof`, `taxonomy`, `uncomputed`, `factsErrors` and
  `factsRegistryVersion`;
* **tier1** — the full compact-search scientific payload per node budget:
  `scoreCpStm`, `mate`, `bestMove`, `pv`, `nodes`, `depth`, `trajectory`,
  `stabilization`, `termination`, `resultSource`, `qNodes`, `avgCutoffMoveIndex`,
  `avgLegalMoves`.

Timing fields are compared with **declared tolerances** (`TOLERANCES`) rather than
exact equality; every non-scientific path that is skipped must be listed there
with a reason. Differences that are intentional must be declared in
`INTENTIONAL_DIFFS`, so drift can never pass silently.

Exit code 0 = parity; 1 = mismatches; 2 = usage error.

Usage:
    python -m cvslab.funnel.parity --run-dir path/to/legacy/run \
        --engine-root F:/Github/chess-vision-studio-rust-engine [--limit 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .providers import AnalyzeFactsProvider, AnalyzeSearchProvider, AnalyzeTransport
from .protocols import Position

# Scientific payload keys compared for tier1 (exact equality).
TIER1_KEYS = ("scoreCpStm", "mate", "bestMove", "pv", "nodes", "depth", "trajectory",
              "stabilization", "termination", "resultSource", "qNodes",
              "avgCutoffMoveIndex", "avgLegalMoves")

# Declared tolerances for timing/cost paths: (path -> (abs_tol, reason)).
# Exact comparison is the default; anything timing-shaped must appear here.
TOLERANCES: dict[str, tuple[float, str]] = {
    "tier1.timeMs": (float("inf"), "wall-clock timing; not scientific payload"),
    "tier1.cost.wallMs": (float("inf"), "wall-clock timing; not scientific payload"),
    "tier1.cost.nodes": (0.0, "node counts are equal by construction; kept for completeness"),
}

# Paths where a difference is accepted, each with a reason.
INTENTIONAL_DIFFS: dict[str, str] = {}

DEFAULT_ANALYZE_ARGS = [
    "--depth", "64",
    "--nnue", "target-cvs/matrix-raw.json",
    "--nnue-cal", "nets/eval-cal.json",
    "--helper-nnue", "target-cvs/matrix-residual.json",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def diff_paths(left, right, prefix: str = "") -> list[str]:
    """Dotted paths where two JSON-ish values differ (tolerance-aware at leaves)."""
    if isinstance(left, dict) and isinstance(right, dict):
        out = []
        for key in sorted(set(left) | set(right)):
            out += diff_paths(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else str(key))
        return out
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{prefix}.length({len(left)}!={len(right)})"]
        out = []
        for index, (a, b) in enumerate(zip(left, right)):
            out += diff_paths(a, b, f"{prefix}[{index}]")
        return out
    if prefix in TOLERANCES:
        tolerance, _reason = TOLERANCES[prefix]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return [] if abs(left - right) <= tolerance else [prefix]
        return [] if left == right else [prefix]
    return [] if left == right else [prefix or "<root>"]


def compare_tier0(legacy: dict, lab: dict) -> list[str]:
    mismatches = []
    for key in ("status", "deterministic_geometry", "bounded_tactical_proof", "taxonomy",
                "uncomputed", "factsErrors", "factsRegistryVersion"):
        if key in INTENTIONAL_DIFFS:
            continue
        mismatches += [path for path in diff_paths(legacy.get(key), lab.get(key), key)
                       if path not in INTENTIONAL_DIFFS]
    return mismatches


def compare_tier1(legacy: dict, lab_search_by_budget: dict[int, dict]) -> list[str]:
    mismatches = []
    for run in legacy.get("search_derived", {}).get("budgets", []):
        budget = run.get("nodeBudget")
        lab = lab_search_by_budget.get(budget)
        if lab is None:
            mismatches.append(f"tier1.budgets[{budget}].missing")
            continue
        for key in TIER1_KEYS:
            if f"tier1.{key}" in INTENTIONAL_DIFFS:
                continue
            mismatches += [path for path in diff_paths(run.get(key), lab.get(key), f"tier1.{key}")
                           if path not in INTENTIONAL_DIFFS]
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cvslab.funnel.parity")
    parser.add_argument("--run-dir", required=True, help="legacy funnel run directory")
    parser.add_argument("--engine-root", required=True, help="legacy engine checkout (artifact root)")
    parser.add_argument("--limit", type=int, default=0, help="compare at most N positions")
    parser.add_argument("--pv-plies", type=int, default=8)
    parser.add_argument("--tier1-max-budget", type=int, default=16000,
                        help="only replay tier1 budgets up to this node budget (cost control)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    positions = read_jsonl(run_dir / "positions.jsonl")
    legacy_tier0 = {row["id"]: row for row in read_jsonl(run_dir / "tier0.jsonl")}
    legacy_tier1 = {row["id"]: row for row in read_jsonl(run_dir / "tier1.jsonl")}
    if args.limit:
        positions = positions[: args.limit]
    if not positions:
        print("no positions to compare", file=sys.stderr)
        return 2

    transport = AnalyzeTransport(args.engine_root, args=DEFAULT_ANALYZE_ARGS, cwd=args.engine_root)
    facts_provider = AnalyzeFactsProvider(transport)
    search_provider = AnalyzeSearchProvider(transport, family="search_shallow_cp")

    counts = {"tier0": {"compared": 0, "mismatched": 0}, "tier1": {"compared": 0, "mismatched": 0}}
    try:
        for row in positions:
            position = Position(fen=row["fen"])
            legacy0 = legacy_tier0.get(row["id"])
            if legacy0 is not None:
                batch = facts_provider.label(position)
                lab_record = batch.rows[0] if batch.rows else {}
                mismatches = (compare_tier0(legacy0, lab_record) if batch.rows
                              else [f"lab.provider.uncomputed={batch.uncomputed}"])
                counts["tier0"]["compared"] += 1
                if mismatches:
                    counts["tier0"]["mismatched"] += 1
                    print(f"MISMATCH tier0 {row['id']}: {mismatches[:6]}")
            legacy1 = legacy_tier1.get(row["id"])
            if legacy1 is not None:
                budgets = sorted(r.get("nodeBudget") for r in legacy1.get("search_derived", {}).get("budgets", []))
                lab_search: dict[int, dict] = {}
                for budget in budgets:
                    if budget and budget <= args.tier1_max_budget:
                        search = search_provider.search(position, node_budget=budget, pv_plies=args.pv_plies)
                        lab_search[budget] = {"scoreCpStm": search.score_cp_stm, "mate": search.mate,
                                              "bestMove": search.best_move, "pv": list(search.pv),
                                              "nodes": search.nodes, **search.extra}
                mismatches = compare_tier1(legacy1, lab_search)
                counts["tier1"]["compared"] += 1
                if mismatches:
                    counts["tier1"]["mismatched"] += 1
                    print(f"MISMATCH tier1 {row['id']}: {mismatches[:6]}")
    finally:
        transport.close()

    # Full identities stay canonical; the manifest records the full sha256.
    print(f"analyze sha256: {transport.analyze_sha256}")
    print(f"taxonomy sha256: {transport.taxonomy_sha256} (schema {transport.taxonomy['schemaVersion']})")
    print(f"tier0: compared {counts['tier0']['compared']}, mismatched {counts['tier0']['mismatched']}")
    print(f"tier1: compared {counts['tier1']['compared']}, mismatched {counts['tier1']['mismatched']}"
          f" (payload keys: {len(TIER1_KEYS)}; declared tolerances: {len(TOLERANCES)};"
          f" intentional diffs: {len(INTENTIONAL_DIFFS)})")
    return 0 if counts["tier0"]["mismatched"] + counts["tier1"]["mismatched"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
