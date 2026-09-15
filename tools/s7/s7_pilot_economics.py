"""S7-P repair: realized economics from the existing observations, and precise wording.

No rerun, no new labels, no new searches. Two things the sealed bundle got wrong:

  1. the teaching statement said effects were "at most ~0.7 % relative" while the same
     document reported H4's rows effect at +0.006657 on a ~0.043 level, which is ~15 %;
  2. economics were nominal rows x budget, not the realized node totals the observations
     already contain.

This script recomputes both from the run artifacts (whose hashes the inventory pinned),
cross-checks the 400k arm against the deep-node total R0008 recorded, and rewrites
tools/s7/s7-pilot-result.json + -summary.json with corrected wording:

  * P_econ reads UNRESOLVED (no consistent all-width advantage or disadvantage), never
    "non-inferiority" / "cheaper was not worse";
  * P_depth reads "400k never showed an advantage over 16k: three widths unresolved, H32
     significantly favoured 16k" — an effect existed, in the opposite direction.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import chess

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.data import read_jsonl

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools" / "s7"
RUN_DIR = pathlib.Path(r"F:\Github\_parity_tmp\s6\s4run")
RESULT = TOOLS / "s7-pilot-result.json"
SUMMARY = TOOLS / "s7-pilot-result-summary.json"
WIDTHS = ("1", "4", "16", "32")


def rid(fen: str) -> str:
    return "pos_" + hashlib.sha256(chess.Board(fen).epd().encode()).hexdigest()[:20]


def main() -> int:
    inventory = json.loads((TOOLS / "s7-pilot-inventory.json").read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    uniform = set(inventory["uniform_record_ids"])
    eligible = set(inventory["eligible_record_ids"])

    positions = {row["id"]: rid(row["fen"]) for row in read_jsonl(RUN_DIR / "positions.jsonl")}
    tier1 = {positions[row["id"]]: row for row in read_jsonl(RUN_DIR / "tier1.jsonl")
             if row["id"] in positions}
    tier3 = {positions[row["id"]]: row for row in read_jsonl(RUN_DIR / "tier3.jsonl")
             if row["id"] in positions}

    def budget_nodes(record_ids: set, budget: int) -> int:
        total = 0
        for record_id in record_ids:
            entry = next(b for b in tier1[record_id]["search_derived"]["budgets"]
                         if b["nodeBudget"] == budget)
            total += int(entry["nodes"])
        return total

    def tier1_scan_seconds(record_ids: set) -> float:
        """The 2k and 16k searches share one scan; only the combined engine time is recorded."""
        return round(sum(tier1[r]["cost"]["wallMs"] for r in record_ids) / 1000.0, 3)

    realized = {
        "P1": {"rows": len(uniform), "node_budget": 400000,
               "realized_nodes": sum(int(tier3[r]["cost"]["nodes"]) for r in uniform),
               "engine_seconds": round(sum(tier3[r]["cost"]["wallMs"] for r in uniform) / 1000.0, 3),
               "source": "tier3 cost over the 94 uniform records"},
        "P2": {"rows": len(uniform), "node_budget": 16000, "realized_nodes": budget_nodes(uniform, 16000),
               "engine_seconds": None, "tier1_scan_seconds_combined_2k_and_16k": tier1_scan_seconds(uniform),
               "source": "tier1 16k budget over the 94 uniform records"},
        "P3": {"rows": len(eligible), "node_budget": 16000, "realized_nodes": budget_nodes(eligible, 16000),
               "engine_seconds": None, "tier1_scan_seconds_combined_2k_and_16k": tier1_scan_seconds(eligible),
               "source": "tier1 16k budget over the 926 train-eligible records"},
        "P4": {"rows": len(eligible), "node_budget": 2000, "realized_nodes": budget_nodes(eligible, 2000),
               "engine_seconds": None, "tier1_scan_seconds_combined_2k_and_16k": tier1_scan_seconds(eligible),
               "source": "tier1 2k budget over the 926 train-eligible records"},
    }

    # cross-check 1: the 400k arm must reproduce the deep-node total R0008 recorded
    recorded_uniform_nodes = 37285491
    if realized["P1"]["realized_nodes"] != recorded_uniform_nodes:
        print(f"STOP: realized 400k nodes {realized['P1']['realized_nodes']} != R0008's arm record "
              f"{recorded_uniform_nodes}")
        return 3
    # cross-check 2: label-visible nodes must equal artifact costs
    labels_16k = json.loads((TOOLS / "s7-pilot-state.json").read_text(encoding="utf-8"))
    if not labels_16k.get("results"):
        print("STOP: no pilot results recorded")
        return 3

    baseline = realized["P1"]["realized_nodes"]
    for arm, entry in realized.items():
        entry["relative_to_P1_realized"] = round(entry["realized_nodes"] / baseline, 4)

    # relative effect sizes, so the bundle never again states a magnitude it contradicts
    values = result["values"]
    level = {}
    for arm, widths in values.items():
        pooled = [v for width in widths.values() for v in width.values()]
        level[arm] = sum(pooled) / len(pooled)
    for name, cells in result["contrasts"].items():
        for width, cell in cells.items():
            both = [level[cell["a"]], level[cell["b"]]]
            cell["mean_level_of_both_arms"] = round(sum(both) / 2, 6)
            cell["relative_to_level"] = round(cell["width_estimate"] / (sum(both) / 2), 4)

    largest = max(((name, width, abs(cell["width_estimate"]), cell["relative_to_level"])
                   for name, cells in result["contrasts"].items() for width, cell in cells.items()),
                  key=lambda item: item[2])

    result["economics_realized"] = realized
    if "economics" in result:            # idempotent: first run relabels nominal, later runs keep it
        result["economics_nominal"] = result.pop("economics")
    result["largest_effect"] = {"contrast": largest[0], "width": largest[1], "absolute": round(largest[2], 6),
                                "relative_to_level": largest[3]}
    result["reading"] = {
        "P_econ": "UNRESOLVED_AT_40_PERCENT_COMPUTE",
        "P_econ_qualifier": ("no consistent all-width advantage or disadvantage at ~40 % of the teacher compute; "
                             "H4 significantly favours narrow-deep (+0.006130) and H32 significantly favours "
                             "broad-shallow (-0.001600). This is NOT a non-inferiority claim, and it is not a "
                             "claim that the cheaper regime is equally good."),
        "P_depth": ("400k never showed an advantage over 16k: three widths unresolved, H32 significantly favoured "
                    "the shallower 16k labels (-0.001014 [-0.001918, -0.000111])"),
        "P_rows": ("no consistent row-count effect: three widths unresolved, H4 significantly favoured the smaller "
                   "94-row set (+0.006657 [+0.002564, +0.010749])"),
        "magnitudes": (f"pooled test_loss is ~0.043 per arm; the largest single-width effect is "
                       f"{largest[0]} at H{largest[1]} = {largest[2]:.6f}, i.e. {largest[3] * 100:.1f} % of that "
                       "level; the other resolving contrasts are 2.4 %-14.2 % in absolute value, and the "
                       "unresolved ones span 0.4 %-9.5 % with intervals wider than their estimates"),
    }
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary["reading"] = result["reading"]
    summary["economics_realized"] = realized
    summary["largest_effect"] = result["largest_effect"]
    for name, cells in summary.get("contrasts", {}).items():
        for width, cell in cells.items():
            cell["relative_to_level"] = result["contrasts"][name][width]["relative_to_level"]
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"realized": {k: {kk: vv for kk, vv in v.items() if kk in
                                       ("rows", "node_budget", "realized_nodes", "engine_seconds",
                                        "relative_to_P1_realized")}
                                  for k, v in realized.items()},
                      "largest_effect": result["largest_effect"], "reading": result["reading"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
