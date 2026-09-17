"""S14 shared: the frozen contract, the geometry manifest, and the matched-budget ladder.

The teacher/exam contract is S13's selected CVS side, unchanged: supervision = CVS-4k over D0036's
18,953 rows (the same TargetSpec X0008/X0010 trained under), exam = E0004 (CVS-DEEP, 400k nodes,
the 200 held-out identities). Representation is the ONLY changing variable.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cvslab.nnue as nnue
from cvslab import families
from cvslab.facts import GEOMETRY_FAMILIES, facts_registry_hash
from cvslab.hashing import hash_obj

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
S14 = LAB / "tools" / "s14"
STATE = S14 / "s14-state.json"

DATASET = "D0036"                    # 18,953 rows, CVS-4k supervision (S12/S13's CVS arm)
EXAM_PROTOCOL = "E0004"              # CVS-DEEP 400k over the 200 held-out identities
CVS_SPEC_HASH = "sha256:6b8bd7dcc1acce86a136ccbf6c6b33e444e58fb4adbde325b799708fed3d3d43"  # X0008's 4k arm
SEEDS = tuple(range(20))
BATCH, MAX_UPDATES = 256, 760
SCALE_NAME, SCALE_NODES, NODE_BUDGET = "2.0x", 74_570_561, 4_000
FAMILIES = ("RAW", "GEO", "HYBRID")
TARGETS = (1_000, 3_000, 12_000, 25_000)

# Named geometry groups for the semantic ablation (the registry's own families, grouped by theme).
GROUPS = {
    "king_safety": ["KING_DANGER", "KING_ZONE_PRESSURE", "KING_OPEN_FILE", "KING_SHIELD",
                    "KING_CENTRAL_EXPOSURE", "ENEMY_QUEEN_NEAR_KING", "OPEN_CENTER_KING",
                    "KING_ESCAPE_DEFICIT"],
    "mobility": ["MOBILITY_KNIGHT", "MOBILITY_BISHOP", "MOBILITY_ROOK", "MOBILITY_QUEEN"],
    "pawn_structure": ["PASSED_PAWN", "CONNECTED_PASSED_PAWN", "DOUBLED_PAWN", "ISOLATED_PAWN"],
    "rook_files": ["ROOK_OPEN_FILE", "ROOK_SEMI_OPEN_FILE", "ROOK_SEVENTH"],
    "hanging_material": ["HANGING_MATERIAL"],
    "bishop_pair": ["BISHOP_PAIR"],
}


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"saved {list(updates)}", flush=True)


def geometry_manifest() -> dict:
    """Every GEO input as a named, semantic entry: what it is, its POV, scale and range."""
    columns = []
    for name, (t0, t1, t2) in GEOMETRY_FAMILIES:
        columns.append({"column": 2 * GEOMETRY_FAMILIES.index((name, (t0, t1, t2))),
                        "family": name, "kind": "value",
                        "semantics": "White-POV signed delta, divided by the family's own bucket-3 "
                                     f"threshold {t2} (the registry's magnitude unit), sign-flipped "
                                     "for black to move"})
        columns.append({"column": 2 * GEOMETRY_FAMILIES.index((name, (t0, t1, t2))) + 1,
                        "family": name, "kind": "bucket",
                        "semantics": "the registry's 0-3 magnitude bucket divided by 3 "
                                     f"(thresholds {t0}/{t1}/{t2})"})
    group_of = {family: group for group, members in GROUPS.items() for family in members}
    for entry in columns:
        entry["group"] = group_of[entry["family"]]
    return {"registry_hash": facts_registry_hash(),
            "registry_version": 1,
            "authority": "deterministic.cvs.geometry.v1",
            "columns": columns, "input_dim": len(columns),
            "missingness": "none: extract_geometry emits every family for every position "
                           "(0.0 value, bucket 0 when absent)",
            "derivation": "position only; no label, teacher score, best move, outcome or test-time "
                          "information enters the representation",
            "pov": "side-to-move relative, exactly like RAW (colour swap + mirror); the registry's "
                   "deltas are White-POV and are sign-flipped for black to move",
            "groups": GROUPS}


def matched_h(family_input: str, target: int) -> tuple[int, int]:
    """The exact hidden width whose learned-parameter count is nearest the target (from the code)."""
    best = (1, None)
    dim = families.INPUT_DIMS[family_input]
    centre = max(1, round((target - 1) / (dim + 2)))
    for hidden in range(max(1, centre - 2), centre + 3):
        count = families.param_count("NNUE", {"INPUT": family_input, "H": hidden, "ARCH": "crelu1"})
        if best[1] is None or abs(count - target) < abs(best[1] - target):
            best = (hidden, count)
    return best


def ladder() -> dict:
    rows = []
    for family_input in FAMILIES:
        for target in TARGETS:
            hidden, count = matched_h(family_input, target)
            rows.append({"input": family_input, "budget_target": target, "H": hidden,
                         "learned_parameters": count, "mismatch": count - target})
    rows.append({"input": "GEO", "budget_target": "linear-floor", "H": 1, "ARCH": "linear",
                 "learned_parameters": families.param_count(
                     "NNUE", {"INPUT": "GEO", "H": 1, "ARCH": "linear"}),
                 "mismatch": None})
    return {"targets": list(TARGETS), "rows": rows,
            "formula": {"crelu1": "inputs*H + 2H + 1", "linear": "inputs + 1"},
            "input_dims": dict(families.INPUT_DIMS),
            "note": "H chosen as the exact-count nearest neighbour of each target, generated from "
                    "cvslab.families.param_count (never hand-entered), frozen before any outcome"}
