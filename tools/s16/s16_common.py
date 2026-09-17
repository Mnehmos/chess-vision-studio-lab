"""S16 shared: the minimal factorial matrix, selected from prior evidence by predeclared rules.

Frozen inputs (all merged and immutable before S16 was frozen):

* teacher A — CVS-4k: dataset D0036, nodeBudget 4000, 74,570,561 realized teacher nodes (2.0x),
  spec sha256:6b8bd7dc… (S12's 4k arm spec, re-pinned by S13 and S14);
* teacher B — Stockfish 32k cold: dataset D0044, {"nodes": 32000}, 599,592,682 nodes (16.1x),
  spec sha256:28d9613d… (S13's PRIMARY Stockfish condition, cold per label);
* exams — CVS-DEEP E0004/D0010 (the in-lab exam the lattice binds) and SF-DEEP E0006/D0043
  (S13's cold 4M/label instrument) over the same 200 held-out identities.

Both teacher datasets carry the SAME 18,953 record ids in the same order (record_ids_hash
sha256:7d5a8a78…), so no teacher-specific row set can enter the comparison.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
S16 = LAB / "tools" / "s16"
STATE = S16 / "s16-state.json"

TEACHERS = {"CVS-4k": {"dataset": "D0036", "node_budget": 4_000, "scale": "2.0x",
                       "scale_nodes": 74_570_561, "authority": "legacy.cvs.search.shallow",
                       "spec_hash": "sha256:6b8bd7dcc1acce86a136ccbf6c6b33e444e58fb4adbde325b799708fed3d3d43",
                       "ms_per_label": 7.5},
            "SF-32k": {"dataset": "D0044", "node_budget": 32_000, "scale": "16.1x",
                       "scale_nodes": 599_592_682, "authority": "oracle.stockfish.18.cold",
                       "spec_hash": "sha256:28d9613d8be373bcfbe64133c225e8bbbbe5938f84e4c01c0f8839f701d478c5",
                       "ms_per_label": 79.73}}
REPRESENTATIONS = ("RAW", "HYBRID")
CAPACITIES = (1, 4)                    # S14's sentinel and knee; exact counts per cell below
SEEDS = tuple(range(20))
BATCH, MAX_UPDATES = 256, 760
EXAM_CVS = {"protocol": "E0004", "dataset": "D0010", "nodes": 400_000}
EXAM_SF = {"protocol": "E0006", "dataset": "D0043", "nodes": 4_000_000}

SELECTION_RULES = {
    "teachers": "exactly the two authorities S13 established, at S13's frozen training contracts; "
                "no recalibration (CVS-4k primary; Stockfish 32k cold primary)",
    "representations": "RAW plus the strongest explicit semantic representation justified by S14. "
                       "GEO is EXCLUDED by the predeclared rule: S14 measured it strictly worse "
                       "than RAW at every matched budget, so it cannot inform an interaction "
                       "question that HYBRID already answers. HYBRID is included: strictly better "
                       "than RAW from the 3K budget up",
    "capacities": "two scales only, S14's sentinel (H=1) and knee (H=4); their exact parameter "
                  "counts are reported per cell and the mismatch between representations is "
                  "disclosed (RAW 771 vs HYBRID 813 at H=1; 3,081 vs 3,249 at H=4)",
    "auxiliary_arm": "EXCLUDED by the predeclared rule: S15's supported effect lives at H16/H32, "
                     "outside this matrix's capacities (H=1/H=4 were unresolved there), so adding "
                     "it would change the factorial's capacity axis rather than extend it",
    "frozen_before": "no S16 outcome existed when these rules were written",
}


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"saved {list(updates)}", flush=True)


def cell_manifest() -> dict:
    """The exact factorial cells: 4 arms (teacher x representation) x 2 capacities x 20 seeds."""
    from cvslab import families
    cells = []
    for teacher, info in TEACHERS.items():
        for representation in REPRESENTATIONS:
            for hidden in CAPACITIES:
                shapes = families.parameter_shapes("NNUE", {"INPUT": representation, "H": hidden})
                cells.append({
                    "arm_id": f"S16-{teacher}-{representation}", "teacher": teacher,
                    "representation": representation, "H": hidden,
                    "dataset": info["dataset"], "node_budget": info["node_budget"],
                    "scale": info["scale"], "scale_nodes": info["scale_nodes"],
                    "train_spec_hash": info["spec_hash"],
                    "persistent_parameters": families.count_parameters(shapes),
                    "training_only_parameters": 0,
                    "inference_parameters": families.count_parameters(
                        families.inference_parameter_shapes(shapes))})
    return {"teachers": list(TEACHERS), "representations": list(REPRESENTATIONS),
            "capacities": list(CAPACITIES), "seeds": list(SEEDS), "cells": cells,
            "cell_count": len(cells) * len(SEEDS),
            "declared_cells": [[info["scale"], info["node_budget"]] for info in TEACHERS.values()],
            "shared": {"positions": 18_953, "record_ids_hash": "sha256:7d5a8a785148b975a06d630da9a3efd5c5aca6aa74b1705e1ac78fb800d15f2a",
                       "student_compute": f"MAX_UPDATES {MAX_UPDATES} x BATCH {BATCH}",
                       "optimizer": "T0004-derived; identical for every cell",
                       "seeds": list(SEEDS), "cold_start": True,
                       "exams": {"CVS-DEEP": EXAM_CVS, "SF-DEEP": EXAM_SF}},
            "selection_rules": SELECTION_RULES}


def economics() -> dict:
    """Per-cell economics under the frozen contracts (teacher side from S13/S14, student measured)."""
    rows = []
    for teacher, info in TEACHERS.items():
        for representation in REPRESENTATIONS:
            for hidden in CAPACITIES:
                from cvslab import families
                params = families.count_parameters(
                    families.parameter_shapes("NNUE", {"INPUT": representation, "H": hidden}))
                rows.append({"cell": f"{teacher}/{representation}/H{hidden}",
                             "teacher_nodes": info["scale_nodes"],
                             "teacher_nodes_per_label": info["node_budget"],
                             "teacher_ms_per_label": info["ms_per_label"],
                             "student_presentations": MAX_UPDATES * BATCH,
                             "persistent_parameters": params, "training_only_parameters": 0})
    return {"rows": rows, "note": "teacher-side costs are S13/S14's frozen measurements; student "
                                  "compute is identical in every cell"}
