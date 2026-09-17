"""S15 shared: the frozen contract, the auxiliary target manifest, and the width ladder.

Teacher and exam are S13/S14's CVS contract unchanged (CVS-4k over D0036; E0004 over the 200
held-out identities). The ONLY change between arms is whether the RAW trunk is additionally trained
to predict named deterministic geometry facts — the deployed evaluator is RAW-768 -> score in both.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cvslab.nnue as nnue
from cvslab.facts import GEOMETRY_FAMILIES, facts_registry_hash
from cvslab.hashing import hash_obj

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
S15 = LAB / "tools" / "s15"
STATE = S15 / "s15-state.json"

DATASET = "D0036"
EXAM_PROTOCOL = "E0004"
CVS_SPEC_HASH = "sha256:6b8bd7dcc1acce86a136ccbf6c6b33e444e58fb4adbde325b799708fed3d3d43"
SEEDS = tuple(range(20))
WIDTHS = (1, 4, 16, 32)
BATCH, MAX_UPDATES = 256, 760
SCALE_NAME, SCALE_NODES, NODE_BUDGET = "2.0x", 74_570_561, 4_000
CONTROL_MODE = "none"
TREATMENT_MODE = "geo"
ABLATION_MODES = tuple(f"geo-no-{group}" for group in
                       ("king", "mobility", "pawns", "rooks", "hanging", "bishop"))
ABLATION_WIDTH = 1
GRADIENT_RATIO_TARGET = 0.25      # aux gradient norm / score gradient norm at initialisation


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"saved {list(updates)}", flush=True)


def aux_manifest() -> dict:
    """The frozen auxiliary target contract: every supervised quantity, named and typed."""
    targets = []
    for index, (name, (t0, t1, t2)) in enumerate(GEOMETRY_FAMILIES):
        targets.append({"family": name, "kind": "value", "column": index, "type": "continuous",
                        "range": f"White-POV delta / {t2} (the family's own bucket-3 threshold), "
                                 "sign-flipped for black to move",
                        "loss": "mean squared error, averaged with the other families",
                        "derivation": "position only"})
        targets.append({"family": name, "kind": "bucket", "column": index, "type": "ordinal(0-3)",
                        "range": f"the registry's 0-3 magnitude bucket (thresholds {t0}/{t1}/{t2}), "
                                 "sign-independent",
                        "loss": "cross-entropy over 4 classes, averaged with the other families",
                        "derivation": "position only"})
    return {"registry_hash": facts_registry_hash(), "registry_version": 1,
            "authority": "deterministic.cvs.geometry.v1",
            "families": [name for name, _t in GEOMETRY_FAMILIES],
            "value_targets": len(GEOMETRY_FAMILIES),
            "bucket_logits": len(GEOMETRY_FAMILIES) * nnue.AUX_BUCKET_CLASSES,
            "targets": targets,
            "missingness": "none: the registry emits every family for every position",
            "never_depends_on": ["teacher search", "the target score", "game outcome",
                                 "best move", "evaluation split membership"],
            "inference": "auxiliary heads are TRAINING-ONLY; the deployed evaluator is RAW-768 -> "
                         "score and predict_cp never consults them"}


def width_ladder() -> list[dict]:
    from cvslab import families as fam
    rows = []
    for hidden in WIDTHS:
        for mode in (CONTROL_MODE, TREATMENT_MODE):
            shapes = fam.parameter_shapes("NNUE", {"INPUT": "RAW", "H": hidden, "AUX": mode})
            total = fam.count_parameters(shapes)
            inference = fam.count_parameters(fam.inference_parameter_shapes(shapes))
            rows.append({"H": hidden, "AUX": mode, "total_parameters": total,
                         "inference_parameters": inference,
                         "training_only_parameters": total - inference})
    return rows


def freeze_hash(payload: dict) -> str:
    return hash_obj(payload)
