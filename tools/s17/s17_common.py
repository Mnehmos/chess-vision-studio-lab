"""S17 shared: the RAW scaling ladder under the frozen S13-S16 contract, plus one control arm.

Primary: H = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 at the EXACT frozen student contract every S13-S16
study used — D0036 (CVS-4k), RAW-768, MAX_UPDATES 760 x BATCH 256 = 194,560 presentations for
every width, same optimizer/LR/init, same initialization and sampling contract (split RNG streams),
cold starts, E0004 exam.

Control: only H = 128, 256, 512 with student updates multiplied by ONE predeclared factor, chosen
before any outcome: **4x** (MAX_UPDATES 3040). H=128 is the anchor, so the report can show whether
extra compute merely rescues the largest models or bends the curve more generally. No per-width
tuning: one rule, applied identically to all three control widths.

Why 4x and not proportional-to-parameters: a rule that scaled updates with parameter count (e.g.
760 x params/3081) would multiply H=512's run time by 128, turning the control into a multi-day
run. A constant factor is the largest predeclared multiplier that keeps the whole control arm
inside a few hours, and its limitation is stated up front: if 4x does not rescue a width, the
result distinguishes "capacity saturated" from "compute helped less than 4x", not from "no amount
of compute helps". The report must not overstate that.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cvslab.nnue as nnue
from cvslab import families
from cvslab.hashing import hash_obj

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
S17 = LAB / "tools" / "s17"
STATE = S17 / "s17-state.json"

DATASET = "D0036"                     # CVS-4k supervision, the S13-S16 contract unchanged
EXAM_PROTOCOL = "E0004"               # CVS-DEEP 400k, the 200 held-out identities
RAW_INPUT = 768
LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
BATCH = 256
PRIMARY_UPDATES = 760                 # 194,560 presentations, the frozen S13-S16 regime
CONTROL_FACTOR = 4                    # the ONE predeclared scaling rule
CONTROL_UPDATES = PRIMARY_UPDATES * CONTROL_FACTOR
CONTROL_WIDTHS = (128, 256, 512)
SEEDS_SMALL = tuple(range(20))        # H <= 32: the S13-S16 seed list, so anchors stay comparable
SEEDS_LARGE = tuple(range(10))        # H >= 64: predeclared cost policy (10 paired seeds, df 9)
SMALL_MAX = 32
SCALE_NAME, SCALE_NODES, NODE_BUDGET = "2.0x", 74_570_561, 4_000
CVS_SPEC_HASH = "sha256:6b8bd7dcc1acce86a136ccbf6c6b33e444e58fb4adbde325b799708fed3d3d43"


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"saved {list(updates)}", flush=True)


def seeds_for(hidden: int) -> tuple[int, ...]:
    return SEEDS_SMALL if hidden <= SMALL_MAX else SEEDS_LARGE


def ladder_table() -> dict:
    """Exact parameter counts and the presentations each rung receives, generated from the code."""
    rows = []
    for hidden in LADDER:
        shapes = families.parameter_shapes("NNUE", {"INPUT": "RAW", "H": hidden})
        params = families.count_parameters(shapes)
        rows.append({"H": hidden, "parameters": params,
                     "inference_parameters": families.count_parameters(
                         families.inference_parameter_shapes(shapes)),
                     "primary_presentations": PRIMARY_UPDATES * BATCH,
                     "primary_presentations_per_parameter": PRIMARY_UPDATES * BATCH / params,
                     "seeds": len(seeds_for(hidden)),
                     "arm": "primary"})
    for hidden in CONTROL_WIDTHS:
        shapes = families.parameter_shapes("NNUE", {"INPUT": "RAW", "H": hidden})
        params = families.count_parameters(shapes)
        rows.append({"H": hidden, "parameters": params, "inference_parameters": params,
                     "primary_presentations": CONTROL_UPDATES * BATCH,
                     "primary_presentations_per_parameter": CONTROL_UPDATES * BATCH / params,
                     "seeds": len(seeds_for(hidden)), "arm": f"control x{CONTROL_FACTOR}"})
    return {"ladder": list(LADDER), "control_widths": list(CONTROL_WIDTHS),
            "control_factor": CONTROL_FACTOR, "primary_updates": PRIMARY_UPDATES,
            "control_updates": CONTROL_UPDATES, "batch": BATCH, "rows": rows,
            "counts_generated_by": "cvslab.families.parameter_shapes / count_parameters",
            "seed_policy": {"rule": f"{len(SEEDS_SMALL)} seeds for H <= {SMALL_MAX}, "
                                    f"{len(SEEDS_LARGE)} for H >= {2 * SMALL_MAX}",
                            "why": "predeclared cost policy; the small-width seed list is the "
                                   "S13-S16 list so those anchor points stay comparable",
                            "note": "the control arms use the same seed lists as their widths' "
                                    "primary arms"}}


def rng_contract() -> dict:
    return {"streams": "np.random.SeedSequence(seed).spawn(3): persistent init, auxiliary init "
                       "(unused here, AUX=none everywhere), minibatch sampling",
            "consequence": "every run of a seed sees the same initialization and example order, so "
                           "width is the only changing variable",
            "test": "tests/test_nnue_aux.py::test_paired_aux_and_control_arms_share_init_and_batch_order"}
