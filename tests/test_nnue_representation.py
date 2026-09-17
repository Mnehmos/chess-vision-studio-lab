"""The S14 representations: named GEO columns, HYBRID composition, and the direct-linear floor.

Everything here pins the contract the representation study depends on: the GEO vector is exactly
two named columns per registry family, side-to-move relative like RAW, derived from the position
alone; HYBRID is RAW ++ GEO with the RAW part identical; the direct linear model has exactly
inputs+1 learned parameters; and the discard path is representation-independent.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cvslab.nnue as nnue  # noqa: E402
from cvslab import families  # noqa: E402
from cvslab.facts import GEOMETRY_FAMILIES, facts_registry_hash  # noqa: E402

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
START_BLACK = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"


def record(fen, cp=20.0):
    return {"record_id": "r" + str(abs(hash(fen)) % 10**8), "fen": fen,
            "labels": [{"family": "eval_cp", "value": cp, "pov": "white"}]}


def test_geo_columns_are_named_registry_entries():
    columns = nnue.geo_columns()
    assert len(columns) == nnue.GEO_INPUT_DIM == 2 * len(GEOMETRY_FAMILIES) == 42
    assert columns[:4] == [("KING_DANGER", "value"), ("KING_DANGER", "bucket"),
                           ("KING_ZONE_PRESSURE", "value"), ("KING_ZONE_PRESSURE", "bucket")]
    assert {name for name, _kind in columns} == {name for name, _t in GEOMETRY_FAMILIES}
    assert families.INPUT_DIMS["GEO"] == nnue.GEO_INPUT_DIM


def test_geo_is_side_to_move_relative_and_position_only():
    vector = nnue.encode_fen_geo(START)
    flipped = nnue.encode_fen_geo(START_BLACK)
    assert vector.shape == (42,)
    assert np.allclose(vector, -flipped), "black to move must see the sign-flipped geometry"
    assert np.allclose(nnue.encode_fen_geo(START), 0.0), "the start position is geometry-neutral"


def test_hybrid_is_raw_plus_geo():
    split, _ = nnue.encode_records([record(START)], input_kind="HYBRID")
    raw, _ = nnue.encode_records([record(START)], input_kind="RAW")
    assert split.X.shape == (1, 810)
    assert np.array_equal(split.X[:, :768], raw.X.astype(np.float64))
    assert np.allclose(split.X[:, 768:], nnue.encode_fen_geo(START))


def test_all_representations_discard_the_same_rows():
    kept, dropped = [record(START)], [record(START_BLACK, cp=None) | {"labels": []}]
    shapes = {}
    for kind in ("RAW", "GEO", "HYBRID"):
        split, count = nnue.encode_records(kept + dropped, input_kind=kind)
        shapes[kind] = (split.X.shape[0], len(split.record_ids), count)
    assert shapes["RAW"] == shapes["GEO"] == shapes["HYBRID"] == (1, 1, 1)


def test_the_direct_linear_model_has_exactly_inputs_plus_one_parameters():
    rng = np.random.default_rng(7)
    model = nnue.LinearEval.initialize(42, 0.05, 400.0, rng)
    assert sum(value.size for value in model.params.values()) == 43
    assert model.hidden == 0 and model.inputs == 42
    payload = nnue.serialize(model, {"OUT_SCALE_CP": 400.0, "K": 256.0, "LAMBDA": 1.0,
                                     "INPUT": "GEO"}, {})
    assert nnue.serialized_param_count(payload) == 43
    restored = nnue.load_serialized(payload)
    assert isinstance(restored, nnue.LinearEval)
    assert np.allclose(restored.predict_cp(np.zeros((3, 42))), 0.0 + restored.params["b"] * 400.0)


def test_a_gradient_step_lowers_the_linear_loss():
    rng = np.random.default_rng(11)
    model = nnue.LinearEval.initialize(42, 0.05, 400.0, rng)
    X = rng.normal(size=(64, 42))
    target = (X[:, 0] > 0).astype(np.float64) * 0.8 + 0.1
    first, grads = model.loss_and_grads(X, target, 256.0)
    model.params["w"] -= 0.5 * grads["w"]
    model.params["b"] -= 0.5 * grads["b"]
    second, _ = model.loss_and_grads(X, target, 256.0)
    assert second < first


@pytest.mark.parametrize("kind,hidden,expected", [
    ("RAW", 1, 768 + 2 + 1), ("GEO", 68, 42 * 68 + 2 * 68 + 1),
    ("HYBRID", 31, 810 * 31 + 2 * 31 + 1), ("GEO", 272, 42 * 272 + 2 * 272 + 1)])
def test_parameter_counts_match_the_formula(kind, hidden, expected):
    config = {"INPUT": kind, "H": hidden, "ARCH": "crelu1"}
    assert families.param_count("NNUE", config) == expected
    shapes = families.parameter_shapes("NNUE", config)
    assert families.count_parameters(shapes) == expected


def test_the_linear_architecture_switch_changes_the_parameter_formula():
    linear = families.parameter_shapes("NNUE", {"INPUT": "GEO", "H": 272, "ARCH": "linear"})
    assert linear == {"w": [42], "b": []}
    assert families.count_parameters(linear) == 43
    crelu = families.parameter_shapes("NNUE", {"INPUT": "GEO", "H": 272})
    assert families.count_parameters(crelu) == 42 * 272 + 2 * 272 + 1


def test_the_registry_identity_travels_with_the_representation():
    """The GEO vocabulary is the facts registry's, pinned by its own hash."""
    assert len(GEOMETRY_FAMILIES) == 21
    assert facts_registry_hash().startswith("sha256:")
    assert {name for name, _ in nnue.geo_columns()} <= {name for name, _ in GEOMETRY_FAMILIES}
