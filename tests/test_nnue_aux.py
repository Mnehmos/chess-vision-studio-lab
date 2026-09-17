"""Auxiliary geometry supervision: target contract, accounting, gradients, serialization.

These tests pin what S15 depends on: the auxiliary targets come from the frozen registry, are
side-to-move signed, carry named family identity, never touch the deployed evaluator, and are
counted separately from the persistent inference parameters.
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
MATE = "8/8/8/8/8/5K2/6Q1/6k1 w - - 0 1"


def test_targets_are_registry_named_and_side_to_move_signed():
    values, buckets = nnue.aux_targets_for_fen(START)
    assert nnue.AUX_FAMILIES == tuple(name for name, _t in GEOMETRY_FAMILIES)
    assert values.shape == (21,) and buckets.shape == (21,)
    assert buckets.min() >= 0 and buckets.max() <= 3
    other, _ = nnue.aux_targets_for_fen(START_BLACK)
    assert np.allclose(values, -other), "black to move must see the sign-flipped geometry"
    rich, _ = nnue.aux_targets_for_fen(MATE)
    assert abs(rich).sum() > 0, "a material/tactical position must carry signal"


def test_every_family_is_visible_in_the_manifest_order():
    values, buckets = nnue.aux_targets_for_fen(MATE)
    assert len(values) == len(buckets) == len(nnue.AUX_FAMILIES) == 21
    assert dict(zip(nnue.AUX_FAMILIES, buckets))["KING_DANGER"] in (0, 1, 2, 3)


def test_aux_modes_remove_exactly_one_named_group():
    assert nnue.aux_families("none") == ()
    assert len(nnue.aux_families("geo")) == 21
    for mode, group in nnue.AUX_SKIP_GROUPS.items():
        kept = nnue.aux_families(mode)
        assert len(kept) == 21 - len(group)
        assert not (set(kept) & set(group))
    with pytest.raises(ValueError):
        nnue.aux_families("geo-no-everything")


def test_parameter_accounting_separates_training_only_capacity():
    config = {"INPUT": "RAW", "H": 4, "AUX": "geo"}
    total = families.count_parameters(families.parameter_shapes("NNUE", config))
    inference = families.count_parameters(families.inference_parameter_shapes(
        families.parameter_shapes("NNUE", config)))
    assert inference == 768 * 4 + 2 * 4 + 1 == 3081
    assert total == inference + 4 * (21 + 84) + (21 + 84) == 3081 + 525
    baseline = families.count_parameters(families.parameter_shapes(
        "NNUE", {"INPUT": "RAW", "H": 4}))
    assert baseline == inference, "the auxiliary arm must not change the deployed model"


def test_auxiliary_gradients_reach_the_trunk_and_lower_the_aux_loss():
    rng = np.random.default_rng(3)
    model = nnue.RawNnue.initialize(768, 8, 0.05, 400.0, rng, aux_dimensions=nnue.aux_dimensions("geo"))
    X = rng.normal(size=(32, 768)).astype(np.float64)
    t = np.full(32, 0.5)
    values, buckets = nnue.encode_aux([{"fen": MATE}] * 32, "geo")
    first, grads = model.loss_and_grads(X, t, 256.0, aux=(values, buckets), aux_weight=1.0)
    assert set(grads) >= {"w1", "b1", "w2", "b2", "wv", "bv", "wb", "bb"}
    assert np.abs(grads["w1"]).sum() > 0, "auxiliary loss must shape the shared trunk"
    for name, grad in grads.items():
        model.params[name] -= 0.05 * grad
    second, _ = model.loss_and_grads(X, t, 256.0, aux=(values, buckets), aux_weight=1.0)
    assert second < first


def test_the_deployed_evaluator_ignores_auxiliary_heads():
    rng = np.random.default_rng(5)
    plain = nnue.RawNnue.initialize(768, 4, 0.05, 400.0, rng)
    with_aux = nnue.RawNnue.initialize(768, 4, 0.05, 400.0, np.random.default_rng(5),
                                       aux_dimensions=nnue.aux_dimensions("geo"))
    X = rng.normal(size=(16, 768))
    assert np.allclose(plain.predict_cp(X), with_aux.predict_cp(X))
    payload = nnue.serialize(with_aux, {"OUT_SCALE_CP": 400.0, "K": 256.0, "LAMBDA": 1.0,
                                        "INPUT": "RAW", "AUX": "geo"}, {})
    assert payload["auxEnabled"] is True and "wv" in payload and "wb" in payload
    restored = nnue.load_serialized(payload)
    assert restored.aux_enabled
    assert np.allclose(restored.predict_cp(X), with_aux.predict_cp(X))
    assert nnue.serialized_param_count(payload) > families.count_parameters(
        families.inference_parameter_shapes(families.parameter_shapes(
            "NNUE", {"INPUT": "RAW", "H": 4, "AUX": "geo"})))


def test_auxiliary_targets_never_depend_on_labels():
    """A record with no labels at all still yields its geometry targets: position-only."""
    values, buckets = nnue.encode_aux([{"fen": MATE, "record_id": "no-labels-here"}], "geo")
    assert values.shape == (1, 21) and buckets.shape == (1, 21)
    assert facts_registry_hash().startswith("sha256:")


def test_paired_aux_and_control_arms_share_init_and_batch_order():
    """PR #46 review: the ONLY difference between paired arms must be the loss.

    Both arms of a seed take their persistent weights from the same stream and their minibatches
    from the same sampler stream, regardless of how many draws the auxiliary heads consume.
    """
    seed, inputs, hidden = 7, 24, 4
    control = nnue.initialize_for_run(seed, inputs, hidden, 0.05, 400.0)
    treatment = nnue.initialize_for_run(seed, inputs, hidden, 0.05, 400.0,
                                       aux_dimensions=nnue.aux_dimensions("geo"))
    for name in ("w1", "b1", "w2", "b2"):
        assert np.array_equal(control.params[name], treatment.params[name]), \
            f"persistent {name} differs between paired arms"
    assert treatment.aux_enabled and "wv" in treatment.params

    # identical first 20 minibatches for the same seed, whatever the aux heads drew
    _p, _a, sampler_control = nnue.run_rng_streams(seed)
    _p2, _a2, sampler_treatment = nnue.run_rng_streams(seed)
    control_batches = [b.copy() for b, _ in zip(nnue.batch_stream(500, 32, sampler_control), range(20))]
    treatment_batches = [b.copy() for b, _ in zip(nnue.batch_stream(500, 32, sampler_treatment), range(20))]
    assert len(control_batches) == len(treatment_batches) == 20
    for index, (left, right) in enumerate(zip(control_batches, treatment_batches)):
        assert np.array_equal(left, right), f"batch {index} differs between paired arms"
    assert not np.array_equal(control_batches[0], control_batches[1])


def test_the_sampler_stream_is_independent_of_other_draws():
    """Drawing persistent and auxiliary parameters must not move the sampler."""
    seed, inputs, hidden = 11, 16, 3
    _p, _a, sampler_before = nnue.run_rng_streams(seed)
    first = next(nnue.batch_stream(200, 16, sampler_before)).copy()
    nnue.initialize_for_run(seed, inputs, hidden, 0.05, 400.0,
                            aux_dimensions=nnue.aux_dimensions("geo"))
    nnue.initialize_for_run(seed, inputs, hidden, 0.05, 400.0)
    _p, _a, sampler_after = nnue.run_rng_streams(seed)
    second = next(nnue.batch_stream(200, 16, sampler_after)).copy()
    assert np.array_equal(first, second)
