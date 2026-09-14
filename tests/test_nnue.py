import numpy as np

from cvslab import nnue


def test_encode_fen_startpos_white():
    indices, white = nnue.encode_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert white
    assert len(indices) == 32
    assert indices[0] == 9 * 64 + 56  # black rook, a8
    assert 5 * 64 + 4 in indices  # white king, e1
    assert indices[-1] == 3 * 64 + 7  # white rook, h1 (last piece in FEN order)


def test_encode_fen_black_perspective():
    indices, white = nnue.encode_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    assert not white
    assert indices[0] == 3 * 64 + 0  # the black rook on a8 becomes a white rook mirrored to a1


def synthetic_split(n=96, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, nnue.INPUT_DIM), dtype=np.uint8)
    for row in range(n):
        X[row, rng.choice(nnue.INPUT_DIM, size=24, replace=False)] = 1
    cp = rng.uniform(-400, 400, n)
    result = np.full(n, np.nan)
    return nnue.EncodedSplit(X, cp, result, [f"pos_{i:04d}" for i in range(n)])


def test_training_reduces_loss_and_stays_finite():
    split = synthetic_split()
    rng = np.random.default_rng(0)
    model = nnue.RawNnue.initialize(nnue.INPUT_DIM, 8, 0.05, 400.0, rng)
    cfg = {"K": 256.0, "LAMBDA": 1.0, "EPOCHS": 4, "BATCH": 32, "LR": 0.01, "OPTIMIZER": "adam"}
    losses = []

    def on_epoch(entry):
        losses.append(entry["train_loss"])

    nnue.train(model, split, split, cfg, rng, on_epoch=on_epoch)
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]
    assert model.all_finite()


def test_serialize_roundtrip_and_exact_param_count():
    rng = np.random.default_rng(1)
    model = nnue.RawNnue.initialize(nnue.INPUT_DIM, 4, 0.05, 400.0, rng)
    cfg = {"OUT_SCALE_CP": 400.0, "K": 256.0, "LAMBDA": 1.0}
    payload = nnue.serialize(model, cfg, {"run_id": "R0001"})
    assert nnue.serialized_param_count(payload) == 768 * 4 + 8 + 1
    reloaded = nnue.load_serialized(payload)
    split = synthetic_split(n=16)
    assert np.allclose(reloaded.predict_cp(split.X), model.predict_cp(split.X))
    assert payload["arch"] == "768x4cReLU-1"
    assert payload["promotable"] is False


def test_bootstrap_mean_is_deterministic_and_bounded():
    values = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    a = nnue.bootstrap_mean(values, 500, 0)
    b = nnue.bootstrap_mean(values, 500, 0)
    assert a == b
    mean, low, high, n = a
    assert n == 5
    assert low <= mean <= high
    assert nnue.bootstrap_mean(np.array([np.nan]), 100, 0)[3] == 0


def test_evaluate_metrics_shapes():
    split = synthetic_split(n=32, seed=2)
    rng = np.random.default_rng(3)
    model = nnue.RawNnue.initialize(nnue.INPUT_DIM, 2, 0.05, 400.0, rng)
    per_position = nnue.evaluate(model, split, {"K": 256.0, "LAMBDA": 1.0})
    assert set(per_position) == {"test_loss", "cp_mae", "sign_agreement"}
    for values in per_position.values():
        assert values.shape == (32,)
