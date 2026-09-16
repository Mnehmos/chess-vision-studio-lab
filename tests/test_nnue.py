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


def test_fixed_update_regime_runs_exactly_the_declared_updates(lab):
    """S8 control: equal student optimizer work per arm, deterministic from the seed."""
    from cvslab.data import _load_canonical
    from cvslab.targets import TargetSpec

    spec = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow", producer="a" * 64,
                      budget={"nodeBudget": 16000}, value_path=("scoreCpStm",))
    source = lab.create_fixture_source(n_games=6, seed=71)
    normalization = lab.normalize([source.id], name="n")
    evaluated = sorted(r["record_id"] for r in _load_canonical(lab.store, lab.store.get(normalization.id)))
    lab.register_labels(normalization.id, family="search_shallow_cp", producer="a" * 64,
                        authority="legacy.cvs.search.shallow", pov="stm",
                        rows=[{"record_id": r, "value": {"scoreCpStm": 20 + i, "bestMove": "e2e4", "nodes": 16000},
                               "budget": {"nodeBudget": 16000, "nodes": 16000}} for i, r in enumerate(evaluated)])
    dataset = lab.freeze_dataset(normalization.id, name="d-fixed", required_labels=["search_shallow_cp"],
                                 target_spec=spec, fractions=(1.0, 0.0, 0.0))
    exam = lab.freeze_dataset(normalization.id, name="d-fixed-exam", required_labels=["search_shallow_cp"],
                              target_spec=spec, fractions=(0.0, 0.0, 1.0))
    protocol = lab.create_eval_protocol(name="e-fixed", dataset_id=exam.id, split="test", k=256.0, lam=1.0,
                                        target_spec=spec)

    counter = {"n": 0}

    def run_with(max_updates: int, epochs: int, seed: int):
        counter["n"] += 1
        recipe = lab.create_training_recipe(name=f"t-u{max_updates}-e{epochs}-s{seed}-{counter['n']}",
                                            params={"K": 256.0, "LAMBDA": 1.0, "EPOCHS": epochs, "BATCH": 16,
                                                    "MAX_UPDATES": max_updates},
                                            target_spec=spec)
        baseline = lab.register_baseline(name=f"FIXED{max_updates}_{epochs}_{seed}_{counter['n']}",
                                         dataset_id=dataset.id,
                                         training_recipe_id=recipe.id, eval_protocol_id=protocol.id,
                                         model_config={"INPUT": "RAW", "H": 4}, supervision_divergence="same spec")
        [queued] = lab.queue_runs(baseline.id, seeds=(seed,))
        return lab.execute_run(queued.id)

    # 10 updates at batch 16 = 160 presentations, whatever the dataset size or epoch count
    first = run_with(max_updates=10, epochs=99, seed=0)
    assert first.status.value == "COMPLETED", [c.detail for c in first.integrity if c.status == "fail"]
    assert first.compute.train_examples_seen == 160
    assert max(log.epoch for log in first.training_curve) == 10        # the log index is the update count
    assert len(first.training_curve) <= 21                             # ~20 checkpoints + the final one

    # deterministic: the same seed reproduces the same curve exactly
    again = run_with(max_updates=10, epochs=99, seed=0)
    assert [log.train_loss for log in again.training_curve] == [log.train_loss for log in first.training_curve]

    # and the epoch-bounded regime keeps its historical definition: every sample in every
    # epoch, partial tail batch included (the S8 patch briefly counted full batches only)
    epoch_bounded = run_with(max_updates=0, epochs=3, seed=0)
    rows = dataset.splits[0].count
    assert epoch_bounded.compute.train_examples_seen == 3 * rows
    assert len(epoch_bounded.training_curve) == 3
