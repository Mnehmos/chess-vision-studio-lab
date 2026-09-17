"""Raw NNUE (768 -> H clipped-ReLU -> 1) trainer and evaluator in numpy.

Encoding, loss and export format mirror ``chess-vision-studio/arena/train-nnue.py``
and the Rust loader in ``chess-vision-studio-rust-engine/src/eval/nnue.rs``:
side-to-move features (black to move: colours swapped, squares mirrored with
``sq ^ 56``), output in centipawns = (h @ w2 + b2) * OUT_SCALE_CP, loss =
MSE(sigmoid(pred / K), target). numpy keeps tiny runs CPU-deterministic.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

from .schemas import MetricDef

INPUT_DIM = 768

# GEO: the deterministic geometry registry (cvslab.facts), pinned by its own hash. One pair of
# columns per registered family - the White-POV signed delta rescaled by the family's OWN bucket-3
# threshold (the registry's magnitude unit, not an invented normalizer) and the registry's 0-3
# magnitude bucket scaled to [0,1] - then sign-flipped for black to move, so the model is
# side-to-move relative exactly like RAW (colour swap + mirror). Every column resolves to a named
# registry entry; extract_geometry always emits all families, so there is no missingness path.
from .facts import GEOMETRY_FAMILIES, extract_geometry  # noqa: E402

GEO_INPUT_DIM = 2 * len(GEOMETRY_FAMILIES)
INPUT_DIMS = {"RAW": INPUT_DIM, "GEO": GEO_INPUT_DIM, "HYBRID": INPUT_DIM + GEO_INPUT_DIM}


def input_dim(kind: str) -> int:
    try:
        return INPUT_DIMS[kind]
    except KeyError:
        raise ValueError(f"unknown input representation {kind!r}; known: {sorted(INPUT_DIMS)}") from None


AUX_FAMILIES: tuple[str, ...] = tuple(name for name, _thresholds in GEOMETRY_FAMILIES)
AUX_BUCKET_CLASSES = 4            # the registry's own 0-3 magnitude bucket

# Named-group removals for the family ablation. The keys are the AUX switch's values, so every
# ablation arm is an explicit, hashable configuration; the sets reuse S14's group vocabulary.
AUX_SKIP_GROUPS: dict[str, tuple[str, ...]] = {
    "geo-no-king": ("KING_DANGER", "KING_ZONE_PRESSURE", "KING_OPEN_FILE", "KING_SHIELD",
                    "KING_CENTRAL_EXPOSURE", "ENEMY_QUEEN_NEAR_KING", "OPEN_CENTER_KING",
                    "KING_ESCAPE_DEFICIT"),
    "geo-no-mobility": ("MOBILITY_KNIGHT", "MOBILITY_BISHOP", "MOBILITY_ROOK", "MOBILITY_QUEEN"),
    "geo-no-pawns": ("PASSED_PAWN", "CONNECTED_PASSED_PAWN", "DOUBLED_PAWN", "ISOLATED_PAWN"),
    "geo-no-rooks": ("ROOK_OPEN_FILE", "ROOK_SEMI_OPEN_FILE", "ROOK_SEVENTH"),
    "geo-no-hanging": ("HANGING_MATERIAL",),
    "geo-no-bishop": ("BISHOP_PAIR",),
}
AUX_MODES: tuple[str, ...] = ("none", "geo", *sorted(AUX_SKIP_GROUPS))


def aux_families(mode: str) -> tuple[str, ...]:
    """The families an auxiliary mode supervises, in registry order (never reordered)."""
    if mode == "none":
        return ()
    if mode == "geo":
        return AUX_FAMILIES
    if mode not in AUX_SKIP_GROUPS:
        raise ValueError(f"unknown auxiliary mode {mode!r}; known: {AUX_MODES}")
    skipped = set(AUX_SKIP_GROUPS[mode])
    return tuple(name for name in AUX_FAMILIES if name not in skipped)


def aux_dimensions(mode: str) -> tuple[int, int]:
    """(continuous value targets, bucket logits) for one auxiliary mode."""
    count = len(aux_families(mode))
    return count, count * AUX_BUCKET_CLASSES


def aux_targets_for_fen(fen: str) -> tuple[np.ndarray, np.ndarray]:
    """(values, buckets) for every supervised family: the S14 normalization, stm-POV signed.

    values = White-POV delta / the family's own bucket-3 threshold, sign-flipped for black to move;
    buckets = the registry's 0-3 magnitude. Derived from the position only — no teacher, score,
    move, outcome or split information enters an auxiliary target.
    """
    import chess

    board = chess.Board(fen)
    geometry = extract_geometry(board)
    sign = 1.0 if board.turn else -1.0
    values = np.empty(len(AUX_FAMILIES), dtype=np.float64)
    buckets = np.empty(len(AUX_FAMILIES), dtype=np.int64)
    for index, (name, (_t0, _t1, t2)) in enumerate(GEOMETRY_FAMILIES):
        entry = geometry[name]
        values[index] = sign * float(entry["value"]) / t2
        buckets[index] = int(entry["bucket"])
    return values, buckets


def encode_aux(records: list[dict], mode: str = "geo") -> tuple[np.ndarray, np.ndarray]:
    """Auxiliary targets for a split, restricted to the mode's families (registry order)."""
    keep = [index for index, name in enumerate(AUX_FAMILIES) if name in aux_families(mode)]
    if not records:
        return np.zeros((0, len(keep))), np.zeros((0, len(keep)), dtype=np.int64)
    pairs = [aux_targets_for_fen(record["fen"]) for record in records]
    values = np.stack([pair[0][keep] for pair in pairs])
    buckets = np.stack([pair[1][keep] for pair in pairs])
    return values, buckets


def geo_columns() -> list[tuple[str, str]]:
    """The named GEO columns in order: (family, kind) with kind in {value, bucket}."""
    return [(name, kind) for name, _thresholds in GEOMETRY_FAMILIES for kind in ("value", "bucket")]


def encode_fen_geo(fen: str) -> np.ndarray:
    """(42,) float64, side-to-move POV, derived from the position ONLY (never a label)."""
    import chess

    board = chess.Board(fen)
    geometry = extract_geometry(board)
    sign = 1.0 if board.turn else -1.0
    out = np.empty(GEO_INPUT_DIM, dtype=np.float64)
    for index, (name, (_t0, _t1, t2)) in enumerate(GEOMETRY_FAMILIES):
        entry = geometry[name]
        out[2 * index] = sign * float(entry["value"]) / t2
        out[2 * index + 1] = float(entry["bucket"]) / 3.0
    return out
PIECE_IDX = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5, "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
CP_CLIP = 2000.0

METRIC_DEFS = [
    MetricDef(
        name="test_loss", lower_is_better=True,
        description="Mean squared error between sigmoid(pred_cp/K) and the protocol target on the frozen evaluation split.",
    ),
    MetricDef(
        name="cp_mae", lower_is_better=True,
        description=f"Mean absolute centipawn error from the side to move, prediction and label clipped to ±{CP_CLIP:g}.",
    ),
    MetricDef(
        name="sign_agreement", lower_is_better=False,
        description="Fraction of positions with |label| >= 100cp where the predicted sign matches the label sign.",
    ),
]


def encode_fen(fen: str) -> tuple[list[int], bool]:
    board, stm = fen.split(" ")[:2]
    white_to_move = stm == "w"
    indices: list[int] = []
    square = 56  # FEN starts at a8 = LERF 56
    for ch in board:
        if ch == "/":
            square -= 16
        elif ch.isdigit():
            square += int(ch)
        else:
            piece, sq = PIECE_IDX[ch], square
            if not white_to_move:
                piece, sq = (piece + 6) % 12, sq ^ 56
            indices.append(piece * 64 + sq)
            square += 1
    return indices, white_to_move


@dataclass
class EncodedSplit:
    X: np.ndarray  # (n, 768) uint8
    cp: np.ndarray  # eval label, side-to-move POV
    result: np.ndarray  # side-to-move result in [0, 1], NaN when unknown
    record_ids: list[str]

    def __len__(self) -> int:
        return len(self.record_ids)


def encode_records(records: list[dict], *, target_spec=None, input_kind: str = "RAW") -> tuple[EncodedSplit, int]:
    """Encode canonical records under an input representation.

    With a frozen ``TargetSpec`` the supervision is that spec's single matching label per
    record (strict: zero or multiple matches raise — an integrity failure, never a silent
    discard). Without a spec the legacy eval_cp behaviour applies (rows lacking the label
    are discarded and counted). ``input_kind`` selects RAW (768 piece-square), GEO (the 42
    named registry columns) or HYBRID (RAW ++ GEO); the supervision is identical for all.
    """
    from .targets import extract_target

    feats, cps, results, ids, fens = [], [], [], [], []
    discarded = 0
    for record in records:
        indices, white_to_move = encode_fen(record["fen"])
        if target_spec is not None:
            cp = extract_target(record, target_spec)   # raises on zero/multiple matches
            if target_spec.pov == "white" and not white_to_move:
                cp = -cp
        else:
            label = next((lab for lab in record["labels"] if lab["family"] == "eval_cp"), None)
            if label is None:
                discarded += 1
                continue
            cp = float(label["value"])
            if label.get("pov", "white") == "white" and not white_to_move:
                cp = -cp
        res = record.get("result")
        feats.append(indices)
        fens.append(record["fen"])
        cps.append(cp)
        results.append(math.nan if res is None else (float(res) if white_to_move else 1.0 - float(res)))
        ids.append(record["record_id"])
    if input_kind == "RAW":
        X = np.zeros((len(feats), INPUT_DIM), dtype=np.uint8)
        for row, indices in enumerate(feats):
            X[row, indices] = 1
    elif input_kind == "GEO":
        X = (np.stack([encode_fen_geo(fen) for fen in fens]) if fens
             else np.zeros((0, GEO_INPUT_DIM), dtype=np.float64))
    elif input_kind == "HYBRID":
        X = np.zeros((len(feats), INPUT_DIMS["HYBRID"]), dtype=np.float64)
        for row, indices in enumerate(feats):
            X[row, indices] = 1.0
        for row, fen in enumerate(fens):
            X[row, INPUT_DIM:] = encode_fen_geo(fen)
    else:
        raise ValueError(f"unknown input representation {input_kind!r}")
    return EncodedSplit(X, np.array(cps, dtype=np.float64), np.array(results, dtype=np.float64), ids), discarded


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def targets(split: EncodedSplit, lam: float, k: float) -> np.ndarray:
    eval_target = sigmoid(split.cp / k)
    has_result = np.isfinite(split.result)
    blended = lam * eval_target + (1.0 - lam) * np.where(has_result, split.result, 0.0)
    return np.where(has_result, blended, eval_target)


def run_rng_streams(seed: int):
    """(persistent-init, auxiliary-init, minibatch-sampling) - INDEPENDENT streams from one seed.

    Splitting them is what makes a paired RAW/AUX comparison causal: auxiliary draws cannot advance
    the sampler's stream, so both arms of a seed see the same examples in the same order and the
    only difference between them is the loss.
    """
    persistent, auxiliary, sampling = np.random.SeedSequence(int(seed)).spawn(3)
    return (np.random.default_rng(persistent), np.random.default_rng(auxiliary),
            np.random.default_rng(sampling))


def initialize_for_run(seed: int, inputs: int, hidden: int, init_std: float, out_scale: float,
                       aux_dimensions: tuple = (0, 0)) -> "RawNnue":
    """The model a run starts from: persistent weights from the persistent stream, training-only
    heads from the auxiliary stream (never from the sampler's)."""
    persistent, auxiliary, _sampling = run_rng_streams(seed)
    return RawNnue.initialize(inputs, hidden, init_std, out_scale, persistent,
                              aux_dimensions=aux_dimensions, aux_rng=auxiliary)


def batch_stream(n_rows: int, batch: int, rng: np.random.Generator):
    """Minibatch indices for the fixed-update regime: shuffle, walk without replacement, drop the
    tail partial batch, reshuffle on exhaustion. Shared by `train` and its regression test."""
    order = np.arange(n_rows)
    while True:
        rng.shuffle(order)
        for start in range(0, n_rows - batch + 1, batch):
            yield order[start:start + batch]


class RawNnue:
    def __init__(self, params: dict[str, np.ndarray], out_scale: float):
        self.params = params
        self.out_scale = out_scale

    @property
    def hidden(self) -> int:
        return int(self.params["b1"].shape[0])

    @classmethod
    def initialize(cls, inputs: int, hidden: int, init_std: float, out_scale: float,
                   rng: np.random.Generator, aux_dimensions: tuple[int, int] = (0, 0),
                   aux_rng: Optional[np.random.Generator] = None) -> "RawNnue":
        """Persistent weights from `rng`; training-only heads from `aux_rng` when given.

        Persistent draws come first and never depend on whether auxiliary heads exist, so two arms
        of one seed start from identical persistent weights; the auxiliary stream is separate so its
        draw count cannot shift anything else.
        """
        bound = 1.0 / math.sqrt(hidden)
        params = {
            "w1": rng.normal(0.0, init_std, size=(inputs, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.uniform(-bound, bound, size=hidden),
            "b2": np.array(rng.uniform(-bound, bound)),
        }
        values_dim, bucket_logits = aux_dimensions
        if values_dim or bucket_logits:
            # TRAINING-ONLY heads: they read the shared hidden layer and never touch predict_cp
            aux_rng = aux_rng if aux_rng is not None else rng
            aux_bound = 1.0 / math.sqrt(hidden)
            if values_dim:
                params["wv"] = aux_rng.uniform(-aux_bound, aux_bound, size=(hidden, values_dim))
                params["bv"] = np.zeros(values_dim)
            if bucket_logits:
                params["wb"] = aux_rng.uniform(-aux_bound, aux_bound, size=(hidden, bucket_logits))
                params["bb"] = np.zeros(bucket_logits)
        return cls(params, out_scale)

    @property
    def aux_enabled(self) -> bool:
        return "wv" in self.params or "wb" in self.params

    def predict_cp(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        """The DEPLOYED evaluator: RAW-768 -> score. Auxiliary heads are never consulted here."""
        p = self.params
        out = np.empty(X.shape[0])
        for start in range(0, X.shape[0], batch):
            xb = X[start:start + batch].astype(np.float64)
            h = np.clip(xb @ p["w1"] + p["b1"], 0.0, 1.0)
            out[start:start + batch] = (h @ p["w2"] + p["b2"]) * self.out_scale
        return out

    def loss_and_grads(self, X: np.ndarray, t: np.ndarray, k: float,
                       aux: Optional[tuple[np.ndarray, np.ndarray]] = None,
                       aux_weight: float = 0.0) -> tuple[float, dict[str, np.ndarray]]:
        """Score loss plus, when auxiliary heads exist and targets are supplied, the geometry loss.

        The auxiliary objectives enter the shared trunk through the same hidden activation as the
        score head, so they shape the representation; they are weighted by `aux_weight` and the
        returned loss is the weighted total the optimizer actually descends.
        """
        p = self.params
        z = X @ p["w1"] + p["b1"]
        h = np.clip(z, 0.0, 1.0)
        y = h @ p["w2"] + p["b2"]
        prob = sigmoid(y * self.out_scale / k)
        diff = prob - t
        g_y = 2.0 * diff / len(t) * prob * (1.0 - prob) * self.out_scale / k
        grads = {"w2": h.T @ g_y, "b2": np.array(g_y.sum())}
        loss = float(np.mean(diff ** 2))
        if aux is not None and self.aux_enabled:
            values, buckets = aux
            g_h_aux = np.zeros_like(h)
            if "wv" in p:
                predicted = h @ p["wv"] + p["bv"]
                residual = predicted - values
                loss += aux_weight * float(np.mean(residual ** 2))
                d_pred = aux_weight * 2.0 * residual / residual.size
                grads["wv"] = h.T @ d_pred
                grads["bv"] = d_pred.sum(axis=0)
                g_h_aux += d_pred @ p["wv"].T
            if "wb" in p:
                classes = AUX_BUCKET_CLASSES
                families = buckets.shape[1]
                logits = (h @ p["wb"] + p["bb"]).reshape(len(h), families, classes)
                shifted = logits - logits.max(axis=2, keepdims=True)
                exp = np.exp(shifted)
                soft = exp / exp.sum(axis=2, keepdims=True)
                onehot = np.zeros_like(soft)
                np.put_along_axis(onehot, buckets[:, :, None], 1.0, axis=2)
                loss += aux_weight * float(-np.mean(np.log(np.clip(
                    np.take_along_axis(soft, buckets[:, :, None], axis=2), 1e-12, 1.0))))
                d_logits = (aux_weight * (soft - onehot) / (len(h) * families)).reshape(len(h), -1)
                grads["wb"] = h.T @ d_logits
                grads["bb"] = d_logits.sum(axis=0)
                g_h_aux += d_logits @ p["wb"].T
            hidden_grad = np.outer(g_y, p["w2"]) + g_h_aux
        else:
            hidden_grad = np.outer(g_y, p["w2"])
        g_z = hidden_grad * ((z > 0.0) & (z < 1.0))
        grads["w1"] = X.T @ g_z
        grads["b1"] = g_z.sum(axis=0)
        return loss, grads

    def all_finite(self) -> bool:
        return all(bool(np.all(np.isfinite(v))) for v in self.params.values())


class LinearEval:
    """inputs -> 1 direct model: the white-box semantic floor (named inputs, no hidden layer).

    Same output convention as RawNnue: centipawns = (X @ w + b) * out_scale, loss =
    MSE(sigmoid(pred / K), target). Learned parameters = inputs + 1, exactly.
    """

    def __init__(self, params: dict[str, np.ndarray], out_scale: float):
        self.params = params
        self.out_scale = out_scale

    @property
    def hidden(self) -> int:
        return 0

    @property
    def inputs(self) -> int:
        return int(self.params["w"].shape[0])

    @classmethod
    def initialize(cls, inputs: int, init_std: float, out_scale: float,
                   rng: np.random.Generator) -> "LinearEval":
        bound = 1.0 / math.sqrt(inputs)
        return cls({"w": rng.normal(0.0, init_std, size=inputs),
                    "b": np.array(rng.uniform(-bound, bound))}, out_scale)

    def predict_cp(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        p = self.params
        out = np.empty(X.shape[0])
        for start in range(0, X.shape[0], batch):
            out[start:start + batch] = (X[start:start + batch].astype(np.float64) @ p["w"] + p["b"]) \
                * self.out_scale
        return out

    def loss_and_grads(self, X: np.ndarray, t: np.ndarray, k: float) -> tuple[float, dict[str, np.ndarray]]:
        p = self.params
        prob = sigmoid((X @ p["w"] + p["b"]) * self.out_scale / k)
        diff = prob - t
        g_y = 2.0 * diff / len(t) * prob * (1.0 - prob) * self.out_scale / k
        return float(np.mean(diff ** 2)), {"w": X.T @ g_y, "b": np.array(g_y.sum())}

    def all_finite(self) -> bool:
        return all(bool(np.all(np.isfinite(v))) for v in self.params.values())


def per_position_loss(pred_cp: np.ndarray, t: np.ndarray, k: float) -> np.ndarray:
    return (sigmoid(pred_cp / k) - t) ** 2


def _finite_or_none(x: float) -> Optional[float]:
    return float(x) if math.isfinite(x) else None


def train(model: RawNnue, train_split: EncodedSplit, val_split: EncodedSplit, cfg: Mapping, rng: np.random.Generator,
          on_epoch: Optional[Callable[[dict], None]] = None,
          aux: Optional[tuple[np.ndarray, np.ndarray]] = None,
          aux_weight: float = 0.0) -> None:
    k, lam = float(cfg["K"]), float(cfg["LAMBDA"])
    epochs, batch, lr, optimizer = int(cfg["EPOCHS"]), int(cfg["BATCH"]), float(cfg["LR"]), str(cfg["OPTIMIZER"])
    max_updates = int(cfg.get("MAX_UPDATES", 0) or 0)
    t_train, t_val = targets(train_split, lam, k), targets(val_split, lam, k)
    order = np.arange(len(train_split))
    moments = {name: (np.zeros_like(v), np.zeros_like(v)) for name, v in model.params.items()}
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
    started = time.perf_counter()

    if max_updates > 0:
        # fixed-update regime: equal student optimizer work per arm. Deterministic sampler:
        # shuffle, walk without replacement, drop the tail partial batch, reshuffle on
        # exhaustion; stop after exactly max_updates updates.
        checkpoint_every = max(1, max_updates // 20)
        updates = 0
        stream = batch_stream(len(train_split), batch, rng)
        for idx in stream:
            if updates >= max_updates:
                break
            batch_aux = None
            if aux is not None:
                batch_aux = (aux[0][idx], aux[1][idx])
            loss, grads = model.loss_and_grads(train_split.X[idx].astype(np.float64), t_train[idx], k,
                                               aux=batch_aux, aux_weight=aux_weight)
            updates += 1
            step += 1          # Adam's bias-correction counter advances here too
            for name, grad in grads.items():
                param = model.params[name]
                if optimizer == "sgd":
                    param -= lr * grad
                    continue
                m, v = moments[name]
                m *= beta1
                m += (1.0 - beta1) * grad
                v *= beta2
                v += (1.0 - beta2) * grad * grad
                param -= lr * (m / (1.0 - beta1 ** step)) / (np.sqrt(v / (1.0 - beta2 ** step)) + eps)
            if on_epoch and (updates % checkpoint_every == 0 or updates == max_updates):
                val_loss = (float(np.mean(per_position_loss(model.predict_cp(val_split.X), t_val, k)))
                            if len(val_split) else math.nan)
                on_epoch({
                    "epoch": updates,             # in this regime the log index IS the update count
                    "train_loss": _finite_or_none(loss),
                    "val_loss": _finite_or_none(val_loss),
                    "examples_per_second": round(batch / max(time.perf_counter() - started, 1e-9) *
                                                 updates, 1),
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                })
        return

    for epoch in range(epochs):
        rng.shuffle(order)
        epoch_start, total = time.perf_counter(), 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            batch_aux = None
            if aux is not None:
                batch_aux = (aux[0][idx], aux[1][idx])
            loss, grads = model.loss_and_grads(train_split.X[idx].astype(np.float64), t_train[idx], k,
                                               aux=batch_aux, aux_weight=aux_weight)
            total += loss * len(idx)
            step += 1
            for name, grad in grads.items():
                param = model.params[name]
                if optimizer == "sgd":
                    param -= lr * grad
                    continue
                m, v = moments[name]
                m *= beta1
                m += (1.0 - beta1) * grad
                v *= beta2
                v += (1.0 - beta2) * grad * grad
                param -= lr * (m / (1.0 - beta1 ** step)) / (np.sqrt(v / (1.0 - beta2 ** step)) + eps)
        val_loss = float(np.mean(per_position_loss(model.predict_cp(val_split.X), t_val, k))) if len(val_split) else math.nan
        elapsed = time.perf_counter() - epoch_start
        if on_epoch:
            on_epoch({
                "epoch": epoch,
                "train_loss": _finite_or_none(total / max(len(order), 1)),
                "val_loss": _finite_or_none(val_loss),
                "examples_per_second": round(len(order) / elapsed, 1) if elapsed > 0 else 0.0,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            })


def evaluate(model: RawNnue, split: EncodedSplit, params: Mapping) -> dict[str, np.ndarray]:
    """Per-position metric values (NaN where a metric does not apply)."""
    k, lam = float(params["K"]), float(params["LAMBDA"])
    pred = model.predict_cp(split.X)
    label = np.clip(split.cp, -CP_CLIP, CP_CLIP)
    decisive = np.abs(split.cp) >= 100.0
    return {
        "test_loss": per_position_loss(pred, targets(split, lam, k), k),
        "cp_mae": np.abs(np.clip(pred, -CP_CLIP, CP_CLIP) - label),
        "sign_agreement": np.where(decisive, (np.sign(pred) == np.sign(split.cp)).astype(np.float64), np.nan),
    }


def bootstrap_mean(values: np.ndarray, samples: int, seed: int) -> tuple[Optional[float], Optional[float], Optional[float], int]:
    """Mean and percentile 95% bootstrap interval over finite values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return None, None, None, 0
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    chunk = max(1, min(samples, 4_000_000 // n))
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        means[start:start + count] = v[rng.integers(0, n, size=(count, n))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(low), float(high), n


def _param_names(payload: dict) -> tuple[str, ...]:
    if "w" in payload:
        return ("w", "b")
    names = ["w1", "b1", "w2", "b2"]
    for extra in ("wv", "bv", "wb", "bb"):
        if extra in payload:
            names.append(extra)
    return tuple(names)


def serialize(model, cfg: Mapping, meta: dict) -> dict:
    """JSON net; the clipped-ReLU format matches the Rust loader's `Nnue::load` (raw 768 mode)."""
    if isinstance(model, LinearEval):
        payload = {
            "arch": f"{model.inputs}-linear-1", "hidden": 0, "inputDim": model.inputs,
            "input": str(cfg.get("INPUT", "GEO")), "archKind": "linear",
            "outputScaleCp": float(cfg["OUT_SCALE_CP"]), "k": float(cfg["K"]),
            "lambda": float(cfg["LAMBDA"]),
            "w": model.params["w"].tolist(), "b": float(model.params["b"]),
        }
    else:
        payload = {
            "arch": f"{model.params['w1'].shape[0]}x{model.hidden}cReLU-1", "hidden": model.hidden,
            "inputDim": int(model.params["w1"].shape[0]), "input": str(cfg.get("INPUT", "RAW")),
            "archKind": "crelu1",
            "outputScaleCp": float(cfg["OUT_SCALE_CP"]), "k": float(cfg["K"]),
            "lambda": float(cfg["LAMBDA"]),
            "w1": model.params["w1"].tolist(), "b1": model.params["b1"].tolist(),
            "w2": model.params["w2"].tolist(), "b2": float(model.params["b2"]),
        }
        if model.aux_enabled:
            payload["auxEnabled"] = True
            payload["auxMode"] = str(cfg.get("AUX", "geo"))
            for extra in ("wv", "bv", "wb", "bb"):
                if extra in model.params:
                    payload[extra] = model.params[extra].tolist()
    payload.update({"note": "stm-POV, mirror+colorswap for black; produced by CVS Lab; UNPROMOTED",
                    "promotable": False, "cvslab": meta})
    return payload


def serialized_shapes(payload: dict) -> dict[str, list[int]]:
    return {name: list(np.asarray(payload[name], dtype=np.float64).shape)
            for name in _param_names(payload)}


def serialized_param_count(payload: dict) -> int:
    """Count learned parameters from the serialized arrays themselves, not from a formula."""
    return sum(int(np.asarray(payload[name], dtype=np.float64).size)
               for name in _param_names(payload))


def load_serialized(payload: dict):
    if payload.get("archKind") == "linear" or "w" in payload:
        params = {"w": np.asarray(payload["w"], dtype=np.float64),
                  "b": np.asarray(payload["b"], dtype=np.float64)}
        expected = {"w": (int(payload.get("inputDim", params["w"].shape[0])),), "b": ()}
        for name, shape in expected.items():
            if params[name].shape != shape:
                raise ValueError(f"serialized {name} has shape {params[name].shape}, expected {shape}")
        return LinearEval(params, float(payload["outputScaleCp"]))
    hidden = int(payload["hidden"])
    inputs = int(payload.get("inputDim", INPUT_DIM))
    params = {
        "w1": np.asarray(payload["w1"], dtype=np.float64),
        "b1": np.asarray(payload["b1"], dtype=np.float64),
        "w2": np.asarray(payload["w2"], dtype=np.float64),
        "b2": np.asarray(payload["b2"], dtype=np.float64),
    }
    expected = {"w1": (inputs, hidden), "b1": (hidden,), "w2": (hidden,), "b2": ()}
    for extra, rows in (("wv", hidden), ("wb", hidden)):
        if extra in payload:
            params[extra] = np.asarray(payload[extra], dtype=np.float64)
            expected[extra] = (rows, params[extra].shape[1])
    for extra in ("bv", "bb"):
        if extra in payload:
            params[extra] = np.asarray(payload[extra], dtype=np.float64)
            expected[extra] = (params[extra].shape[0],)
    for name, shape in expected.items():
        if params[name].shape != shape:
            raise ValueError(f"serialized {name} has shape {params[name].shape}, expected {shape}")
    return RawNnue(params, float(payload["outputScaleCp"]))
