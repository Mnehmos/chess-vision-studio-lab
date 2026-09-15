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


def encode_records(records: list[dict], *, target_spec=None) -> tuple[EncodedSplit, int]:
    """Encode canonical records.

    With a frozen ``TargetSpec`` the supervision is that spec's single matching label per
    record (strict: zero or multiple matches raise — an integrity failure, never a silent
    discard). Without a spec the legacy eval_cp behaviour applies (rows lacking the label
    are discarded and counted).
    """
    from .targets import extract_target

    feats, cps, results, ids = [], [], [], []
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
        cps.append(cp)
        results.append(math.nan if res is None else (float(res) if white_to_move else 1.0 - float(res)))
        ids.append(record["record_id"])
    X = np.zeros((len(feats), INPUT_DIM), dtype=np.uint8)
    for row, indices in enumerate(feats):
        X[row, indices] = 1
    return EncodedSplit(X, np.array(cps, dtype=np.float64), np.array(results, dtype=np.float64), ids), discarded


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def targets(split: EncodedSplit, lam: float, k: float) -> np.ndarray:
    eval_target = sigmoid(split.cp / k)
    has_result = np.isfinite(split.result)
    blended = lam * eval_target + (1.0 - lam) * np.where(has_result, split.result, 0.0)
    return np.where(has_result, blended, eval_target)


class RawNnue:
    def __init__(self, params: dict[str, np.ndarray], out_scale: float):
        self.params = params
        self.out_scale = out_scale

    @property
    def hidden(self) -> int:
        return int(self.params["b1"].shape[0])

    @classmethod
    def initialize(cls, inputs: int, hidden: int, init_std: float, out_scale: float, rng: np.random.Generator) -> "RawNnue":
        bound = 1.0 / math.sqrt(hidden)
        return cls(
            {
                "w1": rng.normal(0.0, init_std, size=(inputs, hidden)),
                "b1": np.zeros(hidden),
                "w2": rng.uniform(-bound, bound, size=hidden),
                "b2": np.array(rng.uniform(-bound, bound)),
            },
            out_scale,
        )

    def predict_cp(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        p = self.params
        out = np.empty(X.shape[0])
        for start in range(0, X.shape[0], batch):
            xb = X[start:start + batch].astype(np.float64)
            h = np.clip(xb @ p["w1"] + p["b1"], 0.0, 1.0)
            out[start:start + batch] = (h @ p["w2"] + p["b2"]) * self.out_scale
        return out

    def loss_and_grads(self, X: np.ndarray, t: np.ndarray, k: float) -> tuple[float, dict[str, np.ndarray]]:
        p = self.params
        z = X @ p["w1"] + p["b1"]
        h = np.clip(z, 0.0, 1.0)
        prob = sigmoid((h @ p["w2"] + p["b2"]) * self.out_scale / k)
        diff = prob - t
        g_y = 2.0 * diff / len(t) * prob * (1.0 - prob) * self.out_scale / k
        g_z = np.outer(g_y, p["w2"]) * ((z > 0.0) & (z < 1.0))
        grads = {"w1": X.T @ g_z, "b1": g_z.sum(axis=0), "w2": h.T @ g_y, "b2": np.array(g_y.sum())}
        return float(np.mean(diff ** 2)), grads

    def all_finite(self) -> bool:
        return all(bool(np.all(np.isfinite(v))) for v in self.params.values())


def per_position_loss(pred_cp: np.ndarray, t: np.ndarray, k: float) -> np.ndarray:
    return (sigmoid(pred_cp / k) - t) ** 2


def _finite_or_none(x: float) -> Optional[float]:
    return float(x) if math.isfinite(x) else None


def train(model: RawNnue, train_split: EncodedSplit, val_split: EncodedSplit, cfg: Mapping, rng: np.random.Generator,
          on_epoch: Optional[Callable[[dict], None]] = None) -> None:
    k, lam = float(cfg["K"]), float(cfg["LAMBDA"])
    epochs, batch, lr, optimizer = int(cfg["EPOCHS"]), int(cfg["BATCH"]), float(cfg["LR"]), str(cfg["OPTIMIZER"])
    t_train, t_val = targets(train_split, lam, k), targets(val_split, lam, k)
    order = np.arange(len(train_split))
    moments = {name: (np.zeros_like(v), np.zeros_like(v)) for name, v in model.params.items()}
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        rng.shuffle(order)
        epoch_start, total = time.perf_counter(), 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            loss, grads = model.loss_and_grads(train_split.X[idx].astype(np.float64), t_train[idx], k)
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


def serialize(model: RawNnue, cfg: Mapping, meta: dict) -> dict:
    """JSON net in the format read by the Rust engine's `Nnue::load` (raw 768 mode)."""
    return {
        "arch": f"{INPUT_DIM}x{model.hidden}cReLU-1",
        "hidden": model.hidden,
        "outputScaleCp": float(cfg["OUT_SCALE_CP"]),
        "k": float(cfg["K"]),
        "lambda": float(cfg["LAMBDA"]),
        "w1": model.params["w1"].tolist(),
        "b1": model.params["b1"].tolist(),
        "w2": model.params["w2"].tolist(),
        "b2": float(model.params["b2"]),
        "note": "stm-POV, mirror+colorswap for black; produced by CVS Lab; UNPROMOTED",
        "promotable": False,
        "cvslab": meta,
    }


def serialized_shapes(payload: dict) -> dict[str, list[int]]:
    return {name: list(np.asarray(payload[name], dtype=np.float64).shape) for name in ("w1", "b1", "w2", "b2")}


def serialized_param_count(payload: dict) -> int:
    """Count learned parameters from the serialized arrays themselves, not from a formula."""
    return sum(int(np.asarray(payload[name], dtype=np.float64).size) for name in ("w1", "b1", "w2", "b2"))


def load_serialized(payload: dict) -> RawNnue:
    hidden = int(payload["hidden"])
    params = {
        "w1": np.asarray(payload["w1"], dtype=np.float64),
        "b1": np.asarray(payload["b1"], dtype=np.float64),
        "w2": np.asarray(payload["w2"], dtype=np.float64),
        "b2": np.asarray(payload["b2"], dtype=np.float64),
    }
    expected = {"w1": (INPUT_DIM, hidden), "b1": (hidden,), "w2": (hidden,), "b2": ()}
    for name, shape in expected.items():
        if params[name].shape != shape:
            raise ValueError(f"serialized {name} has shape {params[name].shape}, expected {shape}")
    return RawNnue(params, float(payload["outputScaleCp"]))
