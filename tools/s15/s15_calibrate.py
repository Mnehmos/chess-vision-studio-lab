"""S15 weight calibration: choose AUX_WEIGHT from training-dynamics scale ONLY.

The preregistered rule: at initialisation, on TRAINING rows only, measure the gradient norm the
score loss puts on the shared trunk and the norm the auxiliary loss puts there at weight 1. Set

    AUX_WEIGHT = GRADIENT_RATIO_TARGET * ||g_score|| / ||g_aux||

so the auxiliary objectives enter training with a declared fraction (25 %) of the score gradient's
influence, then freeze it. Gradients are averaged over the first few deterministic batches of the
frozen training split; nothing here reads validation or exam performance, and no weight is tuned
per seed or per width.

    python tools/s15/s15_calibrate.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s15_common import (BATCH, DATASET, GRADIENT_RATIO_TARGET, LAB, S15, WIDTHS, freeze_hash,
                        state, save, aux_manifest)

sys.path.insert(0, str(LAB))

import numpy as np

import cvslab.data as data
import cvslab.nnue as nnue
from cvslab.service import LabService, protocol_target_spec, recipe_target_spec
from cvslab.store import Store

CALIBRATION_BATCHES = 8
CALIBRATION_WIDTH = 4


def main() -> int:
    started = time.perf_counter()
    service = LabService(Store(str(LAB / "labstore")))
    dataset = service.store.get(DATASET)
    protocol = service.store.get("E0004")
    exam_spec = protocol_target_spec(protocol)
    train_records = data.load_split(service.store, dataset, "train")
    # the training supervision spec: D0036's recipe pins the CVS-4k TargetSpec
    spec = None
    for experiment in service.store.list("X", verify=True):
        for arm in getattr(experiment, "arms", []):
            if arm.dataset_id == DATASET:
                spec = recipe_target_spec(service.store.get(arm.training_recipe_id))
                break
        if spec is not None:
            break
    if spec is None:
        raise SystemExit("no experiment pins a recipe for D0036")
    split, _dropped = nnue.encode_records(train_records, target_spec=spec, input_kind="RAW")
    values, buckets = nnue.encode_aux(train_records, "geo")
    assert len(split) == len(values), "auxiliary targets must align with the encoded training rows"

    ratios = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        model = nnue.RawNnue.initialize(768, CALIBRATION_WIDTH, 0.05, 400.0, rng,
                                        aux_dimensions=nnue.aux_dimensions("geo"))
        t_all = nnue.targets(split, 1.0, 256.0)
        order = np.arange(len(split))
        rng.shuffle(order)
        g_score_norms, g_aux_norms = [], []
        for index in range(CALIBRATION_BATCHES):
            idx = order[index * BATCH:(index + 1) * BATCH]
            X = split.X[idx].astype(np.float64)
            _loss, grads = model.loss_and_grads(X, t_all[idx], 256.0)
            score_trunk = grads["w1"].copy()          # score path only (no aux targets supplied)
            _loss_aux, grads_aux = model.loss_and_grads(X, t_all[idx], 256.0,
                                                        aux=(values[idx], buckets[idx]), aux_weight=1.0)
            aux_trunk = grads_aux["w1"] - score_trunk
            g_score_norms.append(float(np.linalg.norm(score_trunk)))
            g_aux_norms.append(float(np.linalg.norm(aux_trunk)))
        ratios.append(float(np.mean(g_aux_norms) / np.mean(g_score_norms)))
    ratio = float(np.mean(ratios))
    weight = GRADIENT_RATIO_TARGET / ratio
    weight = float(f"{weight:.6g}")                   # a frozen, hashable number, not a tuned one

    artifact = {
        "id": "s15-aux-weight-calibration",
        "purpose": "choose AUX_WEIGHT from training-dynamics gradient scale ONLY; no validation or "
                   "exam performance is read and no weight is tuned per seed or per width",
        "rule": {"target_ratio": GRADIENT_RATIO_TARGET,
                 "formula": "AUX_WEIGHT = target_ratio * ||g_score|| / ||g_aux|| at initialisation",
                 "batches_per_seed": CALIBRATION_BATCHES, "seeds": [0, 1, 2],
                 "calibration_width": CALIBRATION_WIDTH,
                 "data": "the frozen training split's first deterministic batches"},
        "measured_aux_over_score_gradient_ratio_at_weight_1": ratios,
        "mean_ratio": ratio,
        "chosen_aux_weight": weight,
        "realized_ratio_at_chosen_weight": ratio * weight,
        "aux_manifest": aux_manifest(),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    digest = freeze_hash(artifact)
    (S15 / "s15-aux-weight-calibration.json").write_text(
        json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (S15 / "s15-aux-weight-calibration.hash").write_text(digest + "\n", encoding="utf-8")
    save(calibration={"aux_weight": weight, "mean_gradient_ratio": ratio,
                      "realized_ratio": ratio * weight, "hash": digest})
    print(f"AUX_WEIGHT = {weight} (aux/score trunk gradient {ratio:.4f} at weight 1, "
          f"realized {ratio * weight:.4f}); {digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
