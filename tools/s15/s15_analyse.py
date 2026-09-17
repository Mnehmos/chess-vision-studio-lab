"""S15 analysis: does training-only geometry supervision improve the RAW evaluator?

    python tools/s15/s15_analyse.py report   # primary result + aux quality + throughput + gaps
    python tools/s15/s15_analyse.py seal     # seal the primary experiment with the result

Every number comes from stored evidence: the runs' recorded exam metrics, the stored model
artifacts (reloaded through nnue.load_serialized), and deterministic recomputation from the frozen
splits. The family-ablation contrast is computed here (cross-experiment by construction, since the
triggered ablation matrix is its own experiment) with the same paired machinery the Lab uses.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s15_common import (ABLATION_MODES, ABLATION_WIDTH, DATASET, EXAM_PROTOCOL, GRADIENT_RATIO_TARGET,
                        LAB, SEEDS, S15, WIDTHS, save, state)

sys.path.insert(0, str(LAB))

import numpy as np

import cvslab.data as data
import cvslab.nnue as nnue
from cvslab.funnel.campaign import paired_effect
from cvslab.hashing import hash_obj
from cvslab.schemas import Experiment, Run, RunStatus
from cvslab.service import LabService, protocol_target_spec, recipe_target_spec
from cvslab.store import LabError, Store

V2 = "--v2" in sys.argv                       # the split-stream replacement (X0014)
RESULT = S15 / ("s15-result-v2.json" if V2 else "s15-result.json")
SUMMARY = S15 / ("s15-result-summary-v2.json" if V2 else "s15-result-summary.json")
XID_KEY = "xid_v2" if V2 else "xid"
PREREG_KEY = "preregistration_v2" if V2 else "preregistration"
ABLATION_KEY = "ablation_v2" if V2 else "ablation"
ARMS = ("RAW-SCORE", "AUX-GEO")


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def collect(service: LabService):
    experiment: Experiment = service.store.get_as(state()[XID_KEY], Experiment)
    arm_of = {}
    recipe_of = {}
    for arm in experiment.arms:
        for ablation_id in arm.ablations:
            arm_of[ablation_id] = arm.arm_id.replace("S15-", "")
        recipe_of[arm.arm_id.replace("S15-", "")] = arm.training_recipe_id
    by_cell: dict = {}
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id == experiment.id:
            by_cell.setdefault((run.ablation_id, run.seed), []).append(run)
    values: dict = {}
    models: dict = {}
    for cell, runs in by_cell.items():
        completed = [r for r in runs if r.status == RunStatus.COMPLETED]
        if not completed:
            raise LabError(f"cell {cell} has no COMPLETED run")
        run = completed[0]
        label = arm_of[run.ablation_id]
        hidden = int(run.effective_config["H"])
        values.setdefault((label, hidden), {})[run.seed] = next(
            m.value for m in run.metrics if m.name == "test_loss")
        models[(label, hidden, run.seed)] = {
            "payload": json.loads(service.store.abs(f"artifacts/{run.id}/model.json").read_bytes()),
            "wall_seconds": run.compute.wall_seconds if run.compute else None,
            "cp_mae": next(m.value for m in run.metrics if m.name == "cp_mae"),
            "sign_agreement": next(m.value for m in run.metrics if m.name == "sign_agreement"),
            "aux_mode": run.effective_config.get("AUX", "none")}
    for label in ARMS:
        for hidden in WIDTHS:
            if len(values[(label, hidden)]) != len(SEEDS):
                raise LabError(f"{label} H{hidden}: {len(values[(label, hidden)])} seeds")
    return experiment, values, models, recipe_of


def _contrast(values, hidden: int) -> dict:
    a = values[("AUX-GEO", hidden)]
    b = values[("RAW-SCORE", hidden)]
    two = paired_effect(a, b, seeds=SEEDS)
    one = paired_effect(a, b, seeds=SEEDS, one_sided=True)
    return {"d": two["width_estimate"], "n": two["n"], "t_one_sided": one["critical"],
            "one_sided_bounds": [one["ci_low"], one["ci_high"]],
            "two_sided_ci": [two["ci_low"], two["ci_high"]],
            "mean_aux": statistics.mean(a.values()), "mean_control": statistics.mean(b.values()),
            "verdict": ("aux_worse" if one["ci_low"] > 0 else
                        "aux_better" if one["ci_high"] < 0 else "unresolved")}


def _param_accounting(hidden: int) -> dict:
    from cvslab import families as fam
    out = {}
    for label, mode, weight in (("RAW-SCORE", "none", 0.0), ("AUX-GEO", "geo",
                                                             state()["calibration"]["aux_weight"])):
        shapes = fam.parameter_shapes("NNUE", {"INPUT": "RAW", "H": hidden, "AUX": mode})
        total = fam.count_parameters(shapes)
        inference = fam.count_parameters(fam.inference_parameter_shapes(shapes))
        out[label] = {"total_parameters": total, "inference_parameters": inference,
                      "training_only_parameters": total - inference, "aux_weight": weight}
    return out


def _aux_quality(service, models, hidden: int) -> dict:
    """Per-family auxiliary prediction on the training rows (geometry is deterministic)."""
    entry = models.get(("AUX-GEO", hidden, 0))
    if entry is None:
        return {}
    model = nnue.load_serialized(entry["payload"])
    if not model.aux_enabled:
        return {}
    records = data.load_split(service.store, service.store.get(DATASET), "train")
    values, buckets = nnue.encode_aux(records, entry["aux_mode"])
    families = list(nnue.aux_families(entry["aux_mode"]))
    rng = np.random.default_rng(0)
    sample = np.arange(len(records)) if len(records) <= 4000 else rng.choice(len(records), 4000, replace=False)
    X = nnue.encode_records([records[i] for i in sample], target_spec=_train_spec(service),
                            input_kind="RAW")[0].X.astype(np.float64)
    hidden_activation = np.clip(X @ model.params["w1"] + model.params["b1"], 0.0, 1.0)
    predicted_values = hidden_activation @ model.params["wv"] + model.params["bv"]
    logits = (hidden_activation @ model.params["wb"] + model.params["bb"]).reshape(
        len(sample), len(families), nnue.AUX_BUCKET_CLASSES)
    predicted_buckets = logits.argmax(axis=2)
    rows = []
    for index, family in enumerate(families):
        residual = predicted_values[:, index] - values[sample, index]
        rows.append({"family": family,
                     "value_mse": float(np.mean(residual ** 2)),
                     "value_mae": float(np.mean(np.abs(residual))),
                     "bucket_accuracy": float(np.mean(predicted_buckets[:, index] == buckets[sample, index]))})
    return {"sampled_rows": len(sample), "families": rows}


def _train_spec(service: LabService):
    experiment = service.store.get_as(state()[XID_KEY], Experiment)
    return recipe_target_spec(service.store.get(experiment.arms[0].training_recipe_id))


def _throughput(models) -> dict:
    """Deployment cost with the auxiliary heads disabled (they are never consulted)."""
    out = {}
    for label in ARMS:
        entry = models[(label, max(WIDTHS), 0)]
        model = nnue.load_serialized(entry["payload"])
        X = np.random.default_rng(0).normal(size=(200, 768))
        started = time.perf_counter()
        for _ in range(200):
            model.predict_cp(X)
        out[label] = {"positions_per_second": round(200 * 200 / (time.perf_counter() - started), 1),
                      "inference_parameters": _param_accounting(max(WIDTHS))[label]["inference_parameters"]}
    out["note"] = ("both arms deploy the same RAW-768 -> score evaluator; the treatment model's "
                   "auxiliary arrays are present in the artifact but predict_cp never reads them")
    return out


def _ablation_contrasts(service: LabService, values) -> dict:
    st = state()
    record = st.get(ABLATION_KEY)
    if not record or not record.get("fired") or "xid" not in record:
        return {"fired": False, "outcome": (record or {}).get("outcome", "not evaluated")}
    experiment: Experiment = service.store.get_as(record["xid"], Experiment)
    arm_of = {a: arm.arm_id.replace("S15-AUX-", "") for arm in experiment.arms
              for a in arm.ablations}
    per_mode: dict = {}
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != experiment.id or run.status != RunStatus.COMPLETED:
            continue
        mode = arm_of[run.ablation_id]
        per_mode.setdefault(mode, {})[run.seed] = next(
            m.value for m in run.metrics if m.name == "test_loss")
    full = values[("AUX-GEO", ABLATION_WIDTH)]
    rows = []
    for mode, series in sorted(per_mode.items()):
        if len(series) != len(SEEDS):
            raise LabError(f"{mode}: {len(series)} seeds")
        one = paired_effect(full, series, seeds=SEEDS, one_sided=True)
        rows.append({"mode": mode, "removed": mode.replace("geo-no-", ""),
                     "effect_full_minus_removed": one["width_estimate"],
                     "one_sided_bounds": [one["ci_low"], one["ci_high"]],
                     "mean_removed": statistics.mean(series.values()),
                     "verdict": ("removal_worse" if one["ci_low"] > 0 else
                                 "removal_better" if one["ci_high"] < 0 else "unresolved")})
    return {"fired": True, "experiment": experiment.id, "width": ABLATION_WIDTH, "rows": rows,
            "reading": "positive effect = removing that family made the score loss worse, i.e. the "
                       "family contributed training signal at the sentinel width"}


def step_report() -> int:
    service = svc()
    experiment, values, models, recipe_of = collect(service)
    per_width = {}
    for hidden in WIDTHS:
        per_width[str(hidden)] = {
            "contrast": _contrast(values, hidden),
            "params": _param_accounting(hidden),
            "cp_mae": {label: statistics.mean(models[(label, hidden, seed)]["cp_mae"] for seed in SEEDS)
                       for label in ARMS},
            "sign_agreement": {label: statistics.mean(
                models[(label, hidden, seed)]["sign_agreement"] for seed in SEEDS) for label in ARMS},
            "wall_seconds_mean": {label: round(statistics.mean(
                models[(label, hidden, seed)]["wall_seconds"] or 0.0 for seed in SEEDS), 2)
                for label in ARMS},
            "aux_quality": _aux_quality(service, models, hidden)}
    train_spec = _train_spec(service)
    train_rows = data.load_split(service.store, service.store.get(DATASET), "train")
    gaps = {}
    for label in ARMS:
        payload = models[(label, max(WIDTHS), 0)]["payload"]
        model = nnue.load_serialized(payload)
        split, _ = nnue.encode_records(train_rows, target_spec=train_spec, input_kind="RAW")
        per_position = nnue.evaluate(model, split, {"K": 256.0, "LAMBDA": 1.0})
        train_loss = float(np.mean(per_position["test_loss"]))
        gaps[label] = {"train_loss": train_loss,
                       "exam_loss": statistics.mean(values[(label, max(WIDTHS))].values()),
                       "gap": train_loss - statistics.mean(values[(label, max(WIDTHS))].values())}
    result = {
        "experiment": experiment.id,
        "preregistration_hash": experiment.preregistration_hash,
        "question": "does explicit semantic supervision help a RAW evaluator learn a better static "
                    "representation even when no semantic features are supplied at inference?",
        "contract": {"teacher": f"CVS-4k over {DATASET} (unchanged)",
                     "exam": f"{EXAM_PROTOCOL} CVS-DEEP 400k over the 200 held-out identities",
                     "arms": {"RAW-SCORE": "AUX=none", "AUX-GEO": f"AUX=geo, weight "
                               f"{state()['calibration']['aux_weight']} (25% of the score loss's "
                               "trunk gradient at initialisation)"},
                     "identical": ["RAW-768 inputs", "positions and split identities", "score labels",
                                   "optimizer and update count", "persistent score topology"]},
        "per_width": per_width,
        "gap_at_widest": {"per_arm": gaps,
                          "note": "no validation split exists for D0036 (fractions 1.0/0.0/0.0); this "
                                  "is a train-vs-exam distribution gap, not a classic overfit gap"},
        "inference_throughput": _throughput(models),
        "family_ablation": _ablation_contrasts(service, values),
        "random_streams": ("split: persistent init, auxiliary init and minibatch sampling are "
                           "independent streams from the run seed, so paired arms share their "
                           "example order (X0013, which shared one stream, is preserved as "
                           "superseded)" if V2 else
                           "shared generator: X0013's arms did not share a minibatch order — "
                           "preserved as superseded, see s15-x0013-supersession.json"),
        "fairness": {"extra_training_only_params_reported": True, "no_inference_inputs_added": True,
                     "no_per_arm_lr_tuning": True, "no_teacher_change": True},
        "evidence": {"result": f"tools/s15/s15-result{'-v2' if V2 else ''}.json",
                     "preregistration": f"tools/s15/s15-preregistration{'-v2' if V2 else ''}.json",
                     "aux_manifest": "tools/s15/s15-aux-manifest.json",
                     "calibration": "tools/s15/s15-aux-weight-calibration.json",
                     "state": "tools/s15/s15-state.json"},
    }
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary = {"experiment": experiment.id,
               "per_width": {str(h): {"d_aux_minus_control": round(per_width[str(h)]["contrast"]["d"], 6),
                                      "one_sided_bounds": [round(x, 6) for x in per_width[str(h)]["contrast"]["one_sided_bounds"]],
                                      "verdict": per_width[str(h)]["contrast"]["verdict"],
                                      "mean_aux": round(per_width[str(h)]["contrast"]["mean_aux"], 6),
                                      "mean_control": round(per_width[str(h)]["contrast"]["mean_control"], 6),
                                      "training_only_params": per_width[str(h)]["params"]["AUX-GEO"]["training_only_parameters"]}
                              for h in WIDTHS},
               "family_ablation": result["family_ablation"],
               "evidence": result["evidence"]}
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"experiment {experiment.id}", flush=True)
    for hidden in WIDTHS:
        c = per_width[str(hidden)]["contrast"]
        print(f"  H{hidden}: aux {c['mean_aux']:.6f} vs control {c['mean_control']:.6f} "
              f"d={c['d']:+.6f} [{c['one_sided_bounds'][0]:+.6f}, {c['one_sided_bounds'][1]:+.6f}] "
              f"{c['verdict']}", flush=True)
    print("  ablation:", json.dumps(result["family_ablation"].get("fired")), flush=True)
    return 0


def step_seal() -> int:
    service = svc()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    experiment = service.seal_experiment(state()[XID_KEY], result)
    print(f"{experiment.id} sealed: {json.dumps(experiment.result.get('membership'), default=str)}",
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["report", "seal"])
    parser.add_argument("--v2", action="store_true", help="analyse the split-stream replacement")
    args = parser.parse_args()
    return {"report": step_report, "seal": step_seal}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
