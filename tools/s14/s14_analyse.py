"""S14 analysis: the family-by-budget frontier, inference cost, slices, and the named-group
sentinel-sensitivity ablation.

    python tools/s14/s14_analyse.py report   # the full result artifact + summary
    python tools/s14/s14_analyse.py seal     # seal X#### with the result and membership

Every number comes from stored evidence: the runs' recorded exam metrics, the stored model
artifacts (reloaded through nnue.load_serialized), and deterministic recomputation from the exam
FENs. No outcome was inspected before the lattice was frozen.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from s14_common import DATASET, EXAM_PROTOCOL, FAMILIES, GROUPS, LAB, S14, SEEDS, save, state

import cvslab.data as data
import cvslab.nnue as nnue
from cvslab.funnel.campaign import paired_effect
from cvslab.funnel.providers import board_geometry
from cvslab.hashing import hash_obj, sha256_file
from cvslab.schemas import Experiment, Run, RunStatus
from cvslab.service import LabService, protocol_target_spec
from cvslab.store import LabError, Store

RESULT = S14 / "s14-result.json"
SUMMARY = S14 / "s14-result-summary.json"
BUDGETS = ("1000", "3000", "12000", "25000", "linear-floor")


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def _loss(model, split, k: float) -> float:
    per_position = nnue.evaluate(model, split, {"K": k, "LAMBDA": 1.0})
    values = per_position["test_loss"]
    return float(sum(values) / len(values))


def collect(service: LabService):
    experiment: Experiment = service.store.get_as(state()["xid"], Experiment)
    conditions = state()["conditions"]
    ablation_of = {}
    for family in FAMILIES:
        for budget, spec in conditions[family].items():
            ablation_of[spec["ablation_id"]] = (family, budget)
    values: dict = {}
    models: dict = {}
    invalid = []
    by_cell: dict = {}
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != experiment.id:
            continue
        by_cell.setdefault((run.ablation_id, run.seed), []).append(run)
    for cell_runs in by_cell.values():
        completed = [run for run in cell_runs if run.status == RunStatus.COMPLETED]
        if not completed:
            invalid.append(cell_runs[0].id)
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != experiment.id or run.status != RunStatus.COMPLETED:
            continue
        family, budget = ablation_of[run.ablation_id]
        metric = next((m for m in run.metrics if m.name == "test_loss"), None)
        if metric is None:
            invalid.append(run.id)
            continue
        values.setdefault((family, budget), {})[run.seed] = metric.value
        artifact = f"artifacts/{run.id}/model.json"
        models[(family, budget, run.seed)] = {
            "path": artifact, "payload": json.loads(service.store.abs(artifact).read_bytes()),
            "input": run.effective_config["INPUT"],
            "arch": run.effective_config.get("ARCH", "crelu1"),
            "wall_seconds": run.compute.wall_seconds if run.compute else None,
            "bytes": service.store.abs(artifact).stat().st_size,
            "train_examples": run.compute.train_examples_seen if run.compute else None}
    if invalid:
        raise LabError(f"{len(invalid)} cells have no COMPLETED run: {invalid[:3]}")
    for family in FAMILIES:
        for budget in conditions[family]:
            if len(values[(family, budget)]) != len(SEEDS):
                raise LabError(f"{family}@{budget}: {len(values[(family, budget)])} seeds")
    return experiment, conditions, values, models


def _paired(values, a_family, b_family, budget) -> dict:
    a = {seed: values[(a_family, budget)][seed] for seed in values[(a_family, budget)]}
    b = {seed: values[(b_family, budget)][seed] for seed in values[(b_family, budget)]}
    two = paired_effect(a, b, seeds=SEEDS)
    one = paired_effect(a, b, seeds=SEEDS, one_sided=True)
    return {"d": two["width_estimate"], "one_sided_bounds": [one["ci_low"], one["ci_high"]],
            "t_one_sided": one["critical"], "n": two["n"],
            "mean_a": statistics.mean(a.values()), "mean_b": statistics.mean(b.values())}


def _splits(service: LabService) -> dict:
    """Encoded exam and train splits per input kind, under the SAME frozen TargetSpecs the runs
    used (the exam protocol's spec for the exam; the CVS-4k training spec for the train rows)."""
    from cvslab.service import recipe_target_spec
    protocol = service.store.get(EXAM_PROTOCOL)
    exam_spec = protocol_target_spec(protocol)
    # the training spec is the CVS-4k spec X0012's recipe pins; take it from the recipe object
    experiment = service.store.get_as(state()["xid"], Experiment)
    recipe = service.store.get(experiment.arms[0].training_recipe_id)
    train_spec = recipe_target_spec(recipe)
    exam_dataset = service.store.get(protocol.dataset_id)
    train_dataset = service.store.get(DATASET)
    exam_records = data.load_split(service.store, exam_dataset, protocol.split)
    train_records = data.load_split(service.store, train_dataset, "train")
    out = {}
    for kind in ("RAW", "GEO", "HYBRID"):
        out[kind] = {"exam": nnue.encode_records(exam_records, target_spec=exam_spec,
                                                 input_kind=kind)[0],
                     "train": nnue.encode_records(train_records, target_spec=train_spec,
                                                  input_kind=kind)[0]}
    return out, exam_records, train_records


def _slices(model, split_raw_geo, exam_records, k: float) -> dict:
    """Exam loss by phase and material bucket, from the FENs alone (deterministic)."""
    per_position = nnue.evaluate(model, split_raw_geo["exam"], {"K": k, "LAMBDA": 1.0})
    losses = list(per_position["test_loss"])
    phases, materials = {}, {}
    for record, loss in zip(exam_records, losses):
        geometry, _ = board_geometry(record["fen"])
        phases.setdefault(geometry["phase"], []).append(float(loss))
        materials.setdefault(geometry["materialBucket"], []).append(float(loss))
    return {"phase": {key: sum(v) / len(v) for key, v in sorted(phases.items())},
            "material": {key: sum(v) / len(v) for key, v in sorted(materials.items())}}


def _inference_cost(models, splits, exam_records) -> dict:
    """Feature-encoding and forward-pass rates per representation (python, measured as-is)."""
    out = {}
    fens = [record["fen"] for record in exam_records]
    for kind, encoder in (("RAW", lambda fen: nnue.encode_fen(fen)),
                          ("GEO", nnue.encode_fen_geo)):
        started = time.perf_counter()
        for fen in fens * 20:
            encoder(fen)
        out[f"encode_{kind}_positions_per_second"] = round(
            (len(fens) * 20) / (time.perf_counter() - started), 1)
    sentinel = models[("HYBRID", "25000", 0)]["payload"]
    model = nnue.load_serialized(sentinel)
    X = splits["HYBRID"]["exam"].X
    started = time.perf_counter()
    for _ in range(200):
        model.predict_cp(X)
    out["forward_positions_per_second_hybrid25k"] = round(
        (len(X) * 200) / (time.perf_counter() - started), 1)
    raw_model = nnue.load_serialized(models[("RAW", "25000", 0)]["payload"])
    Xr = splits["RAW"]["exam"].X
    started = time.perf_counter()
    for _ in range(200):
        raw_model.predict_cp(Xr)
    out["forward_positions_per_second_raw25k"] = round(
        (len(Xr) * 200) / (time.perf_counter() - started), 1)
    out["note"] = ("encoding measured in the python implementation as shipped; the forward pass is "
                   "batched numpy over the 200 exam positions")
    return out


def _semantic_ablation(models, splits) -> dict:
    """Named-group input lesions at inference on the preregistered sentinel (HYBRID 12K, seed 0)."""
    payload = models[("HYBRID", "12000", 0)]["payload"]
    model = nnue.load_serialized(payload)
    exam = splits["HYBRID"]["exam"]
    base = _loss(model, exam, 256.0)
    rows = []
    for group, members in sorted(GROUPS.items()):
        X = exam.X.copy()
        from cvslab.facts import GEOMETRY_FAMILIES
        for index, (family, _thresholds) in enumerate(GEOMETRY_FAMILIES):
            if family in members:                    # zero BOTH columns of every member family
                X[:, 768 + 2 * index] = 0.0
                X[:, 768 + 2 * index + 1] = 0.0
        lesioned = nnue.EncodedSplit(X, exam.cp, exam.result, exam.record_ids)
        after = _loss(model, lesioned, 256.0)
        rows.append({"group": group, "families": members, "columns_zeroed": 2 * len(members),
                     "full_loss": base, "lesioned_loss": after, "effect": after - base})
    return {"sentinel": "HYBRID at the 12000 budget, seed 0 (preregistered)",
            "reading": "input lesions at INFERENCE on the frozen weights of THIS sentinel; the "
                       "table is sentinel sensitivity, not general feature importance",
            "base_loss": base, "groups": rows}


def step_report() -> int:
    service = svc()
    experiment, conditions, values, models = collect(service)
    splits, exam_records, _train_records = _splits(service)
    per_budget = {}
    for budget in BUDGETS:
        if budget == "linear-floor":
            entry = {"params": conditions["GEO"][budget]["learned_parameters"],
                     "means": {family: (statistics.mean(values[(family, budget)].values())
                                        if budget in conditions[family] else None)
                               for family in FAMILIES},
                     "note": "the 43-parameter direct floor: named inputs, no hidden layer"}
            per_budget[budget] = entry
            continue
        entry = {"params": {family: conditions[family][budget]["learned_parameters"] for family in FAMILIES},
                 "H": {family: conditions[family][budget]["H"] for family in FAMILIES},
                 "mismatch": {family: conditions[family][budget]["mismatch"] for family in FAMILIES},
                 "means": {family: statistics.mean(values[(family, budget)].values())
                           for family in FAMILIES},
                 "contrasts": {"GEO-RAW": _paired(values, "GEO", "RAW", budget),
                               "HYBRID-RAW": _paired(values, "HYBRID", "RAW", budget),
                               "HYBRID-GEO": _paired(values, "HYBRID", "GEO", budget)}}
        # mechanical arithmetic only: loss/parameters shrinks as the denominator grows, so it is
        # NOT an efficiency or capability score; the matched frontier above is the measurement
        entry["loss_per_1000_parameters_mechanical"] = {
            family: entry["means"][family] / (entry["params"][family] / 1000.0) for family in FAMILIES}
        per_budget[budget] = entry
    sizes = {family: {budget: models[(family, budget, 0)]["bytes"] for budget in conditions[family]}
             for family in FAMILIES}
    walls = {family: {budget: round(statistics.mean(
        models[(family, budget, seed)]["wall_seconds"] or 0.0 for seed in SEEDS), 2)
        for budget in conditions[family]} for family in FAMILIES}
    gaps = {family: {budget: round(_loss(nnue.load_serialized(models[(family, budget, 0)]["payload"]),
                                        splits[family]["train"], 256.0)
                                   - statistics.mean(values[(family, budget)].values()), 6)
                     for budget in conditions[family]} for family in FAMILIES}
    slices = {family: {budget: _slices(nnue.load_serialized(models[(family, budget, 0)]["payload"]),
                                       splits[family], exam_records, 256.0)
                       for budget in ("12000", "25000", "linear-floor") if budget in conditions[family]}
              for family in FAMILIES}
    result = {
        "experiment": experiment.id,
        "preregistration_hash": experiment.preregistration_hash,
        "question": "how much opaque learned capacity can named deterministic chess structure "
                    "replace, and does explicit geometry help RAW at matched learned-parameter "
                    "budgets?",
        "contract": {"teacher": f"CVS-4k over {DATASET} (unchanged from S12/S13)",
                     "exam": f"{EXAM_PROTOCOL} CVS-DEEP 400k over the 200 held-out identities",
                     "student_compute": "MAX_UPDATES 760 x BATCH 256 for every cell",
                     "seeds": list(SEEDS)},
        "per_budget": per_budget,
        "distribution_gap_train_minus_exam": gaps,
        "train_wall_seconds_mean": walls,
        "serialized_bytes": sizes,
        "inference_cost": _inference_cost(models, splits, exam_records),
        "semantic_ablation": _semantic_ablation(models, splits),
        "phase_material_slices": slices,
        "interpretability": {"GEO-LINEAR": "named inputs AND readable weights (43 parameters: one "
                                           "signed weight per registry column)",
                             "crelu1 models": "named inputs for GEO/HYBRID, but the hidden layer is "
                                              "opaque; the report never calls them interpretable"},
        "evidence": {"result": "tools/s14/s14-result.json",
                     "preregistration": "tools/s14/s14-preregistration.json",
                     "geometry_manifest": "tools/s14/s14-geometry-manifest.json",
                     "state": "tools/s14/s14-state.json"},
    }
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary = {"experiment": experiment.id,
               "per_budget": {budget: {"params": per_budget[budget].get("params"),
                                       "means": per_budget[budget]["means"],
                                       "contrasts": {name: {"d": round(stats["d"], 6),
                                                            "bounds": [round(x, 6) for x in stats["one_sided_bounds"]]}
                                                     for name, stats in per_budget[budget].get("contrasts", {}).items()}}
                              for budget in BUDGETS},
               "inference_cost": result["inference_cost"],
               "semantic_ablation": {row["group"]: round(row["effect"], 6)
                                     for row in result["semantic_ablation"]["groups"]},
               "evidence": result["evidence"]}
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"experiment {experiment.id}: budgets written", flush=True)
    for budget in BUDGETS:
        if budget == "linear-floor":
            print(f"  linear-floor: GEO-LINEAR loss {per_budget[budget]['means']['GEO']:.6f} "
                  f"({per_budget[budget]['params']} params)", flush=True)
            continue
        row = per_budget[budget]
        print(f"  {budget}: means " + " ".join(f"{f}={row['means'][f]:.6f}" for f in FAMILIES)
              + f" | HYBRID-RAW {row['contrasts']['HYBRID-RAW']['d']:+.6f} "
                f"{[round(x, 6) for x in row['contrasts']['HYBRID-RAW']['one_sided_bounds']]}",
              flush=True)
    return 0


def step_seal() -> int:
    service = svc()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    experiment = service.seal_experiment(state()["xid"], result)
    print(f"{experiment.id} sealed: {json.dumps(experiment.result.get('membership'), default=str)}",
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["report", "seal"])
    args = parser.parse_args()
    return {"report": step_report, "seal": step_seal}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
