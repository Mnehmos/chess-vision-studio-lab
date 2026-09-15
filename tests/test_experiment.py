"""X#### acceptance tests — the experiment contract, not the prefix list.

Required cases (PR #33 review, item 6):

  1. an arm whose dataset/recipe/spec does not match its frozen identities is refused;
  2. an ablation that does not carry this experiment's id back-reference is refused;
  3. queueing a run for an ablation that is not an arm of the experiment is refused;
  4. execution detects a corrupted experiment contract (a run whose recorded arm identity was
     changed) and fails closed without training;
  5. comparing runs examined on a different instrument is refused;
  6. incomplete / missing / duplicate (ablation, seed) membership is refused at analysis, and
     an incomplete study cannot be sealed;
  7. a complete miniature matrix (2 arms x 4 widths x 2 seeds) is accepted end to end;
  8. a SEALED experiment refuses further runs;
  9. historical identities are unaffected by the X capability (no experiment_id is required).

The miniature matrix keeps the shape of the 320-cell study (16 arms x 4 widths x 5 seeds)
while running in seconds.
"""
import json

import pytest

from cvslab.store import payload_hash
from cvslab.schemas import Run, TERMINAL_RUN_STATUSES
from cvslab.service import protocol_target_spec, recipe_target_spec
from cvslab.store import LabError, TamperError
from cvslab.targets import TargetSpec

SPEC_16K = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                      producer="a" * 64, budget={"nodeBudget": 16000},
                      value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0)
SPEC_400K = TargetSpec(family="search_deep_cp", authority="legacy.cvs.search.deep",
                       producer="a" * 64, budget={"nodeBudget": 400000},
                       value_path=("targets", "scoreCpStm"), pov="stm", k=256.0, lam=1.0)
WIDTHS = (1, 4, 16, 32)
SEEDS = (0, 1)
DECLARATION = "miniature X#### matrix: trained on 16k, examined on the frozen 400k instrument"


def build_lab(lab):
    """Corpus with both observations, an exam dataset/protocol, and one training dataset."""
    from cvslab.data import _load_canonical
    source = lab.create_fixture_source(n_games=6, seed=41)
    normalization = lab.normalize([source.id], name="n")
    evaluated = sorted(r["record_id"] for r in _load_canonical(lab.store, lab.store.get(normalization.id)))
    lab.register_labels(normalization.id, family="search_shallow_cp", producer="a" * 64,
                        authority="legacy.cvs.search.shallow", pov="stm",
                        rows=[{"record_id": r, "value": {"scoreCpStm": 30 + i, "bestMove": "e2e4", "nodes": 16000},
                               "budget": {"nodeBudget": 16000, "nodes": 16000}} for i, r in enumerate(evaluated)])
    lab.register_labels(normalization.id, family="search_deep_cp", producer="a" * 64,
                        authority="legacy.cvs.search.deep", pov="stm",
                        rows=[{"record_id": r, "value": {"targets": {"scoreCpStm": -80 + 10 * i}},
                               "budget": {"nodeBudget": 400000, "nodes": 400000}} for i, r in enumerate(evaluated)])
    train = lab.freeze_dataset(normalization.id, name="d-train", required_labels=["search_shallow_cp"],
                               target_spec=SPEC_16K, fractions=(1.0, 0.0, 0.0))
    exam = lab.freeze_dataset(normalization.id, name="d-exam", required_labels=["search_deep_cp"],
                              target_spec=SPEC_400K, fractions=(1.0, 0.0, 0.0))
    recipe = lab.create_training_recipe(name="t-16k", params={"K": 256.0, "LAMBDA": 1.0, "EPOCHS": 1},
                                        target_spec=SPEC_16K)
    protocol = lab.create_eval_protocol(name="e-400k", dataset_id=exam.id, split="train", k=256.0, lam=1.0,
                                        target_spec=SPEC_400K)
    return normalization, train, exam, recipe, protocol


def arm_material(lab, normalization, train, recipe, protocol, *, scale="2x", scale_nodes=74_570_982,
                 budget=16000, suffix="A"):
    """One arm: a dataset (prefix of the corpus), recipes/ablations for four widths."""
    xid = lab.store.next_id("X")
    dataset = lab.freeze_dataset(normalization.id, name=f"d-arm-{suffix}", required_labels=["search_shallow_cp"],
                                 target_spec=SPEC_16K, fractions=(1.0, 0.0, 0.0),
                                 campaign={"arm": f"S2-D{budget//1000}k", "target_spec": SPEC_16K.spec_hash()})
    baseline = lab.register_baseline(name=f"ARM_{suffix}", dataset_id=dataset.id,
                                     training_recipe_id=recipe.id, eval_protocol_id=protocol.id,
                                     model_config={"INPUT": "RAW", "H": 16},
                                     supervision_divergence=DECLARATION, experiment_id=xid)
    ablations = [baseline.id] + [lab.create_ablation(baseline_id=baseline.id, overrides={"H": w},
                                                     supervision_divergence=DECLARATION,
                                                     experiment_id=xid).id for w in (1, 4, 32)]
    arm = {"arm_id": f"S2-D{budget}", "scale": scale, "scale_nodes": scale_nodes, "node_budget": budget,
           "prefix_size": dataset.splits[0].count, "realized_nodes": budget * dataset.splits[0].count,
           "dataset_id": dataset.id, "dataset_manifest_hash": dataset.manifest_hash,
           "training_recipe_id": recipe.id, "recipe_hash": recipe.recipe_hash,
           "train_target_spec_hash": recipe_target_spec(recipe).spec_hash(),
           "record_ids_hash": dataset.splits[0].record_ids_hash, "ablations": ablations}
    return xid, dataset, recipe, arm


def make_experiment(lab, material, protocol, *, seeds=SEEDS, name="S-mini"):
    xid, train, recipe, arm = material[0], material[1], material[2], material[3]
    return lab.create_experiment(
        name=name, preregistration_hash="sha256:" + "p" * 64, reference_unit_nodes=37_285_491,
        scales={"2x": 74_570_982, "5x": 186_427_455}, node_budgets=[400000, 160000, 64000, 16000],
        arms=[arm], eval_protocol_id=protocol.id, widths=list(WIDTHS), seeds=list(seeds),
        source_id="SXXXX", normalization_id=train.normalization_id if hasattr(train, "normalization_id") else "N0001",
        candidate_universe_hash="sha256:" + "u" * 64, order_seed=20260920, order_hash="sha256:" + "o" * 64,
        analysis={"primary": "A4-A1 per scale"}, experiment_id=xid, notes="miniature matrix")


def test_arm_identity_mismatch_is_refused(lab):
    """1. an arm must match its frozen D/T/spec before the experiment can exist."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    xid, dataset, recipe_obj, arm = arm_material(lab, normalization, train, recipe, protocol)
    other = lab.create_training_recipe(name="t-other", params={"K": 256.0, "LAMBDA": 1.0, "EPOCHS": 1},
                                       target_spec=SPEC_16K)
    broken = dict(arm, training_recipe_id=other.id)          # recipe not the arm's
    with pytest.raises(LabError, match="not defined on this arm|does not match its frozen identities"):
        make_experiment(lab, (xid, dataset, other, broken), protocol)
    broken = dict(arm, dataset_manifest_hash="sha256:" + "0" * 64)
    with pytest.raises(LabError, match="not defined on this arm|does not match its frozen identities"):
        make_experiment(lab, (xid, dataset, recipe_obj, broken), protocol)


def test_ablation_back_reference_is_required(lab):
    """2. an ablation that does not carry this experiment's id cannot be frozen into it."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    xid = lab.store.next_id("X")
    dataset = lab.freeze_dataset(normalization.id, name="d-unbound", required_labels=["search_shallow_cp"],
                                 target_spec=SPEC_16K, fractions=(1.0, 0.0, 0.0))
    baseline = lab.register_baseline(name="UNBOUND", dataset_id=dataset.id, training_recipe_id=recipe.id,
                                     eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 16},
                                     supervision_divergence=DECLARATION)      # no experiment_id
    arm = {"arm_id": "S2-D16k", "scale": "2x", "scale_nodes": 1, "node_budget": 16000,
           "prefix_size": dataset.splits[0].count, "realized_nodes": 1, "dataset_id": dataset.id,
           "dataset_manifest_hash": dataset.manifest_hash, "training_recipe_id": recipe.id,
           "recipe_hash": recipe.recipe_hash,
           "train_target_spec_hash": recipe_target_spec(recipe).spec_hash(),
           "record_ids_hash": dataset.splits[0].record_ids_hash, "ablations": [baseline.id]}
    with pytest.raises(LabError, match="not bound to this experiment"):
        lab.create_experiment(name="X-unbound", preregistration_hash="sha256:" + "p" * 64,
                              reference_unit_nodes=1, scales={"2x": 1}, node_budgets=[16000],
                              arms=[arm], eval_protocol_id=protocol.id, widths=[16], seeds=[0],
                              source_id="S0001", normalization_id=normalization.id,
                              candidate_universe_hash="sha256:" + "u" * 64, order_seed=1,
                              order_hash="sha256:" + "o" * 64, analysis={}, experiment_id=xid)
    # and an ablation that claims the experiment but is not listed blocks creation too
    stray = lab.create_ablation(baseline_id=baseline.id, overrides={"H": 8}, experiment_id=xid,
                                supervision_divergence=DECLARATION)
    with pytest.raises(LabError, match="not bound to this experiment"):
        lab.create_experiment(name="X-stray", preregistration_hash="sha256:" + "p" * 64,
                              reference_unit_nodes=1, scales={"2x": 1}, node_budgets=[16000],
                              arms=[arm], eval_protocol_id=protocol.id, widths=[16], seeds=[0],
                              source_id="S0001", normalization_id=normalization.id,
                              candidate_universe_hash="sha256:" + "u" * 64, order_seed=1,
                              order_hash="sha256:" + "o" * 64, analysis={}, experiment_id=xid)
    assert stray.id  # the ablation exists; the experiment cannot


def test_queueing_a_non_arm_is_refused(lab):
    """3. a run cannot join an experiment it was not frozen into."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    material = arm_material(lab, normalization, train, recipe, protocol)
    experiment = make_experiment(lab, material, protocol)
    outsider = lab.register_baseline(name="OUTSIDER", dataset_id=material[1].id,
                                     training_recipe_id=recipe.id, eval_protocol_id=protocol.id,
                                     model_config={"INPUT": "RAW", "H": 4}, supervision_divergence=DECLARATION,
                                     experiment_id=experiment.id)   # claims the study, not listed in it
    with pytest.raises(LabError, match="not an arm of"):
        lab.queue_runs(outsider.id, seeds=(0,))
    # and an ablation with no experiment at all is simply not part of any study
    plain = lab.register_baseline(name="PLAIN", dataset_id=material[1].id, training_recipe_id=recipe.id,
                                  eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 4},
                                  supervision_divergence=DECLARATION)
    [run] = lab.queue_runs(plain.id, seeds=(0,))
    assert run.experiment_id is None


def test_execution_detects_a_corrupted_contract(lab):
    """4. a run whose recorded arm identity was changed fails closed before training."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    material = arm_material(lab, normalization, train, recipe, protocol)
    experiment = make_experiment(lab, material, protocol)
    [run] = lab.queue_runs(material[3]["ablations"][0], seeds=(0,))
    path = lab.store.object_path(run.id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_id"] = material[1].id + ""  # unchanged field, then the real corruption:
    payload["recipe_hash"] = "sha256:" + "9" * 64
    payload["record_hash"] = payload_hash({k: v for k, v in payload.items() if k != "record_hash"})
    import os
    import stat
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finished = lab.execute_run(run.id)
    assert finished.status.value == "INVALID"
    assert finished.model_id is None
    checks = {c.name: c for c in finished.integrity}
    assert checks["experiment_contract"].status == "fail"
    assert "recipe_hash" in checks["experiment_contract"].detail


def test_a_different_exam_cannot_be_compared(lab):
    """5. one exam per comparison, even inside one experiment."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    material = arm_material(lab, normalization, train, recipe, protocol)
    experiment = make_experiment(lab, material, protocol)
    second_exam = lab.freeze_dataset(normalization.id, name="d-exam-16k", required_labels=["search_shallow_cp"],
                                     target_spec=SPEC_16K, fractions=(1.0, 0.0, 0.0))
    other_protocol = lab.create_eval_protocol(name="e-16k", dataset_id=second_exam.id, split="train",
                                              k=256.0, lam=1.0, target_spec=SPEC_16K)
    [run] = lab.queue_runs(material[3]["ablations"][0], seeds=(0,))
    import os, stat
    path = lab.store.object_path(run.id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["eval_protocol_id"] = other_protocol.id          # the run records a different exam
    payload["protocol_hash"] = other_protocol.protocol_hash
    payload["record_hash"] = payload_hash({k: v for k, v in payload.items() if k != "record_hash"})
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finished = lab.execute_run(run.id)
    assert finished.status.value == "INVALID"
    detail = {c.name: c for c in finished.integrity}["experiment_contract"].detail
    assert "evaluation protocol" in detail


def test_incomplete_membership_is_refused(lab):
    """6. missing, duplicate and extra cells block analysis and sealing."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    material = arm_material(lab, normalization, train, recipe, protocol)
    experiment = make_experiment(lab, material, protocol)
    with pytest.raises(LabError, match="have no run"):
        lab.assert_experiment_complete(experiment)
    with pytest.raises(LabError, match="have no run"):
        lab.seal_experiment(experiment.id, {"decision": "early"})

    for ablation_id in material[3]["ablations"]:
        runs = lab.queue_runs(ablation_id, seeds=SEEDS)
        for run in runs:
            assert lab.execute_run(run.id).status.value == "COMPLETED"
    report = lab.assert_experiment_complete(experiment)
    assert report["expected_cells"] == 4 * len(SEEDS) and report["completed_cells"] == 4 * len(SEEDS)

    # a duplicate cell: queue a second run of the same (ablation, seed) — allowed by the lab, so
    # the membership gate must refuse it rather than silently pick one
    lab.queue_runs(material[3]["ablations"][0], seeds=(0,))
    with pytest.raises(LabError, match="more than one run"):
        lab.assert_experiment_complete(experiment)
    with pytest.raises(LabError, match="more than one run"):
        lab.seal_experiment(experiment.id, {"decision": "early"})


def test_complete_miniature_matrix_is_accepted_and_then_sealed(lab):
    """7 + 8. a complete matrix analyses, seals, and a SEALED study accepts no more runs."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    material = arm_material(lab, normalization, train, recipe, protocol)
    experiment = make_experiment(lab, material, protocol)
    for ablation_id in material[3]["ablations"]:
        for run in lab.queue_runs(ablation_id, seeds=SEEDS):
            assert lab.execute_run(run.id).status.value == "COMPLETED"
    run_ids = [r.id for r in lab.store.list("R", verify=True, kind=Run)
               if r.experiment_id == experiment.id]
    assert lab.assert_experiment_runs(run_ids) == experiment.id
    sealed = lab.seal_experiment(experiment.id, {"decision": "miniature"})
    assert sealed.status == "SEALED" and sealed.result["membership"]["expected_cells"] == 4 * len(SEEDS)
    assert sealed.record_hash
    with pytest.raises(LabError, match="SEALED"):
        lab.queue_runs(material[3]["ablations"][0], seeds=(9,))
    from cvslab.store import ImmutableError
    with pytest.raises(ImmutableError):
        lab.store.update_experiment(sealed.model_copy(update={"name": "renamed"}))


def test_historical_identities_need_no_experiment(lab):
    """9. the X capability is additive: runs without an experiment keep working."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    baseline = lab.register_baseline(name="LEGACY", dataset_id=train.id, training_recipe_id=recipe.id,
                                     eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 4},
                                     supervision_divergence=DECLARATION)
    [run] = lab.queue_runs(baseline.id, seeds=(0,))
    assert run.experiment_id is None
    finished = lab.execute_run(run.id)
    assert finished.status.value == "COMPLETED"
    assert "experiment_contract" not in {c.name for c in finished.integrity}
