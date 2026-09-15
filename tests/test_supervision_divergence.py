"""P1 acceptance tests — the lesson and the exam are not the same thing.

The four tests below are the scientific contract of P1 (S7 preregistration §5.8):

1. undeclared training/evaluation TargetSpec divergence still fails, exactly as before;
2. explicitly declared divergence executes, and the run exposes BOTH supervision hashes;
3. strict label cardinality is untouched: zero or multiple matching observations fail;
4. runs measured against different evaluation TargetSpecs are refused as comparable.

Plus two guards on the mechanism itself: the declaration is part of the ablation identity
(and its absence leaves historical identities byte-identical), and the comparison guard
passes when one instrument is genuinely shared.
"""
import json

import pytest

from cvslab.data import _load_canonical
from cvslab.hashing import hash_obj
from cvslab.schemas import Dataset, EvaluationProtocol, TrainingRecipe
from cvslab.service import protocol_target_spec, recipe_target_spec
from cvslab.store import LabError
from cvslab.targets import TargetSpec

TRAIN_SPEC = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                        producer="a" * 64, budget={"nodeBudget": 16000},
                        value_path=("scoreCpStm",))
EVAL_SPEC = TargetSpec(family="search_deep_cp", authority="legacy.cvs.search.deep",
                       producer="a" * 64, budget={"nodeBudget": 400000},
                       value_path=("targets", "scoreCpStm"))
REASON = "S7 arm: trained at 16k, measured on the frozen 400k instrument"


def shallow_rows(evaluated, *, base: int = 40):
    return [{"record_id": record_id, "value": {"scoreCpStm": base + index, "bestMove": "e2e4", "nodes": 16000},
             "budget": {"nodeBudget": 16000, "nodes": 16000}}
            for index, record_id in enumerate(evaluated)]


def deep_rows(evaluated, *, base: int = -120):
    return [{"record_id": record_id, "value": {"targets": {"scoreCpStm": base + 10 * index}},
             "budget": {"nodeBudget": 400000, "nodes": 400000}}
            for index, record_id in enumerate(evaluated)]


def corpus(lab, *, seed: int, shallow: bool = True, deep: bool = True):
    """Canonical corpus optionally carrying a 16k shallow and/or a 400k deep label."""
    source = lab.create_fixture_source(n_games=6, seed=seed)
    normalization = lab.normalize([source.id], name="n")
    records = _load_canonical(lab.store, lab.store.get(normalization.id))
    evaluated = sorted(record["record_id"] for record in records)
    if shallow:
        lab.register_labels(normalization.id, family="search_shallow_cp", producer="a" * 64,
                            authority="legacy.cvs.search.shallow", pov="stm", rows=shallow_rows(evaluated))
    if deep:
        lab.register_labels(normalization.id, family="search_deep_cp", producer="a" * 64,
                            authority="legacy.cvs.search.deep", pov="stm", rows=deep_rows(evaluated))
    return normalization, evaluated


def divergent_pair(lab, normalization):
    """D_TRAIN supervised at 16k, D_EXAM supervised at 400k: different lessons, same exam."""
    train = lab.freeze_dataset(normalization.id, name="d-train-16k", required_labels=["search_shallow_cp"],
                               target_spec=TRAIN_SPEC)
    exam = lab.freeze_dataset(normalization.id, name="d-exam-400k", required_labels=["search_deep_cp"],
                              target_spec=EVAL_SPEC, fractions=(1.0, 0.0, 0.0))
    recipe = lab.create_training_recipe(name="t-16k", params={"K": 256.0, "LAMBDA": 1.0, "EPOCHS": 2},
                                        target_spec=TRAIN_SPEC)
    protocol = lab.create_eval_protocol(name="e-400k", dataset_id=exam.id, split="train", k=256.0, lam=1.0,
                                        target_spec=EVAL_SPEC)
    assert recipe_target_spec(recipe).spec_hash() != protocol_target_spec(protocol).spec_hash()
    return train, exam, recipe, protocol


def test_undeclared_divergence_still_fails_exactly_as_before(lab):
    """P1 test 1: the old fail-closed behaviour is the default, not a removed check."""
    normalization, _ = corpus(lab, seed=11)
    train, exam, recipe, protocol = divergent_pair(lab, normalization)
    baseline = lab.register_baseline(name="UNDECLARED", dataset_id=train.id, training_recipe_id=recipe.id,
                                     eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 4})
    assert baseline.supervision_divergence == ""
    [run] = lab.queue_runs(baseline.id, seeds=(0,))
    assert run.train_target_spec_hash == TRAIN_SPEC.spec_hash()  # recorded even when refused
    assert run.eval_target_spec_hash == EVAL_SPEC.spec_hash()
    finished = lab.execute_run(run.id)
    assert finished.status.value == "INVALID"
    assert not finished.metrics and finished.model_id is None  # never trained, never measured
    checks = {c.name: c for c in finished.integrity}
    assert checks["supervision_divergence"].status == "fail"
    assert "NOT declared" in checks["supervision_divergence"].detail
    assert checks["preflight"].status == "fail"
    assert "TargetSpecs differ" in checks["preflight"].detail


def test_declared_divergence_executes_and_the_run_exposes_both_supervision_hashes(lab):
    """P1 test 2: the declared contract runs, and R#### states what taught and judged it."""
    normalization, evaluated = corpus(lab, seed=12)
    train, exam, recipe, protocol = divergent_pair(lab, normalization)
    baseline = lab.register_baseline(name="DECLARED", dataset_id=train.id, training_recipe_id=recipe.id,
                                     eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 4},
                                     supervision_divergence=REASON)
    assert baseline.supervision_divergence == REASON
    [run] = lab.queue_runs(baseline.id, seeds=(0,))
    # both identities are on the queued run itself, before execution and without inference
    assert run.train_target_spec_hash == TRAIN_SPEC.spec_hash()
    assert run.eval_target_spec_hash == EVAL_SPEC.spec_hash()
    assert run.supervision_divergence == REASON
    finished = lab.execute_run(run.id)
    assert finished.status.value == "COMPLETED", [c.detail for c in finished.integrity if c.status == "fail"]
    detail = {c.name: c for c in finished.integrity}["supervision_divergence"].detail
    assert "declared divergence" in detail
    assert TRAIN_SPEC.spec_hash()[:19] in detail and EVAL_SPEC.spec_hash()[:19] in detail
    # trained on the 16k lesson, examined on the 400k exam's records
    payload = json.loads(lab.store.abs(f"artifacts/{run.id}/eval.json").read_text(encoding="utf-8"))
    assert len(payload["record_ids"]) == len(evaluated)  # examined exactly the exam's records
    assert sorted(payload["record_ids"]) == evaluated
    # the declaration survives on the stored run, not just in memory
    stored = lab.store.get_as(run.id, type(run))
    assert stored.supervision_divergence == REASON
    assert stored.train_target_spec_hash == TRAIN_SPEC.spec_hash()
    assert stored.eval_target_spec_hash == EVAL_SPEC.spec_hash()


def test_declaration_is_part_of_the_ablation_identity_and_absent_leaves_history_alone(lab):
    """The contract must exist before the run and cannot be added once results are visible."""
    normalization, _ = corpus(lab, seed=13)
    train, exam, recipe, protocol = divergent_pair(lab, normalization)
    plain_base = lab.register_baseline(name="IDPLAIN", dataset_id=train.id, training_recipe_id=recipe.id,
                                       eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 4})
    assert plain_base.supervision_divergence == ""
    plain = lab.preview_ablation(baseline_id=plain_base.id, overrides={"H": 8})
    declared = lab.preview_ablation(baseline_id=plain_base.id, overrides={"H": 8}, supervision_divergence=REASON)
    other = lab.preview_ablation(baseline_id=plain_base.id, overrides={"H": 8},
                                 supervision_divergence="different reason")
    assert plain.identity_hash != declared.identity_hash  # declaring changes the object
    assert declared.identity_hash != other.identity_hash  # so does changing the contract
    assert plain.supervision_divergence == "" and declared.supervision_divergence == REASON
    declared_base = lab.register_baseline(name="IDDECL", dataset_id=train.id, training_recipe_id=recipe.id,
                                          eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 4},
                                          supervision_divergence=REASON)
    assert declared_base.identity_hash != plain_base.identity_hash
    # regression guard: an undeclared ablation hashes exactly as it did before P1 existed,
    # so no historical A#### identity is reinterpreted by this capability
    dataset: Dataset = lab.store.get(plain_base.dataset_id)
    rec: TrainingRecipe = lab.store.get_as(plain_base.training_recipe_id, TrainingRecipe)
    proto: EvaluationProtocol = lab.store.get_as(plain_base.eval_protocol_id, EvaluationProtocol)
    legacy = hash_obj({"family": plain_base.family, "generation": plain_base.generation,
                       "effective_config": plain_base.effective_config,
                       "dataset_manifest_hash": dataset.manifest_hash,
                       "recipe_hash": rec.recipe_hash, "protocol_hash": proto.protocol_hash})
    assert plain_base.identity_hash == legacy


def test_strict_label_cardinality_survives_the_separation(lab):
    """P1 test 3: relaxing an experiment-level equality must not loosen record-level matching."""
    # zero shallow observations -> the training spec cannot resolve its lesson
    normalization, evaluated = corpus(lab, seed=14, shallow=False)
    with pytest.raises(LabError):
        lab.freeze_dataset(normalization.id, name="d-missing", required_labels=["search_shallow_cp"],
                           target_spec=TRAIN_SPEC)
    # two shallow observations per record -> ambiguous lesson, refused
    normalization2, evaluated2 = corpus(lab, seed=15)
    lab.register_labels(normalization2.id, family="search_shallow_cp", producer="a" * 64,
                        authority="legacy.cvs.search.shallow", pov="stm",
                        rows=shallow_rows(evaluated2, base=7))
    with pytest.raises(LabError):
        lab.freeze_dataset(normalization2.id, name="d-two", required_labels=["search_shallow_cp"],
                           target_spec=TRAIN_SPEC)
    # zero deep observations -> the exam cannot resolve its target either
    normalization3, _ = corpus(lab, seed=16, deep=False)
    with pytest.raises(LabError):
        lab.freeze_dataset(normalization3.id, name="d-exam-missing", required_labels=["search_deep_cp"],
                           target_spec=EVAL_SPEC, fractions=(1.0, 0.0, 0.0))


def test_runs_with_different_evaluation_specs_refuse_to_be_compared(lab):
    """P1 test 4: two evaluations are not one measurement just because both are centipawns."""
    normalization, _ = corpus(lab, seed=17)
    train, exam400, recipe, protocol400 = divergent_pair(lab, normalization)
    exam16 = lab.freeze_dataset(normalization.id, name="d-exam-16k", required_labels=["search_shallow_cp"],
                                target_spec=TRAIN_SPEC, fractions=(1.0, 0.0, 0.0))
    protocol16 = lab.create_eval_protocol(name="e-16k", dataset_id=exam16.id, split="train", k=256.0, lam=1.0,
                                          target_spec=TRAIN_SPEC)
    # the SAME exam semantics, but a different protocol object (different bootstrap seed)
    protocol400b = lab.create_eval_protocol(name="e-400k-b", dataset_id=exam400.id, split="train", k=256.0,
                                            lam=1.0, target_spec=EVAL_SPEC, bootstrap_seed=7)

    # three arms, one lesson, different examinations; every arm is legitimate on its own
    def arm(name, protocol):
        registered = lab.register_baseline(name=name, dataset_id=train.id, training_recipe_id=recipe.id,
                                           eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 4},
                                           supervision_divergence=REASON)
        [queued] = lab.queue_runs(registered.id, seeds=(0,))
        assert lab.execute_run(queued.id).status.value == "COMPLETED"
        return registered, queued

    arm400, run400 = arm("EXAM400", protocol400)
    arm16, run16 = arm("EXAM16", protocol16)
    arm400b, run400b = arm("EXAM400B", protocol400b)
    assert run400.eval_target_spec_hash != run16.eval_target_spec_hash

    # rule 1 (the exam): different evaluation TargetSpecs cannot be compared, and the refusal
    # names both exam identities so the reason is legible in the evidence
    with pytest.raises(LabError) as excinfo:
        lab.assert_runs_comparable([run400.id, run16.id])
    message = str(excinfo.value)
    assert "not comparable" in message
    assert EVAL_SPEC.spec_hash()[:19] in message and TRAIN_SPEC.spec_hash()[:19] in message

    # rule 2 (the instrument): same exam semantics, different protocol object -> still not one
    # measurement, because the instrument is the protocol, not only its supervision definition
    with pytest.raises(LabError, match="distinct instruments"):
        lab.assert_runs_comparable([run400.id, run400b.id])

    # one arm's own runs are comparable, and the guard returns the shared exam identity
    [other400] = lab.queue_runs(arm400.id, seeds=(1,))
    assert lab.execute_run(other400.id).status.value == "COMPLETED"
    assert lab.assert_runs_comparable([run400.id, other400.id]) == EVAL_SPEC.spec_hash()

    # the finding path evaluates the same predicate on a real control/intervention pair:
    # both arms of one pair share their instrument, so the check passes and the comparison
    # is allowed — while the cross-exam pair above is refused by the guard
    hypothesis = lab.create_hypothesis(title="same exam", statement="one instrument per comparison")
    intervention = lab.create_ablation(baseline_id=arm400.id, overrides={"H": 8}, hypothesis_id=hypothesis.id,
                                       supervision_divergence=REASON)
    [intervention_run] = lab.queue_runs(intervention.id, seeds=(0,))
    assert lab.execute_run(intervention_run.id).status.value == "COMPLETED"
    finding = lab.draft_finding(hypothesis_id=hypothesis.id, control_ablation_id=arm400.id,
                                intervention_ablation_id=intervention.id)
    checks = {c.name: c for c in finding.integrity}
    assert checks["same_evaluation_target_spec"].status == "pass"
    assert checks["same_instrument"].status == "pass"
    assert checks["same_evaluation_target_spec"].detail.count(EVAL_SPEC.spec_hash()[:19]) == 1
    assert finding.effect is not None
