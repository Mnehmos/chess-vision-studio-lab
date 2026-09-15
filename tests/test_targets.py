import json

import pytest

from cvslab.store import LabError
from cvslab.targets import TargetSpec, extract_target, matching_labels

SPEC = TargetSpec(family="search_deep_cp", authority="legacy.cvs.search.deep",
                  producer="a" * 64, budget={"nodeBudget": 400000},
                  value_path=("targets", "scoreCpStm"))


def label(**overrides) -> dict:
    base = {"family": "search_deep_cp", "authority": "legacy.cvs.search.deep", "producer": "a" * 64,
            "pov": "stm", "budget": {"nodeBudget": 400000, "nodes": 400001},
            "value": {"targets": {"scoreCpStm": 55, "scoreCpWhite": 55, "expectedScoreStm": 0.549},
                      "deep": {}, "shallowToDeep": {}}}
    base.update(overrides)
    return base


def record(*labels: dict) -> dict:
    return {"record_id": "pos_x", "labels": list(labels)}


def test_matches_complete_spec_only():
    assert matching_labels(record(label()), SPEC)  # exact match
    assert not matching_labels(record(label(producer="b" * 64)), SPEC)      # wrong producer
    assert not matching_labels(record(label(budget={"nodeBudget": 16000})), SPEC)  # wrong budget
    assert not matching_labels(record(label(value={"targets": {}})), SPEC)  # missing value path
    assert not matching_labels(record(label(family="search_shallow_cp")), SPEC)   # wrong family


def test_extract_is_strict_zero_and_multiple_fail():
    assert extract_target(record(label()), SPEC) == 55.0
    with pytest.raises(LabError, match="no label satisfies"):
        extract_target(record(label(producer="b" * 64)), SPEC)
    with pytest.raises(LabError, match="exactly one is required"):
        extract_target(record(label(), label()), SPEC)  # duplicate matching observations


def test_spec_identity_is_stable_and_sensitive():
    assert SPEC.spec_hash() == TargetSpec(**{**SPEC.__dict__}).spec_hash()
    other = TargetSpec(family=SPEC.family, authority=SPEC.authority, producer=SPEC.producer,
                       budget={"nodeBudget": 16000}, value_path=SPEC.value_path)
    assert other.spec_hash() != SPEC.spec_hash()


def test_freeze_validates_the_spec_before_materialising_runs(lab):
    """Malformed supervision must be caught at freeze time, not at run time."""
    source = lab.create_fixture_source(n_games=6, seed=3)
    normalization = lab.normalize([source.id], name="n")
    records = __import__("cvslab.data", fromlist=["_load_canonical"])._load_canonical(
        lab.store, lab.store.get(normalization.id))
    rid = records[0]["record_id"]
    # a label with the right FAMILY but the wrong producer must not satisfy the spec
    lab.register_labels(normalization.id, family="search_deep_cp", producer="wrong-producer",
                        authority="legacy.cvs.search.deep", pov="stm",
                        rows=[{"record_id": rid, "value": {"targets": {"scoreCpStm": 10}},
                               "budget": {"nodeBudget": 400000}}])
    with pytest.raises(LabError, match="no label satisfies"):
        lab.freeze_dataset(normalization.id, name="bad-spec", record_ids=[rid],
                           required_labels=["search_deep_cp"], target_spec=SPEC)
    lab.register_labels(normalization.id, family="search_deep_cp", producer="a" * 64,
                        authority="legacy.cvs.search.deep", pov="stm",
                        rows=[{"record_id": rid, "value": {"targets": {"scoreCpStm": 55}},
                               "budget": {"nodeBudget": 400000}}])
    dataset = lab.freeze_dataset(normalization.id, name="good-spec", record_ids=[rid],
                                 required_labels=["search_deep_cp"], target_spec=SPEC)
    assert dataset.counts["records"] == 1


def test_protocol_pins_the_spec_in_its_own_hash(lab):
    from cvslab.service import protocol_target_spec
    source = lab.create_fixture_source(n_games=6, seed=4)
    normalization = lab.normalize([source.id], name="n")
    dataset = lab.freeze_dataset(normalization.id, name="d")
    protocol = lab.create_eval_protocol(name="e", dataset_id=dataset.id, target_spec=SPEC)
    assert protocol_target_spec(protocol).spec_hash() == SPEC.spec_hash()
    without = lab.create_eval_protocol(name="e2", dataset_id=dataset.id)
    assert protocol_target_spec(without) is None
    assert protocol.protocol_hash != without.protocol_hash  # the spec is part of the identity


def test_pov_is_part_of_the_spec():
    assert not matching_labels(record(label(pov="white")), SPEC)  # white label, stm spec
    white = TargetSpec(family=SPEC.family, authority=SPEC.authority, producer=SPEC.producer,
                       budget=SPEC.budget, value_path=SPEC.value_path, pov="white")
    assert matching_labels(record(label(pov="white")), white)


def test_unsupported_types_and_bounds_fail_closed():
    with pytest.raises(LabError, match="unsupported target_type"):
        TargetSpec(family="search_deep_cp", authority="a", producer="p", target_type="probability")
    with pytest.raises(LabError, match="unsupported pov"):
        TargetSpec(family="search_deep_cp", authority="a", producer="p", pov="sideways")
    with pytest.raises(LabError, match="K must be positive"):
        TargetSpec(family="search_deep_cp", authority="a", producer="p", k=0.0)
    with pytest.raises(LabError, match="LAMBDA"):
        TargetSpec(family="search_deep_cp", authority="a", producer="p", lam=1.5)


def test_freeze_validates_requested_population_before_family_filtering(lab):
    """One requested id without any matching label must refuse the freeze, not shrink it."""
    source = lab.create_fixture_source(n_games=6, seed=9)
    normalization = lab.normalize([source.id], name="n")
    records = __import__("cvslab.data", fromlist=["_load_canonical"])._load_canonical(
        lab.store, lab.store.get(normalization.id))
    good_id, bad_id = records[0]["record_id"], records[1]["record_id"]
    lab.register_labels(normalization.id, family="search_deep_cp", producer="a" * 64,
                        authority="legacy.cvs.search.deep", pov="stm",
                        rows=[{"record_id": good_id, "value": {"targets": {"scoreCpStm": 42}},
                               "budget": {"nodeBudget": 400000}}])  # bad_id gets NO label at all
    with pytest.raises(LabError, match="no label satisfies"):
        lab.freeze_dataset(normalization.id, name="shrink-attempt", record_ids=[good_id, bad_id],
                           required_labels=["search_deep_cp"], target_spec=SPEC)


def test_recipe_pins_training_spec_and_klambda_must_match(lab):
    from cvslab.service import recipe_target_spec
    recipe = lab.create_training_recipe(name="spec-recipe",
                                        params={"K": 256.0, "LAMBDA": 1.0, "EPOCHS": 2},
                                        target_spec=SPEC)
    assert recipe_target_spec(recipe).spec_hash() == SPEC.spec_hash()
    with pytest.raises(LabError, match="do not match the TargetSpec"):
        lab.create_training_recipe(name="bad-recipe", params={"K": 128.0, "LAMBDA": 1.0}, target_spec=SPEC)
    source = lab.create_fixture_source(n_games=4, seed=2)
    normalization = lab.normalize([source.id], name="n")
    dataset = lab.freeze_dataset(normalization.id, name="d")
    with pytest.raises(LabError, match="do not match the TargetSpec"):
        lab.create_eval_protocol(name="bad-eval", dataset_id=dataset.id, k=64.0, target_spec=SPEC)


def test_split_dataset_run_trains_on_train_and_evaluates_on_common_eval(lab):
    """End-to-end: D_TRAIN for train/val, a distinct D_EVAL for evaluation."""
    import json as _json
    from cvslab.service import protocol_target_spec, recipe_target_spec

    source = lab.create_fixture_source(n_games=14, seed=5)
    normalization = lab.normalize([source.id], name="n")
    records = __import__("cvslab.data", fromlist=["_load_canonical"])._load_canonical(
        lab.store, lab.store.get(normalization.id))
    evaluated = sorted(record["record_id"] for record in records)
    lab.register_labels(normalization.id, family="search_deep_cp", producer="a" * 64,
                        authority="legacy.cvs.search.deep", pov="stm",
                        rows=[{"record_id": record_id,
                               "value": {"targets": {"scoreCpStm": (index % 4) * 200 - 300}},
                               "budget": {"nodeBudget": 400000}} for index, record_id in enumerate(evaluated)])

    train_dataset = lab.freeze_dataset(normalization.id, name="d-train", target_spec=SPEC)
    eval_ids = evaluated[:4]
    eval_dataset = lab.freeze_dataset(normalization.id, name="d-eval", record_ids=eval_ids,
                                      required_labels=["search_deep_cp"], target_spec=SPEC,
                                      fractions=(0.0, 0.0, 1.0))
    assert eval_dataset.splits[2].count == len(eval_ids)  # everything physically in test

    recipe = lab.create_training_recipe(name="t-spec", params={"K": 256.0, "LAMBDA": 1.0, "EPOCHS": 2},
                                        target_spec=SPEC)
    protocol = lab.create_eval_protocol(name="e-common", dataset_id=eval_dataset.id, split="test",
                                        k=256.0, lam=1.0, target_spec=SPEC)
    assert recipe_target_spec(recipe).spec_hash() == protocol_target_spec(protocol).spec_hash()
    baseline = lab.register_baseline(name="COMMON", dataset_id=train_dataset.id,
                                     training_recipe_id=recipe.id, eval_protocol_id=protocol.id,
                                     model_config={"INPUT": "RAW", "H": 4})
    [run] = lab.queue_runs(baseline.id, seeds=(0,))
    assert run.dataset_id == train_dataset.id and run.eval_dataset_id == eval_dataset.id
    assert run.eval_dataset_manifest_hash == eval_dataset.manifest_hash
    finished = lab.execute_run(run.id)
    assert finished.status.value == "COMPLETED", [c.detail for c in finished.integrity if c.status == "fail"]
    payload = _json.loads(lab.store.abs(f"artifacts/{run.id}/eval.json").read_text(encoding="utf-8"))
    assert payload["record_ids"] == eval_ids  # evaluated on exactly D_EVAL's test records
    assert next(m for m in finished.metrics if m.name == "test_loss").n == len(eval_ids)
