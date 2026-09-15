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
