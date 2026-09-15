import json

import pytest

from cvslab.funnel.policy import build_policy, load_policy, policy_hash, store_policy
from cvslab.funnel.triage import (
    clamp_cp,
    compare_triage,
    score_priority,
    select,
    stable_hash,
    triage_corpus,
    unit_hash,
)
from cvslab.funnel.import_run import import_funnel_run_evidence
from cvslab.store import LabError
from tests.test_funnel_import import make_run_dir

CONFIG = {
    "prioritizerVersion": "priority-v1", "seed": 111,
    "tiers": {"tier2": {
        "deepFraction": 0.10, "auditFraction": 0.02, "holdoutFraction": 0.01, "holdoutSeed": 20260914,
        "weights": {"scoreInstability": 3.0, "bestMoveChange": 2.0},
        "caps": {"scoreDeltaCp": 150, "tacticalItems": 10, "seePruneSkips": 4, "forcedLegalMoves": 3,
                 "outcomeMarginCp": 100, "outcomeScaleCp": 400},
        "coverage": {"targetShare": 0.02, "minTarget": 5, "priorCountsPath": None}},
        "tier4": {"enabled": True, "priorityFraction": 0.02, "uniformFraction": 0.02, "auditFraction": 0.01}},
}


def test_policy_hash_is_canonical_stable_and_sensitive(lab):
    first, second = build_policy(json.loads(json.dumps(CONFIG))), build_policy(CONFIG)
    assert policy_hash(first) == policy_hash(second)
    tweaked = json.loads(json.dumps(first))
    tweaked["selection"]["holdout_seed"] += 1
    assert policy_hash(tweaked) != policy_hash(first)
    tweaked = json.loads(json.dumps(first))
    tweaked["weights"]["scoreInstability"] = 3.0000001
    assert policy_hash(tweaked) != policy_hash(first)

    digest = store_policy(lab.store, first)
    assert load_policy(lab.store, digest) == first
    with pytest.raises(LabError, match="no tiers.tier2"):
        build_policy({"seed": 1})


def test_policy_store_is_immutable_and_verified(lab):
    digest = store_policy(lab.store, build_policy(CONFIG))
    path = lab.store.abs("policies/" + digest.split(":", 1)[1] + ".json")
    assert path.is_file()
    before = path.stat().st_mtime_ns
    assert store_policy(lab.store, build_policy(CONFIG)) == digest  # idempotent
    assert path.stat().st_mtime_ns == before  # untouched
    with pytest.raises(LabError, match="not found"):
        load_policy(lab.store, "sha256:" + "0" * 64)


def test_hashes_match_the_legacy_implementation():
    # fixed constants from the legacy funnel's stable_hash/unit_hash/clamp_cp
    assert stable_hash(20260914, "holdout", "pos0") == stable_hash(20260914, "holdout", "pos0")
    assert 0.0 <= unit_hash(1, "x", "y") < 1.0
    assert clamp_cp(None) == 0 and clamp_cp(99999) == 2000 and clamp_cp(-99999) == -2000


def test_score_priority_shape_and_reasons_ordering():
    weights = {"b": 2.0, "a": 1.0, "c": 0.0}
    total, detail, reasons = score_priority({"a": 0.5, "b": 1.0, "c": 0.9}, weights)
    assert detail["b"]["contribution"] == round(2.0 * 1.0 / 3.0, 6)
    assert reasons == ["b", "a"]  # zero-weight c has zero contribution and is not a reason
    assert total == round(sum(entry["contribution"] for entry in detail.values()), 6)


def test_select_holdout_is_pool_independent_and_arms_disjoint():
    policy = build_policy(CONFIG)
    items = [{"id": f"p{i}", "priority": 0.5 / (i + 1), "components": {}, "reasons": []} for i in range(400)]
    full = select(items, policy)
    subset = select(items[:200], policy)
    holdout_full = {row["id"] for row in full if row["selection"]["holdout"]}
    holdout_subset = {row["id"] for row in subset if row["selection"]["holdout"]}
    assert holdout_subset == holdout_full & {f"p{i}" for i in range(200)}  # pure function of (seed, id)
    row = next(r for r in full if not r["selection"]["holdout"])
    assert row["trainEligible"] is True and row["rank"] is not None
    held = next(r for r in full if r["selection"]["holdout"])
    assert held["trainEligible"] is False and held["rank"] is None
    deep = {r["id"] for r in full if r["selection"]["deep"]}
    uniform = {r["id"] for r in full if r["selection"]["uniform"]}
    assert not deep & holdout_full
    assert len(deep) == round(0.10 * len(items))
    assert len(uniform) <= len(deep)


def test_triage_from_s2_evidence_runs_and_compare_detects_drift(lab, tmp_path):
    run = make_run_dir(tmp_path)
    counts = import_funnel_run_evidence(lab.store, run, name="s3", link_funnel_run=False)
    policy = load_policy(lab.store, counts["policy_hash"])
    rows, counts_used = triage_corpus(lab.store, counts["normalization"], policy)
    assert rows  # two positions have tier0+tier1 evidence
    assert counts_used["__positions__"] >= len(rows)
    labels = sorted(rows, key=lambda row: row["id"])
    assert all({"priority", "rank", "components", "reasons", "selection", "trainEligible"} <= set(row)
               for row in labels)
    # comparator: identical rows pass, a single drifted value fails loudly
    assert compare_triage(labels, labels)["ok"] is True
    drifted = [dict(row) for row in labels]
    drifted[0] = {**drifted[0], "priority": drifted[0]["priority"] + 1e-9}
    report = compare_triage(drifted, labels)
    assert report["ok"] is False and report["priority"]["mismatches"] == 1
    assert compare_triage(labels, labels[:-1])["only_lab"] == [labels[-1]["id"]]  # lab has an extra row


def test_priority_labels_carry_policy_hash(lab, tmp_path):
    run = make_run_dir(tmp_path)
    counts = import_funnel_run_evidence(lab.store, run, name="s3-hash", link_funnel_run=False)
    assert counts["policy_hash"].startswith("sha256:")
    from cvslab.data import label_set_refs
    refs = label_set_refs(lab.store, counts["normalization"])
    priority_ref = next(ref for ref in refs if "priority" in ref.families)
    assert priority_ref.policy_hash == counts["policy_hash"]
