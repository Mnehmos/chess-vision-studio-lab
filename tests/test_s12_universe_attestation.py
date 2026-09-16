"""The S12 universe attestation is compared on CONTENT, never on store identifiers.

`store.next_id` allocates a fresh N#### on a clean replay, so a replay that reproduces the same
universe would fail an id comparison while being the same universe. These tests encode the
contract `tools/s12/s12_run.py::assert_universe` must keep: identity is the corpus hash, the
recipe, the source byte hashes, the join outcome, the exclusions and the order hash — and any
drift in those fails closed, naming the field. Read-only; no lab store is opened.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "s12"))

import s12_run  # noqa: E402  (the driver is importable; module level has no side effects)
from cvslab.store import LabError  # noqa: E402

ATTESTATION = ROOT / "tools" / "s12" / "s12-universe-attestation.json"


def recorded() -> dict:
    return json.loads(ATTESTATION.read_text(encoding="utf-8"))


def facts_from(rec: dict, **overrides) -> dict:
    """What `compose_universe` returns for the same content, with optional drift."""
    facts = {field: rec[field] for field in s12_run.UNIVERSE_FIELDS}
    facts["normalization_id_at_execution"] = rec["executed_identifiers"]["normalization"]
    facts.update(overrides)
    return facts


def test_compared_fields_carry_no_store_identifiers():
    assert "normalization" not in s12_run.UNIVERSE_FIELDS
    assert [field for field in s12_run.UNIVERSE_FIELDS
            if field.endswith("_id") or "executed" in field] == []
    assert "normalization_output_hash" in s12_run.UNIVERSE_FIELDS


def test_attestation_is_complete_and_records_the_executed_ids():
    rec = recorded()
    assert [field for field in s12_run.UNIVERSE_FIELDS if field not in rec] == []
    assert s12_run.executed_normalization_id(rec) == "N0014"
    assert rec["executed_identifiers"]["sources"] == ["S0011", "S0012"]
    assert "not part of the fail-closed comparison" in rec["executed_identifiers"]["note"]


def test_a_fresh_normalization_id_passes():
    """The clean-replay case: same content, a new N####."""
    rec = recorded()
    s12_run.assert_universe(facts_from(rec, normalization_id_at_execution="N0099"), rec, complete=True)


@pytest.mark.parametrize("field, drift", [
    ("normalization_output_hash", "sha256:" + "0" * 64),
    ("normalization_recipe_hash", "sha256:" + "1" * 64),
    ("normalization_source_hashes", ["sha256:" + "2" * 64]),
    ("normalization_records", 183481),
    ("records", 183481),
    ("removed", 298),
    ("old_kept", 79448),
    ("excluded_ids", []),
    ("excluded_positions", [0]),
    ("excluded_inside_old_4k_prefix", 8),
    ("excluded_inside_old_2k_prefix", 17),
    ("appended", 103735),
    ("total", 183182),
    ("universe_hash", "sha256:" + "3" * 64),
    ("order_hash", "sha256:" + "4" * 64),
])
def test_content_drift_fails_closed(field, drift):
    rec = recorded()
    with pytest.raises(LabError) as error:
        s12_run.assert_universe(facts_from(rec, **{field: drift}), rec, complete=True)
    assert field in str(error.value)


def test_the_state_subset_is_compared_on_what_it_records():
    """s12-state.json carries a subset of the same facts plus keys that are not compared."""
    rec = recorded()
    state_like = {"records": 183482, "removed": 299, "old": 79483, "old_kept": 79447,
                  "appended": 103736, "total": 183183, "universe_hash": rec["universe_hash"],
                  "order_hash": rec["order_hash"], "normalization": "N0014",
                  "order_kind": "the state's own wording is not a compared fact"}
    s12_run.assert_universe(facts_from(rec), state_like)
    with pytest.raises(LabError):
        s12_run.assert_universe(facts_from(rec, removed=298), state_like)


def test_missing_field_in_the_attestation_is_refused():
    rec = recorded()
    rec.pop("order_hash")
    with pytest.raises(LabError) as error:
        s12_run.assert_universe(facts_from(recorded()), rec, complete=True)
    assert "missing" in str(error.value)
