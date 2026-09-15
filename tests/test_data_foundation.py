import json

import pytest

from cvslab import data
from cvslab.store import LabError


@pytest.fixture()
def corpus(lab, tmp_path):
    """One normalization over two heterogeneous sources (fixture generator + PGN file)."""
    fixture = lab.create_fixture_source(n_games=20, seed=7)
    pgn = tmp_path / "smoke.pgn"
    pgn.write_text("\n".join([
        '[Event "e1"] [Site "local"] [Result "1-0"]', '',
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O", '',
        '[Event "e2"] [Site "local"] [Result "0-1"]', '',
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6", '',
    ]), encoding="utf-8")
    imported = lab.import_pgn_source(pgn, name="pgn-smoke")
    normalization = lab.normalize([fixture.id, imported.id], name="canon-v1")
    lab.fixture_id, lab.pgn_id, lab.norm_id = fixture.id, imported.id, normalization.id
    return normalization


def test_heterogeneous_sources_standardize_into_one_schema(lab, corpus):
    fixture_rows = lab.store.list("S")
    assert len(fixture_rows) == 2
    importers = {snapshot.importer for snapshot in fixture_rows}
    assert importers == {"cvslab.fixture.random_play", "cvslab.import.pgn"}
    catalog = lab.catalog(lab.norm_id, limit=500)
    assert catalog.record_count == corpus.record_count
    assert set(catalog.facets["source_id"]) == {lab.fixture_id, lab.pgn_id}
    for record in catalog.page:
        assert set(record.model_dump()) >= {"record_id", "fen", "epd", "stm", "phase", "material", "labels"}


def test_same_position_identified_across_sources(lab, tmp_path):
    fixture = lab.create_fixture_source(n_games=6, seed=3)
    rows = data.read_jsonl(lab.store.abs(fixture.path))
    echo = tmp_path / "echo.jsonl"
    echo.write_text("\n".join(json.dumps(row) for row in rows[:4]), encoding="utf-8")
    other = lab.import_jsonl_source(echo, name="echo")
    normalization = lab.normalize([fixture.id, other.id], name="n", dedup="none")
    # with dedup off, the echoed rows are separate records sharing one content-derived identity:
    # 4 identity collisions over 232 raw rows -> 228 unique positions
    assert normalization.record_count == 228 + 4
    from cvslab.data import _load_canonical
    records = _load_canonical(lab.store, normalization)
    echo_ids = {r["record_id"] for r in records if r["source_id"] == other.id}
    fixture_ids = {r["record_id"] for r in records if r["source_id"] == fixture.id}
    assert echo_ids <= fixture_ids


def test_labels_from_different_authorities_coexist(lab, corpus):
    page = lab.catalog(lab.norm_id, limit=3).page
    rid = page[0].record_id
    first = lab.register_labels(lab.norm_id, family="eval_cp", producer="engine-a", authority="engine.a.v1",
                                rows=[{"record_id": rid, "value": 42, "budget": {"depth": 18}}])
    second = lab.register_labels(lab.norm_id, family="eval_cp", producer="engine-b", authority="engine.b.v1",
                                 rows=[{"record_id": rid, "value": 37}])
    assert first.path != second.path
    dataset = lab.freeze_dataset(lab.norm_id, name="d")
    split_rows = data.read_jsonl(lab.store.abs(dataset.splits[0].path))
    joined = [row for row in split_rows if row["record_id"] == rid]
    if joined:  # the record landed in this split with every authority's label attached
        authorities = {label["authority"] for label in joined[0]["labels"]}
        assert {"engine.a.v1", "engine.b.v1"} <= authorities
    assert dataset.label_provenance["engine.a.v1"] == 1
    assert dataset.label_provenance["engine.b.v1"] == 1
    assert len(dataset.label_sets) == 2


def test_label_registry_fail_closed(lab, corpus):
    rid = lab.catalog(lab.norm_id, limit=1).page[0].record_id
    with pytest.raises(LabError, match="unknown label family"):
        lab.register_labels(lab.norm_id, family="vibes", producer="p", authority="a",
                            rows=[{"record_id": rid, "value": 1}])
    with pytest.raises(LabError, match="unknown record"):
        lab.register_labels(lab.norm_id, family="eval_cp", producer="p", authority="a",
                            rows=[{"record_id": "pos_does_not_exist", "value": 1}])
    lab.register_labels(lab.norm_id, family="eval_cp", producer="p", authority="a",
                        rows=[{"record_id": rid, "value": 1}])
    # identical rows in one set are accidents and are refused
    with pytest.raises(LabError, match="duplicate label"):
        lab.register_labels(lab.norm_id, family="eval_cp", producer="p", authority="a",
                            rows=[{"record_id": rid, "value": 1}, {"record_id": rid, "value": 1}])
    # distinct values on one record are legitimate (e.g. one row per node budget) and coexist
    ref = lab.register_labels(lab.norm_id, family="eval_cp", producer="p", authority="a",
                              rows=[{"record_id": rid, "value": 1}, {"record_id": rid, "value": 2}])
    assert ref.rows == 2


def test_catalog_filters_and_facets(lab, corpus):
    lab.freeze_dataset(lab.norm_id, name="d", selection={"phase": "opening"})
    catalog = lab.catalog(lab.norm_id, filters={"phase": "opening", "dataset": "D0001"}, limit=500)
    assert catalog.total_matching > 0
    assert all(record.phase == "opening" for record in catalog.page)
    assert "D0001" in catalog.facets["dataset"]
    tier_catalog = lab.catalog(lab.norm_id, filters={"tier": "deterministic"})
    assert tier_catalog.total_matching == catalog.record_count  # source eval labels are deterministic tier
    with pytest.raises(LabError, match="unknown catalog filter"):
        lab.catalog(lab.norm_id, filters={"vibes": "interference"})


def test_stack_preview_and_freeze_reproducible(lab, corpus):
    arms = [
        {"name": "open", "filter": {"phase": "opening"}, "policy": "all"},
        {"name": "mid-frac", "filter": {"phase": "middlegame"}, "policy": "fraction", "fraction": 0.5, "seed": 2},
        {"name": "end-balanced", "filter": {"phase": "endgame"}, "policy": "balance",
         "balance_bucket": "stm", "balance_cap": 2, "seed": 3},
    ]
    preview = lab.stack_preview(lab.norm_id, arms)
    by_name = {arm.name: arm for arm in preview.arms}
    assert by_name["open"].effective == by_name["open"].available
    assert 0 < by_name["mid-frac"].effective < by_name["mid-frac"].available
    assert by_name["end-balanced"].effective <= 4  # 2 per side-to-move at most
    assert preview.effective_total == preview.unique_records

    frozen = lab.freeze_dataset(lab.norm_id, name="stacked", arms=arms)
    assert [arm.name for arm in frozen.stack] == ["open", "mid-frac", "end-balanced"]
    assert frozen.coverage["stack"] == {arm.name: arm.effective for arm in preview.arms}
    assert frozen.counts["records"] == preview.effective_total

    refrozen = lab.freeze_dataset(lab.norm_id, name="stacked", arms=arms)  # same spec, new identity
    assert refrozen.id != frozen.id
    # identical spec + seed -> byte-identical split content, at each identity's own location
    for left, right in zip(frozen.splits, refrozen.splits):
        assert left.file_hash == right.file_hash and left.record_ids_hash == right.record_ids_hash
    assert refrozen.stack == frozen.stack

    with pytest.raises(LabError, match="requested"):
        lab.freeze_dataset(lab.norm_id, name="too-big",
                           arms=[{"name": "x", "filter": {}, "policy": "fixed_rows", "rows": 10_000}])
    with pytest.raises(LabError, match="unknown field"):
        lab.stack_preview(lab.norm_id, [{"name": "x", "filter": {"vibes": "x"}, "policy": "all"}])


def test_migration_produces_real_diff_and_descendant_dataset(lab, corpus):
    original = lab.freeze_dataset(lab.norm_id, name="tiny")
    migrated = lab.migrate(lab.norm_id, name="canon-v2",
                           settings_overrides={"phase_middlegame_min_material": 4000})
    diff = lab.migration_diff(lab.norm_id, migrated.id)
    assert diff.records_from == diff.records_to == corpus.record_count
    assert diff.changed > 0
    assert diff.coverage_from["phase"] != diff.coverage_to["phase"]
    assert any("phase_middlegame_min_material" in note for note in diff.notes)
    assert diff.affected_dataset_ids == [original.id]

    carried = lab.copy_label_sets(lab.norm_id, migrated.id)
    assert carried["labels_dropped"] == 0  # every record identity survived this migration

    rebuilt = lab.rebuild_dataset(original.id, migrated.id)
    assert rebuilt.parent_id == original.id
    assert rebuilt.normalization_id == migrated.id
    assert rebuilt.split_policy == original.split_policy

    # historical evidence unchanged
    assert lab.store.get(original.id).manifest_hash == original.manifest_hash
    with pytest.raises(LabError, match="already rebuilt"):
        lab.rebuild_dataset(original.id, migrated.id)


def test_copy_label_sets_drops_labels_of_removed_records(lab):
    """A migration that removes records must not silently carry their labels over."""
    fixture = lab.create_fixture_source(n_games=10, seed=11)
    first = lab.normalize([fixture.id], name="v1")
    rid = lab.catalog(first.id, limit=1).page[0].record_id
    lab.register_labels(first.id, family="oracle_cp", producer="sf", authority="oracle.sf",
                        rows=[{"record_id": rid, "value": 5}])
    # v2 normalizes only a *different* fixture: the labelled record no longer exists
    other = lab.create_fixture_source(n_games=10, seed=99)
    second = lab.normalize([other.id], name="v2")
    carried = lab.copy_label_sets(first.id, second.id)
    assert carried == {"sets_copied": 0, "labels_kept": 0, "labels_dropped": 1}


def test_training_consumes_only_frozen_manifests_with_joined_labels(lab, corpus):
    """A frozen stack dataset's split rows carry the labels training will actually see."""
    rid = lab.catalog(lab.norm_id, limit=1).page[0].record_id
    lab.register_labels(lab.norm_id, family="search_deep_cp", producer="cvs-deep", authority="search.cvs.deep",
                        rows=[{"record_id": rid, "value": 17, "budget": {"nodes": 200_000}}])
    dataset = lab.freeze_dataset(lab.norm_id, name="with-deep", required_labels=["eval_cp", "search_deep_cp"])
    assert dataset.counts["records"] == 1  # only the record carrying the deep label survives the requirement
    assert dataset.counts["missing_required_labels"] == corpus.record_count - 1
    lab.store.get(dataset.id)  # manifest verifies
    manifest_rows = data.read_jsonl(lab.store.abs(dataset.splits[0].path))
    deep = [row for row in manifest_rows if row["record_id"] == rid]
    if deep:
        assert any(label["family"] == "search_deep_cp" for label in deep[0]["labels"])
