import hashlib
import json
from pathlib import Path

import pytest

from cvslab.funnel.import_run import compare_coverage, import_funnel_run_evidence
from cvslab.data import _load_canonical
from cvslab.store import LabError

def _distinct_fens() -> list[str]:
    """Four genuinely distinct positions (EPD identity strips move counters, so the
    pool must vary in board state, not clocks)."""
    import chess
    board = chess.Board()
    fens = []
    for move in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(move)
        fens.append(board.fen())
    return fens


FENS = _distinct_fens()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _tier0_record(position_id: str, slug: str, family: str) -> dict:
    return {
        "id": position_id, "schemaVersion": 1, "stage": "tier0", "status": "ok",
        "deterministic_geometry": {
            "pieces": {"stm": 16, "opp": 16}, "attacked": {"stm": 0, "opp": 0},
            "loose": {"stm": 0, "opp": 0}, "pawns": {"doubled": {"stm": 0, "opp": 0}},
            "kingSafety": {"stm": {"inCheck": False, "attackers": 0, "pressuredSquares": 0, "escapeSquares": 1}},
            "squareControl": {"stm": 0, "opp": 0}, "structures": {"stm": [slug], "opp": []},
            "phase": "opening", "materialBucket": "Q1R2m4P2-Q1R2m4P2",
        },
        "bounded_tactical_proof": {
            "opportunities": {"stm": [slug], "opp": []}, "kindCounts": {f"stm:Motifs:{slug}": 1},
            "captures": {"stm": {"count": 0, "seePositive": 0}, "opp": {"count": 0, "seePositive": 0}},
            "hanging": {"stm": 0, "opp": 0}, "hazards": {}, "tacticalItems": 1,
        },
        "taxonomy": {"slugs": [slug], "families": [family], "unmapped": []},
        "uncomputed": [], "factsErrors": 0, "factsRegistryVersion": 23, "cost": {"wallMs": 5.0},
    }


def make_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "synthetic-run"
    run.mkdir(parents=True)
    config = {"funnelConfigVersion": 1, "prioritizerVersion": "priority-v1"}
    (run / "funnel-config.json").write_text(json.dumps(config), encoding="utf-8")
    config_sha = hashlib.sha256((run / "funnel-config.json").read_bytes()).hexdigest()
    (run / "manifest.json").write_text(json.dumps({
        "funnelSchemaVersion": 1, "prioritizerVersion": "priority-v1", "configSha256": config_sha,
        "engine": {"binarySha256": "a" * 64}, "stockfish": {"binarySha256": "b" * 64},
    }), encoding="utf-8")
    (run / "report.json").write_text(json.dumps({
        "funnelSchemaVersion": 1, "tiers": {}, "arms": {}, "oracleArms": None, "coverage": {},
    }), encoding="utf-8")
    _write_jsonl(run / "positions.jsonl", [
        {"id": f"pos{i}", "fen": fen, "gameOutcome": outcome, "source": {"file": "fixture", "line": i}}
        for i, (fen, outcome) in enumerate(zip(FENS, [1.0, None, 0.5, 0.0]))
    ])
    _write_jsonl(run / "tier0.jsonl", [
        _tier0_record("pos0", "fork", "tactics"), _tier0_record("pos1", "fork", "tactics"),
        _tier0_record("pos2", "skewer", "tactics"), _tier0_record("pos3", "fork", "tactics"),
    ])
    _write_jsonl(run / "tier1.jsonl", [
        {"id": position_id, "stage": "tier1", "schemaVersion": 1, "search_derived": {
            "profile": {"nodeBudgets": [2000, 16000], "isolation": "cold"},
            "budgets": [
                {"nodeBudget": 2000, "scoreCpStm": 20, "mate": None, "bestMove": "a2a3",
                 "pv": ["a2a3"], "nodes": 2000, "depth": 5, "trajectory": [[5, "a2a3", 20]],
                 "stabilization": {"status": "stable-at-budget"}, "termination": "nodes",
                 "resultSource": "completed-iteration", "qNodes": 100,
                 "avgCutoffMoveIndex": 1.5, "avgLegalMoves": 20.0},
                {"nodeBudget": 16000, "scoreCpStm": 34, "mate": None, "bestMove": "d2d4",
                 "pv": ["d2d4"], "nodes": 16000, "depth": 8, "trajectory": [[8, "d2d4", 34]],
                 "stabilization": {"status": "stable-at-budget"}, "termination": "nodes",
                 "resultSource": "completed-iteration", "qNodes": 700,
                 "avgCutoffMoveIndex": 1.7, "avgLegalMoves": 20.0},
            ]}}
        for position_id in ("pos0", "pos1")
    ])
    _write_jsonl(run / "tier3.jsonl", [
        {"id": "pos0", "stage": "tier3", "schemaVersion": 1, "search_derived": {
            "profile": {"nodeBudget": 400000, "isolation": "cold"},
            "deep": {"scoreCpStm": 55, "bestMove": "g1f3", "pv": ["g1f3"], "nodes": 400000, "depth": 14},
            "shallowToDeep": {"shallowNodeBudget": 16000, "scoreDeltaCp": 21, "bestMoveChanged": True},
            "targets": {"scoreCpStm": 55, "scoreCpWhite": 55, "expectedScoreStm": 0.549834}}}
    ])
    _write_jsonl(run / "tier4.jsonl", [
        {"id": "pos1", "stage": "tier4", "schemaVersion": 1, "scoreCp": 41, "mate": None,
         "bestMove": "e2e4", "pv": ["e2e4", "e7e5"], "depth": 18, "nodes": 123456}
    ])
    _write_jsonl(run / "triage.jsonl", [
        {"id": "pos0", "stage": "tier2", "schemaVersion": 1, "priority": 0.42, "rank": 1,
         "components": {"scoreInstability": {"value": 0.2, "weight": 3.0, "contribution": 0.6}},
         "reasons": ["scoreInstability"], "selection": {"deep": True}, "trainEligible": True},
        {"id": "pos3", "stage": "tier2", "schemaVersion": 1, "priority": 0.05, "rank": 4,
         "components": {}, "reasons": [], "selection": {"audit": True}, "trainEligible": False},
    ])
    (run / "coverage.json").write_text(json.dumps({
        "prioritizerVersion": "priority-v1",
        "counts": {"__positions__": 4, "fork": 3, "skewer": 1, "family:tactics": 4,
                   "phase:opening": 4, "material:Q1R2m4P2-Q1R2m4P2": 4},
    }), encoding="utf-8")
    return run


def test_import_maps_every_tier_with_provenance(lab, tmp_path):
    counts = import_funnel_run_evidence(lab.store, make_run_dir(tmp_path), name="synthetic")
    assert counts["positions"] == 4 and counts["canonical_records"] == 4
    assert counts["label_rows"] == {"facts": 4, "motif": 4, "strategy": 4, "search_shallow_cp": 4,
                                    "search_deep_cp": 1, "oracle_cp": 1, "outcome": 3, "priority": 2}
    assert "legacy.cvs.tier0.facts" in counts["authorities"] and "external.stockfish" in counts["authorities"]
    assert counts["analyze_sha256"] == "a" * 64

    records = _load_canonical(lab.store, lab.store.get(counts["normalization"]))
    by_id = {row["source_row"]: row for row in
             __import__("cvslab.data", fromlist=["read_jsonl"]).read_jsonl(
                 lab.store.abs(lab.store.get(counts["normalization"]).path))}
    pos0 = next(r for r in records if r["record_id"] == by_id[0]["record_id"])
    families = {(label["family"], label["authority"]) for label in pos0["labels"]}
    # one position retains shallow (both budgets), deep, outcome and priority with distinct provenance
    assert ("search_shallow_cp", "legacy.cvs.search.shallow") in families
    assert ("search_deep_cp", "legacy.cvs.search.deep") in families
    assert ("outcome", "outcome.game_result.v1") in families
    assert ("priority", "triage.priority-v1") in families
    shallow = [label for label in pos0["labels"] if label["family"] == "search_shallow_cp"]
    assert {label["budget"]["nodeBudget"] for label in shallow} == {2000, 16000}

    pos1 = next(r for r in records if r["record_id"] == by_id[1]["record_id"])
    oracle = next(label for label in pos1["labels"] if label["family"] == "oracle_cp")
    assert oracle["authority"] == "external.stockfish" and oracle["value"]["depth"] == 18


def test_tier3_targets_round_trip_exactly(lab, tmp_path):
    run = make_run_dir(tmp_path)
    counts = import_funnel_run_evidence(lab.store, run, name="synthetic")
    source_targets = json.loads((run / "tier3.jsonl").read_text(encoding="utf-8").splitlines()[0])["search_derived"]["targets"]
    records = _load_canonical(lab.store, lab.store.get(counts["normalization"]))
    deep = next(label for record in records for label in record["labels"]
                if label["family"] == "search_deep_cp")
    assert deep["value"]["targets"] == source_targets
    assert deep["budget"]["nodeBudget"] == 400000


def test_coverage_parity_and_no_dataset(lab, tmp_path):
    run = make_run_dir(tmp_path)
    counts = import_funnel_run_evidence(lab.store, run, name="synthetic")
    parity = compare_coverage(lab.store, counts["normalization"], run)
    assert parity["ok"], parity["mismatches"]
    assert parity["keys_compared"] == 5
    assert lab.store.list("D", verify=False) == []  # evidence only, never an implicit dataset


def test_import_is_deterministic(lab, tmp_path):
    run = make_run_dir(tmp_path)
    first = import_funnel_run_evidence(lab.store, run, name="a")
    second = import_funnel_run_evidence(lab.store, run, name="b")
    assert first["canonical_records"] == second["canonical_records"]
    assert first["dedup_dropped"] == second["dedup_dropped"]
    assert first["label_rows"] == second["label_rows"]


def test_duplicate_positions_are_deterministic(lab, tmp_path):
    run = make_run_dir(tmp_path)
    _write_jsonl(run / "positions.jsonl", [
        {"id": "dup-a", "fen": FENS[0], "gameOutcome": None, "source": {}},
        {"id": "dup-b", "fen": FENS[0], "gameOutcome": None, "source": {}},
    ])
    _write_jsonl(run / "tier0.jsonl", [_tier0_record("dup-a", "fork", "tactics")])
    counts = import_funnel_run_evidence(lab.store, run, name="dupes", link_funnel_run=False)
    assert counts["canonical_records"] == 1 and counts["dedup_dropped"] == 1  # keep-first


def test_missing_positions_refused(lab, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(LabError, match="positions.jsonl"):
        import_funnel_run_evidence(lab.store, empty)


def test_same_value_different_budget_labels_coexist(lab, tmp_path):
    """Two searches at different node budgets may legitimately return identical values."""
    run = make_run_dir(tmp_path)
    identical = {"scoreCpStm": 20, "mate": None, "bestMove": "a2a3", "pv": ["a2a3"],
                 "nodes": 2000, "depth": 5, "trajectory": [[5, "a2a3", 20]],
                 "stabilization": {"status": "stable"}, "termination": "nodes",
                 "resultSource": "completed", "qNodes": 10,
                 "avgCutoffMoveIndex": 1.0, "avgLegalMoves": 20.0}
    _write_jsonl(run / "tier1.jsonl", [
        {"id": "pos0", "stage": "tier1", "schemaVersion": 1, "search_derived": {
            "profile": {"nodeBudgets": [2000, 16000], "isolation": "cold"},
            "budgets": [{**identical, "nodeBudget": 2000}, {**identical, "nodeBudget": 16000}]}}
    ])
    counts = import_funnel_run_evidence(lab.store, run, name="identical-budgets", link_funnel_run=False)
    assert counts["label_rows"]["search_shallow_cp"] == 2  # identical values, distinct budgets
    records = _load_canonical(lab.store, lab.store.get(counts["normalization"]))
    shallow = [label for record in records for label in record["labels"]
               if label["family"] == "search_shallow_cp"]
    assert {label["budget"]["nodeBudget"] for label in shallow} == {2000, 16000}


def test_game_grouping_preserved_or_fail_safe(lab, tmp_path):
    """Explicit game ids pass through; a run without game identity yields per-position
    groups (leakage-safe) and says so in the pool snapshot."""
    from cvslab.data import read_jsonl
    run = make_run_dir(tmp_path)

    # (a) explicit game ids are preserved into record groups
    _write_jsonl(run / "positions.jsonl", [
        {"id": "pos0", "fen": FENS[0], "gameOutcome": None, "game": "game-42", "source": {}},
        {"id": "pos1", "fen": FENS[1], "gameOutcome": None, "game": "game-42", "source": {}},
        {"id": "pos2", "fen": FENS[2], "gameOutcome": None, "game": "game-43", "source": {}},
    ])
    counts = import_funnel_run_evidence(lab.store, run, name="explicit-games", link_funnel_run=False)
    records = read_jsonl(lab.store.abs(lab.store.get(counts["normalization"]).path))
    groups = {record["group"] for record in records}
    assert groups == {f"{counts['source']}:game-42", f"{counts['source']}:game-43"}
    snapshot = lab.store.get(counts["source"])
    assert snapshot.generator["game_grouping"] == "explicit"

    # (b) no game identity anywhere: every position is its own group, and the snapshot says so
    run2 = make_run_dir(tmp_path / "second")
    counts2 = import_funnel_run_evidence(lab.store, run2, name="no-games", link_funnel_run=False)
    records2 = read_jsonl(lab.store.abs(lab.store.get(counts2["normalization"]).path))
    groups2 = [record["group"] for record in records2]
    assert len(groups2) == len(set(groups2)) == len(records2)  # no two positions claim one game
    assert "no game identity" in lab.store.get(counts2["source"]).generator["game_grouping"]


def test_falsy_outcome_is_preserved(lab, tmp_path):
    """0.0 is a real loss, not a missing value."""
    run = make_run_dir(tmp_path)
    _write_jsonl(run / "positions.jsonl", [
        {"id": "pos0", "fen": FENS[0], "gameOutcome": 0.0, "source": {}},
        {"id": "pos1", "fen": FENS[1], "gameOutcome": 0.5, "source": {}},
        {"id": "pos2", "fen": FENS[2], "gameOutcome": 1.0, "source": {}},
    ])
    counts = import_funnel_run_evidence(lab.store, run, name="falsy", link_funnel_run=False)
    assert counts["label_rows"]["outcome"] == 3
    records = _load_canonical(lab.store, lab.store.get(counts["normalization"]))
    values = sorted(label["value"] for record in records for label in record["labels"]
                    if label["family"] == "outcome")
    assert values == [0.0, 0.5, 1.0]
