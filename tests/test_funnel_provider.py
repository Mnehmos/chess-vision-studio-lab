import sys
import textwrap

import pytest

from cvslab.funnel.leakguard import assert_static_input_safe
from cvslab.funnel.providers import Serve, board_geometry, compact_search, parse_sf_output, sha256_file
from cvslab.funnel.taxonomy import load_taxonomy, resolve_slug, summarize_facts

ENGINE_ROOT = r"F:\Github\chess-vision-studio-rust-engine"
ANALYZE = ENGINE_ROOT + r"\target\release\analyze.exe"
TAXONOMY = ENGINE_ROOT + r"\benchmarks\data\motif-taxonomy.json"

import os


def fake_tier0_position() -> tuple[dict, str]:
    return board_geometry("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")


def test_board_geometry_start_position():
    geometry, first_move = fake_tier0_position()
    assert geometry["sideToMove"] == "white"
    assert geometry["legalMoves"] == 20
    assert geometry["legalCaptures"] == 0
    assert geometry["legalChecks"] == 0
    assert geometry["inCheck"] is False
    assert geometry["nonPawnMaterial"] == 24
    assert geometry["phase"] == "opening"
    assert geometry["materialBucket"] == "Q1R2m4P2-Q1R2m4P2"
    assert first_move == "a2a3"  # lexicographically first legal move


def test_compact_search_maps_engine_response():
    # shape taken from a real engine response: the chosen move is `uci`, telemetry nested
    response = {"scoreCp": -33, "mate": None, "uci": "e2e4", "nodes": 16000, "depth": 9,
                "pv": ["e2e4", "e7e5", "g1f3", "x", "y"], "qNodes": 812,
                "stabilization": {"status": "stable-at-budget", "bestMoveChanges": 1,
                                  "seePruneSkips": 0, "reasons": []},
                "telemetry": {"avgCutoffMoveIndex": 2.5, "avgLegalMoves": 31.0},
                "iterations": [{"depth": 1, "uci": "d2d4", "scoreCp": 20},
                               {"depth": 2, "uci": "e2e4", "scoreCp": -33}]}
    compact = compact_search(response, pv_plies=3)
    assert compact["scoreCpStm"] == -33
    assert compact["bestMove"] == "e2e4"
    assert compact["pv"] == ["e2e4", "e7e5", "g1f3"]
    assert compact["nodes"] == 16000 and compact["qNodes"] == 812
    assert compact["stabilization"]["status"] == "stable-at-budget"
    assert compact["trajectory"] == [[1, "d2d4", 20], [2, "e2e4", -33]]
    assert compact["avgLegalMoves"] == 31.0


def test_parse_sf_output_extracts_depth_and_bestmove():
    text = "\n".join([
        "info depth 10 seldepth 12 nodes 12345 time 40 score cp 15 pv e2e4 e7e5",
        "info depth 12 seldepth 14 nodes 30000 time 90 score cp 22 pv d2d4",
        "bestmove d2d4 ponder d7d5",
    ])
    parsed = parse_sf_output(text)
    assert parsed[-1]["bestmove"] == "d2d4"
    assert parsed[-1]["depth"] == "12"
    assert parsed[-1]["scoreValue"] == "22"


def test_leak_guard_rejects_nested_search_keys():
    nested = {"id": "x", "stage": "tier0", "status": "ok",
              "bounded_tactical_proof": {"opportunities": {"stm": [{"scoreCp": 12}]}}}
    with pytest.raises(ValueError, match="leaks"):
        assert_static_input_safe(nested)
    for key in ("nodes", "pv", "bestMove", "gameOutcome", "oracle", "priority", "depth"):
        payload = {"id": "x", "stage": "tier0", "status": "ok",
                   "deterministic_geometry": {"structures": [{"inner": {key: 1}}]}}
        with pytest.raises(ValueError, match="leaks"):
            assert_static_input_safe(payload)
    # declared timing metadata subtree stays allowed
    assert_static_input_safe({"id": "x", "stage": "tier0", "status": "ok", "cost": {"wallMs": 6.1}})


def test_leak_guard_accepts_clean_and_rejects_leaks():
    clean = {"id": "x", "stage": "tier0", "status": "ok",
             "deterministic_geometry": {"pieces": {}},
             "bounded_tactical_proof": {"opportunities": {}},
             "taxonomy": {"slugs": []}, "uncomputed": [], "factsErrors": 0,
             "factsRegistryVersion": 23, "cost": {"wallMs": 6.1}}
    assert_static_input_safe(clean)

    with pytest.raises(ValueError, match="non-static fields"):
        assert_static_input_safe({**clean, "search_derived": {"scoreCpStm": 3}})
    with pytest.raises(ValueError, match="leaks"):
        assert_static_input_safe({**clean, "bounded_tactical_proof": {"gameOutcome": 1.0}})


def test_serve_round_trip_with_stub_process():
    stub = textwrap.dedent("""
        import json, sys
        for line in sys.stdin:
            req = json.loads(line)
            if req["cmd"] == "facts":
                print(json.dumps({"ok": True, "cmd": "facts"}), flush=True)
            else:
                print(json.dumps({"scoreCp": 7, "bestMove": "e2e4", "pv": [], "nodes": req["nodeBudget"]}), flush=True)
    """)
    serve = Serve([sys.executable, "-c", stub])
    try:
        assert serve.request({"cmd": "facts"})["ok"] is True
        response = serve.request({"cmd": "go", "fen": "x", "nodeBudget": 2000})
        assert response["nodes"] == 2000 and response["scoreCp"] == 7
    finally:
        serve.close()


def test_summarize_facts_sides_unmapped_and_uncomputed():
    bundle = {
        "before": {
            "sideToMove": "white",
            "availableMotifs": {"status": "computed", "items": [{"kind": "fork"}]},
            "opponentAvailableMotifs": {"status": "computed", "items": [{"kind": "unknown_thing"}]},
            "availableCaptures": {"status": "computed", "items": [{"seeCp": 120}, {"seeCp": -30}]},
            "pawnStructure": {"doubled": [{"side": "white"}, {"side": "black"}],
                              "isolated": {"status": "not-computed"}, "islands": [{"side": "white"}]},
            "kingSafety": {"status": "computed", "items": [
                {"side": "white", "inCheck": False, "attackers": [], "pressuredSquares": [1, 2],
                 "legalEscapeSquares": {"status": "computed", "items": [1, 2, 3]}}]},
            "squareFacts": {"status": "computed", "items": [
                {"controlledByWhite": True, "controlledByBlack": False}]},
            "hazards": {"status": "computed", "items": [{"kind": "mate_threat", "side": "black"}]},
            "pieces": [{"side": "black", "pieceType": "rook", "attacked": True, "loose": True},
                       {"side": "white", "pieceType": "bishop", "attacked": True, "loose": False}],
        },
        "errors": [],
        "provenance": {"factsRegistryVersion": 23},
    }
    families = {"fork": "tactics", "hanging-piece": "tactics", "mate-threat": "mate"}
    summary = summarize_facts(bundle, families)
    opportunities = summary["bounded_tactical_proof"]["opportunities"]
    assert "fork" in opportunities["stm"]                       # white can execute
    assert "hanging-piece" in opportunities["stm"]              # black rook hangs to white
    assert "mate-threat" in opportunities["stm"]               # black threatened; white benefits
    assert any(slug.startswith("unmapped:Motifs") for slug in opportunities["opp"])
    assert summary["deterministic_geometry"]["pawns"]["doubled"] == {"stm": 1, "opp": 1}
    assert "pawnStructure.isolated" in summary["uncomputed"]
    assert summary["uncomputed"] == ["pawnStructure.isolated", "pawnStructure.passed"]
    assert summary["factsRegistryVersion"] == 23
    assert summary["taxonomy"]["families"] == ["mate", "tactics"]
    assert summary["bounded_tactical_proof"]["captures"]["stm"] == {"count": 2, "seePositive": 1}


@pytest.mark.skipif(not os.path.isfile(ANALYZE), reason="legacy analyze binary not present")
def test_live_provider_facts_and_search_smoke():
    from cvslab.funnel.providers import AnalyzeFactsProvider, AnalyzeSearchProvider, AnalyzeTransport
    from cvslab.funnel.protocols import Position

    transport = AnalyzeTransport(ENGINE_ROOT, cwd=ENGINE_ROOT, args=[
        "--depth", "64",
        "--nnue", "target-cvs/matrix-raw.json",
        "--nnue-cal", "nets/eval-cal.json",
        "--helper-nnue", "target-cvs/matrix-residual.json",
    ])
    try:
        assert len(transport.analyze_sha256) == 64  # full identity, not truncated
        assert len(transport.analyze_sha256_short) == 16  # display-only form
        assert len(transport.taxonomy_sha256) == 64
        assert transport.taxonomy["family"]

        facts = AnalyzeFactsProvider(transport)
        batch = facts.label(Position(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
        assert batch.rows, batch.uncomputed
        record = batch.rows[0]
        assert record["status"] == "ok"
        assert record["deterministic_geometry"]["legalMoves"] == 20
        assert record["factsRegistryVersion"] and record["factsRegistryVersion"] > 0
        assert facts.producer_hash == transport.analyze_sha256  # full hash in provenance

        search = AnalyzeSearchProvider(transport, family="search_shallow_cp")
        assert search.family == "search_shallow_cp"
        label = search.search(Position(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                              node_budget=2000, pv_plies=8)
        assert label.nodes > 0
        assert label.best_move
        assert label.pv
        assert "trajectory" in label.extra and "stabilization" in label.extra
    finally:
        transport.close()


@pytest.mark.skipif(not os.path.isfile(TAXONOMY), reason="taxonomy file not present")
def test_taxonomy_loads_and_resolves():
    taxonomy = load_taxonomy(TAXONOMY)
    assert taxonomy["schemaVersion"] and taxonomy["sha256"]
    slugs = set(taxonomy["family"])
    assert resolve_slug("Skewers", None, slugs) == "skewer"
    assert resolve_slug("Pins", "absolute", slugs) == "absolute-pin"
    assert resolve_slug("Nope", "what", slugs) == "unmapped:Nope:what"
