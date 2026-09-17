"""The Stockfish teacher: identity by bytes, the CVS-compatible label contract, TargetSpec match.

Nothing here needs a Stockfish binary — the fake UCI engine in `tests/fake_uci_engine.py` speaks
enough of the protocol — but everything here is what a real label set is registered under.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cvslab.funnel.protocols import Position  # noqa: E402
from cvslab.funnel.stockfish import (DEFAULT_OPTIONS, StockfishProvider, StockfishTransport,  # noqa: E402
                                     mate_score_cp)
from cvslab.targets import TargetSpec, extract_target, matching_labels  # noqa: E402

FAKE = str(pathlib.Path(__file__).resolve().parent / "fake_uci_engine.py")
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_FEN = "8/8/8/8/8/5K2/6Q1/6k1 w - - 0 1"


@pytest.fixture(scope="module")
def transport():
    transport = StockfishTransport([sys.executable, FAKE])
    transport._start()
    yield transport
    transport.close()


def test_identity_is_bytes_and_announced_fields(transport):
    identity = transport.identity
    assert identity.name == "FakeFish 1.0"
    assert identity.nets == ("nn-test.nnue",) and identity.net == "nn-test.nnue"
    assert len(identity.binary_sha256) == 64
    assert identity.options == DEFAULT_OPTIONS
    assert identity.canonical()["options"] == {"Hash": 16, "MultiPV": 1, "Threads": 1}
    assert identity.short.startswith("FakeFish 1.0 [")
    assert identity.short.endswith("nn-test.nnue")
    assert "per position" in identity.canonical()["netSelection"]


def test_label_contract_matches_the_cvs_shape(transport):
    provider = StockfishProvider(transport, authority="oracle.stockfish.test")
    label = provider.search(Position(fen=START_FEN), node_budget=4096)
    assert label.score_cp_stm == 42 and label.mate is None
    assert label.best_move == "e2e4"
    assert label.pv == ("e2e4", "e7e5", "g1f3")
    assert label.reached_depth == 7
    assert label.nodes == 4096           # realized work is recorded, not assumed
    assert label.wall_ms >= 0
    row = provider.label_row("pos_test", label, node_budget=4096)
    assert row["budget"] == {"nodes": 4096}
    assert row["value"]["scoreCpStm"] == 42 and row["value"]["depth"] == 7
    assert provider.producer == transport.identity.binary_sha256


def test_mate_scores_use_the_declared_sentinel(transport):
    provider = StockfishProvider(transport, authority="oracle.stockfish.test")
    label = provider.search(Position(fen=MATE_FEN), node_budget=1024)
    assert label.mate == 3
    assert label.score_cp_stm == mate_score_cp(3) == 999_997


def test_a_stockfish_label_row_satisfies_a_frozen_target_spec(transport):
    provider = StockfishProvider(transport, authority="oracle.stockfish.test")
    label = provider.search(Position(fen=START_FEN), node_budget=4096)
    row = provider.label_row("pos_test", label, node_budget=4096)
    stamped = {**row, "family": "oracle_cp", "authority": "oracle.stockfish.test",
               "producer": provider.producer, "pov": "stm"}
    record = {"record_id": "pos_test", "labels": [stamped]}
    spec = TargetSpec(family="oracle_cp", authority="oracle.stockfish.test",
                      producer=provider.producer, budget={"nodes": 4096},
                      value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0)
    assert [match["_target_value"] for match in matching_labels(record, spec)] == [42]
    assert extract_target(record, spec) == 42.0


def test_authority_and_budget_prevent_cross_teacher_matches(transport):
    provider = StockfishProvider(transport, authority="oracle.stockfish.test")
    row = provider.label_row("pos_test", provider.search(Position(fen=START_FEN), node_budget=4096),
                             node_budget=4096)
    stamped = {**row, "family": "oracle_cp", "authority": "oracle.stockfish.test",
               "producer": provider.producer, "pov": "stm"}
    record = {"record_id": "pos_test", "labels": [stamped]}
    cvs_like = TargetSpec(family="oracle_cp", authority="legacy.cvs.search.shallow",
                          producer=provider.producer, budget={"nodes": 4096},
                          value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0)
    other_budget = TargetSpec(family="oracle_cp", authority="oracle.stockfish.test",
                              producer=provider.producer, budget={"nodes": 8192},
                              value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0)
    assert matching_labels(record, cvs_like) == []
    assert matching_labels(record, other_budget) == []


def test_transport_refuses_a_missing_binary():
    with pytest.raises(FileNotFoundError):
        StockfishTransport([str(pathlib.Path("does-not-exist-stockfish"))])._start()


def test_every_label_is_preceded_by_a_search_state_reset(tmp_path):
    """The cold contract: no `go` may ever be sent without a reset first.

    This is the regression test for the S13 warm-TT confound: the transport is reused across a
    whole worker chunk, and Stockfish's `position` command does not clear the transposition table
    (uci.cpp: position -> Engine::set_position; only ucinewgame / Clear Hash -> search_clear), so
    without an explicit reset a label would depend on the positions that worker saw before it.
    """
    transcript = tmp_path / "uci-transcript.txt"
    transport = StockfishTransport([sys.executable, FAKE, "--transcript", str(transcript)])
    try:
        transport._start()
        provider = StockfishProvider(transport, authority="oracle.stockfish.test")
        for fen in (START_FEN, START_FEN, MATE_FEN):
            provider.search(Position(fen=fen), node_budget=2048)
        assert transport.resets == 3, "three labels must mean three resets"

        lines = [line for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
        goes = [index for index, line in enumerate(lines) if line.startswith("go nodes ")]
        # the first go is the 1-node identity probe inside _start(); it cannot affect a label
        # because every label resets first, and it is the only go allowed to be warm
        assert lines[goes[0]] == "go nodes 1", "the identity probe must be the first search"
        labelled = goes[1:]
        assert len(labelled) == 3, f"expected three labelled searches, saw {len(labelled)}"
        for index in labelled:
            assert lines[index - 4:index - 1] == ["ucinewgame", "setoption name Clear Hash", "isready"], \
                f"a go without a full reset: {lines[max(0, index - 5):index + 1]}"
            assert lines[index - 1].startswith("position fen "), lines[index - 1]
    finally:
        transport.close()


def test_warm_mode_is_opt_in_and_visible(tmp_path):
    """`cold=False` exists for cost experiments only; the default must be cold."""
    transcript = tmp_path / "uci-transcript-warm.txt"
    transport = StockfishTransport([sys.executable, FAKE, "--transcript", str(transcript)])
    try:
        transport._start()
        provider = StockfishProvider(transport, authority="oracle.stockfish.test")
        baseline_resets = transport.resets
        label = provider.search(Position(fen=START_FEN), node_budget=1024, cold=False)
        assert transport.resets == baseline_resets, "cold=False must not reset"
        assert "WARM" in provider.label_row("pos_x", label, node_budget=1024, cold=False)["note"]
        assert "cold per label" in provider.label_row("pos_x", label, node_budget=1024)["note"]
    finally:
        transport.close()
