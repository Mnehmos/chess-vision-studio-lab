import json
import os
from pathlib import Path

import chess
import pytest

from cvslab.funnel.pool import (
    OPENING_LINES,
    PoolConfig,
    SelfPlayGenerator,
    engine_identity,
    epd_identity,
    opening_source_sha256,
    snapshot_source,
    write_pool,
)
from cvslab.store import LabError

ENGINE_ROOT = r"F:\Github\chess-vision-studio-rust-engine"
ANALYZE = ENGINE_ROOT + r"\target\release\analyze.exe"

# Two declared opening lines that transpose: the same position after six plies.
ITALIAN_A = {"name": "test-italian-order-a", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]}
ITALIAN_B = {"name": "test-italian-order-b", "moves": ["g1f3", "b8c6", "e2e4", "e7e5", "f1c4", "f8c5"]}


class ScriptedTeacher:
    """Deterministic teacher: scripted early moves per game, first-legal afterwards."""

    producer_hash = "scriptedteacher" * 4

    def __init__(self, scripts: list[list[str]] | None = None):
        self.scripts = scripts or []
        self.game_index = -1
        self.queue: list[str] = []

    def _advance(self) -> None:
        if not self.queue and self.game_index + 1 < len(self.scripts):
            self.game_index += 1
            self.queue = list(self.scripts[self.game_index])

    def search(self, fen: str, *, node_budget: int):
        self._advance()
        if self.queue:
            return self.queue.pop(0), 0
        board = chess.Board(fen)
        return (min(m.uci() for m in board.legal_moves) if board.legal_moves else None), 0

    def root_candidates(self, fen: str, *, candidates: int, node_budget: int):
        board = chess.Board(fen)
        moves = sorted(m.uci() for m in board.legal_moves)[:candidates]
        return [(move, 10 - index) for index, move in enumerate(moves)]  # all within a 25cp window


def config(**overrides) -> PoolConfig:
    base = dict(seed=7, games=2, max_plies=8, sample_every=1, min_ply=1, diversification_plies=(),
                play_node_budget=100)
    base.update(overrides)
    return PoolConfig(**base)


def test_deterministic_replay_is_byte_identical(tmp_path):
    first = SelfPlayGenerator(config(), ScriptedTeacher()).generate()
    second = SelfPlayGenerator(config(), ScriptedTeacher()).generate()
    assert json.dumps(first["positions"], sort_keys=True) == json.dumps(second["positions"], sort_keys=True)
    assert json.dumps(first["games"], sort_keys=True) == json.dumps(second["games"], sort_keys=True)
    cfg = config().canonical()
    manifest_a = write_pool(tmp_path / "a", first, {"config": cfg, "engine": {"commit": "x"}})
    manifest_b = write_pool(tmp_path / "b", second, {"config": cfg, "engine": {"commit": "x"}})
    assert manifest_a["positionsFile"]["sha256"] == manifest_b["positionsFile"]["sha256"]
    assert manifest_a["configSha256"] == manifest_b["configSha256"]


def test_every_position_carries_identity_fields():
    generated = SelfPlayGenerator(config(min_ply=2, sample_every=2, max_plies=8),
                                  ScriptedTeacher()).generate()
    assert generated["positions"]
    for position in generated["positions"]:
        assert position["game_id"].startswith("g")
        assert position["opening_id"] in {line["name"] for line in OPENING_LINES}
        assert isinstance(position["ply"], int) and position["ply"] >= 2 and position["ply"] % 2 == 0


def test_raw_pool_keeps_transposed_position_in_both_games(tmp_path, lab):
    """Adversarial regression for the S2 lesson: two different games reaching the same
    position stay separate in the RAW source; only normalization may collapse them."""
    cfg = config(games=2, max_plies=6, sample_every=1, min_ply=6,
                 opening_lines=(ITALIAN_A, ITALIAN_B))
    generated = SelfPlayGenerator(cfg, ScriptedTeacher()).generate()
    report = generated["report"]
    assert report["rawPositions"] == 2 and report["games"] == 2
    assert report["uniquePositions"] == 1 and report["duplicatePositions"] == 1
    assert report["maxMultiplicity"] == 2
    assert {position["game_id"] for position in generated["positions"]} == {"g000000", "g000001"}
    assert {position["opening_id"] for position in generated["positions"]} == {ITALIAN_A["name"],
                                                                              ITALIAN_B["name"]}
    assert epd_identity(generated["positions"][0]["fen"]) == epd_identity(generated["positions"][1]["fen"])

    manifest = write_pool(tmp_path / "pool", generated, {"config": cfg.canonical(),
                                                         "engine": {"commit": "x"}})
    source = snapshot_source(lab.store, tmp_path / "pool", name="raw", license="test",
                             generated=generated, manifest=manifest)
    normalization = lab.normalize([source.id], name="canon")
    assert normalization.record_count == 1 and normalization.duplicate_count == 1  # dedup happens HERE
    records = [json.loads(line) for line in
               lab.store.abs(normalization.path).read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1 and records[0]["group"].startswith(source.id + ":g")  # grouping survived


def test_opening_diversification_is_auditable_and_seed_deterministic():
    cfg = config(seed=11, games=1, max_plies=12, diversification_plies=(8,),
                 diversification_candidates=3, diversification_window_cp=25)
    first = SelfPlayGenerator(cfg, ScriptedTeacher()).generate()
    second = SelfPlayGenerator(cfg, ScriptedTeacher()).generate()
    audit = first["games"][0]["diversification"]
    assert audit and audit[0]["ply"] == 8
    entry = audit[0]
    assert entry["windowCp"] == 25 and entry["candidates"] and entry["withinWindow"]
    assert entry["selected"] in entry["withinWindow"]
    assert all(entry["bestScoreCpWhite"] - candidate["scoreCpWhite"] <= 25 for candidate in entry["candidates"])
    assert second["games"][0]["diversification"][0]["selected"] == entry["selected"]


def test_sampling_is_declared_and_not_opening_skewed():
    cfg = config(games=1, max_plies=40, sample_every=2, min_ply=6, diversification_plies=())
    generated = SelfPlayGenerator(cfg, ScriptedTeacher()).generate()
    plies = [position["ply"] for position in generated["positions"]]
    assert plies and all(ply >= 6 and ply % 2 == 0 for ply in plies)
    assert max(plies) >= 20  # sampled throughout the game, not only the opening
    assert min(plies) == 6 and max(plies) - min(plies) >= 14


def test_max_plies_shorter_than_opening_fails_loudly():
    cfg = config(max_plies=4)  # built-in openings are 6 plies
    with pytest.raises(LabError, match="raise max_plies above the opening length"):
        SelfPlayGenerator(cfg, ScriptedTeacher()).generate()


def test_manifest_pins_identities_and_coverage(tmp_path):
    cfg = config()
    generated = SelfPlayGenerator(cfg, ScriptedTeacher()).generate()
    identity = {"engineRoot": "x", "commit": "deadbeef", "binarySha256": "b" * 64,
                "args": ["--depth", "64"], "netHashes": {"--nnue": "n" * 64}}
    manifest = write_pool(tmp_path / "pool", generated, {"config": cfg.canonical(), "engine": identity})
    assert manifest["openingSourceSha256"] == opening_source_sha256()
    assert manifest["engine"]["binarySha256"] == "b" * 64 and manifest["engine"]["commit"] == "deadbeef"
    assert manifest["config"]["seed"] == 7 and manifest["config"]["sample_every"] == 1
    assert manifest["config"]["openingSourceSha256"] == opening_source_sha256()
    assert manifest["positionsFile"]["rows"] == len(generated["positions"])
    assert manifest["manifestSha256"].startswith("sha256:")
    report = manifest["report"]
    assert set(report) >= {"games", "rawPositions", "uniquePositions", "duplicatePositions",
                           "openings", "phases", "materialBuckets"}
    assert "NO position dedup" in manifest["note"]


def test_declared_opening_source_is_hashed_differently():
    assert opening_source_sha256([ITALIAN_A, ITALIAN_B]) != opening_source_sha256()


def test_engine_identity_requires_binary(tmp_path):
    with pytest.raises(LabError, match="analyze binary not found"):
        engine_identity(tmp_path)


@pytest.mark.skipif(not os.path.isfile(ANALYZE), reason="legacy analyze binary not present")
def test_live_selfplay_deterministic_replay(tmp_path):
    """Real frozen teacher: one short game, generated twice, byte-identical."""
    from cvslab.funnel.pool import generate_and_write

    cfg = PoolConfig(seed=3, games=1, max_plies=12, sample_every=1, min_ply=4,
                     diversification_plies=(8,), diversification_candidates=3,
                     play_node_budget=2000, diversification_node_budget=1000)
    first = generate_and_write(ENGINE_ROOT, tmp_path / "a", config=cfg)
    second = generate_and_write(ENGINE_ROOT, tmp_path / "b", config=cfg)
    assert first["manifest"]["positionsFile"]["sha256"] == second["manifest"]["positionsFile"]["sha256"]
    assert first["manifest"]["engine"]["binarySha256"]
    assert first["manifest"]["engine"]["netHashes"]
    report = first["report"]
    assert report["games"] == 1 and report["rawPositions"] > 0
