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
UNRELATED_C = {"name": "test-unrelated-c", "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"]}


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
    position stay separate in the RAW source; normalization collapses the chess position
    but keeps the whole transposition-connected family split-safe."""
    cfg = config(games=3, max_plies=8, sample_every=1, min_ply=6,
                 opening_lines=(ITALIAN_A, ITALIAN_B, UNRELATED_C))
    # openings cycle with a seed-derived offset: derive the same rotation the generator
    # uses so each game gets the script written for its line
    import random as _random
    lines = cfg.effective_openings()
    offset = _random.Random(cfg.seed).randrange(len(lines))
    rotation = lines[offset:] + lines[:offset]
    scripts_by_line = {ITALIAN_A["name"]: ["f3g5", "h7h6"],   # diverges after the shared position
                       ITALIAN_B["name"]: ["d2d3", "d7d6"],   # diverges differently
                       UNRELATED_C["name"]: ["g1f3", "f8e7"]}  # unrelated: no shared positions
    scripts = [scripts_by_line[line["name"]] for line in rotation]
    game_of = {line["name"]: f"g{index:06d}" for index, line in enumerate(rotation)}
    game_a, game_b = game_of[ITALIAN_A["name"]], game_of[ITALIAN_B["name"]]
    generated = SelfPlayGenerator(cfg, ScriptedTeacher(scripts)).generate()
    report = generated["report"]
    assert report["games"] == 3 and report["rawPositions"] == 9
    assert report["uniquePositions"] == 8 and report["duplicatePositions"] == 1  # A4..A6 == B4..B6
    from collections import defaultdict
    by_identity: dict[str, set[str]] = defaultdict(set)
    for position in generated["positions"]:
        by_identity[epd_identity(position["fen"])].add(position["game_id"])
    shared_identities = [identity for identity, games in by_identity.items() if games == {game_a, game_b}]
    assert len(shared_identities) == 1  # the sampled shared position (ply 6); later plies diverge
    shared = [position for position in generated["positions"]
              if epd_identity(position["fen"]) == shared_identities[0]]
    assert {position["game_id"] for position in shared} == {game_a, game_b}  # both games retained

    manifest = write_pool(tmp_path / "pool", generated, {"config": cfg.canonical(),
                                                         "engine": {"commit": "x"}},
                          opening_lines=cfg.effective_openings())
    # the EFFECTIVE opening source is pinned (blocker 3 regression)
    assert manifest["openingSourceSha256"] == opening_source_sha256([ITALIAN_A, ITALIAN_B, UNRELATED_C])
    assert manifest["openingSourceSha256"] != opening_source_sha256()

    source = snapshot_source(lab.store, tmp_path / "pool", name="raw", license="test",
                             generated=generated, manifest=manifest)
    normalization = lab.normalize([source.id], name="canon")
    records = [json.loads(line) for line in
               lab.store.abs(normalization.path).read_text(encoding="utf-8").splitlines()]
    assert len(records) == 8 and normalization.duplicate_count == 1  # dedup happens HERE

    prefixed_a, prefixed_b = f"{source.id}:{game_a}", f"{source.id}:{game_b}"
    shared_records = [record for record in records if prefixed_a in record.get("games", [])
                      and prefixed_b in record.get("games", [])]
    assert shared_records and all(record["group"].endswith(record["group"].split(":")[-1])
                                  and ":xc" in record["group"] for record in shared_records)
    # every record of the A/B family carries the full two-game provenance
    ab_records = [record for record in records if prefixed_a in record.get("games", [])]
    assert len(ab_records) == 5 and all(
        record["games"] == sorted([prefixed_a, prefixed_b])
        for record in ab_records)

    # split safety: a group (connected component) never straddles dataset splits
    dataset = lab.freeze_dataset(source.id and normalization.id, name="split-safety",
                                 required_labels=[])
    split_of_group: dict[str, set[str]] = {}
    for manifest_entry in dataset.splits:
        for row in json.loads(json.dumps(__import__("cvslab.data", fromlist=["read_jsonl"])
                                         .read_jsonl(lab.store.abs(manifest_entry.path)))) or []:
            split_of_group.setdefault(row["group"], set()).add(manifest_entry.name)
    assert split_of_group and all(len(splits) == 1 for splits in split_of_group.values())


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
