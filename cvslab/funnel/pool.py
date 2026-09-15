"""S5 (#11): raw candidate-position source — fresh legacy-CVS self-play.

Boundary (established in S4 review): this module produces the **raw source**. It
keeps game multiplicity and provenance and performs **no position dedup** — the
same position reached by different games stays as separate records, because that
grouping is evidence and is required for leakage-safe splitting. Dedup happens at
normalization (`N####`, EPD identity) or, for the labeling workset, in the S4
orchestrator's pool stage.

Determinism contract: same seed + same engine/model/opening identities -> byte-identical
raw pool. All stochastic choices (opening line, near-equal root move, sampling)
are seed-derived; all engine searches run at fixed node budgets with a pinned binary.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import chess

from ..hashing import canonical_json, hash_obj, sha256_file
from ..store import LabError

GENERATOR = "cvslab.pool.selfplay"
GENERATOR_VERSION = 1

# A small declarative opening source. Chosen deterministically per game (seed-derived).
# Its canonical hash is pinned in the manifest as `openingSourceSha256`.
OPENING_LINES: list[dict] = [
    {"name": "open-italian", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]},
    {"name": "open-spanish", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]},
    {"name": "open-scotch", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4"]},
    {"name": "semi-slav", "moves": ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6"]},
    {"name": "qgd", "moves": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"]},
    {"name": "kid", "moves": ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"]},
    {"name": "english", "moves": ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5"]},
    {"name": "reti", "moves": ["g1f3", "d7d5", "c2c4", "e7e6", "g2g3", "g8f6"]},
    {"name": "caro-kann", "moves": ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"]},
    {"name": "french", "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"]},
    {"name": "sicilian", "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"]},
    {"name": "pirc", "moves": ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"]},
]


def opening_source_sha256(lines: Optional[list[dict]] = None) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(lines if lines is not None else OPENING_LINES)).hexdigest()


class Teacher(Protocol):
    """Minimal search contract the generator needs from a frozen engine."""

    producer_hash: str

    def root_candidates(self, fen: str, *, candidates: int, node_budget: int) -> list[tuple[str, Optional[int]]]:
        """Ordered [(move_uci, score_cp_stm)] root candidate set from one search."""

    def search(self, fen: str, *, node_budget: int) -> tuple[str, Optional[int]]:
        """(best_move_uci, score_cp_stm) at a fixed node budget."""


class AnalyzeTeacher:
    """Frozen legacy engine adapter (reuses the S1 transport)."""

    def __init__(self, transport, *, root_candidates_from_order: int = 6):
        from .providers import AnalyzeSearchProvider
        self.transport = transport
        self.provider = AnalyzeSearchProvider(transport, family="search_shallow_cp")
        self.producer_hash = transport.analyze_sha256
        self.root_candidates_from_order = root_candidates_from_order

    def root_candidates(self, fen: str, *, candidates: int, node_budget: int):
        response = self.transport.request_search(fen, node_budget)
        order = list(response.get("rootOrder") or [])
        best_uci, best_score = response.get("uci"), response.get("scoreCp")
        out: list[tuple[str, Optional[int]]] = []
        for move in order[:candidates]:
            if move == best_uci:
                out.append((move, best_score))
                continue
            board = chess.Board(fen)
            board.push_uci(move)
            _, score = self.search(board.fen(), node_budget=node_budget)
            out.append((move, score))
        return out

    def search(self, fen: str, *, node_budget: int) -> tuple[str, Optional[int]]:
        response = self.transport.request_search(fen, node_budget)
        return str(response.get("uci")), response.get("scoreCp")


@dataclass
class PoolConfig:
    seed: int = 20260915
    games: int = 10
    max_plies: int = 80
    # opening diversification: at these plies the generator considers a small root
    # candidate set and picks seed-derived among moves within `score_window_cp` of best
    diversification_plies: tuple[int, ...] = (6, 8, 10)
    diversification_candidates: int = 4
    diversification_window_cp: int = 25
    diversification_node_budget: int = 2000
    play_node_budget: int = 20000
    # sampling: emit every `sample_every`-th ply from `min_ply` onward (uniform through the game)
    sample_every: int = 2
    min_ply: int = 6
    engine_root: Optional[str] = None
    # declared opening source: None = the built-in OPENING_LINES; a tuple of
    # {"name","moves"} replaces it for a run (the effective source is hashed).
    opening_lines: Optional[tuple] = None

    def effective_openings(self) -> list[dict]:
        return list(self.opening_lines) if self.opening_lines else list(OPENING_LINES)

    def canonical(self) -> dict:
        return {
            "generator": GENERATOR, "generator_version": GENERATOR_VERSION,
            "openingSourceSha256": opening_source_sha256(self.effective_openings()),
            "seed": self.seed, "games": self.games, "max_plies": self.max_plies,
            "diversification_plies": list(self.diversification_plies),
            "diversification_candidates": self.diversification_candidates,
            "diversification_window_cp": self.diversification_window_cp,
            "diversification_node_budget": self.diversification_node_budget,
            "play_node_budget": self.play_node_budget,
            "sample_every": self.sample_every, "min_ply": self.min_ply,
        }


def _score_from_stm_pov(score_cp: Optional[int], mover: chess.Color) -> Optional[int]:
    """Teacher scores are stm-POV; normalize to White-POV for windowing on one move list."""
    if score_cp is None:
        return None
    return score_cp if mover == chess.WHITE else -score_cp


class SelfPlayGenerator:
    """Deterministic fresh self-play pool generator. No dedup happens here, by design."""

    def __init__(self, config: PoolConfig, teacher: Teacher):
        self.config = config
        self.teacher = teacher

    # -- one game --------------------------------------------------------------

    def _opening_offset(self, rng: random.Random) -> int:
        return rng.randrange(len(self.config.effective_openings()))

    def _pick_opening(self, offset: int, game_index: int) -> dict:
        """Openings cycle with a seed-derived offset: every declared line is used before
        any repeats, which is broader coverage than iid draws and still deterministic."""
        lines = self.config.effective_openings()
        return lines[(offset + game_index) % len(lines)]

    def _choose_move(self, board: chess.Board, ply: int, rng: random.Random,
                     audit: list[dict]) -> tuple[str, Optional[int]]:
        if ply in self.config.diversification_plies:
            scored: list[tuple[str, Optional[int]]] = []
            for move, score_stm in self.teacher.root_candidates(
                    board.fen(), candidates=self.config.diversification_candidates,
                    node_budget=self.config.diversification_node_budget):
                scored.append((move, _score_from_stm_pov(score_stm, board.turn)))
            known = [(move, score) for move, score in scored if score is not None]
            if known:
                best = max(score for _, score in known)
                within = [move for move, score in known if best - score <= self.config.diversification_window_cp]
                if within:
                    selected = within[rng.randrange(len(within))]
                    audit.append({
                        "ply": ply, "candidates": [{"move": move, "scoreCpWhite": score} for move, score in scored],
                        "windowCp": self.config.diversification_window_cp, "bestScoreCpWhite": best,
                        "withinWindow": within, "selected": selected,
                    })
                    return selected, next(score for move, score in scored if move == selected)
        move, score = self.teacher.search(board.fen(), node_budget=self.config.play_node_budget)
        return move, score

    def _play_game(self, game_index: int, rng: random.Random, offset: int) -> dict:
        opening = self._pick_opening(offset, game_index)
        game_id = f"g{game_index:06d}"
        if self.config.max_plies < len(opening["moves"]):
            raise LabError(
                f"{game_id}: opening {opening['name']!r} needs {len(opening['moves'])} plies but "
                f"max_plies={self.config.max_plies}; raise max_plies above the opening length")
        board = chess.Board()
        audit: list[dict] = []
        plies: list[dict] = []

        def sample(ply: int) -> None:
            if ply >= self.config.min_ply and ply % self.config.sample_every == 0:
                plies.append({"ply": ply, "fen": board.fen()})

        for move_uci in opening["moves"]:
            board.push_uci(move_uci)
            sample(board.ply())
        for ply in range(board.ply() + 1, self.config.max_plies + 1):
            if board.is_game_over():
                break
            move, _score = self._choose_move(board, ply, rng, audit)
            try:
                board.push_uci(move)
            except ValueError as exc:  # teacher returned an illegal/garbled move: fail loudly
                raise LabError(f"{game_id} ply {ply}: teacher move {move!r} is illegal ({exc})") from None
            sample(board.ply())
        result = board.result(claim_draw=True)
        white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result)
        return {"game_id": game_id, "opening_id": opening["name"],
                "opening_moves": list(opening["moves"]), "positions": plies,
                "result": result, "gameOutcome": ({"whiteScore": white_score} if white_score is not None else None),
                "diversification": audit}

    # -- whole pool ------------------------------------------------------------

    def generate(self) -> dict:
        """Returns {games, positions, report}. Raw: duplicate positions across games are kept."""
        rng = random.Random(self.config.seed)
        offset = self._opening_offset(rng)
        games = [self._play_game(index, rng, offset) for index in range(self.config.games)]
        positions = []
        for game in games:
            for entry in game["positions"]:
                positions.append({
                    "game_id": game["game_id"], "opening_id": game["opening_id"], "ply": entry["ply"],
                    "fen": entry["fen"], "gameOutcome": game["gameOutcome"],
                    "source": {"generator": GENERATOR, "generator_version": GENERATOR_VERSION,
                               "seed": self.config.seed},
                })
        return {"games": games, "positions": positions, "report": coverage_report(games, positions)}


def epd_identity(fen: str) -> str:
    """Analysis-only position identity (counters stripped); NOT used to dedup the raw source."""
    board = chess.Board(fen)
    return "pos_" + hashlib.sha256(board.epd().encode()).hexdigest()[:20]


def coverage_report(games: list[dict], positions: list[dict]) -> dict:
    """Opening/phase/material coverage of the raw pool — analysis, never a collapse of it."""
    from ..data import game_phase, material_signature

    by_opening: dict[str, int] = {}
    by_phase: dict[str, int] = {}
    by_material: dict[str, int] = {}
    identities: dict[str, int] = {}
    for game in games:
        by_opening[game["opening_id"]] = by_opening.get(game["opening_id"], 0) + 1
    for position in positions:
        board = chess.Board(position["fen"])
        phase, material = game_phase(board), material_signature(board)
        by_phase[phase] = by_phase.get(phase, 0) + 1
        by_material[material] = by_material.get(material, 0) + 1
        identity = epd_identity(position["fen"])
        identities[identity] = identities.get(identity, 0) + 1
    duplicate_positions = sum(count - 1 for count in identities.values() if count > 1)
    return {
        "games": len(games),
        "rawPositions": len(positions),
        "uniquePositions": len(identities),
        "duplicatePositions": duplicate_positions,
        "maxMultiplicity": max(identities.values()) if identities else 0,
        "openings": dict(sorted(by_opening.items())),
        "phases": dict(sorted(by_phase.items())),
        "materialBuckets": dict(sorted(by_material.items())),
    }


def write_pool(directory: str | Path, generated: dict, manifest_extra: dict) -> dict:
    """Write the raw pool (positions.jsonl, games.jsonl, pool-manifest.json) + return the manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def write_jsonl(name: str, rows: list[dict]) -> str:
        path = directory / name
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        return sha256_file(path)

    positions_hash = write_jsonl("positions.jsonl", generated["positions"])
    games_hash = write_jsonl("games.jsonl", [
        {key: value for key, value in game.items() if key != "positions"} for game in generated["games"]])
    manifest = {
        "generator": GENERATOR, "generatorVersion": GENERATOR_VERSION,
        "configSha256": hash_obj(manifest_extra["config"]),
        "config": manifest_extra["config"],
        "engine": manifest_extra["engine"],
        "openingSourceSha256": opening_source_sha256(),
        "positionsFile": {"name": "positions.jsonl", "sha256": positions_hash,
                          "rows": len(generated["positions"])},
        "gamesFile": {"name": "games.jsonl", "sha256": games_hash, "rows": len(generated["games"])},
        "report": generated["report"],
        "note": "Raw source: game multiplicity preserved; NO position dedup happens here. "
                "Dedup belongs to N#### normalization (EPD) or the S4 labeling workset.",
    }
    manifest["manifestSha256"] = hash_obj(manifest)
    (directory / "pool-manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def snapshot_source(store, pool_directory: str | Path, *, name: str, license: str,
                    generated: dict, manifest: dict):
    """Snapshot the RAW pool as an S#### source: rows carry explicit `game` ids, so
    normalization groups by game and only then collapses duplicate positions."""
    from ..data import write_jsonl
    from ..schemas import Compute, SourceSnapshot, utc_now

    pool_directory = Path(pool_directory)
    source_id = store.next_id("S")
    rel = f"sources/{source_id}/positions.jsonl"
    rows = [{"fen": row["fen"], "game": row["game_id"], "ply": row["ply"],
             "opening_id": row["opening_id"],
             "res": (row.get("gameOutcome") or {}).get("whiteScore")
             if isinstance(row.get("gameOutcome"), dict) else None,
             "chosen_opening": row["opening_id"]}
            for row in generated["positions"]]
    content_hash = write_jsonl(store.abs(rel), rows)
    store.make_readonly(store.abs(rel))
    engine = manifest.get("engine") or {}
    return store.create(SourceSnapshot(
        id=source_id, name=name, source_origin=str(pool_directory.resolve()), locality="local",
        generator={key: value for key, value in {
            "name": GENERATOR, "version": GENERATOR_VERSION,
            "configSha256": manifest["configSha256"],
            "manifestSha256": manifest["manifestSha256"],
            "openingSourceSha256": manifest["openingSourceSha256"],
            "engineCommit": engine.get("commit"), "engineBinarySha256": engine.get("binarySha256"),
            "netHashes": json.dumps(engine.get("netHashes") or {}, sort_keys=True),
            "games": manifest["report"]["games"]}.items() if value is not None},
        license=license, importer=GENERATOR, importer_version=GENERATOR_VERSION, path=rel,
        content_hash=content_hash, row_count=len(rows),
        game_count=manifest["report"]["games"], label_authorities=["outcome.game_result.v1"],
        compute=Compute(), created_at=utc_now()))


DEFAULT_ENGINE_ARGS = [
    "--depth", "64",
    "--nnue", "target-cvs/matrix-raw.json",
    "--nnue-cal", "nets/eval-cal.json",
    "--helper-nnue", "target-cvs/matrix-residual.json",
]


def engine_identity(engine_root: str | Path, *, analyze: str = "target/release/analyze.exe",
                    args: list[str] | None = None) -> dict:
    """Frozen teacher identity: commit, binary sha256, and every net file hash.

    Net paths are taken from the same args the binary runs with (--nnue, --nnue-cal,
    --helper-nnue, ...), so the manifest pins exactly the artifacts that produced moves.
    """
    import subprocess

    root = Path(engine_root)
    binary = root / analyze
    if not binary.is_file():
        raise LabError(f"analyze binary not found: {binary}")
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    net_hashes: dict[str, str] = {}
    arg_list = list(args if args is not None else DEFAULT_ENGINE_ARGS)
    for index, token in enumerate(arg_list):
        if token.startswith("--") and index + 1 < len(arg_list) and not arg_list[index + 1].startswith("--"):
            value = arg_list[index + 1]
            if value.endswith(".json"):
                path = Path(value)
                if not path.is_absolute():
                    path = root / value
                if path.is_file():
                    net_hashes[token] = sha256_file(path)
    return {"engineRoot": str(root.resolve()), "commit": commit,
            "binarySha256": sha256_file(binary), "binary": analyze,
            "args": arg_list, "netHashes": dict(sorted(net_hashes.items()))}


def generate_and_write(engine_root: str | Path, out_directory: str | Path, *, config: PoolConfig,
                       analyze: str = "target/release/analyze.exe",
                       args: list[str] | None = None, store=None, source_name: Optional[str] = None,
                       license: str = "engine self-play corpus") -> dict:
    """CLI/entry helper: frozen teacher -> deterministic pool -> raw files (+ optional S####)."""
    from .providers import AnalyzeTransport

    identity = engine_identity(engine_root, analyze=analyze, args=args)
    transport = AnalyzeTransport(engine_root, analyze=analyze, args=list(identity["args"]),
                                 cwd=engine_root)
    try:
        generator = SelfPlayGenerator(config, AnalyzeTeacher(transport))
        generated = generator.generate()
    finally:
        transport.close()
    manifest = write_pool(out_directory, generated, {"config": config.canonical(), "engine": identity})
    snapshot = (snapshot_source(store, out_directory, name=source_name or f"selfplay-{config.seed}",
                                license=license, generated=generated, manifest=manifest)
                if store is not None else None)
    return {"manifest": manifest, "report": generated["report"],
            "source": getattr(snapshot, "id", None)}
