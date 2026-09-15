"""Legacy-engine providers: shared `analyze --serve` transport + method-scoped identities.

Ports, from `chess-vision-studio-rust-engine/training/funnel/labeling_funnel.py`:
the persistent JSON-lines transport, ``board_geometry`` (python-chess, search-free),
``compact_search`` (search response -> full compact payload) and ``parse_sf_output``
(Stockfish UCI).

Identity rules (review of PR #24):

* the transport is shared; **facts and search are separate providers** with their own
  provenance classes, so label provenance is never guessed by later slices;
* full SHA256 identities for the analyze binary and taxonomy are preserved for
  canonical provenance/manifests — ``*_short`` forms are display-only conveniences.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import chess

from .protocols import LabelBatch, Position, SearchLabel

TAXONOMY_SUFFIX = Path("benchmarks/data/motif-taxonomy.json")
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if os.name == "nt" else 0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(value: str, length: int = 16) -> str:
    """Display-only shortening; canonical provenance always keeps the full hash."""
    return value[:length]


def board_geometry(fen: str) -> tuple[dict, str]:
    """Exact port of the funnel's search-free geometry (python-chess)."""
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    counts = {}
    for color, name in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        counts[name] = {piece: len(board.pieces(piece_type, color)) for piece, piece_type in
                        (("Q", chess.QUEEN), ("R", chess.ROOK), ("B", chess.BISHOP),
                         ("N", chess.KNIGHT), ("P", chess.PAWN))}
    non_pawn = sum(c["N"] + c["B"] + 2 * c["R"] + 4 * c["Q"] for c in counts.values())
    phase = "opening" if non_pawn >= 22 else "endgame" if non_pawn <= 8 else "middlegame"

    def signature(side: dict) -> str:
        return f"Q{side['Q']}R{side['R']}m{side['B'] + side['N']}P{min(side['P'], 8) // 3}"

    bucket = f"{signature(counts['w'])}-{signature(counts['b'])}"
    return {
        "sideToMove": "white" if board.turn else "black",
        "legalMoves": len(legal),
        "legalCaptures": sum(1 for move in legal if board.is_capture(move)),
        "legalChecks": sum(1 for move in legal if board.gives_check(move)),
        "inCheck": board.is_check(),
        "nonPawnMaterial": non_pawn,
        "phase": phase,
        "materialBucket": bucket,
    }, (min(move.uci() for move in legal) if legal else "")


# The scientific payload every search label carries. Timing fields are excluded
# here and handled by declared tolerances in the parity harness.
COMPACT_SEARCH_KEYS = (
    "nodes", "depth", "scoreCpStm", "mate", "bestMove", "pv", "termination",
    "resultSource", "stabilization", "trajectory", "avgCutoffMoveIndex",
    "avgLegalMoves", "qNodes",
)


def compact_search(response: dict, pv_plies: int) -> dict:
    """Search response -> the funnel's compact label shape (verbatim key contract).

    The engine's `go` response carries the chosen move as ``uci`` (not ``bestMove``);
    stabilization/telemetry are nested objects.
    """
    stabilization = response.get("stabilization") or {}
    telemetry = response.get("telemetry") or {}
    return {
        "nodes": response.get("nodes"),
        "depth": response.get("depth"),
        "scoreCpStm": response.get("scoreCp"),
        "mate": response.get("mate"),
        "bestMove": response.get("uci"),
        "pv": (response.get("pv") or [])[:pv_plies],
        "termination": response.get("termination"),
        "resultSource": response.get("resultSource"),
        "stabilization": {key: stabilization.get(key) for key in (
            "status", "bestMoveChanges", "maxAdjacentSwingCp", "scoreRangeCp", "signFlips",
            "oddEvenOscillation", "mateFlip", "seePruneSkips", "reasons")},
        "trajectory": [[iteration.get("depth"), iteration.get("uci"), iteration.get("scoreCp")]
                       for iteration in response.get("iterations") or []],
        "avgCutoffMoveIndex": telemetry.get("avgCutoffMoveIndex"),
        "avgLegalMoves": telemetry.get("avgLegalMoves"),
        "qNodes": response.get("qNodes"),
    }


def parse_sf_output(text: str) -> list[Optional[dict]]:
    """Stockfish `go` output -> one dict per info line (port of the funnel's parser)."""
    out: list[Optional[dict]] = []
    for line in text.splitlines():
        if line.startswith("info depth"):
            parts = line.split()
            info: dict = {}
            for index, token in enumerate(parts):
                if token in ("depth", "seldepth", "nodes", "time", "score", "pv", "multipv", "hashfull"):
                    info[token] = parts[index + 1] if index + 1 < len(parts) else None
                    if token == "score" and index + 2 < len(parts):
                        info["scoreKind"] = parts[index + 1]
                        info["scoreValue"] = parts[index + 2]
            out.append(info)
        elif line.startswith("bestmove") and out:
            parts = line.split()
            out[-1] = {**(out[-1] or {}), "bestmove": parts[1] if len(parts) > 1 else None}
    return out


class Serve:
    """One persistent `analyze --serve`; one JSON request -> one JSON line."""

    def __init__(self, command: list[str], cwd: Optional[str] = None):
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, creationflags=BELOW_NORMAL, cwd=cwd)

    def request(self, obj: dict) -> dict:
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(obj) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("analyze --serve exited")
        return json.loads(line)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()


class AnalyzeTransport:
    """Shared, identity-pinned connection to the frozen legacy engine.

    Owns the binary-sha (full) and taxonomy-sha (full) identities; facts and
    search providers are separate views over one transport so a run never spawns
    two engine processes, while each provider keeps its own provenance class.
    """

    def __init__(self, repo_root: str | Path, analyze: str = "target/release/analyze.exe",
                 args: Optional[list[str]] = None, cwd: Optional[str | Path] = None):
        self.repo_root = Path(repo_root)
        self.analyze_path = self.repo_root / analyze
        if not self.analyze_path.is_file():
            raise FileNotFoundError(f"analyze binary not found: {self.analyze_path}")
        self.args = list(args or [])
        # Relative net paths in args resolve against the artifact root, like the
        # legacy funnel's --artifact-root.
        self.cwd = str(cwd or self.repo_root)
        self.analyze_sha256 = sha256_file(self.analyze_path)  # full identity
        self.taxonomy_path = self.repo_root / TAXONOMY_SUFFIX
        if self.taxonomy_path.is_file():
            from .taxonomy import load_taxonomy  # lazy: taxonomy imports this module's sha256_file
            self.taxonomy = load_taxonomy(self.taxonomy_path)
        else:
            self.taxonomy = {"schemaVersion": None, "sha256": "", "family": {}}
        self.taxonomy_sha256 = self.taxonomy["sha256"]  # full identity
        self._serve: Optional[Serve] = None

    @property
    def analyze_sha256_short(self) -> str:
        return short_hash(self.analyze_sha256)

    @property
    def taxonomy_sha256_short(self) -> str:
        return short_hash(self.taxonomy_sha256)

    @property
    def serve(self) -> Serve:
        if self._serve is None:
            self._serve = Serve([str(self.analyze_path), "--serve", *self.args], cwd=self.cwd)
        return self._serve

    def request_facts(self, fen_before: str, played_move_uci: str, options: dict) -> dict:
        return self.serve.request({"cmd": "facts", "schemaVersion": 1, "fenBefore": fen_before,
                                   "playedMoveUci": played_move_uci, "options": options})

    def request_search(self, fen: str, node_budget: int) -> dict:
        return self.serve.request({"cmd": "go", "fen": fen, "nodeBudget": node_budget})

    def close(self) -> None:
        if self._serve is not None:
            self._serve.close()
            self._serve = None


class AnalyzeFactsProvider:
    """Tier-0 deterministic facts from the legacy engine. Provenance: deterministic_geometry."""

    provenance_class = "deterministic_geometry"

    def __init__(self, transport: AnalyzeTransport):
        self.transport = transport

    @property
    def producer_hash(self) -> str:
        return self.transport.analyze_sha256  # full

    def label(self, position: Position, *, options: Optional[dict] = None) -> LabelBatch:
        """Search-free: the lexicographically first legal move is passed because the
        position-level ``before`` block does not depend on the played move (same
        trick as the legacy funnel)."""
        from .leakguard import assert_static_input_safe
        from .taxonomy import summarize_facts

        options = options or {}
        geo, first_move = board_geometry(position.fen)
        if not first_move:
            return LabelBatch(provenance_class="deterministic_geometry", producer=self.producer_hash,
                              uncomputed=("no-legal-moves",))
        bundle = self.transport.request_facts(
            position.fen, first_move,
            {"includeMotifOpportunities": options.get("includeMotifOpportunities", True),
             "includeCounterfactual": False})
        if "error" in bundle:
            return LabelBatch(provenance_class="deterministic_geometry", producer=self.producer_hash,
                              uncomputed=(f"facts-error: {bundle['error']}",))
        summary = summarize_facts(bundle, self.transport.taxonomy["family"])
        summary["deterministic_geometry"] = {**geo, **summary["deterministic_geometry"]}
        record = {"id": position.fen, "stage": "tier0", "status": "ok", **summary}
        assert_static_input_safe(record)
        return LabelBatch(provenance_class="deterministic_geometry", producer=self.producer_hash,
                          rows=(record,), registry_version=int(summary["factsRegistryVersion"] or 0))


class AnalyzeSearchProvider:
    """Cold fixed-node search from the legacy engine. Provenance: search_derived.

    ``family`` names which label family this instance produces (search_shallow_cp
    for tier1 budgets, search_deep_cp for tier3), so downstream provenance is
    never guessed from the provider type.
    """

    provenance_class = "search_derived"

    def __init__(self, transport: AnalyzeTransport, *, family: str = "search_deep_cp"):
        if family not in ("search_shallow_cp", "search_deep_cp"):
            raise ValueError(f"unknown search label family {family!r}")
        self.transport = transport
        self.family = family

    @property
    def producer_hash(self) -> str:
        return self.transport.analyze_sha256  # full

    def search(self, position: Position, *, node_budget: int, pv_plies: int = 8) -> SearchLabel:
        response = self.transport.request_search(position.fen, node_budget)
        compact = compact_search(response, pv_plies)
        return SearchLabel(
            score_cp_stm=compact["scoreCpStm"], mate=compact["mate"], best_move=compact["bestMove"],
            pv=tuple(compact["pv"]), nodes=compact["nodes"],
            wall_ms=float(response.get("timeMs", 0.0)),
            extra={key: compact[key] for key in COMPACT_SEARCH_KEYS
                   if key not in ("scoreCpStm", "mate", "bestMove", "pv", "nodes")},
        )
