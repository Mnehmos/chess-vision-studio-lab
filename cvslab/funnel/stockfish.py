"""Stockfish UCI provider: the external-oracle teacher, pinned by binary and net identity.

This is the lab's first non-CVS teacher, and two things make it honest rather than convenient:

* **identity is bytes, not a nickname.** ``StockfishIdentity`` carries the full SHA256 of the
  binary actually executed, the ``id name``/``id author`` the binary reported over UCI, the NNUE
  net string it announced (``NNUE evaluation using <file>``), and the exact search options pinned
  for every label. A label's ``producer`` is that binary hash; no version string is trusted.
* **the label contract mirrors the CVS teacher.** ``scoreCpStm`` is side-to-move POV; mate scores
  are carried as a declared sentinel (±1_000_000) plus the mate distance, exactly so a Stockfish
  label and a CVS label are *structurally* comparable while remaining separate authorities. The
  realized ``nodes`` and ``depth`` are recorded per label.

**A Stockfish node is not assumed comparable to a CVS node** — the budget keys differ on purpose
(``{"nodes": N}`` here, ``{"nodeBudget": N}`` for CVS), so no TargetSpec can accidentally match one
teacher's labels with the other's contract.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .protocols import OracleLabel, Position

BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) if os.name == "nt" else 0
MATE_SENTINEL = 1_000_000
DEFAULT_OPTIONS = {"Threads": 1, "Hash": 16, "MultiPV": 1}


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mate_score_cp(mate: int) -> int:
    """Declared mate convention: sign * (1_000_000 - |mate|), closer mates larger.

    Mirrors the CVS teacher's sentinel practice (its own values are ±~1e6) without pretending the
    two engines share an arithmetic: what the student sees is a saturated cp, and the honest
    distance travels in the ``mate`` field.
    """
    sign = 1 if mate >= 0 else -1
    return sign * (MATE_SENTINEL - abs(mate))


@dataclass(frozen=True)
class StockfishIdentity:
    """What the executed binary *is*: bytes, the names it reports, and the options it ran under.

    ``nets`` holds every NNUE net the binary announced. Stockfish 18 embeds two and chooses per
    position (``Eval::use_smallnet``: the small net when the linear simple-eval magnitude exceeds
    962), so a single-net identity would be a half-truth; both are recorded.
    """

    name: str
    author: str
    binary: str
    binary_sha256: str
    nets: tuple[str, ...]
    options: dict = field(default_factory=dict)

    @property
    def net(self) -> str:
        """The primary (full-size) net — the first one the binary announces."""
        return self.nets[0] if self.nets else ""

    def canonical(self) -> dict:
        return {"name": self.name, "author": self.author, "binary": self.binary,
                "binarySha256": self.binary_sha256, "nets": list(self.nets),
                "netSelection": "per position: the small net is used when |simple_eval| > 962 "
                                "(Eval::use_smallnet), deterministically, with no option to disable",
                "options": {k: self.options[k] for k in sorted(self.options)}}

    @property
    def short(self) -> str:
        return f"{self.name} [{self.binary_sha256[:16]}] {'+'.join(self.nets)}"


class StockfishTransport:
    """One pinned UCI subprocess. ``label()`` is the only query; options never change."""

    def __init__(self, command: Sequence[str], *, options: Optional[Mapping[str, object]] = None,
                 timeout: float = 300.0):
        self.command = [str(part) for part in command]
        self.options = dict(DEFAULT_OPTIONS if options is None else options)
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self.identity: Optional[StockfishIdentity] = None
        self.resets = 0                        # search-state clears actually performed

    # -- process ---------------------------------------------------------------
    def _start(self) -> None:
        binary = Path(self.command[0])
        if not binary.is_file():
            raise FileNotFoundError(f"stockfish binary not found: {binary}")
        self._process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         text=True, encoding="utf-8", errors="replace",
                                         bufsize=1, creationflags=BELOW_NORMAL)
        self._process.stdin.write("uci\n")
        self._process.stdin.flush()
        name = author = ""
        nets: list[str] = []
        seen_uciok = False
        for line in self._lines():
            if line.startswith("id name "):
                name = line[len("id name "):]
            elif line.startswith("id author "):
                author = line[len("id author "):]
            elif line.startswith("uciok"):
                seen_uciok = True
                break
        if not seen_uciok:
            raise RuntimeError("the engine never answered 'uciok'")
        for key, value in self.options.items():
            self._send(f"setoption name {key} value {value}")
        self._send("isready")
        for line in self._lines():
            if line.startswith("readyok"):
                break
        # the net announcements arrive with the first search; ask for one cheap search to capture them
        self._send("position startpos")
        self._send("go nodes 1")
        for line in self._lines():
            if "NNUE evaluation using" in line:
                announced = line.split("NNUE evaluation using", 1)[1].strip().split(" ")[0]
                if announced and announced not in nets:
                    nets.append(announced)
            if line.startswith("bestmove"):
                break
        self.identity = StockfishIdentity(name=name, author=author, binary=str(binary),
                                          binary_sha256=sha256_file(binary), nets=tuple(nets),
                                          options=dict(self.options))

    def _send(self, text: str) -> None:
        if self._process is None:
            self._start()
        assert self._process and self._process.stdin
        self._process.stdin.write(text + "\n")
        self._process.stdin.flush()

    def _lines(self):
        assert self._process and self._process.stdout
        started = time.time()
        while True:
            if time.time() - started > self.timeout:
                raise TimeoutError(f"engine silent for {self.timeout}s")
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("engine closed its output")
            yield line.rstrip("\n")

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._process.stdin.write("quit\n")
            self._process.stdin.flush()
            self._process.wait(timeout=10)
        except Exception:
            self._process.kill()

    def _reset_search_state(self) -> None:
        """Make the next search independent of every earlier one on this process.

        Stockfish's `position` command does NOT clear the transposition table: it runs
        `Engine::set_position` (src/uci.cpp, `UCIEngine::position`), while only `ucinewgame` and
        `setoption name Clear Hash` reach `Engine::search_clear` (src/uci.cpp:135/281/350/410,
        src/engine.cpp:99-100). A reused process would therefore make a label a function of the
        positions that worker happened to see before it — a hidden treatment variable whose
        partition depends on the worker count.

        The protocol here is `ucinewgame` + `setoption name Clear Hash` + `isready`, and the
        `readyok` is the sync point: the engine's own comment notes that a clear "may take a
        while", so the next `go` must not race it.
        """
        self._send("ucinewgame")
        self._send("setoption name Clear Hash")
        self._send("isready")
        for line in self._lines():
            if line.startswith("readyok"):
                self.resets += 1
                return
            if line.startswith("bestmove"):     # impossible, but never loop forever
                break
        raise RuntimeError("the engine never acknowledged the search-state reset")

    # -- query -----------------------------------------------------------------
    def label(self, fen: str, *, node_budget: int, pv_plies: int = 8, cold: bool = True) -> OracleLabel:
        """One search at ``go nodes <node_budget>``; the realized work is recorded.

        ``cold=True`` (the default, and the only mode any S13 label was bought in) clears the
        search state first, so the label is a function of (pinned engine, options, position,
        budget) alone.
        """
        if self.identity is None:
            self._start()
        if cold:
            self._reset_search_state()
        self._send(f"position fen {fen}")
        wall_start = time.perf_counter()          # perf_counter: time.time() is ~15ms-grained on Windows
        self._send(f"go nodes {int(node_budget)}")
        depth = nodes = time_ms = None
        cp: Optional[int] = None
        mate: Optional[int] = None
        pv: tuple[str, ...] = ()
        best_move: Optional[str] = None
        for line in self._lines():
            if line.startswith("bestmove"):
                parts = line.split()
                best_move = parts[1] if len(parts) > 1 else None
                break
            if not line.startswith("info ") or " score " not in line:
                continue
            parts = line.split()
            for index, token in enumerate(parts):
                if token == "depth":
                    depth = int(parts[index + 1])
                elif token == "nodes":
                    nodes = int(parts[index + 1])
                elif token == "time":
                    time_ms = int(parts[index + 1])
                elif token == "score" and index + 2 < len(parts):
                    kind, value = parts[index + 1], parts[index + 2]
                    if kind == "cp":
                        cp, mate = int(value), None
                    elif kind == "mate":
                        mate, cp = int(value), mate_score_cp(int(value))
                elif token == "pv":
                    pv = tuple(parts[index + 1:index + 1 + pv_plies])
        return OracleLabel(score_cp_stm=cp, mate=mate, best_move=best_move, pv=pv,
                           reached_depth=int(depth or 0), nodes=int(nodes or 0),
                           wall_ms=(time.perf_counter() - wall_start) * 1000.0)


class StockfishProvider:
    """Search provider over a pinned transport; also the label-row writer.

    ``authority`` is the provenance class the labels are registered under (never ``legacy.cvs.*``),
    and ``producer`` is the executed binary's SHA256 — the same value the TargetSpec pins.
    """

    def __init__(self, transport: StockfishTransport, *, authority: str, family: str = "oracle_cp"):
        self.transport = transport
        self.authority = authority
        self.family = family

    @property
    def producer(self) -> str:
        if self.transport.identity is None:
            self.transport._start()
        return self.transport.identity.binary_sha256

    @property
    def identity(self) -> StockfishIdentity:
        if self.transport.identity is None:
            self.transport._start()
        assert self.transport.identity
        return self.transport.identity

    def search(self, position: Position, *, node_budget: int, pv_plies: int = 8,
               cold: bool = True) -> OracleLabel:
        return self.transport.label(position.fen, node_budget=node_budget, pv_plies=pv_plies,
                                    cold=cold)

    def label_row(self, record_id: str, label: OracleLabel, *, node_budget: int,
                  cold: bool = True) -> dict:
        """The row `register_labels` stores: value + the budget contract of THIS teacher."""
        return {"record_id": record_id,
                "value": {"scoreCpStm": label.score_cp_stm, "mate": label.mate,
                          "bestMove": label.best_move, "pv": list(label.pv),
                          "depth": label.reached_depth, "nodes": label.nodes},
                "budget": {"nodes": int(node_budget)},
                "note": ("cold per label: ucinewgame + Clear Hash + isready before every search"
                         if cold else "WARM search state reused across labels")}
