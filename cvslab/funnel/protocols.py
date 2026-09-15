"""Provider contracts for the lab funnel (plan §6).

Providers are thin adapters around one analysis capability each. They return
evidence — never datasets — and every returned row carries provenance: the
provenance class, the producing identity (binary/provider hash), and measured
cost. Position identity, dedup, standardization and dataset freezing are the
lab's job, not a provider's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol, runtime_checkable

# Provenance classes a provider may declare. Only the two deterministic classes
# may ever feed a static evaluator input (leak guard, leakguard.py).
PROVENANCE_CLASSES = (
    "deterministic_geometry",
    "bounded_tactical_proof",
    "search_derived",
    "external_oracle",
    "outcome",
)


@dataclass(frozen=True)
class Position:
    """One candidate position from a source. Identity is the lab's EPD, not this FEN."""

    fen: str
    game_id: str = ""
    opening_id: str = ""
    ply: int = 0
    source_ref: str = ""  # e.g. "S0001:row1234" — where the candidate came from


@dataclass(frozen=True)
class LabelBatch:
    """Deterministic label rows for one position (facts/motif/strategy families)."""

    provenance_class: str
    rows: tuple[dict, ...] = ()
    producer: str = ""
    registry_version: int = 0
    uncomputed: tuple[str, ...] = ()  # requested-but-not-computed facts, kept honest


@dataclass(frozen=True)
class SearchLabel:
    """One cold search result at a fixed node budget."""

    score_cp_stm: Optional[int]
    mate: Optional[int]
    best_move: Optional[str]
    pv: tuple[str, ...] = ()
    nodes: int = 0
    wall_ms: float = 0.0
    extra: dict = field(default_factory=dict)  # telemetry (trajectory, cutoffs, …)


@dataclass(frozen=True)
class OracleLabel:
    """One external-oracle evaluation (e.g. Stockfish)."""

    score_cp_stm: Optional[int]
    mate: Optional[int]
    best_move: Optional[str]
    pv: tuple[str, ...] = ()
    reached_depth: int = 0
    nodes: int = 0
    wall_ms: float = 0.0


@runtime_checkable
class PositionSource(Protocol):
    id: str

    def positions(self, *, count: int, seed: int, pool: dict) -> Iterator[Position]:
        """Deterministic candidate stream. Dedup/standardization is the lab's job."""


@runtime_checkable
class DeterministicLabelProducer(Protocol):
    provenance_class: str  # one of the two deterministic classes
    producer_hash: str

    def label(self, position: Position, *, options: dict) -> LabelBatch: ...


@runtime_checkable
class SearchLabelProducer(Protocol):
    provenance_class: str  # "search_derived"
    family: str  # search_shallow_cp | search_deep_cp
    producer_hash: str

    def search(self, position: Position, *, node_budget: int, pv_plies: int) -> SearchLabel: ...


@runtime_checkable
class OracleProducer(Protocol):
    provenance_class: str  # "external_oracle"
    producer_hash: str

    def evaluate(self, position: Position, *, depth: int, movetime_ms: int) -> OracleLabel: ...
