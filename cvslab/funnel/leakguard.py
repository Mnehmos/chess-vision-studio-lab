"""Tier-0 static-input leak guard (port of the funnel's ``assert_static_input_safe``).

A Tier-0 record may carry only deterministic/bounded-proof fields. Search,
outcome and oracle evidence must stay in other files and other label families;
this guard makes a violation loud instead of silently training a static model on
leaked search data.
"""
from __future__ import annotations

LEAK_GUARD_KEYS = frozenset({
    "id", "schemaVersion", "stage", "status", "deterministic_geometry",
    "bounded_tactical_proof", "taxonomy", "uncomputed", "factsErrors",
    "factsRegistryVersion", "cost",
})

# Keys that must never appear in a Tier-0 record even nested inside geometry.
FORBIDDEN_KEYS = frozenset({
    "search_derived", "scoreCpStm", "scoreCpWhite", "bestMove", "pv", "nodes",
    "gameOutcome", "outcome", "oracle", "stockfish", "priority",
})


def assert_static_input_safe(tier0_record: dict, *, strict: bool = True) -> None:
    """Raise ValueError when a Tier-0 record carries anything but static fields."""
    extra = set(tier0_record) - LEAK_GUARD_KEYS
    if extra:
        raise ValueError(f"tier0 record carries non-static fields: {sorted(extra)}")
    if not strict:
        return
    serialized = str(tier0_record).lower()
    hit = sorted(key for key in FORBIDDEN_KEYS if key.lower() in serialized
                 and key not in ("nodes", "pv"))  # geometry may legitimately mention these words in docs
    if hit:
        raise ValueError(f"tier0 record leaks search/outcome/oracle evidence: {hit}")
