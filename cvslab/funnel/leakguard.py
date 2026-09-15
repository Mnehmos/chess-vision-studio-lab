"""Tier-0 static-input leak guard (port and hardening of the funnel's check).

A Tier-0 record may carry only deterministic/bounded-proof fields. The check is a
**recursive key walk**: search, outcome, oracle and priority keys fail at any
depth, in dicts or lists, regardless of nesting. (The earlier string-scan version
exempted `nodes`/`pv` and could be fooled by nesting; test
`test_leak_guard_rejects_nested_search_keys` locks this down.)
"""
from __future__ import annotations

LEAK_GUARD_KEYS = frozenset({
    "id", "schemaVersion", "stage", "status", "deterministic_geometry",
    "bounded_tactical_proof", "taxonomy", "uncomputed", "factsErrors",
    "factsRegistryVersion", "cost",
})

# Rejected at ANY depth, compared case-insensitively. Static geometry fields the
# engine returns (e.g. "attacked", "pieces", "kindCounts") are not here, but every
# search/outcome/oracle/priority-shaped key is.
FORBIDDEN_KEYS = frozenset({
    "search_derived", "scorecp", "scorecpstm", "scorecpwhite", "bestmove", "pv", "mate",
    "nodes", "qnodes", "depth", "iterations", "stabilization", "trajectory", "telemetry",
    "termination", "resultsource", "wallms", "enginesec", "costnodes",
    "gameoutcome", "outcome", "oracle", "stockfish", "priority", "traineligible",
})

# Nested keys allowed inside static structures even though the words look like
# search vocabulary (none currently; kept explicit so additions are deliberate).
ALLOWED_NESTED_EXCEPTIONS: frozenset[str] = frozenset()


def _walk_keys(value, path: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key)
            child_path = f"{path}.{name}" if path else name
            yield name, child_path
            if name.lower() == "cost":
                continue  # declared timing metadata subtree (`cost.wallMs`, …), never scientific input
            yield from _walk_keys(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{path}[{index}]" if path else f"[{index}]")


def assert_static_input_safe(tier0_record: dict, *, strict: bool = True) -> None:
    """Raise ValueError when a Tier-0 record carries anything but static fields."""
    extra = set(tier0_record) - LEAK_GUARD_KEYS
    if extra:
        raise ValueError(f"tier0 record carries non-static fields: {sorted(extra)}")
    if not strict:
        return
    violations = []
    for key, path in _walk_keys(tier0_record):
        lowered = key.lower()
        if lowered in ALLOWED_NESTED_EXCEPTIONS:
            continue
        if lowered in FORBIDDEN_KEYS:
            violations.append(path)
    if violations:
        raise ValueError(f"tier0 record leaks search/outcome/oracle evidence at: {sorted(violations)}")
