"""Faithful port of the legacy funnel's Tier-0 taxonomy mapping.

Source: `chess-vision-studio-rust-engine/training/funnel/labeling_funnel.py`
(PIN_KIND, KIND_FIRST, COLLECTION_DEFAULT, PAWN_SLUG, load_taxonomy,
resolve_slug, _items, summarize_facts). Key semantics preserved verbatim:

* opportunity sides are who can EXECUTE ("stm"/"opp"), structure sides are who
  OWNS the structure;
* a motif that resolves to no taxonomy slug becomes ``unmapped:<collection>:<kind>``
  and is reported, never silently dropped;
* ``uncomputed`` (present but not computed) is kept separate from computed-empty;
* hazard ``side`` is the threatened side; the beneficiary can execute mate-threat.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from .providers import sha256_file

PIN_KIND = {"absolute": "absolute-pin", "relative": "relative-pin"}
KIND_FIRST = {"Motifs", "Discoveries", "MatePatterns"}
COLLECTION_DEFAULT = {
    "Skewers": "skewer",
    "Discoveries": "discovered-attack",
    "DiscoveredDefense": "discovered-defense",
    "RemoveGuard": "capturing-defender",
    "Trapped": "trapped-piece",
    "Desperado": "desperado",
    "Overload": "overloading",
    "AttackDefender": "attacking-the-defender",
    "Deflection": "distraction",
    "LureDefender": "luring-the-defender",
    "Interference": "interference",
    "DoubleAttack": "double-attack",
    "XrayAttack": "xray-attack",
    "XrayDefense": "xray-defense",
    "WinExchange": "win-the-exchange",
}
PAWN_SLUG = {"doubled": "doubled-pawns", "isolated": "isolated-pawn", "passed": "passed-pawn"}


def load_taxonomy(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "schemaVersion": data.get("schemaVersion"),
        "sha256": sha256_file(path),
        "family": {motif["slug"]: motif.get("family") for motif in data["motifs"]},
    }


def resolve_slug(collection: str, kind: Optional[str], slugs) -> str:
    if collection == "Pins" and kind in PIN_KIND:
        return PIN_KIND[kind]
    default = COLLECTION_DEFAULT.get(collection)
    candidates = []
    if kind:
        k = kind.replace("_", "-")
        candidates = [k, k + "-mate"]
    order = candidates + [default] if collection in KIND_FIRST else [default] + candidates
    for candidate in order:
        if candidate and candidate in slugs:
            return candidate
    return f"unmapped:{collection}:{kind}"


def _items(collection):
    if isinstance(collection, dict) and collection.get("status") == "computed":
        return collection.get("items") or []
    return None


def summarize_facts(bundle: dict, families: dict) -> dict:
    """TeachingFactBundleV1 ``before`` block -> compact Tier-0 fields (verbatim port)."""
    before = bundle["before"]
    stm = before["sideToMove"]
    owner = lambda side: "stm" if side == stm else "opp"  # noqa: E731
    other = {"stm": "opp", "opp": "stm"}

    uncomputed: list[str] = []
    opportunities = {"stm": set(), "opp": set()}
    structures = {"stm": set(), "opp": set()}
    kind_counts: Counter = Counter()
    captures = {"stm": {"count": 0, "seePositive": 0}, "opp": {"count": 0, "seePositive": 0}}

    for key, coll in before.items():
        if key.startswith("opponentAvailable"):
            side, name = "opp", key[len("opponentAvailable"):]
        elif key.startswith("available"):
            side, name = "stm", key[len("available"):]
        else:
            continue
        items = _items(coll)
        if items is None:
            uncomputed.append(key)
            continue
        if name == "Captures":
            captures[side]["count"] += len(items)
            captures[side]["seePositive"] += sum(1 for item in items if (item.get("seeCp") or 0) > 0)
            continue
        for item in items:
            kind = item.get("kind") if isinstance(item, dict) else None
            kind_counts[f"{side}:{name}:{kind}"] += 1
            opportunities[side].add(resolve_slug(name, kind, families))

    pieces = {"stm": 0, "opp": 0}
    attacked = {"stm": 0, "opp": 0}
    loose = {"stm": 0, "opp": 0}
    hanging = {"stm": 0, "opp": 0}
    for pc in before.get("pieces") or []:
        o = owner(pc.get("side"))
        pieces[o] += 1
        attacked[o] += bool(pc.get("attacked"))
        loose[o] += bool(pc.get("loose"))
        if pc.get("pieceType") != "king" and pc.get("attacked") and pc.get("loose"):
            hanging[o] += 1
            opportunities[other[o]].add("hanging-piece")

    pawns: dict = {}
    pawn_structure = before.get("pawnStructure") or {}
    for key, slug in PAWN_SLUG.items():
        value = pawn_structure.get(key)
        if isinstance(value, list):
            counts = {"stm": 0, "opp": 0}
            for item in value:
                if isinstance(item, dict) and item.get("side"):
                    counts[owner(item["side"])] += 1
                    structures[owner(item["side"])].add(slug)
            pawns[key] = counts
        else:
            uncomputed.append(f"pawnStructure.{key}")
    islands = pawn_structure.get("islands")
    if isinstance(islands, list):
        pawns["islands"] = {side: sum(1 for item in islands if owner(item.get("side")) == side)
                            for side in ("stm", "opp")}

    king = {}
    for item in _items(before.get("kingSafety")) or []:
        escapes = _items(item.get("legalEscapeSquares"))
        king[owner(item.get("side"))] = {
            "inCheck": bool(item.get("inCheck")),
            "attackers": len(item.get("attackers") or []),
            "pressuredSquares": len(item.get("pressuredSquares") or []),
            "escapeSquares": None if escapes is None else len(escapes),
        }

    control = {"stm": 0, "opp": 0}
    squares = _items(before.get("squareFacts"))
    if squares is None:
        uncomputed.append("squareFacts")
    else:
        white = "stm" if stm == "white" else "opp"
        black = other[white]
        for square in squares:
            control[white] += bool(square.get("controlledByWhite"))
            control[black] += bool(square.get("controlledByBlack"))

    hazards: Counter = Counter()
    hazard_items = _items(before.get("hazards"))
    if hazard_items is None:
        uncomputed.append("hazards")
    else:
        for item in hazard_items:
            hazards[item.get("kind")] += 1
            if item.get("kind") == "mate_threat" and item.get("side"):
                opportunities[other[owner(item["side"])]].add("mate-threat")

    all_slugs = sorted(opportunities["stm"] | opportunities["opp"]
                       | structures["stm"] | structures["opp"])
    return {
        "deterministic_geometry": {
            "pieces": pieces, "attacked": attacked, "loose": loose,
            "pawns": pawns, "kingSafety": king, "squareControl": control,
            "structures": {side: sorted(values) for side, values in structures.items()},
        },
        "bounded_tactical_proof": {
            "opportunities": {side: sorted(values) for side, values in opportunities.items()},
            "kindCounts": dict(sorted(kind_counts.items())),
            "captures": captures,
            "hanging": hanging,
            "hazards": dict(sorted(hazards.items())),
            "tacticalItems": sum(kind_counts.values()) + sum(hazards.values()),
        },
        "taxonomy": {
            "slugs": [slug for slug in all_slugs if not slug.startswith("unmapped:")],
            "families": sorted({families[slug] for slug in all_slugs
                                if slug in families and families[slug]}),
            "unmapped": [slug for slug in all_slugs if slug.startswith("unmapped:")],
        },
        "uncomputed": sorted(uncomputed),
        "factsErrors": len(bundle.get("errors") or []),
        "factsRegistryVersion": (bundle.get("provenance") or {}).get("factsRegistryVersion"),
    }
