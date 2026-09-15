"""Import a legacy funnel run's per-position evidence as canonical lab evidence (S2, #9).

Mapping (docs/TRAINING_DATA_RESEARCH_PLAN.md §7):

* candidate pool                -> immutable `S####` snapshot (funnel ids + source refs preserved;
  game grouping inferred from FEN move counters by the normalizer, so later splits stay leakage-safe);
* canonical positions           -> `N####` with EPD identity and exact-EPD dedup;
* tier0 deterministic_geometry  -> `facts` labels (authority legacy.cvs.tier0.facts);
* tier0 tactics/taxonomy        -> `motif` labels (slugs/opportunities/hazards; used for coverage parity);
* tier0 structures/families     -> `strategy` labels;
* tier1 per-budget runs         -> `search_shallow_cp` labels, one row per position, budget per node budget;
* tier3 deep + targets          -> `search_deep_cp` labels (targets round-trip exactly);
* tier4 Stockfish               -> `oracle_cp` labels (authority external.stockfish);
* game outcome                  -> `outcome` labels (own authority, white POV);
* triage priority               -> `priority` labels; per-record component contributions go in the
  dedicated `components` field (never `budget`). The priority-v1 *policy identity* belongs to S3/#15 —
  this importer only records the version string the run declares.

Provenance rules: full analyze/SF sha256 identities from the run manifest are used as label
`producer`; every tier is a separate append-only label set, so multiple authorities and budgets
coexist on one canonical position. Nothing here creates a dataset — only evidence.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from ..data import normalize, register_labels, write_jsonl
from ..schemas import SourceSnapshot, utc_now
from ..store import LabError, Store

IMPORTER = "cvslab.funnel.import"
IMPORTER_VERSION = 1
OUTCOME_AUTHORITY = "outcome.game_result.v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _optional(directory: Path, name: str) -> list[dict]:
    path = directory / name
    return _read_jsonl(path) if path.is_file() else []


def _row_outcome(row: dict):
    """gameOutcome (or res) with falsy values preserved: 0.0 is a real loss, not missing."""
    outcome = row.get("gameOutcome")
    return row.get("res") if outcome is None else outcome


def _row_game(row: dict) -> tuple[Optional[str], bool]:
    """(explicit game identity or None, was explicit). The legacy sample stage stores only
    {file, line} and hash-ranks positions, so game identity often does not exist in a run.
    We never manufacture it: missing identity is marked unknown (`game_unknown`), the
    source_ref is retained for later recovery, and dataset freezing refuses leakage-safe
    splits over unknown-grouping positions unless the caller explicitly opts into a single
    shared group (safe, at the cost of balanced splits)."""
    explicit = row.get("game") or (row.get("source") or {}).get("game")
    return (str(explicit), True) if explicit else (None, False)


def import_funnel_run_evidence(store: Store, run_dir: str | Path, *, name: Optional[str] = None,
                               license: str = "legacy engine research corpus",
                               link_funnel_run: bool = True) -> dict:
    """Import one funnel run directory. Deterministic; call twice for identical counts."""
    directory = Path(run_dir)
    if not (directory / "positions.jsonl").is_file():
        raise LabError(f"{directory} has no positions.jsonl; not a funnel run directory")
    positions = _read_jsonl(directory / "positions.jsonl")
    tier0 = {row["id"]: row for row in _optional(directory, "tier0.jsonl")}
    tier1 = {row["id"]: row for row in _optional(directory, "tier1.jsonl")}
    tier3 = {row["id"]: row for row in _optional(directory, "tier3.jsonl")}
    tier4 = {row["id"]: row for row in _optional(directory, "tier4.jsonl")}
    triage = {row["id"]: row for row in _optional(directory, "triage.jsonl")}
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8")) \
        if (directory / "manifest.json").is_file() else {}
    if not positions:
        raise LabError(f"{directory} has no positions.jsonl rows")

    analyze_sha = str((manifest.get("engine") or {}).get("binarySha256") or "")
    sf_sha = str((manifest.get("stockfish") or {}).get("binarySha256") or "")
    registry_version = int(next((r.get("factsRegistryVersion") or 0 for r in tier0.values()), 0) or 0)

    # -- L0: candidate pool ------------------------------------------------------
    grouped = [_row_game(row) for row in positions]
    game_grouping = "explicit" if all(explicit for _game, explicit in grouped) else         "unknown (run carries no game identity; source_ref retained for recovery; "         "leakage-safe freezing refused until grouping is repaired or shared-group mode is chosen)"
    pool_rows = []
    for row, (game, explicit) in zip(positions, grouped):
        pool_row = {"fen": row["fen"], "res": _row_outcome(row),
                    "funnel_id": row["id"], "source_ref": json.dumps(row.get("source"))}
        if explicit:
            pool_row["game"] = game
        else:
            pool_row["game_unknown"] = True
        pool_rows.append(pool_row)
    source_id = store.next_id("S")
    rel = f"sources/{source_id}/positions.jsonl"
    content_hash = write_jsonl(store.abs(rel), pool_rows)
    store.make_readonly(store.abs(rel))
    from ..data import infer_games
    source = store.create(SourceSnapshot(
        id=source_id, name=name or f"funnel-pool {directory.name}", source_origin=str(directory.resolve()),
        locality="local",
        generator={"name": "legacy-funnel-run", "run_dir": str(directory.resolve()),
                   "configSha256": str(manifest.get("configSha256") or ""),
                   "prioritizerVersion": str(manifest.get("prioritizerVersion") or ""),
                   "analyzeSha256": analyze_sha, "positions": len(pool_rows),
                   "game_grouping": game_grouping},
        license=license, importer=IMPORTER, importer_version=IMPORTER_VERSION, path=rel,
        content_hash=content_hash, row_count=len(pool_rows),
        game_count=len(set(infer_games(pool_rows))), label_authorities=[OUTCOME_AUTHORITY],
        compute={}, created_at=utc_now()))

    # -- L1: canonical records ---------------------------------------------------
    normalization = normalize(store, [source.id], name=f"{source.id}-canonical")

    def record_ids_by_row() -> dict[int, str]:
        from ..data import read_jsonl
        return {row["source_row"]: row["record_id"] for row in read_jsonl(store.abs(normalization.path))}

    by_row = record_ids_by_row()
    kept = len(by_row)
    dropped_by_dedup = len(pool_rows) - kept

    def fid(record: dict) -> str:
        return str(record["record_id"])

    def lookup(row_index: int) -> Optional[str]:
        return by_row.get(row_index)

    keyed: dict[str, dict] = {}
    for index, row in enumerate(positions):
        record_id = lookup(index)
        if record_id:
            keyed[record_id] = row

    sets: dict[str, str] = {}

    def add_set(family: str, authority: str, producer: str, rows: list[dict], **kwargs) -> Optional[str]:
        if not rows:
            return None
        ref = register_labels(store, normalization.id, family=family, producer=producer,
                              authority=authority, rows=rows, **kwargs)
        sets[f"{authority}:{family}"] = ref.path
        return ref.path

    # tier0 -> facts / motif / strategy
    facts_rows, motif_rows, strategy_rows = [], [], []
    for record_id, row in keyed.items():
        t0 = tier0.get(row["id"])
        if not t0 or t0.get("status") != "ok":
            continue
        geometry = t0.get("deterministic_geometry") or {}
        tactics = t0.get("bounded_tactical_proof") or {}
        taxonomy = t0.get("taxonomy") or {}
        note = f"registry {t0.get('factsRegistryVersion')}"
        facts_rows.append({"record_id": record_id, "value": geometry, "note": note,
                           "registry_version": int(t0.get("factsRegistryVersion") or 0)})
        motif_rows.append({"record_id": record_id, "value": {
            "slugs": taxonomy.get("slugs") or [], "families": taxonomy.get("families") or [],
            "opportunities": tactics.get("opportunities") or {}, "hazards": tactics.get("hazards") or {},
            "kindCounts": tactics.get("kindCounts") or {}}, "note": note})
        strategy_rows.append({"record_id": record_id, "value": {
            "structures": geometry.get("structures") or {},
            "unmapped": taxonomy.get("unmapped") or [], "uncomputed": t0.get("uncomputed") or []}})
    add_set("facts", "legacy.cvs.tier0.facts", analyze_sha, facts_rows, registry_version=registry_version)
    add_set("motif", "legacy.cvs.tier0.tactics", analyze_sha, motif_rows, registry_version=registry_version)
    add_set("strategy", "legacy.cvs.tier0.structures", analyze_sha, strategy_rows, registry_version=registry_version)

    # tier1 -> search_shallow_cp (budget per node budget, full compact payload preserved)
    shallow_rows = []
    for record_id, row in keyed.items():
        t1 = tier1.get(row["id"])
        if not t1:
            continue
        for run in (t1.get("search_derived") or {}).get("budgets") or []:
            shallow_rows.append({"record_id": record_id,
                                 "value": {k: v for k, v in run.items() if k != "cost"},
                                 "budget": {"nodeBudget": run.get("nodeBudget"), "nodes": run.get("nodes")},
                                 "note": f"budget {run.get('nodeBudget')} nodes"})
    add_set("search_shallow_cp", "legacy.cvs.search.shallow", analyze_sha, shallow_rows, pov="stm")

    # tier3 -> search_deep_cp with exact targets
    deep_rows = []
    for record_id, row in keyed.items():
        t3 = tier3.get(row["id"])
        if not t3:
            continue
        derived = t3.get("search_derived") or {}
        deep_rows.append({"record_id": record_id,
                          "value": {"targets": derived.get("targets") or {}, "deep": derived.get("deep") or {},
                                    "shallowToDeep": derived.get("shallowToDeep") or {}},
                          "budget": {"nodeBudget": (derived.get("profile") or {}).get("nodeBudget")},
                          "note": "tier3 deep target"})
    add_set("search_deep_cp", "legacy.cvs.search.deep", analyze_sha, deep_rows, pov="stm")

    # tier4 -> oracle_cp
    oracle_rows = [{"record_id": record_id, "value": {k: v for k, v in (tier4[row["id"]] or {}).items()
                                                       if k not in ("cost", "stage", "schemaVersion", "id")},
                    "note": "stockfish oracle"}
                   for record_id, row in keyed.items() if row["id"] in tier4]
    add_set("oracle_cp", "external.stockfish", sf_sha or "stockfish", oracle_rows, pov="stm")

    # outcome (own authority, white POV)
    outcome_rows = [{"record_id": record_id, "value": float(_row_outcome(row)),
                     "note": "funnel gameOutcome"} for record_id, row in keyed.items()
                    if _row_outcome(row) is not None]
    add_set("outcome", OUTCOME_AUTHORITY, "legacy.funnel.sample", outcome_rows)

    # triage -> priority (per-record evidence only; policy identity belongs to S3/#15)
    policy_version = str(manifest.get("prioritizerVersion") or "unknown")
    priority_rows = []
    for record_id, row in keyed.items():
        t2 = triage.get(row["id"])
        if not t2:
            continue
        note = json.dumps({"reasons": t2.get("reasons") or [], "selection": t2.get("selection") or {},
                           "trainEligible": t2.get("trainEligible"), "rank": t2.get("rank")},
                          sort_keys=True)
        priority_rows.append({"record_id": record_id, "value": float(t2["priority"]),
                              "components": t2.get("components") or {}, "note": note})
    add_set("priority", f"triage.{policy_version}", "legacy.funnel.triage", priority_rows)

    counts = {
        "source": source.id, "normalization": normalization.id,
        "positions": len(positions), "canonical_records": kept, "dedup_dropped": dropped_by_dedup,
        "label_rows": {"facts": len(facts_rows), "motif": len(motif_rows), "strategy": len(strategy_rows),
                       "search_shallow_cp": len(shallow_rows), "search_deep_cp": len(deep_rows),
                       "oracle_cp": len(oracle_rows), "outcome": len(outcome_rows),
                       "priority": len(priority_rows)},
        "authorities": {ref.split(":")[0] for ref in sets},
        "policy_version": policy_version,
        "analyze_sha256": analyze_sha,
    }

    if link_funnel_run:
        from .. import intake
        try:
            funnel_run = intake.import_funnel_run(store, directory, name=directory.name,
                                                  position_pool=source.id)
            counts["funnel_run"] = funnel_run.id
        except Exception as exc:  # already imported (sealed) — report instead of silently skipping
            existing = next((r.id for r in store.list("R", verify=False, kind=intake.FunnelRun)
                             if any(f.name == "manifest.json" and
                                    f.sha256 == intake._hex_sha256(directory / "manifest.json")
                                    for f in r.files)), None)
            counts["funnel_run"] = existing
            counts["funnel_run_note"] = (f"existing FunnelRun {existing}; pool linkage "
                                         f"{source.id} recorded on this import") if existing else str(exc)
    return counts


def compare_coverage(store: Store, normalization_id: str, run_dir: str | Path) -> dict:
    """Lab motif-slug counts vs the run's coverage.json (acceptance gate #1)."""
    from ..data import _load_canonical
    from ..schemas import Normalization

    coverage_path = Path(run_dir) / "coverage.json"
    if not coverage_path.is_file():
        raise LabError(f"{coverage_path} not found")
    source_counts = json.loads(coverage_path.read_text(encoding="utf-8"))["counts"]
    normalization: Normalization = store.get_as(normalization_id, Normalization)
    lab_counts: Counter = Counter()
    positions_seen = 0
    for record in _load_canonical(store, normalization):
        positions_seen += 1
        for label in record["labels"]:
            if label["family"] != "motif":
                continue
            value = label["value"] or {}
            for slug in value.get("slugs") or []:
                lab_counts[slug] += 1
            for family in value.get("families") or []:
                lab_counts[f"family:{family}"] += 1
        for label in record["labels"]:
            if label["family"] == "facts":
                geometry = label["value"] or {}
                lab_counts[f"phase:{geometry.get('phase')}"] += 1
                lab_counts[f"material:{geometry.get('materialBucket')}"] += 1
    lab_counts["__positions__"] = positions_seen
    compared = [key for key in source_counts if key != "__positions__"]
    mismatches = {key: {"lab": lab_counts.get(key, 0), "source": source_counts[key]}
                  for key in compared if lab_counts.get(key, 0) != source_counts[key]}
    return {"positions": {"lab": positions_seen, "source": source_counts.get("__positions__")},
            "keys_compared": len(compared), "mismatches": mismatches, "ok": not mismatches}
