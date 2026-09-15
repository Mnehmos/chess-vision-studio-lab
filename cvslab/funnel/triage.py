"""priority-v1 triage, ported verbatim from the frozen legacy funnel (S3, #15).

Source: `chess-vision-studio-rust-engine/training/funnel/labeling_funnel.py`
(`stable_hash`, `unit_hash`, `clamp_cp`, `coverage_counts`, `priority_components`,
`score_priority`, `select`, `stage_triage` inputs). This is a **parity port**: the
policy weights/caps/fractions are read from a content-addressed policy document and
never improved here. Inputs are the Lab's own S2 evidence (facts/motif/strategy
labels, per-budget search_shallow_cp labels, outcome labels), with the tier1
`derived` block recomputed exactly as `_tier1_job` did.

Known legacy ambiguities (reported in the S3 PR):
* `gameOutcome` in `positions.jsonl` is `{"whiteScore": float}` (or null), not a
  scalar; the S2 importer's scalar assumption is corrected alongside this port.
* `rank` is defined only for non-holdout positions; holdout rows rank as n+1 and
  sort after the ranked pool by id.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Optional

from ..store import LabError, Store

CP_CLAMP = 2000
STATUS_INSTABILITY = {
    "stable-at-budget": 0.0,
    "exact-tablebase": 0.0,
    "verified-forced-mate": 0.0,
    "unresolved-at-budget": 0.75,
    "omission-risk": 0.75,
    "verifier-conflict": 1.0,
    "unstable-trajectory": 1.0,
}


def stable_hash(*parts) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


def unit_hash(*parts) -> float:
    return stable_hash(*parts) / float(1 << 64)


def clamp_cp(value) -> int:
    if value is None:
        return 0
    return max(-CP_CLAMP, min(CP_CLAMP, int(value)))


def common_prefix(left: list, right: list) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def white_pov(fen: str, stm_cp: Optional[int]) -> Optional[int]:
    if stm_cp is None:
        return None
    return stm_cp if fen.split()[1] == "w" else -stm_cp


def coverage_counts(tier0_records) -> Counter:
    """Verbatim port: slugs, family:, phase:, material:, __positions__ over ok records."""
    counts: Counter = Counter()
    for record in tier0_records:
        if record.get("status") != "ok":
            continue
        for slug in record["taxonomy"]["slugs"]:
            counts[slug] += 1
        for family in record["taxonomy"]["families"]:
            counts["family:" + family] += 1
        geometry = record["deterministic_geometry"]
        counts["phase:" + geometry["phase"]] += 1
        counts["material:" + geometry["materialBucket"]] += 1
        counts["__positions__"] += 1
    return counts


def priority_components(tier0: dict, tier1: dict, outcome, counts: Counter, policy: dict) -> dict:
    """Verbatim port of `priority_components` (the policy supplies caps + coverage prior)."""
    caps = policy["caps"]
    search = tier1["search_derived"]
    hi, lo, derived = search["budgets"][-1], search["budgets"][0], search["derived"]
    geometry = tier0["deterministic_geometry"]
    tactics = tier0["bounded_tactical_proof"]

    coverage = policy["coverage_prior"]
    total = counts.get("__positions__", 0)
    target = max(coverage["min_target"], coverage["target_share"] * total)
    slug_rarity = max((min(1.0, target / max(1, counts.get(slug, 0)))
                       for slug in tier0["taxonomy"]["slugs"]), default=0.0)
    bucket_rarity = min(1.0, target / max(1, counts.get("material:" + geometry["materialBucket"], 0)))

    shorter = min(len(lo["pv"]), len(hi["pv"]))
    pv_disagreement = 1.0 - derived["pvCommonPrefix"] / shorter if shorter else 0.0

    outcome_value = 0.0
    if outcome is not None and derived.get("scoreCpWhite") is not None:
        score = clamp_cp(derived["scoreCpWhite"])
        against = -score if outcome == 1 else score if outcome == 0 else abs(score) * 0.5
        outcome_value = min(1.0, max(0.0, against - caps["outcomeMarginCp"]) / caps["outcomeScaleCp"])

    stabilization = hi["stabilization"] or {}
    selectivity = min(1.0, (stabilization.get("seePruneSkips") or 0) / caps["seePruneSkips"])
    if "partial-iteration-move-shift" in (stabilization.get("reasons") or []):
        selectivity = 1.0

    return {
        "scoreInstability": min(1.0, abs(derived["scoreDeltaCp"]) / caps["scoreDeltaCp"]),
        "bestMoveChange": 1.0 if derived["bestMoveChanged"] else 0.0,
        "trajectoryUnstable": STATUS_INSTABILITY.get(stabilization.get("status"), 0.5),
        "pvDisagreement": pv_disagreement,
        "tacticalDensity": min(1.0, tactics["tacticalItems"] / caps["tacticalItems"]),
        "rarity": max(slug_rarity, 0.5 * bucket_rarity),
        "outcomeDisagreement": outcome_value,
        "selectivityEdge": selectivity,
        "forcedness": 1.0 if geometry["inCheck"] or geometry["legalMoves"] <= caps["forcedLegalMoves"] else 0.0,
    }


def score_priority(components: dict, weights: dict) -> tuple[float, dict, list[str]]:
    """Verbatim port: rounded contributions, weighted mean, top-4 reasons by contribution."""
    weight_sum = sum(weights.values()) or 1.0
    detail = {}
    for name, weight in sorted(weights.items()):
        value = components.get(name, 0.0)
        detail[name] = {"value": round(value, 6), "weight": weight,
                        "contribution": round(weight * value / weight_sum, 6)}
    total = sum(entry["contribution"] for entry in detail.values())
    reasons = [name for name, entry in sorted(detail.items(),
                                              key=lambda item: (-item[1]["contribution"], item[0]))
               if entry["contribution"] > 0][:4]
    return round(total, 6), detail, reasons


def select(items: list[dict], policy: dict, *, stockfish: Optional[dict] = None) -> list[dict]:
    """Verbatim port of `select` (holdout is a pure function of (holdout_seed, id))."""
    selection = policy["selection"]
    tier4 = stockfish if stockfish is not None else policy["stockfish"]
    n = len(items)
    holdout_seed = selection["holdout_seed"]
    holdout = {item["id"] for item in items
               if unit_hash(holdout_seed, "holdout", item["id"]) < selection["holdout_fraction"]}
    pool = [item for item in items if item["id"] not in holdout]
    ranked = sorted(pool, key=lambda item: (-item["priority"], stable_hash(policy["seed"], "tie", item["id"])))
    k_deep = round(selection["deep_fraction"] * n)
    deep = [item["id"] for item in ranked[:k_deep]]
    remainder = ranked[k_deep:]
    audit = [item["id"] for item in sorted(remainder, key=lambda item: stable_hash(policy["seed"], "audit", item["id"]))
             [:round(selection["audit_fraction"] * n)]]
    uniform = [item["id"] for item in sorted(pool, key=lambda item: stable_hash(policy["seed"], "uniform", item["id"]))
               [:k_deep]]

    stockfish_tags: dict[str, list[str]] = defaultdict(list)
    if tier4.get("enabled", True):
        for tag, ids, fraction in (("priority", deep, tier4["priority_fraction"]),
                                   ("uniform", uniform, tier4["uniform_fraction"]),
                                   ("audit", audit, tier4["audit_fraction"])):
            for position_id in ids[:round(fraction * n)]:
                stockfish_tags[position_id].append(tag)
        for position_id in sorted(holdout):
            stockfish_tags[position_id].append("holdout")

    rank = {item["id"]: index for index, item in enumerate(ranked, 1)}
    deep_set, audit_set, uniform_set = set(deep), set(audit), set(uniform)
    out = []
    for item in sorted(items, key=lambda item: (rank.get(item["id"], n + 1), item["id"])):
        position_id = item["id"]
        out.append({
            "id": position_id, "priority": item["priority"], "rank": rank.get(position_id),
            "components": item["components"], "reasons": item["reasons"],
            "selection": {"deep": position_id in deep_set, "uniform": position_id in uniform_set,
                          "audit": position_id in audit_set, "holdout": position_id in holdout,
                          "stockfish": stockfish_tags.get(position_id, [])},
            "trainEligible": position_id not in holdout,
        })
    return out


# ---------------------------------------------------------------------------
# Building triage inputs from S2 lab evidence
# ---------------------------------------------------------------------------


def _label_value(record: dict, family: str):
    for label in record["labels"]:
        if label["family"] == family:
            return label["value"]
    return None


def _tier1_view(record: dict) -> Optional[dict]:
    """Rebuild the legacy tier1 record shape from per-budget search_shallow_cp labels."""
    budgets = [dict(label["value"], nodeBudget=label["budget"]["nodeBudget"])
               for label in record["labels"] if label["family"] == "search_shallow_cp"]
    if not budgets:
        return None
    budgets.sort(key=lambda run: run["nodeBudget"])
    low, high = budgets[0], budgets[-1]
    fen = record["fen"]
    return {"search_derived": {
        "budgets": budgets,
        "derived": {
            "scoreDeltaCp": clamp_cp(high["scoreCpStm"]) - clamp_cp(low["scoreCpStm"]),
            "bestMoveChanged": low["bestMove"] != high["bestMove"],
            "pvCommonPrefix": common_prefix(low["pv"], high["pv"]),
            "scoreCpWhite": white_pov(fen, high["scoreCpStm"]),
        },
    }}


def _tier0_view(record: dict) -> Optional[dict]:
    facts = _label_value(record, "facts")
    motif = _label_value(record, "motif")
    if facts is None or motif is None:
        return None
    kind_counts = motif.get("kindCounts") or {}
    hazards = motif.get("hazards") or {}
    return {
        "status": "ok",
        "deterministic_geometry": facts,
        "bounded_tactical_proof": {
            "tacticalItems": sum(kind_counts.values()) + sum(hazards.values()),
        },
        "taxonomy": {"slugs": motif.get("slugs") or [], "families": motif.get("families") or []},
    }


def triage_corpus(store: Store, normalization_id: str, policy: dict, *, coverage_prior: Optional[Counter] = None):
    """Compute priority-v1 triage for a canonical corpus from its S2 evidence.

    Returns (rows_by_funnel_id, counts) where counts is the coverage Counter used for rarity.
    """
    from ..data import _load_canonical, read_jsonl
    from ..schemas import Normalization, SourceSnapshot

    normalization: Normalization = store.get_as(normalization_id, Normalization)
    records = _load_canonical(store, normalization)
    # funnel ids live in the S#### pool snapshots the normalization was built from
    pool_rows: list[dict] = []
    for source_id in normalization.source_ids:
        source: SourceSnapshot = store.get_as(source_id, SourceSnapshot)
        pool_rows.extend(read_jsonl(store.abs(source.path)))
    funnel_id_by_row = {index: row.get("funnel_id") for index, row in enumerate(pool_rows)}

    tier0_records = {record["record_id"]: _tier0_view(record) for record in records}
    tier0_records = {key: value for key, value in tier0_records.items() if value is not None}
    counts = coverage_counts(tier0_records.values())
    if coverage_prior:
        counts.update(coverage_prior)

    items = []
    for record in records:
        tier0, tier1 = tier0_records.get(record["record_id"]), _tier1_view(record)
        if tier0 is None or tier1 is None:
            continue
        outcome = _label_value(record, "outcome")
        components = priority_components(tier0, tier1, outcome, counts, policy)
        priority, detail, reasons = score_priority(components, policy["weights"])
        items.append({"id": funnel_id_by_row.get(record["source_row"], record["record_id"]),
                      "priority": priority, "components": detail, "reasons": reasons})
    return select(items, policy), counts


def compare_triage(lab_rows: list[dict], legacy_rows: list[dict]) -> dict:
    """Per-field parity between lab triage and legacy triage.jsonl (S3 gate).

    Fields compared: priority, rank, reasons, components, trainEligible, selection.
    Returns {field: {"mismatches": n, "examples": [...]}} plus ids missing on either side.
    """
    lab = {row["id"]: row for row in lab_rows}
    legacy = {row["id"]: row for row in legacy_rows}
    only_lab = sorted(set(lab) - set(legacy))
    only_legacy = sorted(set(legacy) - set(lab))
    fields = ("priority", "rank", "reasons", "components", "trainEligible", "selection")
    report = {field: {"mismatches": 0, "examples": []} for field in fields}
    for position_id in sorted(set(lab) & set(legacy)):
        for field in fields:
            if lab[position_id].get(field) != legacy[position_id].get(field):
                report[field]["mismatches"] += 1
                if len(report[field]["examples"]) < 3:
                    report[field]["examples"].append(
                        {"id": position_id, "lab": lab[position_id].get(field), "legacy": legacy[position_id].get(field)})
    report["only_lab"] = only_lab
    report["only_legacy"] = only_legacy
    report["compared"] = len(set(lab) & set(legacy))
    report["ok"] = (not only_lab and not only_legacy
                    and all(report[field]["mismatches"] == 0 for field in fields))
    return report
