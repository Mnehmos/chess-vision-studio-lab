"""Staged funnel orchestration: the lab's reproducible corpus-generation machine (S4, #10).

Runs `pool -> tier0 -> tier1 -> triage -> tier3 -> tier4 -> report` over injected
providers (S1 contract), with:

* **resume** — every stage skips ids already present in its output file, so an
  interrupted run continues without duplicates;
* **cost accounting** — per-stage engine-seconds/wall-ms/nodes, plus totals;
* **an immutable lab-shaped manifest** pinning every identity that mattered:
  full analyze/taxonomy sha256, the content-addressed triage policy
  (policy_version + policy_hash, #15), the *resolved* coverage-prior identity
  (`counts_hash`, never a bare path), seed, pool source hash, and the manifest's
  own content hash.

Candidate-pool *selection* (hash-ranked sampling from shards) is S5's job; this
slice orchestrates whatever pool it is given. A run directory is **evidence,
never a dataset**: importing goes through `cvslab.funnel.import_run`, and
training inputs are only ever produced by `cvslab.data.freeze_dataset`.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..hashing import canonical_json, hash_obj, sha256_file
from ..store import LabError
from .policy import build_policy, policy_hash, store_policy
from .import_run import row_outcome
from .triage import clamp_cp, common_prefix, coverage_counts, priority_components, score_priority, select, white_pov

FUNNEL_SCHEMA_VERSION = 1
CREATOR = "cvslab.funnel.orchestrator"
IDENTITY_FILE = "run-identity.json"


def position_identity(fen: str) -> str:
    """Legacy sample-stage identity: sha1 over the first four FEN fields (clocks stripped).

    Using the same formula keeps orchestrated ids interoperable with legacy runs and
    makes pool dedup identity-based rather than string-based.
    """
    digest = hashlib.sha1(" ".join(fen.split()[:4]).encode())
    return digest.hexdigest()[:16]


def expected_score(cp) -> float:
    return 1.0 / (1.0 + 10 ** (-clamp_cp(cp) / 400.0))


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(fh, row: dict) -> None:
    fh.write(json.dumps(row, sort_keys=True) + "\n")
    fh.flush()


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


@dataclass
class FunnelRunConfig:
    """Everything that shapes a generation run, pinned into the manifest."""

    seed: int
    pool_files: list[str]
    positions: int
    engine_args: list[str] = field(default_factory=list)
    include_motif_opportunities: bool = True
    tier1_budgets: list[int] = field(default_factory=lambda: [2000, 16000])
    tier1_pv_plies: int = 8
    tier3_node_budget: int = 400_000
    tier3_pv_plies: int = 12
    tier4_enabled: bool = True
    tier4_depth: int = 18
    tier4_movetime_ms: int = 3000
    report_informative_delta_cp: int = 60
    workers: int = 1
    artifact_root: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict, *, artifact_root: Optional[str] = None) -> "FunnelRunConfig":
        source = config["source"]
        tiers = config["tiers"]
        tier4 = tiers.get("tier4", {})
        return cls(
            seed=int(config["seed"]),
            pool_files=list(source.get("positionsFiles") or source.get("shards") or []),
            positions=int(source.get("positions", 0)),
            engine_args=list((config.get("engine") or {}).get("args") or []),
            include_motif_opportunities=bool(tiers.get("tier0", {}).get("includeMotifOpportunities", True)),
            tier1_budgets=[int(b) for b in tiers.get("tier1", {}).get("nodeBudgets", [2000, 16000])],
            tier1_pv_plies=int(tiers.get("tier1", {}).get("pvPlies", 8)),
            tier3_node_budget=int(tiers.get("tier3", {}).get("nodeBudget", 400_000)),
            tier3_pv_plies=int(tiers.get("tier3", {}).get("pvPlies", 12)),
            tier4_enabled=bool(tier4.get("enabled", True)),
            tier4_depth=int(tier4.get("depth", 18)),
            tier4_movetime_ms=int(tier4.get("movetimeMs", 3000)),
            report_informative_delta_cp=int((config.get("report") or {}).get("informativeDeltaCp", 60)),
            workers=int(config.get("workers", 1)),
            artifact_root=artifact_root,
        )

    def canonical(self) -> dict:
        return {
            "seed": self.seed, "pool_files": self.pool_files, "positions": self.positions,
            "engine_args": self.engine_args, "include_motif_opportunities": self.include_motif_opportunities,
            "tier1_budgets": self.tier1_budgets, "tier1_pv_plies": self.tier1_pv_plies,
            "tier3_node_budget": self.tier3_node_budget, "tier3_pv_plies": self.tier3_pv_plies,
            "tier4_enabled": self.tier4_enabled,
            "tier4_depth": self.tier4_depth,
            "tier4_movetime_ms": self.tier4_movetime_ms,
            "report_informative_delta_cp": self.report_informative_delta_cp,
        }


class FunnelOrchestrator:
    """Executes the staged funnel with resume, cost accounting and an immutable manifest."""

    def __init__(self, config: FunnelRunConfig, run_dir: str | Path, *, policy: dict,
                 facts_provider, search_provider, oracle_provider=None, positions: Optional[list[dict]] = None):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.facts_provider = facts_provider
        self.search_provider = search_provider
        self.oracle_provider = oracle_provider
        self._positions = positions
        self.costs: dict[str, dict] = {}

    # -- paths -----------------------------------------------------------------

    def path(self, name: str) -> Path:
        return self.run_dir / name

    # -- stages ----------------------------------------------------------------

    def stage_pool(self) -> list[dict]:
        """Load the candidate pool, dedup by position id (keep-first). Pool *selection*
        is S5's job; this stage just materialises and pins what it was given."""
        target = self.path("positions.jsonl")
        if target.is_file():
            return _read_jsonl(target)
        rows: list[dict] = []
        for file_name in self.config.pool_files:
            path = Path(file_name)
            if not path.is_absolute() and self.config.artifact_root:
                path = Path(self.config.artifact_root) / file_name
            rows.extend(_read_jsonl(path))
        seen: set[str] = set()
        unique = []
        for row in rows:
            # Identity-based dedup (legacy semantics): the same position reached via
            # different games or move orders is one candidate, not several.
            position_id = position_identity(row["fen"])
            if position_id in seen:
                continue
            seen.add(position_id)
            unique.append({"id": position_id, "fen": row["fen"],
                           "gameOutcome": row_outcome(row),  # 0.0 preserved; dict shape normalized
                           "source": row.get("source"),
                           **({"sourceId": row["id"]} if row.get("id") and row.get("id") != position_id else {})})
            if self.config.positions and len(unique) >= self.config.positions:
                break
        _write_jsonl(target, unique)
        return unique

    def stage_tier0(self) -> None:
        from .protocols import Position
        positions = self.stage_pool()
        done = {row["id"] for row in _read_jsonl(self.path("tier0.jsonl"))}
        pending = [row for row in positions if row["id"] not in done]
        started = time.perf_counter()
        engine_seconds = 0.0
        with open(self.path("tier0.jsonl"), "a", encoding="utf-8", newline="\n") as fh:
            for row in pending:
                tick = time.perf_counter()
                batch = self.facts_provider.label(Position(fen=row["fen"]),
                                                  options={"includeMotifOpportunities": self.config.include_motif_opportunities})
                wall_ms = round((time.perf_counter() - tick) * 1000, 3)
                if batch.rows:
                    record = {**batch.rows[0], "id": row["id"], "stage": "tier0",
                              "schemaVersion": FUNNEL_SCHEMA_VERSION, "cost": {"wallMs": wall_ms}}
                else:
                    record = {"id": row["id"], "stage": "tier0", "schemaVersion": FUNNEL_SCHEMA_VERSION,
                              "status": "uncomputed", "uncomputed": list(batch.uncomputed)}
                _append_jsonl(fh, record)
                engine_seconds += wall_ms / 1000.0
        self._record_cost("tier0", positions=len(positions), processed=len(pending),
                          engine_seconds=engine_seconds, wall_seconds=time.perf_counter() - started, nodes=0)

    def stage_tier1(self) -> None:
        from .protocols import Position
        positions = _by_id(self.stage_pool())
        done = {row["id"] for row in _read_jsonl(self.path("tier1.jsonl"))}
        pending = [row for row in positions.values() if row["id"] not in done]
        started = time.perf_counter()
        engine_seconds = 0.0
        total_nodes = 0
        with open(self.path("tier1.jsonl"), "a", encoding="utf-8", newline="\n") as fh:
            for row in pending:
                tick = time.perf_counter()
                budgets = []
                for budget in self.config.tier1_budgets:
                    label = self.search_provider.search(Position(fen=row["fen"]), node_budget=budget,
                                                        pv_plies=self.config.tier1_pv_plies)
                    budgets.append({"nodeBudget": budget, "scoreCpStm": label.score_cp_stm, "mate": label.mate,
                                    "bestMove": label.best_move, "pv": list(label.pv), "nodes": label.nodes,
                                    **label.extra})
                low, high = budgets[0], budgets[-1]
                record = {"id": row["id"], "stage": "tier1", "schemaVersion": FUNNEL_SCHEMA_VERSION,
                          "search_derived": {
                              "profile": {"nodeBudgets": self.config.tier1_budgets, "isolation": "cold"},
                              "budgets": budgets,
                              "derived": {
                                  "scoreDeltaCp": clamp_cp(high["scoreCpStm"]) - clamp_cp(low["scoreCpStm"]),
                                  "bestMoveChanged": low["bestMove"] != high["bestMove"],
                                  "pvCommonPrefix": common_prefix(low["pv"], high["pv"]),
                                  "scoreCpWhite": white_pov(row["fen"], high["scoreCpStm"]),
                              }},
                          "cost": {"wallMs": round((time.perf_counter() - tick) * 1000, 3),
                                   "nodes": sum(b["nodes"] or 0 for b in budgets)}}
                _append_jsonl(fh, record)
                engine_seconds += record["cost"]["wallMs"] / 1000.0
                total_nodes += record["cost"]["nodes"]
        self._record_cost("tier1", positions=len(positions), processed=len(pending),
                          engine_seconds=engine_seconds, wall_seconds=time.perf_counter() - started, nodes=total_nodes)

    def stage_triage(self) -> None:
        tier0 = {row["id"]: row for row in _read_jsonl(self.path("tier0.jsonl")) if row.get("status") == "ok"}
        tier1 = _by_id(_read_jsonl(self.path("tier1.jsonl")))
        positions = _by_id(self.stage_pool())
        counts = coverage_counts(tier0.values())
        items = []
        for position_id in sorted(set(tier0) & set(tier1)):
            outcome_row = positions.get(position_id, {}).get("gameOutcome")
            outcome = outcome_row.get("whiteScore") if isinstance(outcome_row, dict) else outcome_row
            components = priority_components(tier0[position_id], tier1[position_id], outcome, counts, self.policy)
            priority, detail, reasons = score_priority(components, self.policy["weights"])
            items.append({"id": position_id, "priority": priority, "components": detail, "reasons": reasons})
        records = select(items, self.policy)
        for record in records:
            record["schemaVersion"] = FUNNEL_SCHEMA_VERSION
            record["stage"] = "tier2"
            record["prioritizerVersion"] = self.policy["policy_version"]
        _write_jsonl(self.path("triage.jsonl"), records)
        self.path("coverage.json").write_text(json.dumps({
            "note": "position counts per taxonomy slug / family / phase / material bucket for this pool",
            "prioritizerVersion": self.policy["policy_version"],
            "policyHash": policy_hash(self.policy),
            "counts": dict(sorted(counts.items())),
        }, indent=1), encoding="utf-8")
        selected = {key: sum(1 for row in records if row["selection"][key]) for key in ("deep", "uniform", "audit", "holdout")}
        self.costs["triage"] = {"positions": len(records), "engine_seconds": 0.0, "wall_seconds": 0.0,
                                "nodes": 0, "selected": selected}

    def stage_tier3(self) -> None:
        from .protocols import Position
        triage = _read_jsonl(self.path("triage.jsonl"))
        wanted = {row["id"]: row for row in triage
                  if any(row["selection"][key] for key in ("deep", "uniform", "audit", "holdout"))}
        done = {row["id"] for row in _read_jsonl(self.path("tier3.jsonl"))}
        tier1 = _by_id(_read_jsonl(self.path("tier1.jsonl")))
        positions = _by_id(self.stage_pool())
        started = time.perf_counter()
        engine_seconds = 0.0
        total_nodes = 0
        with open(self.path("tier3.jsonl"), "a", encoding="utf-8", newline="\n") as fh:
            for position_id in sorted(wanted):
                if position_id in done:
                    continue
                row = positions[position_id]
                shallow = tier1[position_id]["search_derived"]["budgets"][-1]
                tick = time.perf_counter()
                deep = self.search_provider.search(Position(fen=row["fen"]),
                                                   node_budget=self.config.tier3_node_budget,
                                                   pv_plies=self.config.tier3_pv_plies)
                payload = {"nodeBudget": self.config.tier3_node_budget, "scoreCpStm": deep.score_cp_stm,
                           "mate": deep.mate, "bestMove": deep.best_move, "pv": list(deep.pv),
                           "nodes": deep.nodes, **deep.extra}
                record = {"id": position_id, "stage": "tier3", "schemaVersion": FUNNEL_SCHEMA_VERSION,
                          "search_derived": {
                              "profile": {"nodeBudget": self.config.tier3_node_budget, "isolation": "cold"},
                              "deep": payload,
                              "shallowToDeep": {
                                  "shallowNodeBudget": shallow["nodeBudget"],
                                  "scoreDeltaCp": clamp_cp(deep.score_cp_stm) - clamp_cp(shallow["scoreCpStm"]),
                                  "bestMoveChanged": deep.best_move != shallow["bestMove"]},
                              "targets": {"scoreCpStm": deep.score_cp_stm,
                                          "scoreCpWhite": white_pov(row["fen"], deep.score_cp_stm),
                                          "expectedScoreStm": round(expected_score(deep.score_cp_stm), 6)}},
                          "cost": {"wallMs": round((time.perf_counter() - tick) * 1000, 3), "nodes": deep.nodes}}
                _append_jsonl(fh, record)
                engine_seconds += record["cost"]["wallMs"] / 1000.0
                total_nodes += deep.nodes
        self._record_cost("tier3", positions=len(wanted), processed=len(wanted) - len(done & set(wanted)),
                          engine_seconds=engine_seconds, wall_seconds=time.perf_counter() - started, nodes=total_nodes)

    def stage_tier4(self) -> None:
        from .protocols import Position
        triage = _read_jsonl(self.path("triage.jsonl"))
        positions = _by_id(self.stage_pool())
        done = {row["id"] for row in _read_jsonl(self.path("tier4.jsonl"))}
        started = time.perf_counter()
        engine_seconds = 0.0
        total_nodes = 0
        with open(self.path("tier4.jsonl"), "a", encoding="utf-8", newline="\n") as fh:
            for row in sorted(triage, key=lambda entry: entry["id"]):
                if row["id"] in done or not row["selection"].get("stockfish"):
                    continue
                if self.oracle_provider is None:
                    continue
                tick = time.perf_counter()
                label = self.oracle_provider.evaluate(Position(fen=positions[row["id"]]["fen"]),
                                                      depth=self.config.tier4_depth,
                                                      movetime_ms=self.config.tier4_movetime_ms)
                record = {"id": row["id"], "stage": "tier4", "schemaVersion": FUNNEL_SCHEMA_VERSION,
                          "scoreCp": label.score_cp_stm, "mate": label.mate, "bestMove": label.best_move,
                          "pv": list(label.pv), "depth": label.reached_depth, "nodes": label.nodes,
                          "cost": {"wallMs": round((time.perf_counter() - tick) * 1000, 3)}}
                _append_jsonl(fh, record)
                engine_seconds += record["cost"]["wallMs"] / 1000.0
                total_nodes += label.nodes
        self._record_cost("tier4", positions=0, processed=0, engine_seconds=engine_seconds,
                          wall_seconds=time.perf_counter() - started, nodes=total_nodes)

    def _record_cost(self, stage: str, **fields) -> None:
        self.costs[stage] = {"engine_seconds": round(fields.pop("engine_seconds"), 4),
                             **{key: round(value, 4) if isinstance(value, float) else value
                                for key, value in fields.items()}}

    # -- report and manifest ---------------------------------------------------

    def write_report(self) -> dict:
        tier0 = _read_jsonl(self.path("tier0.jsonl"))
        tier1 = _by_id(_read_jsonl(self.path("tier1.jsonl")))
        tier3 = _by_id(_read_jsonl(self.path("tier3.jsonl")))
        triage = _read_jsonl(self.path("triage.jsonl"))
        informative_delta = self.config.report_informative_delta_cp
        arms: dict[str, dict] = {}
        for arm in ("deep", "uniform", "audit", "holdout"):
            ids = [row["id"] for row in triage if row["selection"].get(arm)]
            labeled = [tier3[i] for i in ids if i in tier3]
            informative = 0
            move_change = 0
            delta_sum = 0.0
            for record in labeled:
                derived = record["search_derived"]
                delta = abs(derived["shallowToDeep"]["scoreDeltaCp"])
                delta_sum += delta
                informative += delta >= informative_delta
                move_change += bool(derived["shallowToDeep"]["bestMoveChanged"])
            arms[arm] = {
                "n": len(ids), "deepLabeled": len(labeled),
                "deepNodes": sum(tier3[i]["cost"]["nodes"] or 0 for i in ids if i in tier3),
                "deepEngineSec": round(sum(tier3[i]["cost"]["wallMs"] for i in ids if i in tier3) / 1000.0, 3),
                "informative": informative,
                "informativeRate": round(informative / len(labeled), 4) if labeled else 0.0,
                "moveChangeRate": round(move_change / len(labeled), 4) if labeled else 0.0,
                "absShallowDeepDeltaCp": {"mean": round(delta_sum / len(labeled), 3)} if labeled else {},
            }
        file_costs = self._stage_costs_from_files()
        total_engine = round(sum(entry["engine_seconds"] for entry in file_costs.values()), 3)
        report = {
            "funnelSchemaVersion": FUNNEL_SCHEMA_VERSION,
            "prioritizerVersion": self.policy["policy_version"],
            "policyHash": policy_hash(self.policy),
            "tiers": {stage: {"positions": entry.get("positions", 0),
                              "engineSec": entry["engine_seconds"],
                              "nodes": entry.get("nodes", 0),
                              "msPerPosition": round(entry["engine_seconds"] * 1000 / entry["positions"], 3)
                              if entry.get("positions") else 0.0}
                      for stage, entry in file_costs.items()},
            "totalEngineSec": total_engine,
            "arms": arms,
            "oracleArms": {},
            "coverage": {"provenanceRecords": {}},
            "experiment": {
                "question": "Does selective information-gain labeling produce a stronger evaluator than "
                            "uniform deep labeling at equal compute?",
                "informativeYieldRatio": (round(arms["deep"]["informativeRate"] / arms["uniform"]["informativeRate"], 4)
                                          if arms["uniform"]["informativeRate"] else None),
                "caveat": "Selection-level metrics are process evidence, not strength evidence; the causal test "
                          "requires training matched models on frozen priority vs uniform datasets.",
            },
        }
        self.path("report.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
        return report

    def write_manifest(self, *, facts_provider_hash: str = "", oracle_provider_hash: str = "") -> dict:
        prior = self.policy["coverage_prior"].get("prior_counts")
        positions_path = self.path("positions.jsonl")
        file_costs = self._stage_costs_from_files()  # full-corpus costs, resume-safe
        manifest = {
            "funnelSchemaVersion": FUNNEL_SCHEMA_VERSION,
            "creator": CREATOR,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": self.config.seed,
            "config": self.config.canonical(),
            "configSha256": hash_obj(self.config.canonical()),
            "policyVersion": self.policy["policy_version"],
            "policyHash": policy_hash(self.policy),
            "coveragePrior": ({"source_path": prior["source_path"], "counts_hash": prior["counts_hash"]}
                              if prior else None),
            "identity": self._identity(),
            "engine": {"analyzeSha256": facts_provider_hash or None,
                       "searchProviderHash": getattr(self.search_provider, "producer_hash", None),
                       "taxonomySha256": getattr(self.facts_provider, "taxonomy_sha256", None),
                       "taxonomySchemaVersion": getattr(self.facts_provider, "taxonomy_schema_version", None)},
            "oracle": {"providerHash": oracle_provider_hash or None,
                       "enabled": bool(self.config.tier4_enabled and self.oracle_provider),
                       "depth": self.config.tier4_depth, "movetimeMs": self.config.tier4_movetime_ms},
            "positionsSource": {"path": str(positions_path),
                                "sha256": sha256_file(positions_path) if positions_path.is_file() else None},
            "stages": {stage: {**entry,
                               "processedThisInvocation": self.costs.get(stage, {}).get("processed", 0),
                               "wallSecondsThisInvocation": self.costs.get(stage, {}).get("wall_seconds", 0.0)}
                       for stage, entry in file_costs.items()},
            "compute": {"engineSeconds": round(sum(e["engine_seconds"] for e in file_costs.values()), 4),
                        "nodes": sum(e.get("nodes", 0) for e in file_costs.values())},
            "note": "Evidence, not a dataset: import with cvslab.funnel.import_run; "
                    "training inputs are produced only by cvslab.data.freeze_dataset.",
        }
        manifest["manifestSha256"] = hash_obj(manifest)
        self.path("manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        return manifest

    # -- run identity ----------------------------------------------------------

    def _identity(self) -> dict:
        """Everything that decides what evidence this directory may contain."""
        source_files = []
        for file_name in self.config.pool_files:
            path = Path(file_name)
            if not path.is_absolute() and self.config.artifact_root:
                path = Path(self.config.artifact_root) / file_name
            source_files.append({"path": str(path), "sha256": sha256_file(path) if path.is_file() else None})
        return {
            "creator": CREATOR,
            "seed": self.config.seed,
            "configSha256": hash_obj(self.config.canonical()),
            "policyHash": policy_hash(self.policy),
            "factsProviderHash": getattr(self.facts_provider, "producer_hash", None),
            "searchProviderHash": getattr(self.search_provider, "producer_hash", None),
            "searchFamily": getattr(self.search_provider, "family", None),
            "oracle": {"providerHash": getattr(self.oracle_provider, "producer_hash", None),
                       "enabled": bool(self.config.tier4_enabled and self.oracle_provider is not None),
                       "depth": self.config.tier4_depth, "movetimeMs": self.config.tier4_movetime_ms},
            "taxonomySha256": getattr(self.facts_provider, "taxonomy_sha256", None),
            "taxonomySchemaVersion": getattr(self.facts_provider, "taxonomy_schema_version", None),
            "tier1Budgets": self.config.tier1_budgets,
            "tier3NodeBudget": self.config.tier3_node_budget,
            "sourceFiles": source_files,
        }

    def _init_or_verify_identity(self) -> None:
        """Pin the run identity before any stage runs; fail closed on resume mismatches.

        Changing the policy, a provider binary, the taxonomy, the config, the seed, the
        oracle settings or the source files makes resuming unsafe: old and new evidence
        would mix under one manifest. A directory holding stage files but no pinned
        identity is a foreign/older run and is refused rather than adopted.
        """
        identity_path = self.path(IDENTITY_FILE)
        current = self._identity()
        if identity_path.is_file():
            pinned = json.loads(identity_path.read_text(encoding="utf-8")).get("identity") or {}
            differences = sorted(key for key in set(pinned) | set(current) if pinned.get(key) != current.get(key))
            if differences:
                raise LabError(
                    "run identity differs from the pinned identity for this directory; refusing to mix "
                    f"evidence (fields: {', '.join(differences)}). Start a fresh run directory or restore "
                    "the pinned configuration.")
            return
        stage_files = [name for name in ("tier0.jsonl", "tier1.jsonl", "triage.jsonl",
                                         "tier3.jsonl", "tier4.jsonl")
                       if self.path(name).is_file() and self.path(name).stat().st_size > 0]
        if stage_files:
            raise LabError(f"{self.run_dir} already holds stage files ({', '.join(stage_files)}) but no "
                           f"{IDENTITY_FILE}; refusing to adopt a foreign run directory (fail closed)")
        identity_path.write_text(json.dumps({"identity": current}, indent=1, sort_keys=True), encoding="utf-8")

    # -- cost accounting -------------------------------------------------------

    def _stage_costs_from_files(self) -> dict[str, dict]:
        """Costs derived from the stage files themselves, so a resumed run reports the
        compute that produced the FULL corpus, not just this invocation's share."""
        positions = len(_read_jsonl(self.path("positions.jsonl")))
        costs: dict[str, dict] = {}
        for stage, file_name in (("tier0", "tier0.jsonl"), ("tier1", "tier1.jsonl"),
                                 ("tier3", "tier3.jsonl"), ("tier4", "tier4.jsonl")):
            rows = _read_jsonl(self.path(file_name))
            costs[stage] = {
                "positions": positions if stage == "tier0" else len(rows),
                "engine_seconds": round(sum(float((row.get("cost") or {}).get("wallMs", 0.0) or 0.0)
                                            for row in rows) / 1000.0, 4),
                "nodes": sum(int((row.get("cost") or {}).get("nodes", 0) or 0) for row in rows),
            }
        costs["triage"] = {"positions": len(_read_jsonl(self.path("triage.jsonl"))),
                           "engine_seconds": 0.0, "nodes": 0}
        return costs

    # -- driver ----------------------------------------------------------------

    def run(self) -> dict:
        self._init_or_verify_identity()
        stages = [("tier0", self.stage_tier0), ("tier1", self.stage_tier1), ("triage", self.stage_triage),
                  ("tier3", self.stage_tier3), ("tier4", self.stage_tier4)]
        for name, stage in stages:
            stage()
        self.write_report()
        return self.write_manifest(facts_provider_hash=getattr(self.facts_provider, "producer_hash", ""),
                                   oracle_provider_hash=getattr(self.oracle_provider, "producer_hash", ""))


def verify_manifest(run_dir: str | Path) -> tuple[bool, str]:
    """Recompute the manifest's own content hash; returns (ok, stored_hash)."""
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        return False, ""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stored = manifest.pop("manifestSha256", "")
    return hash_obj(manifest) == stored, stored


def prepare_policy(config: dict, *, artifact_root: Optional[str] = None, store=None) -> tuple[dict, Optional[str]]:
    """Build the run's policy, resolving a configured coverage prior **explicitly**.

    A non-null ``priorCountsPath`` is read (relative to the artifact root when not
    absolute); its content hash becomes part of the policy identity. Returns
    (policy, stored_policy_hash_if_store_given).
    """
    tier2 = config.get("tiers", {}).get("tier2", {})
    prior_path = (tier2.get("coverage") or {}).get("priorCountsPath")
    prior_counts = None
    if prior_path:
        resolved = Path(prior_path)
        if not resolved.is_absolute() and artifact_root:
            resolved = Path(artifact_root) / prior_path
        if not resolved.is_file():
            raise LabError(f"coverage prior not found: {resolved} (configured as {prior_path})")
        prior_counts = json.loads(resolved.read_text(encoding="utf-8"))["counts"]
    policy = build_policy(config, prior_counts=prior_counts)
    digest = store_policy(store, policy) if store is not None else None
    return policy, digest
