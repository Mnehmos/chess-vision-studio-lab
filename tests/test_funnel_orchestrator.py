import json
from pathlib import Path

import pytest

from cvslab.funnel.orchestrator import (
    FunnelOrchestrator,
    FunnelRunConfig,
    prepare_policy,
    verify_manifest,
)
from cvslab.funnel.policy import build_policy, policy_hash
from cvslab.funnel.providers import board_geometry
from cvslab.funnel.protocols import LabelBatch, Position, SearchLabel
from cvslab.funnel.triage import triage_corpus
from cvslab.store import LabError

CONFIG = {
    "prioritizerVersion": "priority-v1", "seed": 111,
    "tiers": {
        "tier0": {"includeMotifOpportunities": True},
        "tier1": {"nodeBudgets": [2000], "pvPlies": 8},
        "tier2": {"deepFraction": 0.5, "auditFraction": 0.1, "holdoutFraction": 0.1, "holdoutSeed": 7,
                  "weights": {"scoreInstability": 3.0, "bestMoveChange": 2.0, "trajectoryUnstable": 1.5,
                              "pvDisagreement": 0.5, "tacticalDensity": 1.0, "rarity": 1.5,
                              "outcomeDisagreement": 1.0, "selectivityEdge": 0.5, "forcedness": 0.25},
                  "caps": {"scoreDeltaCp": 150, "tacticalItems": 10, "seePruneSkips": 4,
                           "forcedLegalMoves": 3, "outcomeMarginCp": 100, "outcomeScaleCp": 400},
                  "coverage": {"targetShare": 0.02, "minTarget": 5, "priorCountsPath": None}},
        "tier3": {"nodeBudget": 10000, "pvPlies": 12},
        "tier4": {"enabled": False},
    },
}


class StubFacts:
    """Deterministic facts provider with the real tier-0 record shape (search-free)."""

    provenance_class = "deterministic_geometry"
    producer_hash = "stubfacts" * 8
    taxonomy_sha256 = "taxonomy" * 8
    taxonomy_schema_version = 2

    def label(self, position: Position, *, options=None) -> LabelBatch:
        geometry, _first_move = board_geometry(position.fen)
        slug = "fork" if geometry["legalMoves"] % 2 else "skewer"
        record = {
            "status": "ok",
            "deterministic_geometry": geometry,
            # schema-faithful tactics block: `tacticalItems` is recomputed from
            # kindCounts + hazards on import, so a stub omitting them would silently
            # change triage (this bit the first version of this test)
            "bounded_tactical_proof": {
                "opportunities": {"stm": [slug], "opp": []},
                "kindCounts": {f"stm:Motifs:{slug}": 1},
                "captures": {"stm": {"count": geometry["legalCaptures"], "seePositive": 0},
                             "opp": {"count": 0, "seePositive": 0}},
                "hanging": {"stm": 0, "opp": 0}, "hazards": {}, "tacticalItems": 1},
            "taxonomy": {"slugs": [slug], "families": ["tactics"], "unmapped": []},
            "uncomputed": [], "factsErrors": 0, "factsRegistryVersion": 23,
        }
        return LabelBatch(provenance_class="deterministic_geometry", producer=self.producer_hash,
                          rows=(record,), registry_version=23)


class StubSearch:
    provenance_class = "search_derived"
    family = "search_shallow_cp"
    producer_hash = "stubsearch" * 6

    def search(self, position: Position, *, node_budget: int, pv_plies: int = 8) -> SearchLabel:
        seed = int(position.fen.split()[5]) + node_budget % 97 + len(position.fen)
        score = (seed * 13) % 400 - 200
        return SearchLabel(score_cp_stm=score, mate=None, best_move="e2e4", pv=("e2e4", "e7e5")[:pv_plies],
                           nodes=node_budget, wall_ms=1.0, extra={"depth": 4, "trajectory": [[4, "e2e4", score]],
                                                                  "stabilization": {"status": "stable-at-budget"},
                                                                  "termination": "nodes", "resultSource": "completed",
                                                                  "qNodes": 0})


class StubOracle:
    provenance_class = "external_oracle"
    producer_hash = "stuboracle" * 6

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def evaluate(self, position: Position, *, depth: int, movetime_ms: int):
        from cvslab.funnel.protocols import OracleLabel
        self.calls.append((depth, movetime_ms))
        return OracleLabel(score_cp_stm=11, mate=None, best_move="e2e4", pv=("e2e4",),
                           reached_depth=depth, nodes=1000, wall_ms=2.0)


def make_pool(tmp_path: Path, count: int = 12) -> Path:
    """Three distinct opening lines cycling across games (so positions differ), plus one
    deliberate identity-duplicate to prove the pool stage dedups by position identity."""
    import chess
    tmp_path.mkdir(parents=True, exist_ok=True)
    lines = [("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"),
             ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"),
             ("c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5")]
    rows = []
    for game in range(count):
        board = chess.Board()
        for ply, move in enumerate(lines[game % len(lines)], start=1):
            board.push_uci(move)
            if ply % 2 == 0:
                rows.append({"id": f"pos{game}x{ply}", "fen": board.fen(),
                             "gameOutcome": {"whiteScore": 1.0 if game % 2 else 0.0},
                             "source": {"file": "stub", "line": game * 10 + ply}})
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return pool


def make_orchestrator(tmp_path: Path, *, count: int = 12, policy: dict | None = None) -> FunnelOrchestrator:
    config = json.loads(json.dumps(CONFIG))
    config["source"] = {"positionsFiles": [str(make_pool(tmp_path, count))], "positions": 0}
    run_config = FunnelRunConfig.from_dict(config)
    return FunnelOrchestrator(run_config, tmp_path / "run", policy=policy or build_policy(config),
                              facts_provider=StubFacts(), search_provider=StubSearch())


def test_staged_run_produces_evidence_and_verifiable_manifest(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    manifest = orchestrator.run()
    for name in ("positions.jsonl", "tier0.jsonl", "tier1.jsonl", "triage.jsonl", "tier3.jsonl",
                 "coverage.json", "report.json", "manifest.json"):
        assert (tmp_path / "run" / name).is_file(), name
    ok, stored = verify_manifest(tmp_path / "run")
    assert ok and stored == manifest["manifestSha256"]
    assert manifest["policyHash"] == policy_hash(orchestrator.policy)
    assert manifest["policyVersion"] == "priority-v1"
    assert manifest["coveragePrior"] is None
    assert "not a dataset" in manifest["note"]

    report = json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))
    assert report["policyHash"] == manifest["policyHash"]
    assert set(report["arms"]) == {"deep", "uniform", "audit", "holdout"}
    tier3_rows = [json.loads(line) for line in (tmp_path / "run" / "tier3.jsonl").read_text().splitlines()]
    assert tier3_rows and all(row["search_derived"]["targets"]["expectedScoreStm"] for row in tier3_rows)


def test_resume_continues_without_duplicates(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    orchestrator.run()
    full_tier1 = (tmp_path / "run" / "tier1.jsonl").read_text(encoding="utf-8").splitlines()

    # simulate an interruption: the last 3 tier1 rows and the whole tier3 stage are missing
    (tmp_path / "run" / "tier1.jsonl").write_text("\n".join(full_tier1[:-3]) + "\n", encoding="utf-8")
    (tmp_path / "run" / "tier3.jsonl").unlink()

    resumed = make_orchestrator(tmp_path)
    resumed.run()
    rows = [json.loads(line) for line in (tmp_path / "run" / "tier1.jsonl").read_text().splitlines()]
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)) == len(full_tier1)  # no duplicates, nothing lost
    strip = lambda row: {k: v for k, v in row.items() if k != "cost"}
    first = sorted(json.dumps(strip(json.loads(line)), sort_keys=True) for line in full_tier1)
    again = sorted(json.dumps(strip(row), sort_keys=True) for row in rows)
    assert first == again  # identical final content modulo per-attempt timing


def test_cost_accounting_totals_match_stage_sums(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    manifest = orchestrator.run()
    stages = manifest["stages"]
    assert manifest["compute"]["engineSeconds"] == round(sum(s["engine_seconds"] for s in stages.values()), 4)
    assert manifest["compute"]["nodes"] == sum(s.get("nodes", 0) for s in stages.values())
    assert stages["tier1"]["nodes"] > 0 and stages["tier3"]["processedThisInvocation"] > 0

    # totals equal the sums over the stage FILES (not just this invocation)
    import json as _json
    tier1_rows = [_json.loads(line) for line in (tmp_path / "run" / "tier1.jsonl").read_text().splitlines()]
    assert stages["tier1"]["nodes"] == sum(row["cost"]["nodes"] for row in tier1_rows)
    assert stages["tier1"]["engine_seconds"] == round(sum(row["cost"]["wallMs"] for row in tier1_rows) / 1000.0, 4)


def test_resume_manifest_reports_full_corpus_compute(tmp_path):
    """A resumed run must not underreport: totals come from the files, so a run split
    across two invocations reports exactly what a single clean invocation reports."""
    clean = make_orchestrator(tmp_path / "clean")
    clean_manifest = clean.run()

    split = make_orchestrator(tmp_path / "split")
    split.run()
    tier1_lines = (tmp_path / "split" / "run" / "tier1.jsonl").read_text().splitlines()
    (tmp_path / "split" / "run" / "tier1.jsonl").write_text("\n".join(tier1_lines[:-3]) + "\n", encoding="utf-8")
    (tmp_path / "split" / "run" / "tier3.jsonl").unlink()
    resumed = make_orchestrator(tmp_path / "split")
    resumed_manifest = resumed.run()

    # node totals are exact (budget-derived); wall times are measured, so they only agree closely
    assert resumed_manifest["compute"]["nodes"] == clean_manifest["compute"]["nodes"]
    clean_seconds = clean_manifest["compute"]["engineSeconds"]
    resumed_seconds = resumed_manifest["compute"]["engineSeconds"]
    assert abs(resumed_seconds - clean_seconds) <= max(0.002, clean_seconds * 0.05)
    assert resumed_manifest["stages"]["tier3"]["positions"] == clean_manifest["stages"]["tier3"]["positions"]
    # the resumed invocation redid only the missing rows, yet the manifest reports the full corpus:
    assert resumed_manifest["stages"]["tier1"]["processedThisInvocation"] == 3
    assert resumed_manifest["stages"]["tier1"]["positions"] == clean_manifest["stages"]["tier1"]["positions"]


def test_resume_refuses_changed_identity(tmp_path):
    orchestrator = make_orchestrator(tmp_path)
    orchestrator.run()

    # change the policy -> the pinned identity no longer matches
    changed = build_policy(json.loads(json.dumps(CONFIG)))
    changed["weights"]["rarity"] = 1.25
    tampered = make_orchestrator(tmp_path, policy=changed)
    with pytest.raises(LabError, match="run identity differs"):
        tampered.run()

    # a directory with stage files but no pinned identity is refused, not adopted
    (tmp_path / "run" / "run-identity.json").unlink()
    fresh = make_orchestrator(tmp_path)
    with pytest.raises(LabError, match="no .*run-identity"):
        fresh.run()


def test_manifest_pins_taxonomy_oracle_and_tier4_config(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    config["tiers"]["tier4"] = {"enabled": True, "depth": 21, "movetimeMs": 1500,
                                "priorityFraction": 0.1, "uniformFraction": 0.1, "auditFraction": 0.1}
    config["source"] = {"positionsFiles": [str(make_pool(tmp_path))], "positions": 0}
    oracle = StubOracle()
    orchestrator = FunnelOrchestrator(FunnelRunConfig.from_dict(config), tmp_path / "run",
                                      policy=build_policy(config),
                                      facts_provider=StubFacts(), search_provider=StubSearch(),
                                      oracle_provider=oracle)
    manifest = orchestrator.run()
    assert manifest["engine"]["taxonomySha256"] == StubFacts.taxonomy_sha256
    assert manifest["engine"]["taxonomySchemaVersion"] == 2
    assert manifest["oracle"]["providerHash"] == StubOracle.producer_hash
    assert manifest["oracle"]["depth"] == 21 and manifest["oracle"]["movetimeMs"] == 1500
    assert manifest["config"]["tier4_depth"] == 21 and manifest["config"]["tier4_movetime_ms"] == 1500
    assert oracle.calls and all(call == (21, 1500) for call in oracle.calls)


def test_pool_preserves_falsy_outcome(tmp_path):
    import json as _json
    pool = tmp_path / "pool.jsonl"
    pool.write_text(_json.dumps({"id": "l1", "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                                 "gameOutcome": {"whiteScore": 0.0}}) + "\n", encoding="utf-8")
    config = _json.loads(_json.dumps(CONFIG))
    config["source"] = {"positionsFiles": [str(pool)], "positions": 0}
    orchestrator = FunnelOrchestrator(FunnelRunConfig.from_dict(config), tmp_path / "run",
                                      policy=build_policy(config),
                                      facts_provider=StubFacts(), search_provider=StubSearch())
    positions = orchestrator.stage_pool()
    assert positions[0]["gameOutcome"] == 0.0  # a real loss is not "missing"


def test_manifest_detects_tampering(tmp_path):
    make_orchestrator(tmp_path).run()
    path = tmp_path / "run" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["seed"] = manifest["seed"] + 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    ok, _stored = verify_manifest(tmp_path / "run")
    assert ok is False


def test_prior_counts_are_resolved_and_pinned(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    config["tiers"]["tier2"]["coverage"]["priorCountsPath"] = "prior.json"
    with pytest.raises(LabError, match="coverage prior not found"):
        prepare_policy(config, artifact_root=str(tmp_path))
    (tmp_path / "prior.json").write_text(json.dumps({"counts": {"__positions__": 5, "fork": 2}}), encoding="utf-8")
    policy, _digest = prepare_policy(config, artifact_root=str(tmp_path))
    prior = policy["coverage_prior"]["prior_counts"]
    assert prior["source_path"] == "prior.json" and prior["counts_hash"].startswith("sha256:")
    base = build_policy(json.loads(json.dumps(CONFIG)))
    assert policy_hash(policy) != policy_hash(base)  # the prior is part of the identity

    orchestrator = make_orchestrator(tmp_path, policy=policy)
    manifest = orchestrator.run()
    assert manifest["coveragePrior"]["counts_hash"] == prior["counts_hash"]


def test_orchestrated_evidence_flows_through_s2_import(tmp_path, lab):
    """The run dir is evidence: importing via S2 yields canonical records + labels."""
    from cvslab.funnel.import_run import import_funnel_run_evidence

    make_orchestrator(tmp_path).run()
    counts = import_funnel_run_evidence(lab.store, tmp_path / "run", name="s4-evidence", link_funnel_run=False)
    assert counts["canonical_records"] > 0
    assert counts["label_rows"]["facts"] == counts["canonical_records"]
    assert counts["label_rows"]["search_shallow_cp"] >= counts["canonical_records"]
    assert lab.store.list("D", verify=False) == []  # still no implicit dataset


def test_orchestrated_triage_is_reproducible_from_imported_evidence(tmp_path, lab):
    """Lab-produced triage over lab-produced evidence equals the run's own triage.jsonl."""
    from cvslab.funnel.import_run import import_funnel_run_evidence
    from cvslab.funnel.triage import compare_triage

    orchestrator = make_orchestrator(tmp_path)
    orchestrator.run()
    counts = import_funnel_run_evidence(lab.store, tmp_path / "run", name="s4-parity", link_funnel_run=False)
    lab_rows, _counts = triage_corpus(lab.store, counts["normalization"], orchestrator.policy)
    legacy_rows = [json.loads(line) for line in (tmp_path / "run" / "triage.jsonl").read_text().splitlines()]
    report = compare_triage(lab_rows, legacy_rows)
    assert report["ok"], {field: report[field] for field in ("priority", "rank", "reasons", "selection")}


def test_pool_dedups_by_position_identity_not_row_id(tmp_path):
    """Two records with different ids but the same position are one candidate."""
    import json as _json
    pool = tmp_path / "pool.jsonl"
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    rows = [{"id": "a", "fen": fen, "source": {"line": 1}},
            {"id": "b", "fen": fen.replace("0 1", "0 9"), "source": {"line": 2}},  # same position, clocks differ
            {"id": "c", "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "source": {"line": 3}}]
    pool.write_text("\n".join(_json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = _json.loads(_json.dumps(CONFIG))
    config["source"] = {"positionsFiles": [str(pool)], "positions": 0}
    orchestrator = FunnelOrchestrator(FunnelRunConfig.from_dict(config), tmp_path / "run",
                                      policy=build_policy(config),
                                      facts_provider=StubFacts(), search_provider=StubSearch())
    positions = orchestrator.stage_pool()
    assert len(positions) == 2  # a and b collapse (same position); c differs by side to move
    assert len({row["id"] for row in positions}) == 2  # identity unique
    keep_first = next(row for row in positions if row.get("sourceId") == "a")
    assert keep_first["fen"].split()[5] == "1"  # the first occurrence of the position was kept


# --- S4 review pass 2: S4->S2 provenance, complete cost accounting, workers, workset ---


def test_orchestrated_import_preserves_producer_and_policy_identity(tmp_path, lab):
    """The critical boundary: orchestrated evidence must import WITH its real identities."""
    from cvslab.data import label_set_refs
    from cvslab.funnel.import_run import import_funnel_run_evidence

    orchestrator = make_orchestrator(tmp_path)
    orchestrator.run()
    counts = import_funnel_run_evidence(lab.store, tmp_path / "run", name="provenance")
    assert counts["analyze_sha256"] == StubFacts.producer_hash  # modern manifest name was read
    assert counts["policy_hash"] == policy_hash(orchestrator.policy)

    refs = label_set_refs(lab.store, counts["normalization"])
    facts_ref = next(ref for ref in refs if "facts" in ref.families)
    assert set(facts_ref.producers) == {StubFacts.producer_hash}  # labels keep the true producer
    priority_ref = next(ref for ref in refs if "priority" in ref.families)
    assert priority_ref.policy_hash == counts["policy_hash"]

    funnel_run = lab.store.get(counts["funnel_run"])
    assert funnel_run.triage_policy_hash == counts["policy_hash"]
    assert funnel_run.triage_policy_version == "priority-v1"
    assert funnel_run.position_pool == counts["source"]


def test_import_refuses_policy_hash_mismatch(tmp_path, lab):
    """A run whose pinned policy cannot be reproduced from its config must not import."""
    import json as _json
    from cvslab.funnel.import_run import import_funnel_run_evidence

    make_orchestrator(tmp_path).run()
    manifest_path = tmp_path / "run" / "manifest.json"
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["policyHash"] = "sha256:" + "d" * 64
    manifest_path.write_text(_json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LabError, match="policy hash mismatch"):
        import_funnel_run_evidence(lab.store, tmp_path / "run", name="mismatch", link_funnel_run=False)


def test_tier4_nodes_are_counted_in_compute(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    config["tiers"]["tier4"] = {"enabled": True, "depth": 18, "movetimeMs": 3000,
                                "priorityFraction": 0.5, "uniformFraction": 0.5, "auditFraction": 0.5}
    config["source"] = {"positionsFiles": [str(make_pool(tmp_path))], "positions": 0}
    oracle = StubOracle()
    orchestrator = FunnelOrchestrator(FunnelRunConfig.from_dict(config), tmp_path / "run",
                                      policy=build_policy(config), facts_provider=StubFacts(),
                                      search_provider=StubSearch(), oracle_provider=oracle)
    manifest = orchestrator.run()
    assert oracle.calls
    assert manifest["stages"]["tier4"]["nodes"] == 1000 * len(oracle.calls)  # cost.nodes is the source
    assert manifest["compute"]["nodes"] == sum(stage.get("nodes", 0) for stage in manifest["stages"].values())


def test_uncomputed_tier0_rows_keep_measured_cost(tmp_path):
    from cvslab.funnel.protocols import LabelBatch

    class EmptyFacts:
        provenance_class = "deterministic_geometry"
        producer_hash = "emptyfacts" * 6
        taxonomy_sha256 = "0" * 64
        taxonomy_schema_version = 2

        def label(self, position, *, options=None):
            return LabelBatch(provenance_class="deterministic_geometry", producer=self.producer_hash,
                              rows=(), uncomputed=("unsupported-position",))

    config = json.loads(json.dumps(CONFIG))
    config["source"] = {"positionsFiles": [str(make_pool(tmp_path))], "positions": 0}
    orchestrator = FunnelOrchestrator(FunnelRunConfig.from_dict(config), tmp_path / "run",
                                      policy=build_policy(config), facts_provider=EmptyFacts(),
                                      search_provider=StubSearch())
    manifest = orchestrator.run()
    tier0_rows = [json.loads(line) for line in (tmp_path / "run" / "tier0.jsonl").read_text().splitlines()]
    assert tier0_rows and all("cost" in row for row in tier0_rows)
    assert manifest["stages"]["tier0"]["engine_seconds"] == round(
        sum(row["cost"]["wallMs"] for row in tier0_rows) / 1000.0, 4)


def test_workers_other_than_one_fail_loudly(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    config["workers"] = 4
    config["source"] = {"positionsFiles": [str(make_pool(tmp_path))], "positions": 0}
    orchestrator = FunnelOrchestrator(FunnelRunConfig.from_dict(config), tmp_path / "run",
                                      policy=build_policy(config), facts_provider=StubFacts(),
                                      search_provider=StubSearch())
    with pytest.raises(LabError, match="workers=4 is not supported"):
        orchestrator.run()


def test_workset_boundary_is_explicit(tmp_path):
    """positions.jsonl is a labeling workset; the raw S5 source keeps game multiplicity."""
    config = json.loads(json.dumps(CONFIG))
    config["source"] = {"positionsFiles": [str(make_pool(tmp_path))], "positions": 0}
    orchestrator = FunnelOrchestrator(FunnelRunConfig.from_dict(config), tmp_path / "run",
                                      policy=build_policy(config), facts_provider=StubFacts(),
                                      search_provider=StubSearch())
    manifest = orchestrator.run()
    source = manifest["positionsSource"]
    assert source["kind"] == "labeling_workset"
    assert "raw S5 source" in source["note"]
    assert source["inputRows"] > source["uniqueIdentities"]  # the stub pool carries a duplicate
