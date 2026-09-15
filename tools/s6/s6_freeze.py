"""S6 real freeze: S5 pool -> S4 labeling -> S2 import -> Universe -> T -> D pair -> E pair -> report.

Idempotent per step (artifacts cached under OUT), so a rerun resumes instead of redoing compute.
Runs with the configured S4 budgets: tier1 [2000, 16000], tier3 400000 nodes.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, r"F:\Github\chess-vision-studio-lab")
from cvslab.data import _load_canonical, read_jsonl
from cvslab.funnel.campaign import (CampaignPlan, eval_semantics_hash, freeze_arm_datasets,
                                    freeze_universe, verify_arm_protocols)
from cvslab.funnel.import_run import import_funnel_run_evidence
from cvslab.funnel.orchestrator import FunnelOrchestrator, FunnelRunConfig, prepare_policy
from cvslab.funnel.pool import (DEFAULT_ENGINE_ARGS, PoolConfig, SelfPlayGenerator, AnalyzeTeacher,
                                snapshot_source, write_pool)
from cvslab.funnel.providers import AnalyzeFactsProvider, AnalyzeSearchProvider, AnalyzeTransport
from cvslab.service import LabService
from cvslab.store import Store

ENGINE = r"F:\Github\chess-vision-studio-rust-engine"
HOME = r"F:\Github\chess-vision-studio-lab\labstore"
OUT = Path(r"F:\Github\_parity_tmp\s6")
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "progress.json"


def log(step: str, payload: dict) -> None:
    state = json.loads(PROGRESS.read_text()) if PROGRESS.is_file() else {}
    state[step] = payload
    PROGRESS.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] {step}: {json.dumps(payload, default=str)[:300]}", flush=True)


def cached(name: str) -> dict:
    """Read a cached step payload, unwrapping any legacy {"cached": …} entries."""
    entry = json.loads(PROGRESS.read_text())[name]
    return entry.get("cached", entry)


def record_id_for_fen(fen: str) -> str:
    epd = chess.Board(fen).epd()
    return "pos_" + hashlib.sha256(epd.encode()).hexdigest()[:20]


def main() -> int:
    state = json.loads(PROGRESS.read_text()) if PROGRESS.is_file() else {}
    svc = LabService(Store(HOME))

    # -- step 1: S5 pool -----------------------------------------------------
    pool_dir = OUT / "pool"
    if "pool" not in state:
        transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
        try:
            generator = SelfPlayGenerator(
                PoolConfig(seed=20260915, games=70, max_plies=70, sample_every=2, min_ply=6,
                           diversification_plies=(8, 10, 12), diversification_candidates=3,
                           diversification_window_cp=25, play_node_budget=20000,
                           diversification_node_budget=2000), AnalyzeTeacher(transport))
            generated = generator.generate()
        finally:
            transport.close()
        manifest = write_pool(pool_dir, generated, {"config": PoolConfig(seed=20260915).canonical(),
                                                    "engine": {"commit": "pinned"}},
                              opening_lines=PoolConfig(seed=20260915).effective_openings())
        source = snapshot_source(svc.store, pool_dir, name="s6-pool", license="engine self-play corpus",
                                 generated=generated, manifest=manifest)
        log("pool", {"source": source.id, "rows": source.row_count, "games": source.game_count,
                     "positions_sha256": manifest["positionsFile"]["sha256"],
                     "report": manifest["report"]})
    else:
        print(f"[cached] pool", flush=True)

    # -- step 2: S4 labeling run --------------------------------------------
    run_dir = OUT / "s4run"
    if "s4run" not in state:
        config = {
            "funnelConfigVersion": 1, "prioritizerVersion": "priority-v1", "seed": 111,
            "source": {"positionsFiles": [str(pool_dir / "positions.jsonl")], "positions": 0},
            "engine": {"args": DEFAULT_ENGINE_ARGS},
            "workers": 1,
            "tiers": {
                "tier0": {"includeMotifOpportunities": True},
                "tier1": {"nodeBudgets": [2000, 16000], "pvPlies": 8},
                "tier2": {"deepFraction": 0.10, "auditFraction": 0.02, "holdoutFraction": 0.01,
                          "holdoutSeed": 20260914,
                          "weights": {"scoreInstability": 3.0, "bestMoveChange": 2.0, "trajectoryUnstable": 1.5,
                                      "pvDisagreement": 0.5, "tacticalDensity": 1.0, "rarity": 1.5,
                                      "outcomeDisagreement": 1.0, "selectivityEdge": 0.5, "forcedness": 0.25},
                          "caps": {"scoreDeltaCp": 150, "tacticalItems": 10, "seePruneSkips": 4,
                                   "forcedLegalMoves": 3, "outcomeMarginCp": 100, "outcomeScaleCp": 400},
                          "coverage": {"targetShare": 0.02, "minTarget": 5, "priorCountsPath": None}},
                "tier3": {"nodeBudget": 400000, "pvPlies": 12},
                "tier4": {"enabled": False}},
            "report": {"informativeDeltaCp": 60},
        }
        policy, _digest = prepare_policy(config)
        transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
        try:
            orchestrator = FunnelOrchestrator(
                FunnelRunConfig.from_dict(config, artifact_root=ENGINE), run_dir, policy=policy,
                facts_provider=AnalyzeFactsProvider(transport),
                search_provider=AnalyzeSearchProvider(transport, family="search_deep_cp"))
            manifest = orchestrator.run()
        finally:
            transport.close()
        log("s4run", {"manifest_sha256": manifest["manifestSha256"],
                      "positions": manifest["positionsSource"], "stages": manifest["stages"],
                      "compute": manifest["compute"]})
    else:
        print(f"[cached] s4run", flush=True)

    # -- step 3: S2 import + universe reconstruction -------------------------
    if "import" not in state:
        counts = import_funnel_run_evidence(svc.store, run_dir, name="s6-run")
        log("import", {k: counts[k] for k in ("source", "normalization", "records" if "records" in counts else "canonical_records",
                                              "policy_hash", "funnel_run", "label_rows")})
    else:
        print(f"[cached] import", flush=True)
    imported = cached("import")
    funnel_run_id = imported["funnel_run"]
    policy_hash = imported["policy_hash"]

    # replay the raw pool into the lab so the canonical corpus keeps GAME identity
    if "canonical" not in state:
        s5 = cached("pool")["source"]
        normalization = svc.normalize([s5], name="s6-pool-canonical")
        log("canonical", {"normalization": normalization.id, "records": normalization.record_count,
                          "duplicates": normalization.duplicate_count})
    else:
        print(f"[cached] canonical", flush=True)
    normalization_id = cached("canonical")["normalization"]
    records = _load_canonical(svc.store, svc.store.get(normalization_id))
    group_of = {record["record_id"]: record["group"] for record in records}

    run_positions = {row["id"]: row["fen"] for row in read_jsonl(run_dir / "positions.jsonl")}
    triage = read_jsonl(run_dir / "triage.jsonl")
    tier3 = read_jsonl(run_dir / "tier3.jsonl")

    selection = {"deep": [], "uniform": []}
    eligible = {}
    for row in triage:
        record_id = record_id_for_fen(run_positions[row["id"]])
        eligible[record_id] = bool(row.get("trainEligible", True))
        if row["selection"].get("deep"):
            selection["deep"].append(record_id)
        if row["selection"].get("uniform"):
            selection["uniform"].append(record_id)
    deep_nodes, deep_seconds = {}, {"PRIORITY": 0.0, "UNIFORM": 0.0}
    arm_of = {record_id: "PRIORITY" for record_id in selection["deep"]}
    arm_of.update({record_id: "UNIFORM" for record_id in selection["uniform"] if record_id not in arm_of})
    for row in tier3:
        record_id = record_id_for_fen(run_positions[row["id"]])
        deep_nodes[record_id] = int(row["cost"]["nodes"])
        arm = arm_of.get(record_id)
        if arm:
            deep_seconds[arm] += row["cost"]["wallMs"] / 1000.0

    candidates = [{"record_id": record_id, "group": group_of[record_id], "train_eligible": True}
                  for record_id in sorted(group_of) if eligible.get(record_id, False)]
    existing_candidates = {row["record_id"] for row in candidates}
    selection = {arm: [rid for rid in ids if rid in existing_candidates]
                 for arm, ids in selection.items()}
    if len(selection["deep"]) != len(selection["uniform"]):
        trimmed = min(len(selection["deep"]), len(selection["uniform"]))
        selection["deep"], selection["uniform"] = selection["deep"][:trimmed], selection["uniform"][:trimmed]
    universe = freeze_universe(source_id=cached("pool")["source"], normalization_id=normalization_id,
                               funnel_run_id=funnel_run_id, policy_version="priority-v1",
                               policy_hash=policy_hash, candidates=candidates, deep_nodes=deep_nodes,
                               selections=selection, split_seed=20260915)
    log("universe", {"candidates": len(universe.candidate_ids),
                     "priority": len(universe.arms["PRIORITY"]), "uniform": len(universe.arms["UNIFORM"]),
                     "universe_hash": universe.universe_hash(),
                     "split_policy_hash": universe.split_policy_hash(),
                     "candidate_universe_hash": universe.candidate_universe_hash()})

    # labels registered under the run's normalization must reach the game-aware corpus;
    # record identities are EPD-derived so they survive the carry (counted, never assumed)
    if "labels_carried" not in state:
        carried = svc.copy_label_sets(imported["normalization"], normalization_id)
        log("labels_carried", carried)
    else:
        print("[cached] labels_carried", flush=True)

    # -- step 4: T + D pair (+ engine seconds) -------------------------------
    if "datasets" not in state:
        recipe = svc.create_training_recipe(name="s6-raw-nnue-ladder",
                                            params={"EPOCHS": 40, "BATCH": 256, "LR": 0.003,
                                                    "OPTIMIZER": "adam", "INIT_STD": 0.05,
                                                    "LAMBDA": 1.0, "K": 256.0})
        report = freeze_arm_datasets(svc.store, universe, CampaignPlan(), recipe=recipe,
                                     deep_engine_seconds=deep_seconds, name_prefix="s6")
        log("datasets", {"recipe": recipe.id, "recipe_hash": recipe.recipe_hash, "report": report})
    else:
        print(f"[cached] datasets", flush=True)

    # -- step 5: arm-bound E protocols + semantics proof ---------------------
    datasets = cached("datasets")
    if "protocols" not in state:
        protocols = {}
        for arm in ("PRIORITY", "UNIFORM"):
            dataset_id = datasets["report"]["arms"][arm]["dataset_id"]
            protocols[arm] = svc.create_eval_protocol(name=f"s6-{arm.lower()}-eval", dataset_id=dataset_id,
                                                      split="test", bootstrap_samples=1000, bootstrap_seed=0,
                                                      k=256.0, lam=1.0)
        semantics = verify_arm_protocols(protocols)
        log("protocols", {"ids": {arm: p.id for arm, p in protocols.items()},
                          "protocol_hashes": {arm: p.protocol_hash for arm, p in protocols.items()},
                          "semantics_hash": semantics,
                          "semantics_equal": len({eval_semantics_hash(p) for p in protocols.values()}) == 1})
    else:
        print(f"[cached] protocols", flush=True)

    print("FINAL REPORT", json.dumps(json.loads(PROGRESS.read_text()), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
