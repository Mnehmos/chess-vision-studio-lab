"""S6 round 2: corrected lineage -> T/D pair -> common eval -> smoke -> 40 cells -> analysis.

Fail-closed scientific conditions abort with a clear stop reason; ordinary bugs are fixed
and rerun. Every artifact lands under OUT so the run is inspectable and resumable.
"""
import json
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, r"F:\Github\chess-vision-studio-lab")
from cvslab.data import _load_canonical, read_jsonl
from cvslab.funnel.campaign import (CampaignPlan, DEFAULT_SEEDS, EXPECTED_PARAMS, WIDTHS,
                                    decide, freeze_arm_datasets, freeze_universe,
                                    interaction_effects, paired_effect, split_identity)
from cvslab.funnel.pool import snapshot_source
from cvslab.funnel.providers import AnalyzeSearchProvider, AnalyzeTransport
from cvslab.funnel.pool import DEFAULT_ENGINE_ARGS
from cvslab.service import LabService
from cvslab.schemas import RunStatus
from cvslab.store import LabError, Store
from cvslab.targets import TargetSpec

ENGINE = r"F:\Github\chess-vision-studio-rust-engine"
OUT = Path(r"F:\Github\_parity_tmp\s6")
POOL = OUT / "pool-corrected"
STATE = OUT / "round2-state.json"
RESULT = OUT / "s6-campaign-result-v2.json"
SPEC = None  # set after the R0008 analyze sha is known


def stop(reason: str) -> int:
    payload = {"stopped": reason, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    STATE.write_text(json.dumps({**state(), "stop": payload}, indent=1, default=str), encoding="utf-8")
    print(f"STOP: {reason}", flush=True)
    return 3


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def save(**updates) -> None:
    data = {**state(), **updates}
    STATE.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] saved {list(updates)}", flush=True)


def record_id_for_fen(fen: str) -> str:
    return "pos_" + __import__("hashlib").sha256(chess.Board(fen).epd().encode()).hexdigest()[:20]


def main() -> int:
    global SPEC
    svc = LabService(Store(r"F:\Github\chess-vision-studio-lab\labstore"))
    prior = json.loads((OUT / "progress.json").read_text())
    old = {name: (entry.get("cached", entry)) for name, entry in prior.items()}
    run_dir = OUT / "s4run"
    analyze_sha = json.loads((run_dir / "manifest.json").read_text())["engine"]["binarySha256"] \
        if "engine" in json.loads((run_dir / "manifest.json").read_text()) else \
        json.loads((run_dir / "manifest.json").read_text())["engine"]["binarySha256"]
    SPEC = TargetSpec(family="search_deep_cp", authority="legacy.cvs.search.deep", producer=analyze_sha,
                      budget={"nodeBudget": 400000}, value_path=("targets", "scoreCpStm"),
                      pov="stm", target_type="cp", k=256.0, lam=1.0)
    save(analyze_sha=analyze_sha, spec_hash=SPEC.spec_hash())

    # ---- 1. corrected truthful S/N lineage ---------------------------------
    if "s_corrected" not in state():
        pool_manifest = json.loads((POOL / "pool-manifest.json").read_text())
        generated = {"positions": [json.loads(line) for line in
                                   (POOL / "positions.jsonl").read_text(encoding="utf-8").splitlines()],
                     "report": pool_manifest["report"]}
        source = snapshot_source(svc.store, POOL, name="s6-pool-corrected",
                                 license="engine self-play corpus", generated=generated,
                                 manifest=pool_manifest)
        normalization = svc.normalize([source.id], name="s6-corrected-canonical")
        if normalization.record_count != old["canonical"]["records"]:
            return stop(f"corrected lineage record count {normalization.record_count} != original "
                        f"{old['canonical']['records']}")
        carried = svc.copy_label_sets(old["import"]["normalization"], normalization.id)
        if carried["labels_dropped"]:
            return stop(f"label carry lost {carried['labels_dropped']} labels")
        save(s_corrected=source.id, n_corrected=normalization.id, carry=carried)
    st = state()

    # ---- 2. universe reconstruction (recorded selections, no resampling) ----
    if "universe_v2" not in state():
        records = _load_canonical(svc.store, svc.store.get(st["n_corrected"]))
        group_of = {record["record_id"]: record["group"] for record in records}
        run_positions = {row["id"]: row["fen"] for row in read_jsonl(run_dir / "positions.jsonl")}
        triage = read_jsonl(run_dir / "triage.jsonl")
        tier3 = read_jsonl(run_dir / "tier3.jsonl")
        selection = {"deep": [], "uniform": []}
        eligible = {}
        for row in triage:
            rid = record_id_for_fen(run_positions[row["id"]])
            eligible[rid] = bool(row.get("trainEligible", True))
            if row["selection"].get("deep"):
                selection["deep"].append(rid)
            if row["selection"].get("uniform"):
                selection["uniform"].append(rid)
        raw_counts = {arm: len(ids) for arm, ids in selection.items()}
        if raw_counts["deep"] != raw_counts["uniform"]:
            return stop(f"recorded arm counts differ at source: {raw_counts}")
        deep_nodes, deep_seconds = {}, {"PRIORITY": 0.0, "UNIFORM": 0.0}
        arm_of = {rid: "PRIORITY" for rid in selection["deep"]}
        for rid in selection["uniform"]:
            arm_of.setdefault(rid, "UNIFORM")
        for row in tier3:
            rid = record_id_for_fen(run_positions[row["id"]])
            deep_nodes[rid] = int(row["cost"]["nodes"])
            if arm_of.get(rid):
                deep_seconds[arm_of[rid]] += row["cost"]["wallMs"] / 1000.0
        candidates = [{"record_id": rid, "group": group_of[rid], "train_eligible": True}
                      for rid in sorted(group_of) if eligible.get(rid, False)]
        keep = {row["record_id"] for row in candidates}
        selection = {arm: [rid for rid in ids if rid in keep] for arm, ids in selection.items()}
        joined = {arm: len(ids) for arm, ids in selection.items()}
        if joined["deep"] != joined["uniform"]:
            return stop(f"arm counts differ after join: raw {raw_counts} -> joined {joined}")
        universe = freeze_universe(source_id=st["s_corrected"], normalization_id=st["n_corrected"],
                                   funnel_run_id=old["import"]["funnel_run"], policy_version="priority-v1",
                                   policy_hash=old["import"]["policy_hash"], candidates=candidates,
                                   deep_nodes=deep_nodes, selections=selection, split_seed=20260915)
        save(universe_v2={"candidates": len(universe.candidate_ids),
                          "arms": {a: len(ids) for a, ids in universe.arms.items()},
                          "universe_hash": universe.universe_hash(),
                          "split_policy_hash": universe.split_policy_hash(),
                          "candidate_universe_hash": universe.candidate_universe_hash()},
             deep_seconds=deep_seconds)
    uni_state = state()["universe_v2"]

    # rebuild the universe object for the remaining steps
    records = _load_canonical(svc.store, svc.store.get(st["n_corrected"]))
    group_of = {record["record_id"]: record["group"] for record in records}
    run_positions = {row["id"]: row["fen"] for row in read_jsonl(run_dir / "positions.jsonl")}
    triage = read_jsonl(run_dir / "triage.jsonl")
    tier3 = read_jsonl(run_dir / "tier3.jsonl")
    selection = {"deep": [], "uniform": []}
    eligible = {}
    for row in triage:
        rid = record_id_for_fen(run_positions[row["id"]])
        eligible[rid] = bool(row.get("trainEligible", True))
        if row["selection"].get("deep"):
            selection["deep"].append(rid)
        if row["selection"].get("uniform"):
            selection["uniform"].append(rid)
    deep_nodes = {record_id_for_fen(run_positions[row["id"]]): int(row["cost"]["nodes"]) for row in tier3}
    candidates = [{"record_id": rid, "group": group_of[rid], "train_eligible": True}
                  for rid in sorted(group_of) if eligible.get(rid, False)]
    keep = {row["record_id"] for row in candidates}
    selection = {arm: [rid for rid in ids if rid in keep] for arm, ids in selection.items()}
    universe = freeze_universe(source_id=st["s_corrected"], normalization_id=st["n_corrected"],
                               funnel_run_id=old["import"]["funnel_run"], policy_version="priority-v1",
                               policy_hash=old["import"]["policy_hash"], candidates=candidates,
                               deep_nodes=deep_nodes, selections=selection, split_seed=20260915)

    # ---- 3. T with the exact spec + TargetSpec-validated arm D pair --------
    if "t_v2" not in state():
        recipe = svc.create_training_recipe(
            name="s6-raw-ladder-v2",
            params={"EPOCHS": 40, "BATCH": 256, "LR": 0.003, "OPTIMIZER": "adam",
                    "INIT_STD": 0.05, "K": 256.0, "LAMBDA": 1.0},
            target_spec=SPEC, description="S6 supervision: deep-search targets, stm POV")
        report = freeze_arm_datasets(svc.store, universe, CampaignPlan(), recipe=recipe,
                                     deep_engine_seconds=state().get("deep_seconds"), name_prefix="s6v2",
                                     target_spec=SPEC)
        save(t_v2={"id": recipe.id, "hash": recipe.recipe_hash},
             d_v2={arm: {"id": report["arms"][arm]["dataset_id"],
                         "manifest": report["arms"][arm]["manifest_hash"],
                         "rows": report["arms"][arm]["rows"]} for arm in ("PRIORITY", "UNIFORM")},
             parity=report["parity"])

    # ---- 4. component-safe common evaluation -------------------------------
    if "d_eval" not in state():
        test_components = {component for component, split in universe.split_by_component.items()
                           if split == "test"}
        eval_ids = sorted(rid for rid, component in universe.components.items()
                          if component in test_components)
        arm_ids = set(universe.arms["PRIORITY"]) | set(universe.arms["UNIFORM"])
        overlap = sorted(set(eval_ids) & arm_ids)
        if overlap:
            return stop(f"common evaluation overlaps arm membership: {overlap[:3]}")
        by_id = {record["record_id"]: record for record in records}
        missing = [rid for rid in eval_ids if rid not in deep_nodes]
        bought_nodes, bought_seconds = 0, 0.0
        if missing:
            transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
            try:
                searcher = AnalyzeSearchProvider(transport, family="search_deep_cp")
                rows = []
                from cvslab.funnel.protocols import Position
                for rid in missing:
                    position = Position(fen=by_id[rid]["fen"])
                    label = searcher.search(position, node_budget=400000, pv_plies=12)
                    rows.append({"record_id": rid, "value": {"targets": {"scoreCpStm": label.score_cp_stm,
                                                                          "scoreCpWhite": label.score_cp_stm,
                                                                          "expectedScoreStm": None},
                                                              "deep": {"nodes": label.nodes}, "shallowToDeep": {}},
                                 "budget": {"nodeBudget": 400000, "nodes": label.nodes}})
                    bought_nodes += label.nodes
                    bought_seconds += label.wall_ms / 1000.0
            finally:
                transport.close()
            svc.register_labels(st["n_corrected"], family="search_deep_cp", producer=analyze_sha,
                                authority="legacy.cvs.search.deep", pov="stm", rows=rows)
        eval_dataset = svc.freeze_dataset(st["n_corrected"], name="s6v2-common-eval", record_ids=eval_ids,
                                          required_labels=["search_deep_cp"], target_spec=SPEC,
                                          fractions=(0.0, 0.0, 1.0),
                                          campaign={"purpose": "common evaluation instrument",
                                                    "universe_hash": universe.universe_hash(),
                                                    "split_policy_hash": universe.split_policy_hash(),
                                                    "test_components": len(test_components)})
        protocol = svc.create_eval_protocol(name="s6v2-common-eval", dataset_id=eval_dataset.id, split="test",
                                            k=256.0, lam=1.0, target_spec=SPEC,
                                            description="shared evaluation for every S6 arm")
        save(d_eval={"id": eval_dataset.id, "manifest": eval_dataset.manifest_hash,
                     "rows": eval_dataset.counts["records"]},
             e_common={"id": protocol.id, "hash": protocol.protocol_hash, "spec": SPEC.spec_hash()},
             eval_compute={"labels_bought": len(missing), "nodes": bought_nodes,
                           "engine_seconds": round(bought_seconds, 2)})
    st = state()
    eval_dataset = svc.store.get(st["d_eval"]["id"])
    protocol = svc.store.get(st["e_common"]["id"])
    recipe = svc.store.get(st["t_v2"]["id"])

    # ---- 5. smoke cell, then the remaining 39 ------------------------------
    plan = CampaignPlan()
    ablations = {}
    for arm in ("PRIORITY", "UNIFORM"):
        baseline = svc.register_baseline(name=f"S6V2_{arm}", dataset_id=st["d_v2"][arm]["id"],
                                         training_recipe_id=recipe.id, eval_protocol_id=protocol.id,
                                         model_config={"INPUT": "RAW", "H": 1}, notes=f"S6 v2 {arm}, H=1")
        ablations[(arm, 1)] = baseline.id
        for width in WIDTHS:
            if width != 1:
                ablations[(arm, width)] = svc.create_ablation(baseline_id=baseline.id,
                                                              overrides={"H": width},
                                                              notes=f"S6 v2 {arm}, H={width}").id

    def cell(width, arm, seed):
        [run] = svc.queue_runs(ablations[(arm, width)], seeds=(seed,))
        finished = svc.execute_run(run.id)
        entry = {"status": finished.status.value, "run_id": run.id, "width": width, "arm": arm, "seed": seed}
        if finished.status == RunStatus.COMPLETED:
            metric = next((m for m in finished.metrics if m.name == "test_loss"), None)
            if metric is None or not (metric.value == metric.value):
                entry.update(status="INVALID", cause="no finite primary metric")
            else:
                entry["test_loss"] = metric.value
                payload = json.loads(svc.store.abs(f"artifacts/{run.id}/eval.json").read_text(encoding="utf-8"))
                entry["eval_ids_match"] = payload["record_ids"] == sorted(
                    rid for rid in eval_dataset.splits[2].record_ids_hash and
                    [row["record_id"] for row in read_jsonl(svc.store.abs(eval_dataset.splits[2].path))])
        else:
            entry["cause"] = "; ".join(c.detail for c in finished.integrity if c.status == "fail") \
                or (finished.error or "unknown")
        return entry

    results = state().get("cells", {})
    smoke_id = "H01-PRIORITY-s0"
    if smoke_id not in results:
        smoke = cell(1, "PRIORITY", 0)
        results[smoke_id] = smoke
        save(cells=results)
        if smoke["status"] != "COMPLETE" or smoke.get("eval_ids_match") is False:
            return stop(f"smoke cell failed: {json.dumps(smoke)[:300]}")
        print(f"SMOKE OK: {json.dumps(smoke)[:200]}", flush=True)

    for width in WIDTHS:
        for arm in ("PRIORITY", "UNIFORM"):
            for seed in DEFAULT_SEEDS:
                cid = f"H{width:02d}-{arm}-s{seed}"
                if cid in results:
                    continue
                results[cid] = cell(width, arm, seed)
                save(cells=results)

    invalid = {cid: entry for cid, entry in results.items() if entry["status"] != "COMPLETE"}
    if invalid:
        return stop(f"{len(invalid)} campaign cells INVALID: {json.dumps(list(invalid.items())[:2])[:300]}")

    losses = {cid: entry["test_loss"] for cid, entry in results.items()}
    effects = {}
    for width in WIDTHS:
        effects[width] = paired_effect({s: losses[f"H{width:02d}-PRIORITY-s{s}"] for s in DEFAULT_SEEDS},
                                       {s: losses[f"H{width:02d}-UNIFORM-s{s}"] for s in DEFAULT_SEEDS})
    verdict = decide(effects)
    payload = {"campaign": "S6 round 2 — PRIORITY vs UNIFORM", "pins": {
                   "universe_hash": uni_state["universe_hash"], "split_policy_hash": uni_state["split_policy_hash"],
                   "candidate_universe_hash": uni_state["candidate_universe_hash"],
                   "policy_hash": old["import"]["policy_hash"], "spec_hash": SPEC.spec_hash(),
                   "s_corrected": st["s_corrected"], "n_corrected": st["n_corrected"],
                   "t": st["t_v2"], "d_priority": st["d_v2"]["PRIORITY"], "d_uniform": st["d_v2"]["UNIFORM"],
                   "d_eval": st["d_eval"], "e_common": st["e_common"], "parity": st["parity"],
                   "eval_compute": st.get("eval_compute")},
               "cells_total": 40, "cells_complete": 40, "invalid": {},
               "losses": losses, "effects_by_width": effects, "interaction": interaction_effects(effects),
               "decision": verdict,
               "preregistration": "paired seed-level differences, t 95% CI, every width required"}
    RESULT.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print("DECISION:", verdict, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
