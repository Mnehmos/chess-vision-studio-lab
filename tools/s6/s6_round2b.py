"""S6 round 2b: all-train arms, component-disjoint D_EVAL, smoke, 40 cells, analysis."""
import hashlib
import json
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, r"F:\Github\chess-vision-studio-lab")
from cvslab.data import _load_canonical, read_jsonl
from cvslab.funnel.campaign import (CampaignPlan, DEFAULT_SEEDS, WIDTHS, decide, freeze_arm_datasets,
                                    freeze_universe, interaction_effects, paired_effect)
from cvslab.funnel.pool import DEFAULT_ENGINE_ARGS
from cvslab.funnel.protocols import Position
from cvslab.funnel.providers import AnalyzeSearchProvider, AnalyzeTransport
from cvslab.schemas import RunStatus
from cvslab.service import LabService
from cvslab.store import Store
from cvslab.targets import TargetSpec

ENGINE = r"F:\Github\chess-vision-studio-rust-engine"
OUT = Path(r"F:\Github\_parity_tmp\s6")
STATE = OUT / "round2b-state.json"
RESULT = OUT / "s6-campaign-result-v3.json"


def state():
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def save(**updates):
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] saved {list(updates)}", flush=True)


def stop(reason):
    save(stop={"reason": reason, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"STOP: {reason}", flush=True)
    return 3


def rid(fen):
    return "pos_" + hashlib.sha256(chess.Board(fen).epd().encode()).hexdigest()[:20]


def main():
    svc = LabService(Store(r"F:\Github\chess-vision-studio-lab\labstore"))
    r2 = json.loads((OUT / "round2-state.json").read_text())
    round1 = {name: entry.get("cached", entry)
              for name, entry in json.loads((OUT / "progress.json").read_text()).items()}
    r2["import"] = round1["import"]
    run_dir = OUT / "s4run"
    spec = TargetSpec(family="search_deep_cp", authority="legacy.cvs.search.deep",
                      producer=r2["analyze_sha"], budget={"nodeBudget": 400000},
                      value_path=("targets", "scoreCpStm"), pov="stm", k=256.0, lam=1.0)

    records = _load_canonical(svc.store, svc.store.get(r2["n_corrected"]))
    group_of = {record["record_id"]: record["group"] for record in records}
    run_positions = {row["id"]: row["fen"] for row in read_jsonl(run_dir / "positions.jsonl")}
    triage = read_jsonl(run_dir / "triage.jsonl")
    tier3 = read_jsonl(run_dir / "tier3.jsonl")
    selection, eligible = {"deep": [], "uniform": []}, {}
    for row in triage:
        record_id = rid(run_positions[row["id"]])
        eligible[record_id] = bool(row.get("trainEligible", True))
        if row["selection"].get("deep"):
            selection["deep"].append(record_id)
        if row["selection"].get("uniform"):
            selection["uniform"].append(record_id)
    deep_nodes = {rid(run_positions[row["id"]]): int(row["cost"]["nodes"]) for row in tier3}
    candidates = [{"record_id": record_id, "group": group_of[record_id], "train_eligible": True}
                  for record_id in sorted(group_of) if eligible.get(record_id, False)]
    keep = {row["record_id"] for row in candidates}
    selection = {arm: [record_id for record_id in ids if record_id in keep] for arm, ids in selection.items()}
    if len(selection["deep"]) != len(selection["uniform"]):
        return stop(f"arm counts differ after join: {len(selection['deep'])} vs {len(selection['uniform'])}")
    universe = freeze_universe(source_id=r2["s_corrected"], normalization_id=r2["n_corrected"],
                              funnel_run_id=r2["import"]["funnel_run"], policy_version="priority-v1",
                              policy_hash=r2["import"]["policy_hash"], candidates=candidates,
                              deep_nodes=deep_nodes, selections=selection, split_seed=20260915)

    # ---- arms as treatments: all selected records train (1,0,0) -------------
    if "arms_alltrain" not in state():
        recipe = svc.store.get(r2["t_v2"]["id"])
        report = freeze_arm_datasets(svc.store, universe, CampaignPlan(), recipe=recipe,
                                     deep_engine_seconds=r2.get("deep_seconds"), name_prefix="s6v3",
                                     target_spec=spec, training_fractions=(1.0, 0.0, 0.0))
        save(arms_alltrain={arm: {"id": report["arms"][arm]["dataset_id"],
                                  "manifest": report["arms"][arm]["manifest_hash"],
                                  "rows": report["arms"][arm]["rows"]} for arm in ("PRIORITY", "UNIFORM")},
             parity=report["parity"])
    arms = state()["arms_alltrain"]
    for arm in ("PRIORITY", "UNIFORM"):
        dataset = svc.store.get(arms[arm]["id"])
        if dataset.splits[0].count != 94 or dataset.splits[1].count or dataset.splits[2].count:
            return stop(f"{arm}: not all-train layout: {[s.count for s in dataset.splits]}")

    # ---- common evaluation: test components with zero arm members -----------
    if "d_eval" not in state():
        arm_ids = set(universe.arms["PRIORITY"]) | set(universe.arms["UNIFORM"])
        components_touched = {universe.components[record_id] for record_id in arm_ids}
        eval_components = {component for component, split in universe.split_by_component.items()
                          if split == "test" and component not in components_touched}
        eval_ids = sorted(record_id for record_id, component in universe.components.items()
                          if component in eval_components)
        if not eval_ids:
            return stop("no component-disjoint common evaluation population exists")
        overlap = set(eval_components) & components_touched
        if overlap:
            return stop(f"component overlap between evaluation and arms: {sorted(overlap)[:3]}")
        by_id = {record["record_id"]: record for record in records}
        missing = [record_id for record_id in eval_ids if record_id not in deep_nodes]
        bought_nodes, bought_seconds = 0, 0.0
        if missing:
            transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
            try:
                searcher = AnalyzeSearchProvider(transport, family="search_deep_cp")
                rows = []
                for record_id in missing:
                    label = searcher.search(Position(fen=by_id[record_id]["fen"]),
                                            node_budget=400000, pv_plies=12)
                    rows.append({"record_id": record_id,
                                 "value": {"targets": {"scoreCpStm": label.score_cp_stm,
                                                       "scoreCpWhite": label.score_cp_stm,
                                                       "expectedScoreStm": None},
                                           "deep": {"nodes": label.nodes}, "shallowToDeep": {}},
                                 "budget": {"nodeBudget": 400000, "nodes": label.nodes}})
                    bought_nodes += label.nodes
                    bought_seconds += label.wall_ms / 1000.0
            finally:
                transport.close()
            svc.register_labels(r2["n_corrected"], family="search_deep_cp", producer=r2["analyze_sha"],
                                authority="legacy.cvs.search.deep", pov="stm", rows=rows)
        dataset = svc.freeze_dataset(r2["n_corrected"], name="s6v3-common-eval", record_ids=eval_ids,
                                     required_labels=["search_deep_cp"], target_spec=spec,
                                     fractions=(0.0, 0.0, 1.0),
                                     campaign={"purpose": "common evaluation instrument",
                                               "selection_split_policy_hash": universe.split_policy_hash(),
                                               "training_layout": "not-applicable",
                                               "eval_components": len(eval_components),
                                               "arm_components_excluded": len(components_touched)})
        protocol = svc.create_eval_protocol(name="s6v3-common-eval", dataset_id=dataset.id, split="test",
                                            k=256.0, lam=1.0, target_spec=spec,
                                            description="shared evaluation instrument for every S6 arm")
        save(d_eval={"id": dataset.id, "manifest": dataset.manifest_hash, "rows": dataset.counts["records"],
                     "components": len(eval_components)},
             e_common={"id": protocol.id, "hash": protocol.protocol_hash, "spec": spec.spec_hash()},
             eval_compute={"labels_bought": len(missing), "nodes": bought_nodes,
                           "engine_seconds": round(bought_seconds, 2)})
    st = state()
    eval_dataset = svc.store.get(st["d_eval"]["id"])
    protocol = svc.store.get(st["e_common"]["id"])
    recipe = svc.store.get(r2["t_v2"]["id"])
    eval_component_set = {universe.components[record_id] for record_id in
                          [row["record_id"] for row in read_jsonl(svc.store.abs(eval_dataset.splits[2].path))]}
    arm_component_set = {universe.components[record_id] for record_id in
                         set(universe.arms["PRIORITY"]) | set(universe.arms["UNIFORM"])}
    if eval_component_set & arm_component_set:
        return stop("component overlap re-check failed before the smoke run")

    # ---- smoke, then the remaining cells -----------------------------------
    plan = CampaignPlan()
    ablations = {}
    for arm in ("PRIORITY", "UNIFORM"):
        baseline = svc.register_baseline(name=f"S6V3_{arm}", dataset_id=arms[arm]["id"],
                                         training_recipe_id=recipe.id, eval_protocol_id=protocol.id,
                                         model_config={"INPUT": "RAW", "H": 1}, notes=f"S6 v3 {arm}, H=1")
        ablations[(arm, 1)] = baseline.id
        for width in WIDTHS:
            if width != 1:
                ablations[(arm, width)] = svc.create_ablation(baseline_id=baseline.id,
                                                              overrides={"H": width},
                                                              notes=f"S6 v3 {arm}, H={width}").id

    eval_row_ids = sorted(row["record_id"] for row in read_jsonl(svc.store.abs(eval_dataset.splits[2].path)))

    def cell(width, arm, seed):
        [run] = svc.queue_runs(ablations[(arm, width)], seeds=(seed,))
        finished = svc.execute_run(run.id)
        entry = {"status": finished.status.value, "run_id": run.id}
        if finished.status == RunStatus.COMPLETED:
            metric = next((m for m in finished.metrics if m.name == "test_loss"), None)
            if metric is None or metric.value != metric.value:
                entry.update(status="INVALID", cause="no finite primary metric")
            else:
                payload = json.loads(svc.store.abs(f"artifacts/{run.id}/eval.json").read_text(encoding="utf-8"))
                entry.update(test_loss=metric.value, param_count=finished.param_count,
                             dataset_id=finished.dataset_id, eval_dataset_id=finished.eval_dataset_id,
                             eval_ids_match=payload["record_ids"] == eval_row_ids)
        else:
            entry["cause"] = "; ".join(c.detail for c in finished.integrity if c.status == "fail") \
                or (finished.error or "unknown")
        return entry

    results = state().get("cells", {})
    if "H01-PRIORITY-s0" not in results:
        smoke = cell(1, "PRIORITY", 0)
        results["H01-PRIORITY-s0"] = smoke
        save(cells=results)
        if (smoke["status"] != "COMPLETE" or smoke.get("eval_ids_match") is not True
                or smoke.get("dataset_id") != arms["PRIORITY"]["id"]
                or smoke.get("eval_dataset_id") != st["d_eval"]["id"]):
            return stop(f"smoke cell failed: {json.dumps(smoke)[:300]}")
        print(f"SMOKE OK: test_loss={smoke['test_loss']:.6f} params={smoke['param_count']}", flush=True)

    for width in WIDTHS:
        for arm in ("PRIORITY", "UNIFORM"):
            for seed in DEFAULT_SEEDS:
                cid = f"H{width:02d}-{arm}-s{seed}"
                if cid not in results:
                    results[cid] = cell(width, arm, seed)
                    save(cells=results)

    invalid = {cid: entry for cid, entry in results.items() if entry["status"] != "COMPLETE"}
    if invalid:
        return stop(f"{len(invalid)} cells INVALID: {json.dumps(list(invalid.items())[:2])[:300]}")
    losses = {cid: entry["test_loss"] for cid, entry in results.items()}
    effects = {width: paired_effect({s: losses[f"H{width:02d}-PRIORITY-s{s}"] for s in DEFAULT_SEEDS},
                                    {s: losses[f"H{width:02d}-UNIFORM-s{s}"] for s in DEFAULT_SEEDS})
               for width in WIDTHS}
    payload = {"campaign": "S6 — PRIORITY vs UNIFORM (round 2, all-train arms + common eval)",
               "preregistration_amendment": "tools/s6/s6-preregistration-amendment-1.json",
               "pins": {"s_corrected": r2["s_corrected"], "n_corrected": r2["n_corrected"],
                        "funnel_run": r2["import"]["funnel_run"], "policy_hash": r2["import"]["policy_hash"],
                        "universe_hash": universe.universe_hash(),
                        "selection_split_policy_hash": universe.split_policy_hash(),
                        "candidate_universe_hash": universe.candidate_universe_hash(),
                        "spec_hash": spec.spec_hash(), "t": r2["t_v2"], "arms": arms,
                        "d_eval": st["d_eval"], "e_common": st["e_common"], "parity": st["parity"],
                        "eval_compute": st.get("eval_compute")},
               "cells_total": 40, "cells_complete": 40, "losses": losses,
               "effects_by_width": effects, "interaction": interaction_effects(effects),
               "decision": decide(effects)}
    RESULT.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print("DECISION:", payload["decision"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
