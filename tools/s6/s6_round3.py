"""S6 Amendment 2: independent evaluation population -> smoke -> 40 cells -> analysis."""
import hashlib
import json
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, r"F:\Github\chess-vision-studio-lab")
from cvslab.data import _load_canonical, read_jsonl
from cvslab.funnel.campaign import DEFAULT_SEEDS, WIDTHS, decide, interaction_effects, paired_effect
from cvslab.funnel.pool import (DEFAULT_ENGINE_ARGS, PoolConfig, AnalyzeTeacher, SelfPlayGenerator,
                                snapshot_source, write_pool)
from cvslab.funnel.protocols import Position
from cvslab.funnel.providers import AnalyzeSearchProvider, AnalyzeTransport
from cvslab.schemas import RunStatus
from cvslab.service import LabService
from cvslab.store import Store
from cvslab.targets import TargetSpec

ENGINE = r"F:\Github\chess-vision-studio-rust-engine"
OUT = Path(r"F:\Github\_parity_tmp\s6")
EVAL_DIR = OUT / "eval-pool"
STATE = OUT / "round3-state.json"
RESULT = OUT / "s6-campaign-result-final.json"
EVAL_SEED = 777001
EVAL_ROWS = 200
HOLDOUT_OPENINGS = [
    {"name": "eval-bird", "moves": ["f2f4", "d7d5", "g1f3", "g8f6", "e2e3", "e7e6"]},
    {"name": "eval-nimzo-larsen", "moves": ["b2b3", "e7e5", "c1b2", "b8c6", "e2e3", "g8f6"]},
    {"name": "eval-kia", "moves": ["e2e4", "d7d6", "d2d3", "g8f6", "g1f3", "g7g6"]},
    {"name": "eval-trompowsky", "moves": ["d2d4", "g8f6", "c1g5", "e7e6", "e2e4", "h7h6"]},
    {"name": "eval-dutch", "moves": ["d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "e7e6"]},
    {"name": "eval-scandinavian", "moves": ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"]},
    {"name": "eval-alekhine", "moves": ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6"]},
    {"name": "eval-modern", "moves": ["e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "d7d6"]},
    {"name": "eval-petroff", "moves": ["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "d7d6"]},
    {"name": "eval-benoni", "moves": ["d2d4", "c7c5", "d4d5", "e7e6", "c2c4", "f8d6"]},
]


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


def deterministic_order(record_ids, seed):
    return sorted(record_ids, key=lambda r: hashlib.sha256(f"{seed}:{r}".encode()).hexdigest())


def main():
    svc = LabService(Store(r"F:\Github\chess-vision-studio-lab\labstore"))
    r2 = json.loads((OUT / "round2-state.json").read_text())
    r2b = json.loads((OUT / "round2b-state.json").read_text())
    round1 = {n: e.get("cached", e) for n, e in json.loads((OUT / "progress.json").read_text()).items()}
    spec = TargetSpec(family="search_deep_cp", authority="legacy.cvs.search.deep", producer=r2["analyze_sha"],
                      budget={"nodeBudget": 400000}, value_path=("targets", "scoreCpStm"), pov="stm",
                      k=256.0, lam=1.0)
    arms = r2b["arms_alltrain"]

    # ---- training-side components and members -----------------------------
    train_records = _load_canonical(svc.store, svc.store.get(r2["n_corrected"]))
    train_group = {record["record_id"]: record["group"] for record in train_records}
    run_positions = {row["id"]: row["fen"] for row in read_jsonl(OUT / "s4run" / "positions.jsonl")}
    arm_ids = set()
    for row in read_jsonl(OUT / "s4run" / "triage.jsonl"):
        if row["selection"].get("deep") or row["selection"].get("uniform"):
            arm_ids.add(rid(run_positions[row["id"]]))
    train_components = {train_group[record_id] for record_id in arm_ids}

    # ---- 1. fresh evaluation-only pool (held-out opening partition) --------
    if "eval_selection" not in state():
        games = 40
        for attempt in range(3):
            transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
            try:
                config = PoolConfig(seed=EVAL_SEED + attempt, games=games, max_plies=70, sample_every=2,
                                    min_ply=6, diversification_plies=(8, 10, 12), diversification_candidates=3,
                                    play_node_budget=20000, diversification_node_budget=2000,
                                    opening_lines=tuple(HOLDOUT_OPENINGS))
                generated = SelfPlayGenerator(config, AnalyzeTeacher(transport)).generate()
            finally:
                transport.close()
            manifest = write_pool(EVAL_DIR / f"attempt{attempt}", generated,
                                  {"config": config.canonical(), "engine": {"commit": "pinned"}},
                                  opening_lines=config.effective_openings())
            source = snapshot_source(svc.store, EVAL_DIR / f"attempt{attempt}", name=f"s6-eval-pool-{attempt}",
                                     license="engine self-play corpus", generated=generated, manifest=manifest)
            normalization = svc.normalize([source.id], name=f"s6-eval-canonical-{attempt}")
            eval_records = _load_canonical(svc.store, svc.store.get(normalization.id))

            # joint component graph: union source-components that share any canonical record
            parent = {}
            def find(node):
                parent.setdefault(node, node)
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            component_of = {record["record_id"]: record["group"] for record in train_records}
            for record in eval_records:
                if record["record_id"] in component_of:  # shared EPD across pools -> joint component
                    union(component_of[record["record_id"]], record["group"])
                component_of.setdefault(record["record_id"], record["group"])
            for record in train_records:
                find(record["group"])
            arm_joint = {find(component) for component in train_components}
            clean = [record["record_id"] for record in eval_records
                     if find(record["group"]) not in arm_joint]
            rejected = len(eval_records) - len(clean)
            save(eval_attempt={"attempt": attempt, "games": games, "source": source.id,
                               "normalization": normalization.id, "records": len(eval_records),
                               "clean": len(clean), "rejected_by_joint_component": rejected})
            if len(clean) >= EVAL_ROWS:
                break
            games *= 2  # fail closed: enlarge the pool without examining any target values
        else:
            return stop(f"fewer than {EVAL_ROWS} clean evaluation records after enlarging the pool")
        selected = deterministic_order(clean, EVAL_SEED)[:EVAL_ROWS]
        save(eval_selection={"source": source.id, "normalization": normalization.id,
                             "clean": len(clean), "selected": len(selected), "seed": EVAL_SEED})
    st = state()
    eval_source = st["eval_selection"]["source"]
    eval_normalization = st["eval_selection"]["normalization"]
    # selection is recomputed deterministically from the recorded clean set
    eval_records = _load_canonical(svc.store, svc.store.get(eval_normalization))
    component_of = {record["record_id"]: record["group"] for record in train_records}
    parent = {}
    def find2(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    def union2(a, b):
        ra, rb = find2(a), find2(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for record in eval_records:
        if record["record_id"] in component_of:
            union2(component_of[record["record_id"]], record["group"])  # shared EPD -> joint component
        component_of.setdefault(record["record_id"], record["group"])
    for record in train_records:
        find2(record["group"])
    arm_joint = {find2(component) for component in train_components}
    clean = [record["record_id"] for record in eval_records if find2(record["group"]) not in arm_joint]
    selected = deterministic_order(clean, EVAL_SEED)[:EVAL_ROWS]
    if len(selected) < EVAL_ROWS:
        return stop("clean evaluation population shrank below the required count")

    # ---- 2. buy exactly the required labels (same analyze identity) --------
    by_id = {record["record_id"]: record for record in eval_records}
    existing_labels = set()
    label_dir = svc.store.abs(f"canonical/{eval_normalization}/labels")
    if label_dir.is_dir():
        for path in sorted(label_dir.glob("*.jsonl")):
            existing_labels |= {row["record_id"] for row in read_jsonl(path)}
    to_buy = [record_id for record_id in selected if record_id not in existing_labels]
    if to_buy:
        transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
        bought_nodes, bought_seconds = 0, 0.0
        try:
            searcher = AnalyzeSearchProvider(transport, family="search_deep_cp")
            rows = []
            for record_id in to_buy:
                fen = by_id[record_id]["fen"]
                label = searcher.search(Position(fen=fen), node_budget=400000, pv_plies=12)
                white = label.score_cp_stm if fen.split()[1] == "w" else -label.score_cp_stm
                rows.append({"record_id": record_id,
                             "value": {"targets": {"scoreCpStm": label.score_cp_stm,
                                                   "scoreCpWhite": white, "expectedScoreStm": None},
                                       "deep": {"nodes": label.nodes}, "shallowToDeep": {}},
                             "budget": {"nodeBudget": 400000, "nodes": label.nodes}})
                bought_nodes += label.nodes
                bought_seconds += label.wall_ms / 1000.0
        finally:
            transport.close()
        svc.register_labels(eval_normalization, family="search_deep_cp", producer=r2["analyze_sha"],
                            authority="legacy.cvs.search.deep", pov="stm", rows=rows)
        save(eval_compute={"labels_bought": len(to_buy), "nodes": bought_nodes,
                           "engine_seconds": round(bought_seconds, 2)})

    # ---- 3. D_EVAL + E_COMMON, disjointness asserted -----------------------
    if "d_eval" not in state():
        dataset = svc.freeze_dataset(eval_normalization, name="s6-common-eval",
                                     record_ids=selected, required_labels=["search_deep_cp"],
                                     target_spec=spec, fractions=(0.0, 0.0, 1.0),
                                     campaign={"purpose": "independent common evaluation instrument",
                                               "amendment": "tools/s6/s6-preregistration-amendment-2.json",
                                               "eval_source": eval_source, "eval_seed": EVAL_SEED,
                                               "selection_rule": "lowest sha256(seed:record_id) order",
                                               "clean_population": len(clean),
                                               "rejected_by_joint_component": st["eval_attempt"]["rejected_by_joint_component"],
                                               "arm_components_excluded": len(train_components)})
        protocol = svc.create_eval_protocol(name="s6-common-eval", dataset_id=dataset.id, split="test",
                                            k=256.0, lam=1.0, target_spec=spec,
                                            description="independent component-disjoint evaluation, 400k labels")
        save(d_eval={"id": dataset.id, "manifest": dataset.manifest_hash, "rows": dataset.counts["records"]},
             e_common={"id": protocol.id, "hash": protocol.protocol_hash, "spec": spec.spec_hash()})
    st = state()
    eval_dataset = svc.store.get(st["d_eval"]["id"])
    protocol = svc.store.get(st["e_common"]["id"])
    recipe = svc.store.get(r2["t_v2"]["id"])
    eval_rows = [row["record_id"] for row in read_jsonl(svc.store.abs(eval_dataset.splits[2].path))]
    if set(eval_rows) & arm_ids:
        return stop("record-level overlap between evaluation and arms")
    eval_joint = {find2(component_of[record_id]) for record_id in eval_rows}
    if eval_joint & arm_joint:
        return stop("joint-component overlap between evaluation and arms")

    # ---- 4. smoke cell, then 39 --------------------------------------------
    ablation_ids = {}
    if "ablations" not in state():
        for arm in ("PRIORITY", "UNIFORM"):
            baseline = svc.register_baseline(name=f"S6V4_{arm}", dataset_id=arms[arm]["id"],
                                             training_recipe_id=recipe.id, eval_protocol_id=protocol.id,
                                             model_config={"INPUT": "RAW", "H": 1}, notes=f"S6 v3 {arm}, H=1")
            ablation_ids[(arm, 1)] = baseline.id
            for width in WIDTHS:
                if width != 1:
                    ablation_ids[(arm, width)] = svc.create_ablation(baseline_id=baseline.id,
                                                                     overrides={"H": width},
                                                                     notes=f"S6 v3 {arm}, H={width}").id
        save(ablations={f"{a}-{w}": i for (a, w), i in ablation_ids.items()})
    ablation_ids = {tuple(k.split("-")): v for k, v in state()["ablations"].items()}
    ablation_ids = {(arm, int(width)): ident for (arm, width), ident in ablation_ids.items()}

    def cell(width, arm, seed):
        [run] = svc.queue_runs(ablation_ids[(arm, width)], seeds=(seed,))
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
                             eval_ids_match=(sorted(payload["record_ids"]) == sorted(eval_rows)
                                             and len(payload["record_ids"]) == len(eval_rows)))
        else:
            entry["cause"] = "; ".join(c.detail for c in finished.integrity if c.status == "fail") \
                or (finished.error or "unknown")
        return entry

    results = state().get("cells", {})
    if "H01-PRIORITY-s0" not in results:
        smoke = cell(1, "PRIORITY", 0)
        results["H01-PRIORITY-s0"] = smoke
        save(cells=results)
        if (smoke["status"] != RunStatus.COMPLETED.value or smoke.get("eval_ids_match") is not True
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

    invalid = {cid: entry for cid, entry in results.items() if entry["status"] != RunStatus.COMPLETED.value}
    if invalid:
        return stop(f"{len(invalid)} cells INVALID: {json.dumps(list(invalid.items())[:2])[:300]}")
    losses = {cid: entry["test_loss"] for cid, entry in results.items()}
    effects = {width: paired_effect({s: losses[f"H{width:02d}-PRIORITY-s{s}"] for s in DEFAULT_SEEDS},
                                    {s: losses[f"H{width:02d}-UNIFORM-s{s}"] for s in DEFAULT_SEEDS})
               for width in WIDTHS}
    payload = {"campaign": "S6 — PRIORITY vs UNIFORM (independent evaluation population)",
               "amendments": ["tools/s6/s6-preregistration-amendment-1.json",
                              "tools/s6/s6-preregistration-amendment-2.json"],
               "pins": {"s_train": r2["s_corrected"], "n_train": r2["n_corrected"],
                        "funnel_run": round1["import"]["funnel_run"], "policy_hash": round1["import"]["policy_hash"],
                        "eval_source": eval_source, "eval_normalization": eval_normalization,
                        "eval_selected": len(selected), "clean_population": len(clean),
                        "spec_hash": spec.spec_hash(), "recipe": r2["t_v2"], "arms": arms,
                        "d_eval": st["d_eval"], "e_common": st["e_common"], "parity": r2b["parity"],
                        "eval_compute": st.get("eval_compute")},
               "cells_total": 40, "cells_complete": 40, "losses": losses,
               "effects_by_width": effects, "interaction": interaction_effects(effects),
               "decision": decide(effects)}
    RESULT.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print("DECISION:", payload["decision"], flush=True)
    for width, effect in sorted(effects.items()):
        print(f"H{width}: {effect['width_estimate']:+.6f} [{effect['ci_low']:+.6f}, {effect['ci_high']:+.6f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
