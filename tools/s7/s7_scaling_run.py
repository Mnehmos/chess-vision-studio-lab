"""S7-S execution: universe -> calibration -> the four 20x label streams -> nested prefix arms -> X####.

Steps (each resumable, state in tools/s7/s7-scaling-state.json):

    universe    snapshot the pool (S####), normalize (N####), leakage-join against all 575
                N0010 records, freeze the universe and its one deterministic order
    calibrate   64 records x four depths, NODE COUNTS ONLY (never a chess value)
    labels      buy each depth ONCE out to the 20x crossing (parallel, margin only to ensure
                the labels exist; the stop is computed from realized nodes)
    arms        construct the sixteen closest-prefix arms, assert both nesting dimensions,
                freeze the sixteen D####
    experiment  four T#### recipes, 64 A#### nulled to the reserved X####, X#### with the
                320-cell membership
    run         queue and execute the matrix
    analyse     frontier per scale, scaling per depth, interaction; seal X####

Firewall: no training outcome is computed until `arms` and `experiment` are finished, and no
chess value ever influences membership.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import chess

from cvslab.data import _load_canonical, read_jsonl
from cvslab.funnel.campaign import paired_effect
from cvslab.funnel.pool import DEFAULT_ENGINE_ARGS, engine_identity, snapshot_source
from cvslab.funnel.providers import AnalyzeSearchProvider, AnalyzeTransport
from cvslab.hashing import hash_obj, sha256_file
from cvslab.schemas import Dataset, TERMINAL_RUN_STATUSES, Run, TrainingRecipe
from cvslab.service import LabService, protocol_target_spec, recipe_target_spec
from cvslab.store import LabError, Store
from cvslab.targets import TargetSpec

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools" / "s7"
OUT = pathlib.Path(r"F:\Github\_parity_tmp\s7scaling")
POOL = OUT / "pool"
STATE = TOOLS / "s7-scaling-state.json"
PREREG = TOOLS / "s7-scaling-preregistration.json"
ENGINE = r"F:\Github\chess-vision-studio-rust-engine"

SCALES = {"2x": 74_570_982, "5x": 186_427_455, "10x": 372_854_910, "20x": 745_709_820}
DEPTHS = (400_000, 160_000, 64_000, 16_000)
DEPTH_LABEL = {400_000: "400k", 160_000: "160k", 64_000: "64k", 16_000: "16k"}
WIDTHS = (1, 4, 16, 32)
SEEDS = (0, 1, 2, 3, 4)
ORDER_SEED = 20260920
UNIVERSE_TARGET = 60_000
PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"
DECLARATION = ("S7-S scaling arm: trained on its own depth's observation, examined on the frozen "
               "instrument E0004/D0010")
SPECS = {
    400_000: TargetSpec(family="search_deep_cp", authority="legacy.cvs.search.deep", producer=PRODUCER,
                        budget={"nodeBudget": 400_000}, value_path=("targets", "scoreCpStm"), pov="stm",
                        k=256.0, lam=1.0),
    160_000: TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow", producer=PRODUCER,
                        budget={"nodeBudget": 160_000}, value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0),
    64_000: TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow", producer=PRODUCER,
                       budget={"nodeBudget": 64_000}, value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0),
    16_000: TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow", producer=PRODUCER,
                       budget={"nodeBudget": 16_000}, value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0),
}
EXPECTED_PRIMARY_NODES = 2_982_839_280


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] saved {list(updates)}", flush=True)


def record_id(fen: str) -> str:
    return "pos_" + hashlib.sha256(chess.Board(fen).epd().encode()).hexdigest()[:20]


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


# ------------------------------------------------------------------ universe ----
def step_universe() -> int:
    st = state()
    if "universe" in st:
        print("[cached] universe")
        return 0
    service = svc()
    generated = {"games": read_jsonl(POOL / "games.jsonl"), "positions": read_jsonl(POOL / "positions.jsonl")}
    manifest = json.loads((POOL / "pool-manifest.json").read_text(encoding="utf-8"))
    source = snapshot_source(service.store, POOL, name="s7s-pool", license="engine self-play corpus",
                             generated=generated, manifest=manifest)
    normalization = service.normalize([source.id], name="s7s-canonical")
    records = _load_canonical(service.store, service.store.get(normalization.id))
    print(f"source {source.id}: {source.row_count} rows; normalization {normalization.id}: "
          f"{len(records)} records", flush=True)

    # leakage join against the FULL 575-record evaluation source (not only D0010)
    eval_records = _load_canonical(service.store, service.store.get("N0010"))
    d10: Dataset = service.store.get_as("D0010", Dataset)
    instrument_ids = sorted(row["record_id"] for row in read_jsonl(service.store.abs(d10.splits[2].path)))
    parent: dict = {}

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

    component_of = {record["record_id"]: record["group"] for record in records}
    for record in eval_records:
        if record["record_id"] in component_of:
            union(component_of[record["record_id"]], record["group"])
        component_of.setdefault(record["record_id"], record["group"])
    for record in records:
        find(record["group"])
    protected = {find(component_of[rid]) for rid in instrument_ids}
    population = {find(component_of[record["record_id"]]) for record in eval_records}
    touching_instrument = [r["record_id"] for r in records if find(component_of[r["record_id"]]) in protected]
    touching_population = [r["record_id"] for r in records if find(component_of[r["record_id"]]) in population]
    removed = set(touching_instrument) | set(touching_population)
    clean = sorted(record["record_id"] for record in records if record["record_id"] not in removed)
    share = len(removed) / max(len(records), 1)
    print(f"leakage: {len(removed)} of {len(records)} records removed ({share:.3%}); "
          f"instrument-adjacent {len(touching_instrument)}; population-adjacent {len(touching_population)}",
          flush=True)
    if share > 0.05:
        print("STOP: leakage removal exceeds the 5% tolerance")
        return 3
    if len(clean) < UNIVERSE_TARGET:
        print(f"STOP: only {len(clean)} clean candidates (< {UNIVERSE_TARGET}); generate more")
        return 3
    order = sorted(clean, key=lambda rid: hashlib.sha256(f"{ORDER_SEED}:{rid}".encode()).hexdigest())
    save(universe={"source": source.id, "normalization": normalization.id,
                   "records": len(records), "removed": len(removed), "clean": len(clean),
                   "universe_hash": hash_obj(clean), "order_hash": hash_obj(order),
                   "instrument_adjacent": len(touching_instrument),
                   "population_adjacent": len(touching_population)},
         order=order)
    print(f"universe {len(clean)} clean; order hash {hash_obj(order)[:24]}…", flush=True)
    return 0


# ---------------------------------------------------------------- calibration ---
def _buy(rows: list[tuple[str, int]], workers: int) -> list[dict]:
    """Buy labels for (record_id, fen, budget) triples in parallel; node counts + wall only."""
    chunks = [rows[index::workers] for index in range(workers)]

    def run(chunk: list[tuple[str, str, int]]) -> list[dict]:
        if not chunk:
            return []
        transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
        out: list[dict] = []
        try:
            providers = {budget: AnalyzeSearchProvider(
                transport, family="search_deep_cp" if budget == 400_000 else "search_shallow_cp")
                for budget in DEPTHS}
            for rid, fen, budget in chunk:
                label = providers[budget].search(_Position(fen), node_budget=budget,
                                                  pv_plies=12 if budget == 400_000 else 8)
                white = label.score_cp_stm if fen.split()[1] == "w" else -label.score_cp_stm
                if budget == 400_000:
                    value = {"targets": {"scoreCpStm": label.score_cp_stm, "scoreCpWhite": white,
                                         "expectedScoreStm": None},
                             "deep": {"nodes": label.nodes}, "shallowToDeep": {}}
                else:
                    # the tier1-shaped row: the target plus its telemetry, exactly like the
                    # labels the funnel import produced (scoreCpStm lives at the top level)
                    value = {"scoreCpStm": label.score_cp_stm, "bestMove": label.best_move,
                             "mate": label.mate, "pv": list(label.pv), "nodes": label.nodes,
                             "nodeBudget": budget, **dict(label.extra)}
                out.append({"record_id": rid, "budget_nodes": int(label.nodes),
                            "wall_ms": float(label.wall_ms),
                            "row": {"record_id": rid, "value": value,
                                    "budget": {"nodeBudget": budget, "nodes": int(label.nodes)}}})
        finally:
            transport.close()
        return out

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, chunk in enumerate(pool.map(run, chunks), start=1):
            results.extend(chunk)
            if index % max(1, workers // 2) == 0:
                print(f"    ... {len(results)} labels bought", flush=True)
    return results


class _Position:
    def __init__(self, fen: str):
        self.fen = fen


def step_calibrate() -> int:
    st = state()
    if "calibration" in st:
        print("[cached] calibration")
        return 0
    order = st["order"]
    sample = order[:64]
    records = {r["record_id"]: r for r in _load_canonical(svc().store, svc().store.get(st["universe"]["normalization"]))}
    jobs = [(rid, records[rid]["fen"], budget) for rid in sample for budget in DEPTHS]
    bought = _buy(jobs, workers=8)
    means = {}
    for budget in DEPTHS:
        nodes = [row["budget_nodes"] for row in bought if row["row"]["budget"]["nodeBudget"] == budget]
        means[str(budget)] = sum(nodes) / len(nodes)
        print(f"calibration {budget}: mean {means[str(budget)]:.0f} nodes over {len(nodes)} labels")
    save(calibration={"sample": len(sample), "means": means,
                      "predicted_rows_for_20x": {str(budget): math.ceil(SCALES["20x"] / means[str(budget)])
                                                 for budget in DEPTHS}})
    return 0


# -------------------------------------------------------------------- labels ----
def step_labels() -> int:
    st = state()
    if "labels" in st:
        print("[cached] labels")
        return 0
    service = svc()
    order = st["order"]
    records = {r["record_id"]: r for r in _load_canonical(service.store,
                                                          service.store.get(st["universe"]["normalization"]))}
    means = {int(k): v for k, v in st["calibration"]["means"].items()}
    purchased: dict[str, list[dict]] = {}
    for budget in DEPTHS:
        target = SCALES["20x"]
        need = math.ceil(target / means[budget])
        bought: list[dict] = []
        done = 0
        while True:
            want = min(len(order), max(need + 32, int(need * 1.03)))
            if want <= done:
                print(f"STOP: depth {budget} ran out of candidates ({done} of {len(order)})")
                return 3
            jobs = [(rid, records[rid]["fen"], budget) for rid in order[done:want]]
            print(f"depth {budget}: buying {len(jobs)} labels ({done} → {want})", flush=True)
            bought.extend(_buy(jobs, workers=8))
            done = want
            cumulative = []
            total = 0
            for row in bought:
                total += row["budget_nodes"]
                cumulative.append(total)
            best = min(range(len(cumulative)), key=lambda index: abs(cumulative[index] - target))
            if best < len(cumulative) - 2 or done >= len(order):
                break
            need = best + 1
        purchased[str(budget)] = bought
        with open(OUT / f"labels-{budget}.jsonl", "w", encoding="utf-8", newline="\n") as handle:
            for row in bought:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        total = sum(row["budget_nodes"] for row in bought)
        print(f"depth {budget}: {len(bought)} labels, {total:,} realized nodes "
              f"(target {target:,}; ratio {total / target:.3f})", flush=True)
        family = "search_deep_cp" if budget == 400_000 else "search_shallow_cp"
        authority = "legacy.cvs.search.deep" if budget == 400_000 else "legacy.cvs.search.shallow"
        service.register_labels(st["universe"]["normalization"], family=family, producer=PRODUCER,
                                authority=authority, pov="stm",
                                rows=[row["row"] for row in bought])
    realized = sum(sum(row["budget_nodes"] for row in rows) for rows in purchased.values())
    save(labels={"per_depth": {str(budget): {"labels": len(purchased[str(budget)]),
                                             "realized_nodes": sum(r["budget_nodes"]
                                                                   for r in purchased[str(budget)])}
                               for budget in DEPTHS},
                 "total_realized_nodes": realized,
                 "expected_primary_nodes": EXPECTED_PRIMARY_NODES})
    print(f"total realized teacher nodes: {realized:,} "
          f"(expected {EXPECTED_PRIMARY_NODES:,}; ratio {realized / EXPECTED_PRIMARY_NODES:.3f})", flush=True)
    return 0


# ---------------------------------------------------------------------- arms ----
def step_arms() -> int:
    st = state()
    if "arms" in st:
        print("[cached] arms")
        return 0
    service = svc()
    order = st["order"]
    xid = service.store.next_id("X")
    arms: list[dict] = []
    for budget in DEPTHS:
        bought = read_jsonl(OUT / f"labels-{budget}.jsonl")
        cumulative, total = [], 0
        for row in bought:
            total += row["budget_nodes"]
            cumulative.append(total)
        for scale, target in SCALES.items():
            best = min(range(len(cumulative)), key=lambda index: abs(cumulative[index] - target))
            prefix = order[: best + 1]
            realized = cumulative[best]
            family = "search_deep_cp" if budget == 400_000 else "search_shallow_cp"
            dataset = service.freeze_dataset(
                st["universe"]["normalization"], name=f"s7s-{scale}-{DEPTH_LABEL[budget]}",
                record_ids=prefix, required_labels=[family], target_spec=SPECS[budget],
                fractions=(1.0, 0.0, 0.0),
                campaign={"purpose": "S7-S supervision-density scaling arm", "scale": scale,
                          "scale_nodes": target, "node_budget": budget, "prefix_size": len(prefix),
                          "realized_nodes": realized,
                          "target_spec": SPECS[budget].spec_hash(),
                          "order_hash": st["universe"]["order_hash"],
                          "preregistration": "tools/s7/s7-scaling-preregistration.json"})
            split = next(entry for entry in dataset.splits if entry.name == "train")
            arms.append({"arm_id": f"{scale}-{DEPTH_LABEL[budget]}", "scale": scale, "scale_nodes": target,
                         "node_budget": budget, "prefix_size": split.count, "realized_nodes": realized,
                         "dataset_id": dataset.id, "dataset_manifest_hash": dataset.manifest_hash,
                         "record_ids_hash": split.record_ids_hash})
            print(f"{scale}/{DEPTH_LABEL[budget]}: prefix {len(prefix)} rows, {realized:,} nodes "
                  f"({realized / target:.4f} of {target:,})", flush=True)

    index = {(arm["scale"], arm["node_budget"]): arm for arm in arms}
    problems = []
    for scale in SCALES:                                     # fixed scale: 400k ⊂ 160k ⊂ 64k ⊂ 16k
        sizes = [index[(scale, budget)]["prefix_size"] for budget in DEPTHS]
        if sizes != sorted(sizes):
            problems.append(f"depth nesting fails at {scale}: {sizes}")
    for budget in DEPTHS:                                    # fixed depth: 2x ⊂ 5x ⊂ 10x ⊂ 20x
        sizes = [index[(scale, budget)]["prefix_size"] for scale in SCALES]
        if sizes != sorted(sizes):
            problems.append(f"scale nesting fails at {budget}: {sizes}")
    if problems:
        print("STOP: nesting failed: " + "; ".join(problems))
        return 3
    save(xid=xid, arms=arms, nesting="asserted in both dimensions")
    print(f"arms frozen; reserved experiment id {xid}", flush=True)
    return 0


# ---------------------------------------------------------------- experiment ----
def step_experiment() -> int:
    st = state()
    if "experiment" in st:
        print("[cached] experiment")
        return 0
    service = svc()
    xid = st["xid"]
    control = service.store.get_as("T0004", TrainingRecipe)
    recipe_params = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    recipes = {}
    for budget in DEPTHS:
        recipe = service.create_training_recipe(name=f"s7s-{DEPTH_LABEL[budget]}", params={**recipe_params},
                                                target_spec=SPECS[budget],
                                                description=f"S7-S supervision at {budget} nodes")
        recipes[str(budget)] = recipe
    ablations: dict = {}
    for arm in st["arms"]:
        suffix = f"{arm['scale']}_{DEPTH_LABEL[arm['node_budget']]}"
        baseline = service.register_baseline(
            name=f"S7S_{suffix.upper()}", dataset_id=arm["dataset_id"],
            training_recipe_id=recipes[str(arm["node_budget"])].id, eval_protocol_id="E0004",
            model_config={"INPUT": "RAW", "H": 16}, supervision_divergence=DECLARATION,
            experiment_id=xid, notes=f"{arm['arm_id']}: {arm['prefix_size']} rows")
        ids = {16: baseline.id}
        for width in WIDTHS:
            if width == 16:
                continue
            ids[width] = service.create_ablation(baseline_id=baseline.id, overrides={"H": width},
                                                 supervision_divergence=DECLARATION,
                                                 experiment_id=xid,
                                                 notes=f"{arm['arm_id']} H={width}").id
        arm["ablations"] = [ids[width] for width in WIDTHS]
        ablations[arm["arm_id"]] = ids
    preregistration_hash = (TOOLS / "s7-scaling-preregistration.hash").read_text(encoding="utf-8").strip()
    experiment = service.create_experiment(
        name="S7-S supervision-density scaling", preregistration_hash=preregistration_hash,
        reference_unit_nodes=37_285_491, scales=SCALES, node_budgets=list(DEPTHS),
        arms=[{key: arm[key] for key in ("arm_id", "scale", "scale_nodes", "node_budget", "prefix_size",
                                         "realized_nodes", "dataset_id", "dataset_manifest_hash",
                                         "record_ids_hash", "ablations")}
              | {"training_recipe_id": recipes[str(arm["node_budget"])].id,
                 "recipe_hash": recipes[str(arm["node_budget"])].recipe_hash,
                 "train_target_spec_hash": SPECS[arm["node_budget"]].spec_hash()}
              for arm in st["arms"]],
        eval_protocol_id="E0004", widths=list(WIDTHS), seeds=list(SEEDS),
        source_id=st["universe"]["source"], normalization_id=st["universe"]["normalization"],
        candidate_universe_hash=st["universe"]["universe_hash"], order_seed=ORDER_SEED,
        order_hash=st["universe"]["order_hash"],
        analysis={"primary_endpoint": "test_loss on E0004/D0010",
                  "frontier_per_scale": ["160k-400k", "64k-160k", "16k-64k", "16k-400k"],
                  "scaling_per_depth": ["5x-2x", "10x-5x", "20x-10x", "20x-2x"],
                  "interaction": "depth profile at 2x versus 20x",
                  "student_compute": "reported separately"},
        experiment_id=xid, notes="16 arms x 4 widths x 5 seeds = 320 cells")
    expected = service.expected_membership(experiment)
    if len(expected) != 320:
        print(f"STOP: expected membership is {len(expected)}, not 320")
        return 3
    save(experiment={"id": experiment.id, "arms": len(experiment.arms), "cells": len(expected),
                     "recipe_ids": {key: recipe.id for key, recipe in recipes.items()}})
    print(f"experiment {experiment.id}: {len(experiment.arms)} arms, {len(expected)} cells", flush=True)
    return 0


# ----------------------------------------------------------------------- run ----
def step_run() -> int:
    st = state()
    experiment_id = st["experiment"]["id"]
    service = svc()
    queued = st.get("queued")
    if not queued:
        experiment = service.store.get(experiment_id)
        cells = []
        for arm in experiment.arms:
            for ablation_id in arm.ablations:
                cells.append(ablation_id)
        runs = {}
        for ablation_id in cells:
            runs[ablation_id] = [run.id for run in service.queue_runs(ablation_id, seeds=SEEDS)]
        queued = runs
        save(queued=queued)
        print(f"queued {sum(len(v) for v in queued.values())} runs", flush=True)
    counts: dict = {}
    for ablation_id, run_ids in queued.items():
        for run_id in run_ids:
            run: Run = service.store.get_as(run_id, Run)
            if run.status in TERMINAL_RUN_STATUSES:
                counts[run.status.value] = counts.get(run.status.value, 0) + 1
                continue
            finished = service.execute_run(run_id)
            counts[finished.status.value] = counts.get(finished.status.value, 0) + 1
            if finished.status.value != "COMPLETED":
                print(f"  {run_id} {finished.status.value}: "
                      f"{[c.detail for c in finished.integrity if c.status == 'fail'][:2]}", flush=True)
    save(run_statuses=counts)
    print("run statuses:", counts, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["universe", "calibrate", "labels", "arms", "experiment", "run"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"universe": step_universe, "calibrate": step_calibrate, "labels": step_labels,
            "arms": step_arms, "experiment": step_experiment, "run": step_run}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
