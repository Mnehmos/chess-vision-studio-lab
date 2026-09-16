"""S12: extend the population append-only, then freeze the 4k/2k/1k/512 frontier.

    python tools/s12/s12_run.py extend    # generate new games, snapshot S####, normalize N####,
                                          # leakage-join, build the append-only order, carry labels
    python tools/s12/s12_run.py calibrate # cost-only calibration at 1k and 512
    python tools/s12/s12_run.py labels    # buy the 1k and 512 streams to the 2x crossing
    python tools/s12/s12_run.py arms      # four nested prefixes + nesting assertions + D####
    python tools/s12/s12_run.py experiment
    python tools/s12/s12_run.py run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import chess

from cvslab.data import _load_canonical, read_jsonl
from cvslab.funnel.pool import (DEFAULT_ENGINE_ARGS, AnalyzeTeacher, PoolConfig, SelfPlayGenerator,
                                engine_identity, snapshot_source, write_pool)
from cvslab.funnel.providers import AnalyzeSearchProvider, AnalyzeTransport
from cvslab.hashing import hash_obj, sha256_file
from cvslab.schemas import Ablation, Dataset, Run, TERMINAL_RUN_STATUSES, TrainingRecipe
from cvslab.service import LabService
from cvslab.store import Store
from cvslab.targets import TargetSpec

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools"
S12 = TOOLS / "s12"
S7_STATE = TOOLS / "s7" / "s7-scaling-state.json"
S8_LABELS = pathlib.Path(r"F:\Github\_parity_tmp\s8")
OUT = pathlib.Path(r"F:\Github\_parity_tmp\s12")
BATCHES = OUT / "batches"
POOL = OUT / "pool"
STATE = S12 / "s12-state.json"
ENGINE = r"F:\Github\chess-vision-studio-rust-engine"
EPD = ENGINE + r"\benchmarks\suites\openings-inv1-20260910.epd"
EPD_SHA = "sha256:4d96e552fd257e66fe39d78420e1e48cfed6753795dd0a3e20cd222a3757e6b3"
SEED = 20260920
SCALE = 74_570_982
DEPTHS = (4_000, 2_000, 1_000, 512)
LABEL = {4_000: "4k", 2_000: "2k", 1_000: "1k", 512: "512"}
OLD_4K_ROWS, OLD_2K_ROWS = 18_961, 37_898
WIDTHS = (1, 4, 16, 32)
SEEDS = tuple(range(20))
BATCH, MAX_UPDATES = 256, 760
PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"
DECLARATION = ("S12 frontier arm: same student compute as its paired arms, one teacher budget, "
               "differing only in label depth and how many rows that budget buys")
SPECS = {depth: TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                           producer=PRODUCER, budget={"nodeBudget": depth}, value_path=("scoreCpStm",),
                           pov="stm", k=256.0, lam=1.0) for depth in DEPTHS}


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] saved {list(updates)}", flush=True)


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def record_id(fen: str) -> str:
    return "pos_" + hashlib.sha256(chess.Board(fen).epd().encode()).hexdigest()[:20]


def _config(count: int) -> PoolConfig:
    return PoolConfig(seed=SEED, games=count, max_plies=70, sample_every=2, min_ply=6,
                      diversification_plies=(8, 10, 12), diversification_candidates=3,
                      diversification_window_cp=25, play_node_budget=20000,
                      diversification_node_budget=2000, rng_mode="per-game",
                      start_source="epd-file", epd_path=EPD, epd_sha256=EPD_SHA)


class _Position:
    def __init__(self, fen: str):
        self.fen = fen


def _worker(start: int, count: int) -> None:
    directory = BATCHES / f"b{start:06d}"
    if (directory / "done.json").is_file():
        return
    directory.mkdir(parents=True, exist_ok=True)
    transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
    try:
        generated = SelfPlayGenerator(_config(count), AnalyzeTeacher(transport)).generate(
            start_index=start, count=count)
    finally:
        transport.close()
    with open(directory / "positions.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in generated["positions"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with open(directory / "games.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for game in generated["games"]:
            handle.write(json.dumps({k: v for k, v in game.items() if k != "positions"},
                                    sort_keys=True) + "\n")
    (directory / "done.json").write_text(json.dumps({"start": start, "count": count}) + "\n", encoding="utf-8")


def _buy(jobs: list[tuple[str, str, int]], workers: int) -> list[dict]:
    chunks = [jobs[index::workers] for index in range(workers)]

    def run(chunk):
        if not chunk:
            return []
        transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
        out = []
        try:
            provider = AnalyzeSearchProvider(transport, family="search_shallow_cp")
            for rid, fen, budget in chunk:
                label = provider.search(_Position(fen), node_budget=budget, pv_plies=8)
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

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(run, chunks):
            results.extend(chunk)
    return results


def step_extend() -> int:
    st = state()
    if "universe" in st:
        print("[cached] extend")
        return 0
    service = svc()
    s7 = json.loads(S7_STATE.read_text(encoding="utf-8"))
    old_order = s7["order"]
    old_ids = set(old_order)

    # generate fresh games until the projected universe comfortably covers a ~150k 512-prefix
    OUT.mkdir(parents=True, exist_ok=True)
    BATCHES.mkdir(parents=True, exist_ok=True)
    target_new = 100_000                      # ~179k total, ~1.15x a ~150k 512-prefix
    # S7-S generated games 0..2599 with this same seed; the extension must start PAST them or
    # per-game determinism regenerates the identical pool
    batch, workers, made = 200, 8, 2600
    new_ids: set[str] = set()
    while len(new_ids) < target_new:
        starts = [made + index * batch for index in range(workers)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda start: _worker(start, batch), starts))
        made += workers * batch
        for done in sorted(BATCHES.glob("b*/done.json")):
            meta = json.loads(done.read_text(encoding="utf-8"))
            with open(done.parent / "positions.jsonl", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rid = record_id(json.loads(line)["fen"])
                        if rid not in old_ids:
                            new_ids.add(rid)
        print(f"[{time.strftime('%H:%M:%S')}] {made} games generated, {len(new_ids)} new unique", flush=True)

    positions, games = [], []
    for done in sorted(BATCHES.glob("b*/done.json")):
        with open(done.parent / "positions.jsonl", "r", encoding="utf-8") as handle:
            positions.extend(json.loads(line) for line in handle if line.strip())
        with open(done.parent / "games.jsonl", "r", encoding="utf-8") as handle:
            games.extend(json.loads(line) for line in handle if line.strip())
    POOL.mkdir(parents=True, exist_ok=True)
    identity = engine_identity(ENGINE, args=DEFAULT_ENGINE_ARGS)
    manifest = write_pool(POOL, {"games": games, "positions": positions,
                                 "report": {"games": len(games), "rawPositions": len(positions)}},
                          {"config": _config(made).canonical(), "engine": identity}, opening_lines=None)
    source = snapshot_source(service.store, POOL, name="s12-extension", license="engine self-play corpus",
                             generated={"games": games, "positions": positions}, manifest=manifest)
    normalization = service.normalize([source.id], name="s12-extension-canonical")
    records = _load_canonical(service.store, service.store.get(normalization.id))
    print(f"source {source.id}: {len(positions)} rows; normalization {normalization.id}: {len(records)} records",
          flush=True)

    # leakage join against the FULL 575-record evaluation source
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
    clean = [record["record_id"] for record in records
             if find(component_of[record["record_id"]]) not in protected
             and find(component_of[record["record_id"]]) not in population]
    removed = len(records) - len(clean)
    if removed / max(len(records), 1) > 0.05:
        print(f"STOP: leakage removal {removed}/{len(records)} exceeds tolerance")
        return 3

    fresh = sorted(rid for rid in clean if rid not in old_ids)
    new_order = sorted(fresh, key=lambda rid: hashlib.sha256(f"{SEED}:{rid}".encode()).hexdigest())
    order = list(old_order) + new_order              # append-only: old prefixes preserved exactly
    if order[: len(old_order)] != old_order:
        print("STOP: the old order's prefix does not reproduce")
        return 3
    carried = service.copy_label_sets(s7["universe"]["normalization"], normalization.id)
    print(f"labels carried into {normalization.id}: {carried}", flush=True)
    save(universe={"source": source.id, "normalization": normalization.id,
                   "records": len(records), "removed": removed, "old": len(old_order),
                   "appended": len(new_order), "total": len(order),
                   "universe_hash": hash_obj(order), "order_hash": hash_obj(order),
                   "order_kind": "append-only: S7-S order ++ hash order of the new clean records"},
         order=order)
    return 0


def _stream_nodes(depth: int) -> dict:
    rows = read_jsonl(S8_LABELS / f"labels-{depth}.jsonl")
    return {row["record_id"]: int(row["budget_nodes"]) for row in rows}


def step_calibrate() -> int:
    st = state()
    if "calibration" in st:
        print("[cached] calibrate")
        return 0
    service = svc()
    records = {r["record_id"]: r for r in _load_canonical(service.store,
                                                          service.store.get(st["universe"]["normalization"]))}
    sample = st["order"][:64]
    jobs = [(rid, records[rid]["fen"], depth) for rid in sample for depth in (1_000, 512)]
    bought = _buy(jobs, workers=8)
    means = {}
    for depth in (1_000, 512):
        nodes = [row["budget_nodes"] for row in bought if row["row"]["budget"]["nodeBudget"] == depth]
        means[str(depth)] = sum(nodes) / len(nodes)
        print(f"calibration {depth}: mean {means[str(depth)]:.0f} nodes over {len(nodes)} labels")
    predicted = {depth: int(SCALE / means[str(depth)]) for depth in (1_000, 512)}
    if len(st["order"]) < 1.15 * predicted[512]:
        print(f"STOP: universe {len(st['order'])} < 1.15 x predicted 512 prefix {predicted[512]}")
        return 3
    save(calibration={"means": means, "predicted": predicted})
    print(f"predicted prefixes: {predicted} (universe {len(st['order'])})", flush=True)
    return 0


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
    purchased = {}
    for depth in (1_000, 512):
        target, bought, done = SCALE, [], 0
        while True:
            need = int(target / means[depth])
            want = min(len(order), max(need + 32, int(need * 1.03)))
            if want <= done:
                print(f"STOP: depth {depth} ran out of candidates")
                return 3
            jobs = [(rid, records[rid]["fen"], depth) for rid in order[done:want]]
            print(f"depth {depth}: buying {len(jobs)} labels ({done} -> {want})", flush=True)
            bought.extend(_buy(jobs, workers=8))
            done = want
            cumulative, total = [], 0
            for row in bought:
                total += row["budget_nodes"]
                cumulative.append(total)
            best = min(range(len(cumulative)), key=lambda index: abs(cumulative[index] - target))
            if best < len(cumulative) - 2 or done >= len(order):
                break
        purchased[str(depth)] = bought
        with open(OUT / f"labels-{depth}.jsonl", "w", encoding="utf-8", newline="\n") as handle:
            for row in bought:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        total = sum(row["budget_nodes"] for row in bought)
        print(f"depth {depth}: {len(bought)} labels, {total:,} nodes (ratio {total / target:.3f})", flush=True)
        service.register_labels(st["universe"]["normalization"], family="search_shallow_cp",
                                producer=PRODUCER, authority="legacy.cvs.search.shallow", pov="stm",
                                rows=[row["row"] for row in bought])
    save(labels={"per_depth": {depth: {"labels": len(purchased[str(depth)]),
                                       "realized_nodes": sum(r["budget_nodes"] for r in purchased[str(depth)])}
                               for depth in (1_000, 512)}})
    return 0


def step_arms() -> int:
    st = state()
    if "arms" in st:
        print("[cached] arms")
        return 0
    service = svc()
    order = st["order"]
    normalization = st["universe"]["normalization"]
    carried = {depth: _stream_nodes(depth) for depth in (4_000, 2_000)}
    new_streams = {depth: {row["record_id"]: int(row["budget_nodes"])
                           for row in read_jsonl(OUT / f"labels-{depth}.jsonl")} for depth in (1_000, 512)}
    arms = []
    for depth in DEPTHS:
        stream = carried.get(depth) or new_streams[depth]
        cumulative, total = [], 0
        for rid in order:
            nodes = stream.get(rid)
            if nodes is None:
                break                              # the stream ends where purchased
            total += nodes
            cumulative.append(total)
        best = min(range(len(cumulative)), key=lambda index: abs(cumulative[index] - SCALE))
        prefix, realized = order[: best + 1], cumulative[best]
        missing = [rid for rid in prefix if rid not in stream]
        if missing:
            print(f"STOP: depth {depth} prefix lacks {len(missing)} observations")
            return 3
        dataset = service.freeze_dataset(normalization, name=f"s12-2x-{LABEL[depth]}", record_ids=prefix,
                                         required_labels=["search_shallow_cp"], target_spec=SPECS[depth],
                                         fractions=(1.0, 0.0, 0.0),
                                         campaign={"purpose": "S12 lower-frontier arm", "scale": "2x",
                                                   "scale_nodes": SCALE, "node_budget": depth,
                                                   "prefix_size": len(prefix), "realized_nodes": realized,
                                                   "target_spec": SPECS[depth].spec_hash(),
                                                   "order_hash": st["universe"]["order_hash"],
                                                   "preregistration": "tools/s12/s12-preregistration.json"})
        split = next(entry for entry in dataset.splits if entry.name == "train")
        arms.append({"arm_id": LABEL[depth], "depth": depth, "prefix_size": split.count,
                     "realized_nodes": realized, "dataset_id": dataset.id,
                     "dataset_manifest_hash": dataset.manifest_hash, "record_ids_hash": split.record_ids_hash})
        print(f"2x/{LABEL[depth]}: prefix {split.count} rows, {realized:,} nodes "
              f"({realized / SCALE:.4f} of target)", flush=True)
    sizes = [arm["prefix_size"] for arm in arms]
    if sizes != sorted(sizes):
        print(f"STOP: nesting failed: {sizes}")
        return 3
    # the carried streams were purchased along the pre-S12 order; 36 of its records were later
    # dropped as instrument-adjacent, so the prefixes reproduce up to those removals
    for arm, old_rows in ((arms[0], OLD_4K_ROWS), (arms[1], OLD_2K_ROWS)):
        if arm["prefix_size"] > old_rows:
            print(f"STOP: {arm['arm_id']} prefix {arm['prefix_size']} exceeds the purchased {old_rows}")
            return 3
        print(f"{arm['arm_id']}: {old_rows - arm['prefix_size']} purchased rows fell outside the "
              f"leakage-clean prefix", flush=True)
    save(arms=arms, nesting="4k subset 2k subset 1k subset 512 asserted; 4k/2k prefixes reproduce S8")
    return 0


def step_experiment() -> int:
    st = state()
    if "experiment" in st:
        print("[cached] experiment")
        return 0
    service = svc()
    while True:
        candidate = service.store.next_id("X")
        if not [a for a in service.store.list("A", verify=True, kind=Ablation) if a.experiment_id == candidate]:
            xid = candidate
            break
    control: TrainingRecipe = service.store.get_as("T0004", TrainingRecipe)
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    recipes = {depth: service.create_training_recipe(
        name=f"s12-{LABEL[depth]}", params={**base, "EPOCHS": 1, "BATCH": BATCH, "MAX_UPDATES": MAX_UPDATES},
        target_spec=SPECS[depth], description=f"S12 fixed-update regime at {depth} nodes") for depth in DEPTHS}
    ablations = {}
    for arm in st["arms"]:
        baseline = service.register_baseline(
            name=f"S12_{arm['arm_id'].upper()}", dataset_id=arm["dataset_id"],
            training_recipe_id=recipes[arm["depth"]].id, eval_protocol_id="E0004",
            model_config={"INPUT": "RAW", "H": 16}, supervision_divergence=DECLARATION,
            experiment_id=xid, notes=f"S12 {arm['arm_id']}: {arm['prefix_size']} rows, one teacher budget")
        ids = {16: baseline.id}
        for width in WIDTHS:
            if width == 16:
                continue
            ids[width] = service.create_ablation(baseline_id=baseline.id, overrides={"H": width},
                                                 supervision_divergence=DECLARATION, experiment_id=xid,
                                                 notes=f"S12 {arm['arm_id']} H={width}").id
        ablations[arm["arm_id"]] = [ids[width] for width in WIDTHS]
    arms_payload = []
    for arm in st["arms"]:
        arms_payload.append({"arm_id": f"S12-{arm['arm_id']}", "scale": "2x", "scale_nodes": SCALE,
                             "node_budget": arm["depth"], "prefix_size": arm["prefix_size"],
                             "realized_nodes": arm["realized_nodes"], "dataset_id": arm["dataset_id"],
                             "dataset_manifest_hash": arm["dataset_manifest_hash"],
                             "record_ids_hash": arm["record_ids_hash"],
                             "training_recipe_id": recipes[arm["depth"]].id,
                             "recipe_hash": recipes[arm["depth"]].recipe_hash,
                             "train_target_spec_hash": SPECS[arm["depth"]].spec_hash(),
                             "ablations": ablations[arm["arm_id"]]})
    experiment = service.create_experiment(
        name="S12 lower economic frontier: 4k/2k/1k/512", preregistration_hash=(S12 / "s12-preregistration.hash")
        .read_text(encoding="utf-8").strip(), reference_unit_nodes=37_285_491,
        scales={"2x": SCALE}, node_budgets=list(DEPTHS), arms=arms_payload, eval_protocol_id="E0004",
        widths=list(WIDTHS), seeds=list(SEEDS), source_id=st["universe"]["source"],
        normalization_id=st["universe"]["normalization"], candidate_universe_hash=st["universe"]["universe_hash"],
        order_seed=SEED, order_hash=st["universe"]["order_hash"],
        analysis={"design": "one teacher budget, four label depths along one append-only order; "
                            "equal student compute (760 x 256)",
                  "primary": "d = test_loss(512) - test_loss(4k) per width, paired over twenty seeds",
                  "bounds": "separate one-sided 95% (exact Student-t per df)",
                  "decision": ["SHALLOWER_FRONTIER if upper < 0 at every width",
                               "FRONTIER_EXHAUSTED if lower > 0 at every width", "INCONCLUSIVE otherwise"],
                  "adjacent": ["2k-4k", "1k-2k", "512-1k"],
                  "acceleration": "does the per-halving tax grow? that bend is where savings stop paying",
                  "order_kind": "append-only: S7-S order ++ hash order of the new clean records"},
        experiment_id=xid, notes="4 depths x 4 widths x 20 seeds = 320 cells")
    expected = service.expected_membership(experiment)
    if len(expected) != 320:
        print(f"STOP: expected membership is {len(expected)}, not 320")
        return 3
    save(xid=xid, experiment={"id": experiment.id, "arms": len(experiment.arms), "cells": len(expected)})
    print(f"experiment {experiment.id}: {len(experiment.arms)} arms, {len(expected)} cells", flush=True)
    return 0


def step_run() -> int:
    st = state()
    service = svc()
    queued = st.get("queued")
    if not queued:
        experiment = service.store.get(st["experiment"]["id"])
        queued = {}
        for arm in experiment.arms:
            for ablation_id in arm.ablations:
                queued[ablation_id] = [run.id for run in service.queue_runs(ablation_id, seeds=SEEDS)]
        save(queued=queued)
        print(f"queued {sum(len(v) for v in queued.values())} runs", flush=True)
    counts: dict = {}
    for run_ids in queued.values():
        for run_id in run_ids:
            run: Run = service.store.get_as(run_id, Run)
            if run.status in TERMINAL_RUN_STATUSES:
                counts[run.status.value] = counts.get(run.status.value, 0) + 1
                continue
            finished = service.execute_run(run_id)
            counts[finished.status.value] = counts.get(finished.status.value, 0) + 1
            if finished.status.value != "COMPLETED":
                print(f"  {run_id} {finished.status.value}", flush=True)
    save(run_statuses=counts)
    print("run statuses:", counts, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["extend", "calibrate", "labels", "arms", "experiment", "run"])
    args = parser.parse_args()
    S12.mkdir(parents=True, exist_ok=True)
    return {"extend": step_extend, "calibrate": step_calibrate, "labels": step_labels,
            "arms": step_arms, "experiment": step_experiment, "run": step_run}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
