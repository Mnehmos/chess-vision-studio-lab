"""S8: the lower supervision frontier with a student-compute control.

Steps (state in tools/s8/s8-state.json):

    verify    re-prove the S7-S universe, order, instrument and leakage invariants; prove the
              existing 2x/16k arm reproduces exactly (records, spec, realized nodes, order)
    calibrate cost-only calibration at 8k/4k/2k (node counts and runtime only)
    labels    buy the 8k/4k/2k streams along the frozen order until the 2x crossing resolves
    arms      build 16k/8k/4k/2k prefixes, assert 16k subset 8k subset 4k subset 2k, freeze D####
    experiment  two recipes per depth (fixed epochs / fixed updates), 32 ablations bound to the
              reserved X####, X#### with the 160-cell membership
    run       queue and execute the matrix
    analyse   primary matched-student contrast + adjacent steps + the fixed-epoch branch
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import chess

from cvslab.data import _load_canonical, read_jsonl
from cvslab.funnel.pool import DEFAULT_ENGINE_ARGS
from cvslab.funnel.providers import AnalyzeSearchProvider, AnalyzeTransport
from cvslab.hashing import hash_obj
from cvslab.schemas import Dataset, Run, TERMINAL_RUN_STATUSES, TrainingRecipe
from cvslab.service import LabService, recipe_target_spec
from cvslab.store import LabError, Store
from cvslab.targets import TargetSpec

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools"
S8 = TOOLS / "s8"
S7 = TOOLS / "s7"
OUT = pathlib.Path(r"F:\Github\_parity_tmp\s8")
STATE = S8 / "s8-state.json"
S7_STATE = S7 / "s7-scaling-state.json"
S7_LABELS = pathlib.Path(r"F:\Github\_parity_tmp\s7scaling")
ENGINE = r"F:\Github\chess-vision-studio-rust-engine"

SCALE = {"name": "2x", "nodes": 74_570_982}
DEPTHS = (16_000, 8_000, 4_000, 2_000)
LABEL = {16_000: "16k", 8_000: "8k", 4_000: "4k", 2_000: "2k"}
REGIMES = {"epochs": {"EPOCHS": 40}, "updates": {"EPOCHS": 1, "MAX_UPDATES": 760}}
WIDTHS = (1, 4, 16, 32)
SEEDS = (0, 1, 2, 3, 4)
BATCH = 256
PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"
DECLARATION = ("S8 lower-frontier arm: trained on its own depth's observation, examined on the frozen "
               "instrument E0004/D0010")
SPECS = {depth: TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                           producer=PRODUCER, budget={"nodeBudget": depth}, value_path=("scoreCpStm",),
                           pov="stm", k=256.0, lam=1.0) for depth in DEPTHS}
EXPECTED_NEW_NODES = 223_712_946


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] saved {list(updates)}", flush=True)


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


class _Position:
    def __init__(self, fen: str):
        self.fen = fen


def _buy(jobs: list[tuple[str, str, int]], workers: int) -> list[dict]:
    chunks = [jobs[index::workers] for index in range(workers)]

    def run(chunk: list[tuple[str, str, int]]) -> list[dict]:
        if not chunk:
            return []
        transport = AnalyzeTransport(ENGINE, args=DEFAULT_ENGINE_ARGS, cwd=ENGINE)
        out: list[dict] = []
        try:
            provider = AnalyzeSearchProvider(transport, family="search_shallow_cp")
            for rid, fen, budget in chunk:
                label = provider.search(_Position(fen), node_budget=budget, pv_plies=8)
                value = {"scoreCpStm": label.score_cp_stm, "bestMove": label.best_move,
                         "mate": label.mate, "pv": list(label.pv), "nodes": label.nodes,
                         "nodeBudget": budget, **dict(label.extra)}
                out.append({"record_id": rid, "budget_nodes": int(label.nodes), "wall_ms": float(label.wall_ms),
                            "row": {"record_id": rid, "value": value,
                                    "budget": {"nodeBudget": budget, "nodes": int(label.nodes)}}})
        finally:
            transport.close()
        return out

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(run, chunks):
            results.extend(chunk)
    return results


# ------------------------------------------------------------------- verify ----
def step_verify() -> int:
    st = state()
    if "verify" in st:
        print("[cached] verify")
        return 0
    service = svc()
    s7 = json.loads(S7_STATE.read_text(encoding="utf-8"))
    normalization = s7["universe"]["normalization"]
    records = _load_canonical(service.store, service.store.get(normalization))
    ids = sorted(record["record_id"] for record in records)
    order = s7["order"]
    problems = []
    if hash_obj(ids) != s7["universe"]["universe_hash"]:
        problems.append("universe hash changed")
    if hash_obj(order) != s7["universe"]["order_hash"]:
        problems.append("order hash changed")
    if len(records) != s7["universe"]["records"]:
        problems.append("record count changed")
    if len(order) != s7["universe"]["clean"]:
        problems.append("order length does not cover the clean universe")

    # leakage invariant: 0 collisions against the full 575-record evaluation source
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
    collisions = [record["record_id"] for record in records
                  if find(component_of[record["record_id"]]) in protected
                  or find(component_of[record["record_id"]]) in population]
    if collisions:
        problems.append(f"{len(collisions)} records touch the protected evaluation population")

    # the existing 2x/16k arm must reproduce exactly if it is to be reused
    existing = next((arm for arm in s7["arms"] if arm["arm_id"] == "2x-16k"), None)
    reuse = None
    if existing is None:
        problems.append("the S7-S 2x/16k arm is missing from the state")
    else:
        dataset: Dataset = service.store.get_as(existing["dataset_id"], Dataset)
        split = next(entry for entry in dataset.splits if entry.name == "train")
        rows = read_jsonl(service.store.abs(split.path))
        expected_ids = order[: existing["prefix_size"]]
        actual_ids = sorted(row["record_id"] for row in rows)
        checks = {
            "dataset_manifest_matches": dataset.manifest_hash == existing["dataset_manifest_hash"],
            "rows": split.count == existing["prefix_size"],
            "record_ids_are_the_order_prefix": actual_ids == sorted(expected_ids),
            "record_ids_hash": split.record_ids_hash == existing["record_ids_hash"],
        }
        stream = read_jsonl(S7_LABELS / "labels-16000.jsonl")
        realized = 0
        for row in stream[: existing["prefix_size"]]:
            realized += row["budget_nodes"]
        checks["realized_nodes"] = realized == existing["realized_nodes"]
        if not all(checks.values()):
            problems.append(f"the S7-S 2x/16k arm does not reproduce: {checks}")
        else:
            reuse = {"arm_id": "2x-16k", "dataset_id": dataset.id, "manifest": dataset.manifest_hash,
                     "rows": split.count, "realized_nodes": realized,
                     "record_ids_hash": split.record_ids_hash, "checks": checks}
    # cardinality: every one of the 2k prefix's records will need exactly one 2k observation
    if problems:
        print("STOP: " + "; ".join(problems))
        return 3
    save(verify={"universe": s7["universe"], "order_hash": s7["universe"]["order_hash"],
                 "collisions": 0, "instrument": d10.id, "reuse_16k": reuse,
                 "order": "unchanged"}, order=order)
    print(f"verified: {len(records)} records, order {s7['universe']['order_hash'][:24]}…, reuse 16k = "
          f"{reuse['arm_id']} ({reuse['rows']} rows, {reuse['realized_nodes']:,} nodes)", flush=True)
    return 0


# ---------------------------------------------------------------- calibrate ----
def step_calibrate() -> int:
    st = state()
    if "calibration" in st:
        print("[cached] calibration")
        return 0
    service = svc()
    records = {r["record_id"]: r for r in _load_canonical(service.store, service.store.get(
        st["verify"]["universe"]["normalization"]))}
    sample = st["order"][:64]
    jobs = [(rid, records[rid]["fen"], depth) for rid in sample for depth in (8_000, 4_000, 2_000)]
    bought = _buy(jobs, workers=8)
    means = {}
    for depth in (8_000, 4_000, 2_000):
        nodes = [row["budget_nodes"] for row in bought if row["row"]["budget"]["nodeBudget"] == depth]
        means[str(depth)] = sum(nodes) / len(nodes)
        print(f"calibration {depth}: mean {means[str(depth)]:.0f} nodes over {len(nodes)} labels")
    predicted = {depth: math.ceil(SCALE["nodes"] / means[str(depth)]) for depth in (8_000, 4_000, 2_000)}
    headroom_ok = st["verify"]["universe"]["clean"] >= 1.15 * predicted[2_000]
    if not headroom_ok:
        print(f"STOP: universe {st['verify']['universe']['clean']} < 1.15 x predicted 2k prefix "
              f"{predicted[2_000]}")
        return 3
    save(calibration={"means": means, "predicted_prefix": predicted, "headroom_ok": headroom_ok})
    print(f"predicted prefixes: {predicted} (universe {st['verify']['universe']['clean']})", flush=True)
    return 0


# -------------------------------------------------------------------- labels ----
def step_labels() -> int:
    st = state()
    if "labels" in st:
        print("[cached] labels")
        return 0
    service = svc()
    order = st["order"]
    records = {r["record_id"]: r for r in _load_canonical(service.store, service.store.get(
        st["verify"]["universe"]["normalization"]))}
    means = {int(k): v for k, v in st["calibration"]["means"].items()}
    purchased: dict = {}
    for depth in (8_000, 4_000, 2_000):
        target = SCALE["nodes"]
        bought: list[dict] = []
        done = 0
        while True:
            need = math.ceil(target / means[depth])
            want = min(len(order), max(need + 32, int(need * 1.03)))
            if want <= done:
                print(f"STOP: depth {depth} ran out of candidates")
                return 3
            jobs = [(rid, records[rid]["fen"], depth) for rid in order[done:want]]
            print(f"depth {depth}: buying {len(jobs)} labels ({done} → {want})", flush=True)
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
        service.register_labels(st["verify"]["universe"]["normalization"], family="search_shallow_cp",
                                producer=PRODUCER, authority="legacy.cvs.search.shallow", pov="stm",
                                rows=[row["row"] for row in bought])
    realized = sum(sum(row["budget_nodes"] for row in rows) for rows in purchased.values())
    save(labels={"per_depth": {depth: {"labels": len(purchased[str(depth)]),
                                       "realized_nodes": sum(r["budget_nodes"]
                                                             for r in purchased[str(depth)])}
                               for depth in (8_000, 4_000, 2_000)},
                 "new_nodes": realized, "expected_new_nodes": EXPECTED_NEW_NODES})
    print(f"new teacher nodes: {realized:,} (expected {EXPECTED_NEW_NODES:,}; ratio "
          f"{realized / EXPECTED_NEW_NODES:.3f})", flush=True)
    return 0


# ---------------------------------------------------------------------- arms ----
def step_arms() -> int:
    st = state()
    if "arms" in st:
        print("[cached] arms")
        return 0
    service = svc()
    order = st["order"]
    normalization = st["verify"]["universe"]["normalization"]
    reuse = st["verify"]["reuse_16k"]
    arms = []

    # 16k: the reused S7-S arm (already proven identical to this order's prefix)
    arms.append({"arm_id": "16k", "depth": 16_000, "prefix_size": reuse["rows"],
                 "realized_nodes": reuse["realized_nodes"], "dataset_id": reuse["dataset_id"],
                 "dataset_manifest_hash": reuse["manifest"], "record_ids_hash": reuse["record_ids_hash"],
                 "reused_from": "S7-S 2x/16k"})
    for depth in (8_000, 4_000, 2_000):
        bought = read_jsonl(OUT / f"labels-{depth}.jsonl")
        cumulative, total = [], 0
        for row in bought:
            total += row["budget_nodes"]
            cumulative.append(total)
        best = min(range(len(cumulative)), key=lambda index: abs(cumulative[index] - SCALE["nodes"]))
        prefix = order[: best + 1]
        realized = cumulative[best]
        dataset = service.freeze_dataset(normalization, name=f"s8-2x-{LABEL[depth]}", record_ids=prefix,
                                         required_labels=["search_shallow_cp"], target_spec=SPECS[depth],
                                         fractions=(1.0, 0.0, 0.0),
                                         campaign={"purpose": "S8 lower-frontier arm", "scale": SCALE["name"],
                                                   "scale_nodes": SCALE["nodes"], "node_budget": depth,
                                                   "prefix_size": len(prefix), "realized_nodes": realized,
                                                   "target_spec": SPECS[depth].spec_hash(),
                                                   "order_hash": st["verify"]["order_hash"],
                                                   "preregistration": "tools/s8/s8-preregistration.json"})
        split = next(entry for entry in dataset.splits if entry.name == "train")
        arms.append({"arm_id": LABEL[depth], "depth": depth, "prefix_size": split.count,
                     "realized_nodes": realized, "dataset_id": dataset.id,
                     "dataset_manifest_hash": dataset.manifest_hash,
                     "record_ids_hash": split.record_ids_hash,
                     "target_spec": SPECS[depth].spec_hash()})
        print(f"2x/{LABEL[depth]}: prefix {split.count} rows, {realized:,} nodes "
              f"({realized / SCALE['nodes']:.4f} of target)", flush=True)

    sizes = [arm["prefix_size"] for arm in arms]
    if sizes != sorted(sizes):
        print(f"STOP: nesting failed: {sizes}")
        return 3
    save(arms=arms, nesting="16k ⊂ 8k ⊂ 4k ⊂ 2k asserted")
    print(f"nesting holds: {{arm['arm_id']: arm['prefix_size'] for arm in arms}}".replace("{", "{"), flush=True)
    return 0


# ---------------------------------------------------------------- experiment ----
def step_experiment() -> int:
    st = state()
    if "experiment" in st:
        print("[cached] experiment")
        return 0
    service = svc()
    xid = service.store.next_id("X")
    control: TrainingRecipe = service.store.get_as("T0004", TrainingRecipe)
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    recipes: dict = {}
    for arm in st["arms"]:
        for regime, regime_params in REGIMES.items():
            recipe = service.create_training_recipe(
                name=f"s8-{arm['arm_id']}-{regime}",
                params={**base, **regime_params, "BATCH": BATCH},
                target_spec=SPECS[arm["depth"]],
                description=f"S8 {arm['arm_id']} under {regime}")
            recipes[f"{arm['arm_id']}-{regime}"] = recipe
    ablations: dict = {}
    for arm in st["arms"]:
        for regime in REGIMES:
            suffix = f"{arm['arm_id'].upper()}_{regime.upper()}"
            baseline = service.register_baseline(
                name=f"S8_{suffix}", dataset_id=arm["dataset_id"],
                training_recipe_id=recipes[f"{arm['arm_id']}-{regime}"].id, eval_protocol_id="E0004",
                model_config={"INPUT": "RAW", "H": 16}, supervision_divergence=DECLARATION,
                experiment_id=xid, notes=f"{arm['arm_id']} / {regime}: {arm['prefix_size']} rows")
            ids = {16: baseline.id}
            for width in WIDTHS:
                if width == 16:
                    continue
                ids[width] = service.create_ablation(baseline_id=baseline.id, overrides={"H": width},
                                                     supervision_divergence=DECLARATION,
                                                     experiment_id=xid,
                                                     notes=f"{arm['arm_id']}/{regime} H={width}").id
            ablations[f"{arm['arm_id']}-{regime}"] = [ids[width] for width in WIDTHS]
    prereg = hash_obj(json.loads((S8 / "s8-preregistration.json").read_text(encoding="utf-8")))
    (S8 / "s8-preregistration.hash").write_text(prereg + "\n", encoding="utf-8")
    experiment = service.create_experiment(
        name="S8 lower supervision frontier", preregistration_hash=prereg,
        reference_unit_nodes=37_285_491,
        scales={f"2x-{regime}": SCALE["nodes"] for regime in REGIMES},
        node_budgets=[depth for depth in DEPTHS],
        arms=[{"arm_id": f"{arm['arm_id']}-{regime}", "scale": f"2x-{regime}",
               "scale_nodes": SCALE["nodes"], "node_budget": arm["depth"],
               "prefix_size": arm["prefix_size"], "realized_nodes": arm["realized_nodes"],
               "dataset_id": arm["dataset_id"], "dataset_manifest_hash": arm["dataset_manifest_hash"],
               "record_ids_hash": arm["record_ids_hash"],
               "training_recipe_id": recipes[f"{arm['arm_id']}-{regime}"].id,
               "recipe_hash": recipes[f"{arm['arm_id']}-{regime}"].recipe_hash,
               "train_target_spec_hash": SPECS[arm["depth"]].spec_hash(),
               "ablations": ablations[f"{arm['arm_id']}-{regime}"]}
              for arm in st["arms"] for regime in REGIMES],
        eval_protocol_id="E0004", widths=list(WIDTHS), seeds=list(SEEDS),
        source_id=st["verify"]["universe"]["source"], normalization_id=st["verify"]["universe"]["normalization"],
        candidate_universe_hash=st["verify"]["universe"]["universe_hash"], order_seed=20260920,
        order_hash=st["verify"]["order_hash"],
        analysis={"primary_branch": "fixed_updates", "primary_contrast": "2k-16k",
                  "adjacent": ["8k-16k", "4k-8k", "2k-4k"],
                  "decision": ["LOWER_FRONTIER_EXTENDS", "TWO_K_TOO_SHALLOW", "INCONCLUSIVE"],
                  "second_axis_note": "the fixed-epoch and fixed-update regimes are carried as the two "
                                      "'scales'; both freeze the same 2x teacher target"},
        experiment_id=xid, notes="4 depths x 2 student regimes x 4 widths x 5 seeds = 160 cells")
    expected = service.expected_membership(experiment)
    if len(expected) != 160:
        print(f"STOP: expected membership is {len(expected)}, not 160")
        return 3
    save(xid=xid, experiment={"id": experiment.id, "arms": len(experiment.arms), "cells": len(expected),
                              "regimes": list(REGIMES), "recipe_ids": {k: r.id for k, r in recipes.items()}})
    print(f"experiment {experiment.id}: {len(experiment.arms)} arms, {len(expected)} cells", flush=True)
    return 0


# ----------------------------------------------------------------------- run ----
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
                print(f"  {run_id} {finished.status.value}: "
                      f"{[c.detail for c in finished.integrity if c.status == 'fail'][:2]}", flush=True)
    save(run_statuses=counts)
    print("run statuses:", counts, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["verify", "calibrate", "labels", "arms", "experiment", "run"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    S8.mkdir(parents=True, exist_ok=True)
    return {"verify": step_verify, "calibrate": step_calibrate, "labels": step_labels,
            "arms": step_arms, "experiment": step_experiment, "run": step_run}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
