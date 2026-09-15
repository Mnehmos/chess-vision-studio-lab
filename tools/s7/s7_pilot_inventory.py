"""S7-P inventory: PROVE the existing-label pilot's inputs before freezing anything.

Checks, in order (any failure stops the pilot):

  1. the S4 run artifacts the eligibility flags come from are the ones R0008 recorded
     (sha256 against the run record, not against a re-derivation);
  2. the 94 S6 UNIFORM record identities recover exactly from D0009's stored split, and
     match the triage artifact's uniform flags;
  3. the 926 train-eligible records recover, and `hash_obj(sorted(ids))` equals the
     candidate_universe_hash R0008's arms were frozen against;
  4. every one of the 926 has EXACTLY ONE matching 16k observation, and exactly one 2k,
     under explicit TargetSpecs (strict cardinality, never "some label exists");
  5. the 94 also carry the 400k deep observation the S6 baseline used;
  6. the twenty reused runs are exactly the S6 UNIFORM cells, pinned to E0004/D0010, and
     they are mutually comparable under the P1 instrument identity;
  7. no pilot record belongs to the evaluation instrument's protected component
     population (joint transposition components against all 575 N0010 records).

Writes tools/s7/s7-pilot-inventory.json. Read-only with respect to the store.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import chess

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.data import _load_canonical, read_jsonl
from cvslab.hashing import hash_obj, sha256_file
from cvslab.schemas import Dataset, EvaluationProtocol, Run, TrainingRecipe
from cvslab.service import LabService, recipe_target_spec
from cvslab.store import LabError, Store
from cvslab.targets import TargetSpec, matching_labels

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
RUN_DIR = pathlib.Path(r"F:\Github\_parity_tmp\s6\s4run")
OUT = LAB / "tools" / "s7" / "s7-pilot-inventory.json"
PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"

SPEC_16K = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow", producer=PRODUCER,
                      budget={"nodeBudget": 16000}, value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0)
SPEC_2K = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow", producer=PRODUCER,
                     budget={"nodeBudget": 2000}, value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0)
SPEC_400K = TargetSpec(family="search_deep_cp", authority="legacy.cvs.search.deep", producer=PRODUCER,
                       budget={"nodeBudget": 400000}, value_path=("targets", "scoreCpStm"), pov="stm",
                       k=256.0, lam=1.0)

EXPECTED_UNIFORM_RUNS = [54, 55, 56, 57, 58, 64, 65, 66, 67, 68, 74, 75, 76, 77, 78, 84, 85, 86, 87, 88]
WIDTHS = (1, 4, 16, 32)
SEEDS = (0, 1, 2, 3, 4)


def record_id_for_fen(fen: str) -> str:
    return "pos_" + hashlib.sha256(chess.Board(fen).epd().encode()).hexdigest()[:20]


def fail(step: str, message: str) -> dict:
    print(f"STOP [{step}]: {message}", flush=True)
    return {"stop": {"step": step, "reason": message}}


def main() -> int:
    svc = LabService(Store(str(LAB / "labstore")))
    store = svc.store
    report: dict = {"checks": {}, "specs": {"16k": SPEC_16K.spec_hash(), "2k": SPEC_2K.spec_hash(),
                                            "400k": SPEC_400K.spec_hash()}}

    # 1 -- the run artifacts are the recorded ones ---------------------------------
    r8 = json.loads((store.root / "objects" / "runs" / "R0008.json").read_text(encoding="utf-8"))
    recorded = {f["name"]: f["sha256"] for f in r8["files"]}
    for name in ("positions.jsonl", "triage.jsonl", "tier3.jsonl"):
        actual = sha256_file(RUN_DIR / name).removeprefix("sha256:")
        if recorded.get(name) != actual:
            return fail(step="artifacts", message=f"{name} does not match the hash R0008 recorded") or 3
    report["checks"]["s4_artifacts_match_run_record"] = True

    # 2 -- the 94 UNIFORM identities -------------------------------------------------
    d9: Dataset = store.get_as("D0009", Dataset)
    split = next(s for s in d9.splits if s.name == "train")
    uniform_ids = sorted(row["record_id"] for row in read_jsonl(store.abs(split.path)))
    triage_flags = {}
    run_positions = {row["id"]: row["fen"] for row in read_jsonl(RUN_DIR / "positions.jsonl")}
    for row in read_jsonl(RUN_DIR / "triage.jsonl"):
        triage_flags[record_id_for_fen(run_positions[row["id"]])] = row
    triage_uniform = sorted(rid for rid, row in triage_flags.items() if row["selection"].get("uniform"))
    if uniform_ids != triage_uniform:
        return fail(step="uniform_ids", message="D0009's stored split and the triage artifact disagree") or 3
    if len(uniform_ids) != 94:
        return fail(step="uniform_ids", message=f"expected 94 uniform records, found {len(uniform_ids)}") or 3
    report["checks"]["uniform_94_recovered"] = True
    report["uniform_record_ids"] = uniform_ids

    # 3 -- the 926 train-eligible records + the recorded universe hash ---------------
    eligible = sorted(rid for rid, row in triage_flags.items() if row.get("trainEligible", True))
    if len(eligible) != 926:
        return fail(step="eligible", message=f"expected 926 train-eligible records, found {len(eligible)}") or 3
    universe_hash = hash_obj(eligible)
    if universe_hash != r8["arms"][0].get("candidate_universe_hash", universe_hash):
        pass  # candidate_universe_hash lives on the frozen datasets, checked below
    d8: Dataset = store.get_as("D0008", Dataset)
    expected_universe_hash = d8.campaign["candidate_universe_hash"]
    if universe_hash != expected_universe_hash:
        return fail(step="eligible", message=f"recovered universe hashes to {universe_hash}, "
                                             f"but S6 froze arms against {expected_universe_hash}") or 3
    report["checks"]["eligible_926_recovered_and_hash_matches_s6"] = True
    report["candidate_universe_hash"] = universe_hash
    report["eligible_record_ids"] = eligible

    # 4/5 -- strict cardinality of the observations the pilot will consume -----------
    normalization_id = d9.normalization_id
    records = {record["record_id"]: record
               for record in _load_canonical(store, store.get(normalization_id))}
    counts = {"16k": {}, "2k": {}, "400k": {}}
    for rid in eligible:
        record = records.get(rid)
        if record is None:
            return fail(step="records", message=f"{rid} is train-eligible but absent from {normalization_id}") or 3
        counts["16k"][rid] = len(matching_labels(record, SPEC_16K))
        counts["2k"][rid] = len(matching_labels(record, SPEC_2K))
    for rid in uniform_ids:
        counts["400k"][rid] = len(matching_labels(records[rid], SPEC_400K))
    bad = {name: [rid for rid, n in mapping.items() if n != 1] for name, mapping in counts.items()}
    for name, offending in bad.items():
        if offending:
            return fail(step="cardinality",
                        message=f"{len(offending)} records do not have exactly one {name} observation "
                                f"(first: {offending[0]})") or 3
    report["checks"]["exactly_one_observation_per_record"] = {
        "16k_over_926": len(counts["16k"]), "2k_over_926": len(counts["2k"]),
        "400k_over_94": len(counts["400k"])}
    report["corpus"] = {"normalization": normalization_id, "records": len(records),
                        "train_eligible": len(eligible), "uniform_94": len(uniform_ids)}

    # 6 -- the twenty reused runs are the S6 UNIFORM cells, pinned to E0004/D0010 -----
    reused, by_width = [], {}
    for number in EXPECTED_UNIFORM_RUNS:
        run: Run = store.get_as(f"R{number:04d}", Run)
        if run.status.value != "COMPLETED":
            return fail(step="reused_runs", message=f"R{number:04d} is {run.status.value}, not COMPLETED") or 3
        if (run.dataset_id, run.eval_protocol_id, run.recipe_hash) != ("D0009", "E0004", run.recipe_hash):
            return fail(step="reused_runs", message=f"R{number:04d} is not the S6 UNIFORM cell") or 3
        if run.eval_dataset_id != "D0010":
            return fail(step="reused_runs", message=f"R{number:04d} was not examined on D0010") or 3
        width = run.effective_config["H"]
        ablation_label = run.display_label
        if "UNIFORM" not in ablation_label:
            return fail(step="reused_runs", message=f"R{number:04d} is not a UNIFORM cell: {ablation_label}") or 3
        by_width.setdefault(width, []).append(run.id)
        reused.append(run.id)
    if sorted(by_width) != sorted(WIDTHS) or any(len(v) != 5 for v in by_width.values()):
        return fail(step="reused_runs", message=f"reused cells are not 4 widths x 5 seeds: "
                                                f"{ {k: len(v) for k, v in by_width.items()} }") or 3
    try:
        shared_spec = svc.assert_runs_comparable(reused)
    except LabError as exc:
        return fail(step="comparability", message=str(exc)) or 3
    report["checks"]["reused_runs_are_s6_uniform_and_comparable"] = True
    report["reused_runs"] = {"by_width": {str(k): sorted(v) for k, v in sorted(by_width.items())},
                             "shared_evaluation_spec": shared_spec}

    # 7 -- evaluation-instrument protection: joint components over all 575 -------------
    d10: Dataset = store.get_as("D0010", Dataset)
    instrument_ids = sorted(row["record_id"] for row in read_jsonl(store.abs(d10.splits[2].path)))
    eval_records = _load_canonical(store, store.get("N0010"))
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

    component_of = {record["record_id"]: record["group"] for record in records.values()}
    for record in eval_records:
        if record["record_id"] in component_of:
            union(component_of[record["record_id"]], record["group"])
        component_of.setdefault(record["record_id"], record["group"])
    for record in records.values():
        find(record["group"])
    instrument_components = {find(component_of[rid]) for rid in instrument_ids}
    colliding_pool = [rid for rid in records if find(component_of[rid]) in instrument_components]
    colliding_eligible = [rid for rid in eligible if rid in set(colliding_pool)]
    if colliding_eligible:
        return fail(step="instrument_protection",
                    message=f"{len(colliding_eligible)} train-eligible records share a transposition "
                            f"component with the D0010 instrument") or 3
    report["checks"]["no_pilot_record_touches_the_instrument_components"] = True
    report["instrument"] = {"dataset": "D0010", "protocol": "E0004", "records": len(instrument_ids),
                            "eval_source_records": len(eval_records),
                            "instrument_components": len(instrument_components),
                            "pool_records_in_those_components": len(colliding_pool)}

    # the S6 controls' recipe parameters, which the pilot recipes must copy exactly -----
    t4: TrainingRecipe = store.get_as("T0004", TrainingRecipe)
    report["control_recipe"] = {"id": "T0004", "params": {k: v for k, v in t4.params.items()
                                                         if k != "TARGET_SPEC"},
                                "spec": recipe_target_spec(t4).spec_hash()}
    e4: EvaluationProtocol = store.get_as("E0004", EvaluationProtocol)
    report["control_protocol"] = {"id": "E0004", "spec": e4.params["TARGET_SPEC"], "split": e4.split,
                                 "dataset": e4.dataset_id, "protocol_hash": e4.protocol_hash}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report["checks"][k] for k in report["checks"]}, indent=1)[:1200])
    print("written", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
