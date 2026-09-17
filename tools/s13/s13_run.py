"""S13: the teacher-authority experiment — two exams, three teachers, one student family.

    python tools/s13/s13_run.py exams       # freeze the SF-DEEP exam (D#### + E####) beside E0004
    python tools/s13/s13_run.py preregister # freeze the S13 preregistration (hash-bound by X####)
    python tools/s13/s13_run.py experiment  # freeze the lattice: 3 teacher conditions x 4 widths x 20 seeds
    python tools/s13/s13_run.py run         # execute every queued cell

`dual` and `analyse` live in tools/s13/s13_analyse.py.

The lab's Run carries one exam, so X#### binds CVS-DEEP (E0004) and every model is additionally
scored on SF-DEEP by the `dual` step: the same nnue evaluation code, the same held-out identities,
the SF TargetSpec instead of the CVS one, validated by reproducing the run's own recorded CVS
metrics exactly before any SF number is believed.

The three conditions are NOT a grid, which is why this study uses an explicit cell list: each arm
declares the teacher spend it actually made (scale value) and its own per-label budget, rather than
inheriting a nominal shared scale that two of the arms do not spend.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from s13_common import (AUTHORITY, AUTHORITY_COLD, CVS_ARM_DATASET, EXAM_PROTOCOL_CVS, FAMILY, LAB,
                        OUT, S13, SF_NETS, SF_OPTIONS, SF_SHA256, write_json)

from cvslab.hashing import read_jsonl
from cvslab.schemas import Ablation, Experiment, Run
from cvslab.service import LabService
from cvslab.store import LabError, Store
from cvslab.targets import TargetSpec

REFERENCE_UNIT = 37_285_491
WIDTHS = (1, 4, 16, 32)
SEEDS = tuple(range(20))
BATCH, MAX_UPDATES = 256, 760
ARMS = (("CVS-4k", 4_000), ("SF-32k", 32_000), ("SF-1M", 1_000_000))
CONTRACT = "cold"                 # the replacement contract: every SF label is cold per label
ACTIVE_AUTHORITY = AUTHORITY_COLD if CONTRACT == "cold" else AUTHORITY
SUFFIX = "-cold" if CONTRACT == "cold" else ""
LABELS_KEY = "labels_cold" if CONTRACT == "cold" else "labels"
EXAM_KEY = "exam_cold" if CONTRACT == "cold" else "exam"
PREREG_PATH = S13 / ("s13-preregistration-v2.json" if CONTRACT == "cold"
                     else "s13-preregistration.json")
PREREG_HASH = S13 / ("s13-preregistration-v2.hash" if CONTRACT == "cold"
                     else "s13-preregistration.hash")
PREREG_KEY = "preregistration_cold" if CONTRACT == "cold" else "preregistration"
CVS_PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"
CVS_SPEC = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                      producer=CVS_PRODUCER, budget={"nodeBudget": 4_000},
                      value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0)
EXAM_BUDGET = 4_000_000
SF_SPECS = {label: TargetSpec(family=FAMILY, authority=ACTIVE_AUTHORITY, producer=SF_SHA256,
                              budget={"nodes": budget}, value_path=("scoreCpStm",), pov="stm",
                              k=256.0, lam=1.0) for label, budget in ARMS if label.startswith("SF")}
SF_EXAM_SPEC = TargetSpec(family=FAMILY, authority=ACTIVE_AUTHORITY, producer=SF_SHA256,
                          budget={"nodes": EXAM_BUDGET}, value_path=("scoreCpStm",), pov="stm",
                          k=256.0, lam=1.0)
DECLARATION = ("S13 teacher arm: same student, same positions, same student compute as its paired "
               "arms; the only change is which authority produced the supervision and at what price")
STATE = S13 / "s13-state.json"


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] saved {list(updates)}", flush=True)


def _train_ids(service: LabService, dataset_id: str) -> list[str]:
    dataset = service.store.get(dataset_id)
    split = next(entry for entry in dataset.splits if entry.name == "train")
    return [row["record_id"] for row in read_jsonl(service.store.abs(split.path))]


def step_exams() -> int:
    """Freeze the SF-DEEP exam: a dataset over the 200 held-out ids + a protocol, beside E0004."""
    st = state()
    if EXAM_KEY in st:
        print("[cached] exams")
        return 0
    service = svc()
    d0010 = service.store.get("D0010")
    split = next(entry for entry in d0010.splits if entry.name == "test")
    ids = [row["record_id"] for row in read_jsonl(service.store.abs(split.path))]
    dataset = service.freeze_dataset(
        "N0006", name=f"s13-exam-sf-deep{SUFFIX}", record_ids=ids, required_labels=[FAMILY],
        target_spec=SF_EXAM_SPEC, fractions=(0.0, 0.0, 1.0),
        campaign={"purpose": f"S13 SF-DEEP exam{SUFFIX}", "authority": ACTIVE_AUTHORITY,
                  "producer": SF_SHA256, "nodes": EXAM_BUDGET,
                  "contract": CONTRACT + (" per label: ucinewgame + Clear Hash + isready" if
                                          CONTRACT == "cold" else " (warm, superseded)"),
                  "identities": "the exact D0010 held-out identities"})
    protocol = service.create_eval_protocol(
        name=f"s13-sf-deep-exam{SUFFIX}", dataset_id=dataset.id, split="test", k=256.0, lam=1.0,
        target_spec=SF_EXAM_SPEC,
        description="Stockfish 18 at 4M nodes/label over the SAME 200 held-out identities as E0004; "
                    "a second static instrument, never averaged with CVS-DEEP and never called truth",
        non_claims=["a deeper Stockfish label is another frozen instrument, not ground truth",
                    "static fit is not playing strength"])
    save(**{EXAM_KEY: {"dataset_id": dataset.id, "dataset_manifest_hash": dataset.manifest_hash,
                       "protocol_id": protocol.id, "protocol_hash": protocol.protocol_hash,
                       "spec_hash": SF_EXAM_SPEC.spec_hash(), "identities": len(ids),
                       "nodes": EXAM_BUDGET, "contract": CONTRACT,
                       "authority": ACTIVE_AUTHORITY}})
    print(f"SF-DEEP exam: {dataset.id} + {protocol.id} over {len(ids)} identities", flush=True)
    return 0


def step_preregister() -> int:
    st = state()
    if PREREG_KEY in st:
        print("[cached] preregister")
        return 0
    labels, exam, cvs_cost = st[LABELS_KEY], st[EXAM_KEY], st["cvs_cost"]
    disagreement = (json.loads((S13 / "s13-disagreement.json").read_text(encoding="utf-8"))
                    if (S13 / "s13-disagreement.json").is_file() else {})
    supersession = json.loads((S13 / "s13-x0009-supersession.json").read_text(encoding="utf-8"))
    prereg = {
        "id": "s13-teacher-authority",
        "version": 2 if CONTRACT == "cold" else 1,
        "status": "materialized before the lattice is frozen and before any student exists",
        "supersedes": ("X0009 (preserved, sealed, and recorded as INVALID for causal claims about "
                       "teacher budget because its Stockfish labels were not cold per label: "
                       "tools/s13/s13-x0009-supersession.json)" if CONTRACT == "cold" else None),
        "contract": ("cold per label: ucinewgame + Clear Hash + isready before every search, 12 "
                     "workers for every stream, worker partition proven irrelevant by "
                     "tools/s13/s13-cold-worker-check.json" if CONTRACT == "cold" else "warm (superseded)"),
        "issue": "Mnehmos/chess-vision-studio-lab#40 (slice of #20)",
        "question": "If the student architecture, positions, optimizer, student compute and evaluation "
                    "questions are held fixed, does changing the teacher from CVS to Stockfish change "
                    "what the student learns?",
        "secondary_question": "Does a student become better according to the teacher that trained it "
                              "while becoming worse according to the other authority, revealing "
                              "teacher-worldview disagreement rather than a generic quality gain?",
        "population": {
            "identities": "the 18,953 rows of D0036 (S12's 4k arm; N0014, order-preserving filtered "
                          "extension) — identical for every arm; a teacher-specific missing row fails "
                          "closed, it never shrinks a dataset",
            "rows": 18_953, "normalization": "N0014"},
        "teachers": {
            "CVS-4k": {"authority": "legacy.cvs.search.shallow", "producer": CVS_PRODUCER,
                       "budget": {"nodeBudget": 4_000}, "dataset": CVS_ARM_DATASET,
                       "spec_hash": CVS_SPEC.spec_hash(),
                       "measured_cost_ms_per_label": cvs_cost["mean_wall_ms"],
                       "realized_nodes": cvs_cost["realized_nodes_total"],
                       "realized_nodes_mean_per_label": cvs_cost["mean_realized_nodes"]},
            "SF-32k": {"authority": ACTIVE_AUTHORITY, "producer": SF_SHA256, "budget": {"nodes": 32_000},
                       "role": "PRIMARY Stockfish condition (cheapest calibrated rung; more expensive "
                               "per label than CVS-4k, so a CVS win cannot be a spending artifact)",
                       "spec_hash": SF_SPECS["SF-32k"].spec_hash(),
                       **{key: labels["SF-32k"][key] for key in
                          ("mean_wall_ms", "mean_realized_nodes", "mean_depth", "realized_nodes")}},
            "SF-1M": {"authority": ACTIVE_AUTHORITY, "producer": SF_SHA256, "budget": {"nodes": 1_000_000},
                      "role": "DOSE condition: bounds whether an effect is authority or budget",
                      "spec_hash": SF_SPECS["SF-1M"].spec_hash(),
                      **{key: labels["SF-1M"][key] for key in
                         ("mean_wall_ms", "mean_realized_nodes", "mean_depth", "realized_nodes")}},
            "pinned_identity": {"binarySha256": SF_SHA256, "nets": list(SF_NETS),
                                "options": SF_OPTIONS,
                                "netSelection": "per position: the small net when |simple_eval| > 962"},
        },
        "student": {"family": "RAW-768", "widths": list(WIDTHS), "seeds": list(SEEDS),
                    "regime": f"fixed updates: MAX_UPDATES {MAX_UPDATES} x BATCH {BATCH} = "
                              f"{MAX_UPDATES * BATCH} presentations for every cell",
                    "recipe": "T0004-derived; identical optimizer/LR/init/K/LAMBDA for every arm",
                    "divergence_declaration": DECLARATION},
        "exams": {
            "CVS-DEEP": {"protocol": EXAM_PROTOCOL_CVS, "nodes": 400_000,
                         "role": "the in-lab exam X#### binds; keeps every run comparable to S7-S12"},
            "SF-DEEP": {"dataset": exam["dataset_id"], "protocol": exam["protocol_id"],
                        "nodes": EXAM_BUDGET, "identities": exam["identities"],
                        "role": "second frozen instrument over the SAME held-out identities"},
            "rule": "the two losses are never averaged and neither exam is called ground truth"},
        "lattice": {"conditions": [label for label, _ in ARMS], "widths": list(WIDTHS),
                    "seeds": list(SEEDS), "cells": len(ARMS) * len(WIDTHS) * len(SEEDS),
                    "cell_list": "explicit (scale, node_budget) pairs; each scale value is the arm's "
                                 "realized teacher spend / the 37,285,491 reference unit"},
        "analysis": {
            "primary": "per width, paired by seed: d = test_loss(SF-32k) - test_loss(CVS-4k), on each "
                       "exam separately, with separate one-sided 95% exact-t bounds at that width's df",
            "interaction": "teacher x exam: all six (teacher, exam) cells per width; the key patterns "
                           "are 'each student best matches its own authority' (worldview disagreement) "
                           "versus 'SF-trained improves on both exams' (generic quality)",
            "dose": ["SF-1M - SF-32k", "SF-1M - CVS-4k"],
            "no_pooling": "widths are reported, never pooled; no single winner is declared",
            "mates": "|cp| >= 100k sentinel rows are counted and reported per exam split"},
        "economics": {"CVS-4k": {"mean_wall_ms_per_label": cvs_cost["mean_wall_ms"],
                                 "realized_nodes": cvs_cost["realized_nodes_total"]},
                      **{label: {"mean_wall_ms_per_label": labels[label]["mean_wall_ms"],
                                 "realized_nodes": labels[label]["realized_nodes"]}
                         for label in labels if label.startswith("SF")}},
        "calibration": {"attempt_1": "tools/s13/s13-calibration.json (rule failed: no converged rung)",
                        "amendment": "tools/s13/s13-calibration-amendment.json",
                        "cold_rerun": ("tools/s13/s13-calibration-cold.json" if CONTRACT == "cold"
                                       else None),
                        "chosen": {"primary": "SF-32k", "dose": "SF-1M", "exam": EXAM_BUDGET}},
        "disagreement_before_training": disagreement.get("blocks", {}),
        "scope_exclusions": ["no games", "no playing-strength claim", "no GEO/HYBRID inputs",
                             "no outcome or blended targets", "no per-arm optimizer tuning",
                             "no post-hoc teacher-budget tuning", "no Stockfish-as-truth language",
                             "sealed S7-S12 experiments are immutable"],
        "stop_conditions": ["a training row lacks a teacher label", "the dual-exam pass cannot "
                            "reproduce the in-lab CVS metrics exactly", "membership or the explicit "
                            "cell list fails", "the SF binary or net identity drifts from the pin"],
    }
    digest = write_json(PREREG_PATH, prereg)
    PREREG_HASH.write_text(digest + "\n", encoding="utf-8")
    save(**{PREREG_KEY: {"file": str(PREREG_PATH.relative_to(LAB)).replace(chr(92), "/"),
                        "hash": digest, "contract": CONTRACT}})
    print(f"preregistration {digest}", flush=True)
    return 0


def labels_cost_cvs(st: dict) -> float:
    return float(st["cvs_cost"]["mean_wall_ms"])


def step_experiment() -> int:
    """Freeze the lattice: three teacher conditions, each with its own honest spend stratum."""
    st = state()
    if ("experiment" if CONTRACT == "warm" else "experiment_cold") in st:
        print("[cached] experiment")
        return 0
    service = svc()
    labels = st[LABELS_KEY]
    sealed = json.loads((LAB / "tools" / "s12" / "s12-result.json").read_text(encoding="utf-8"))
    sealed_cvs_hash = sealed["instrument_contract"]["train_spec_by_arm"]["4k"][0]
    if CVS_SPEC.spec_hash() != sealed_cvs_hash:
        raise LabError(f"the CVS condition would not supervise exactly as X0008's 4k arm did: "
                       f"{CVS_SPEC.spec_hash()} != {sealed_cvs_hash}")
    cvs_realized = next(arm for arm in json.loads((LAB / "tools" / "s12" / "s12-state.json")
                                                  .read_text(encoding="utf-8"))["arms"]
                        if arm["arm_id"] == "4k")["realized_nodes"]
    rows = _train_ids(service, CVS_ARM_DATASET)
    while True:
        candidate = service.store.next_id("X")
        if not [a for a in service.store.list("A", verify=True, kind=Ablation)
                if a.experiment_id == candidate]:
            xid = candidate
            break
    control = service.store.get("T0004")
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    conditions = []
    for label, budget in ARMS:
        if label == "CVS-4k":
            dataset, spec, realized = service.store.get(CVS_ARM_DATASET), CVS_SPEC, cvs_realized
            price = labels_cost_cvs(st)
        else:
            spec = SF_SPECS[label]
            dataset = service.freeze_dataset(
                "N0014", name=f"s13-{label.lower()}{SUFFIX}", record_ids=rows, required_labels=[FAMILY],
                target_spec=spec, fractions=(1.0, 0.0, 0.0),
                campaign={"purpose": "S13 teacher condition", "teacher": label,
                          "authority": ACTIVE_AUTHORITY, "contract": CONTRACT,
                          "producer": SF_SHA256, "budget": {"nodes": budget},
                          "target_spec": spec.spec_hash(), "rows": len(rows),
                          "preregistration": "tools/s13/s13-preregistration.json"})
            realized, price = labels[label]["realized_nodes"], labels[label]["mean_wall_ms"]
        split = next(entry for entry in dataset.splits if entry.name == "train")
        if split.count != len(rows):
            raise LabError(f"{label}: dataset holds {split.count} rows, the population has {len(rows)}")
        scale_name = f"{realized / REFERENCE_UNIT:.1f}x"
        recipe = service.create_training_recipe(
            name=f"s13-{label.lower()}{SUFFIX}",
            params={**base, "EPOCHS": 1, "BATCH": BATCH, "MAX_UPDATES": MAX_UPDATES},
            target_spec=spec, description=f"S13 {label}: same student compute, teacher {label}")
        baseline = service.register_baseline(
            name=f"S13_{label.upper().replace(chr(45), chr(95))}{SUFFIX.upper().replace(chr(45), chr(95))}",
            dataset_id=dataset.id, training_recipe_id=recipe.id,
            eval_protocol_id=EXAM_PROTOCOL_CVS, model_config={"INPUT": "RAW", "H": 16},
            supervision_divergence=DECLARATION, experiment_id=xid,
            notes=f"S13 {label}: {split.count} rows, authority {spec.authority}, "
                  f"{price} ms/label, {realized:,} teacher nodes")
        ablation_ids = {16: baseline.id}
        for width in WIDTHS:
            if width == 16:
                continue
            ablation_ids[width] = service.create_ablation(
                baseline_id=baseline.id, overrides={"H": width}, supervision_divergence=DECLARATION,
                experiment_id=xid, notes=f"S13 {label} H={width}").id
        conditions.append({
            "label": label,
            "arm": {"arm_id": f"S13-{label}", "scale": scale_name, "scale_nodes": realized,
                    "node_budget": budget, "prefix_size": split.count, "realized_nodes": realized,
                    "dataset_id": dataset.id, "dataset_manifest_hash": dataset.manifest_hash,
                    "training_recipe_id": recipe.id, "recipe_hash": recipe.recipe_hash,
                    "train_target_spec_hash": spec.spec_hash(),
                    "record_ids_hash": split.record_ids_hash,
                    "ablations": [ablation_ids[width] for width in WIDTHS]},
            "cell": (scale_name, budget),
            "record": {"dataset_id": dataset.id, "spec_hash": spec.spec_hash(),
                       "authority": spec.authority, "realized_nodes": realized, "rows": split.count,
                       "ms_per_label": price,
                       "unit_multiples": realized / REFERENCE_UNIT}})
    experiment = service.create_experiment(
        name=f"S13 teacher authority: CVS vs Stockfish under dual frozen static exams{SUFFIX}",
        preregistration_hash=PREREG_HASH.read_text(encoding="utf-8").strip(),
        reference_unit_nodes=REFERENCE_UNIT,
        scales={row["arm"]["scale"]: row["arm"]["scale_nodes"] for row in conditions},
        node_budgets=[budget for _, budget in ARMS],
        declared_cells=[row["cell"] for row in conditions],
        arms=[row["arm"] for row in conditions], eval_protocol_id=EXAM_PROTOCOL_CVS,
        widths=list(WIDTHS), seeds=list(SEEDS), source_id="S0012", normalization_id="N0014",
        candidate_universe_hash=json.loads((LAB / "tools" / "s12" / "s12-state.json")
                                           .read_text(encoding="utf-8"))["universe"]["universe_hash"],
        order_seed=20260920,
        order_hash=json.loads((LAB / "tools" / "s12" / "s12-state.json")
                              .read_text(encoding="utf-8"))["universe"]["order_hash"],
        analysis={"design": "three teacher conditions over one population: CVS at 4k nodes, Stockfish "
                            "at 32k and 1M nodes; identical rows and identical student compute",
                  "scale_axis": "each arm's realized teacher spend / the 37,285,491 reference unit, "
                                "declared exactly; the conditions are not a grid, so the cell list is "
                                "explicit rather than a product",
                  "exam": "X#### binds CVS-DEEP (E0004); every model is additionally scored on the "
                          "frozen SF-DEEP instrument by a recorded dual-exam pass",
                  "primary": "per width, paired by seed: SF-32k - CVS-4k on each exam separately",
                  "interaction": "teacher x exam reported explicitly; no single winner",
                  "dose": ["SF-1M - SF-32k", "SF-1M - CVS-4k"]},
        experiment_id=xid, notes=f"3 teacher conditions x 4 widths x 20 seeds = 240 cells; two "
                                 f"frozen static exams, never averaged; contract {CONTRACT}")
    expected = service.expected_membership(experiment)
    if len(expected) != len(ARMS) * len(WIDTHS) * len(SEEDS):
        print(f"STOP: expected membership {len(expected)}")
        return 3
    save(**{("xid" if CONTRACT == "warm" else "xid_cold"): xid,
            ("experiment" if CONTRACT == "warm" else "experiment_cold"):
                {"id": experiment.id, "arms": len(experiment.arms), "cells": len(expected)},
            ("conditions" if CONTRACT == "warm" else "conditions_cold"):
                {row["label"]: row["record"] for row in conditions}})
    print(f"experiment {experiment.id}: {len(experiment.arms)} arms, {len(expected)} cells", flush=True)
    for row in conditions:
        print(f"  {row['label']}: scale {row['arm']['scale']} ({row['arm']['scale_nodes']:,} teacher "
              f"nodes), budget {row['arm']['node_budget']}, {row['arm']['prefix_size']} rows, "
              f"{row['record']['ms_per_label']} ms/label", flush=True)
    return 0


def step_run() -> int:
    st = state()
    service = svc()
    queued = st.get("queued") if CONTRACT == "warm" else st.get("queued_cold")
    if not queued:
        experiment = service.store.get_as(st["xid" if CONTRACT == "warm" else "xid_cold"],
                                          Experiment)
        queued = {ablation_id: [run.id for run in service.queue_runs(ablation_id, seeds=SEEDS)]
                  for arm in experiment.arms for ablation_id in arm.ablations}
        total = sum(len(ids) for ids in queued.values())
        if total != len(ARMS) * len(WIDTHS) * len(SEEDS):
            raise LabError(f"queued {total} runs, expected 240")
        save(**{("queued" if CONTRACT == "warm" else "queued_cold"): queued})
        print(f"queued {total} runs", flush=True)
    counts: dict[str, int] = {}
    done = 0
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != st["xid" if CONTRACT == "warm" else "xid_cold"]:
            continue
        counts[run.status.value] = counts.get(run.status.value, 0) + 1
        if run.status.value in ("COMPLETED", "FAILED", "INVALID"):
            continue
        finished = service.execute_run(run.id)
        counts[finished.status.value] = counts.get(finished.status.value, 0) + 1
        done += 1
        if finished.status.value != "COMPLETED":
            print(f"  {run.id} {finished.status.value}: "
                  f"{[c.detail for c in finished.integrity if c.status == 'fail'][:2]}", flush=True)
        elif done % 20 == 0:
            print(f"    ... {done} runs executed", flush=True)
    save(**{("run_statuses" if CONTRACT == "warm" else "run_statuses_cold"): counts})
    print("run statuses:", counts, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["exams", "preregister", "experiment", "run"])
    args = parser.parse_args()
    return {"exams": step_exams, "preregister": step_preregister,
            "experiment": step_experiment, "run": step_run}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
