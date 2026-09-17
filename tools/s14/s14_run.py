"""S14: the representation study — RAW vs GEO vs HYBRID at matched learned-parameter budgets.

    python tools/s14/s14_run.py manifest      # freeze the geometry column manifest + hash
    python tools/s14/s14_run.py preregister   # freeze the preregistration (ladder + contract)
    python tools/s14/s14_run.py experiment    # freeze the lattice: 3 family arms x budgets x 20 seeds
    python tools/s14/s14_run.py run           # execute every cell

The teacher and exam are S13's selected CVS contract, byte-for-byte: supervision = CVS-4k over
D0036's 18,953 rows (TargetSpec hash asserted equal to X0008's sealed 4k arm), exam = E0004
(CVS-DEEP 400k). Representation is the only changing variable.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from s14_common import (BATCH, CVS_SPEC_HASH, DATASET, EXAM_PROTOCOL, FAMILIES, LAB, MAX_UPDATES,
                        NODE_BUDGET, SCALE_NAME, SCALE_NODES, SEEDS, S14, STATE, geometry_manifest,
                        ladder, save, state)

from cvslab.hashing import hash_obj, read_jsonl
from cvslab.schemas import Ablation, Experiment, Run
from cvslab.service import LabService
from cvslab.store import LabError, Store
from cvslab.targets import TargetSpec

MANIFEST = S14 / "s14-geometry-manifest.json"
PREREG = S14 / "s14-preregistration.json"
DECLARATION = ("S14 representation arm: same positions, same teacher labels, same student compute "
               "and same exam as its paired arms; only the input representation and the matched "
               "learned-parameter budget change")


def svc() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def write_artifact(path: pathlib.Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return hash_obj(payload)


def step_manifest() -> int:
    if MANIFEST.is_file():
        print("[cached] manifest")
        return 0
    manifest = geometry_manifest()
    digest = write_artifact(MANIFEST, manifest)
    (S14 / "s14-geometry-manifest.hash").write_text(digest + "\n", encoding="utf-8")
    print(f"geometry manifest: {len(manifest['columns'])} named columns, registry "
          f"{manifest['registry_hash'][:19]}…, {digest}", flush=True)
    return 0


def step_preregister() -> int:
    if "preregistration" in state():
        print("[cached] preregister")
        return 0
    rows = ladder()
    manifest_digest = (S14 / "s14-geometry-manifest.hash").read_text(encoding="utf-8").strip()
    prereg = {
        "id": "s14-representation",
        "version": 1,
        "status": "materialized before the lattice is frozen and before any outcome is inspected",
        "issue": "Mnehmos/chess-vision-studio-lab#41 (controlled slice of #18)",
        "dependency": {"teacher": f"CVS-4k over {DATASET} (S12/S13's CVS arm; TargetSpec hash "
                                  f"{CVS_SPEC_HASH[:19]}… asserted against X0008's sealed 4k arm)",
                       "exam": f"{EXAM_PROTOCOL} CVS-DEEP 400k over the 200 held-out identities",
                       "why": "S13's selected static contract reused unchanged so representation is "
                              "the ONLY changing variable"},
        "questions": ["how much opaque learned capacity can be replaced by named deterministic "
                      "chess structure?",
                      "does explicit geometry add useful information to RAW-768, or can the raw "
                      "network already recover it efficiently?",
                      "at matched learned-parameter budgets, which representation gives the best "
                      "static judgment per parameter and per unit inference cost?"],
        "representations": {"RAW": "768 piece-square columns, side-to-move relative (existing)",
                            "GEO": "the 42 named registry columns (see the manifest)",
                            "HYBRID": "RAW ++ GEO in one input vector (810 columns)",
                            "architectures": {"crelu1": "inputs -> H clipped-ReLU -> 1",
                                              "linear": "inputs -> 1 (the white-box semantic floor; "
                                                        "GEO only, Stage A)"}},
        "geometry_manifest": {"file": "tools/s14/s14-geometry-manifest.json", "hash": manifest_digest,
                              "derivation": "position only; no label, teacher, move or outcome "
                                            "information enters the representation"},
        "ladder": rows,
        "student": {"seeds": list(SEEDS), "regime": f"fixed updates: MAX_UPDATES {MAX_UPDATES} x "
                                                    f"BATCH {BATCH} = {MAX_UPDATES * BATCH} "
                                                    "presentations for every cell",
                    "recipe": "T0004-derived; identical optimizer/LR/init/K/LAMBDA for every arm; "
                              "cold start every run; no per-family tuning",
                    "divergence_declaration": DECLARATION},
        "lattice": {"arms": [f"S14-{family}" for family in FAMILIES],
                    "budgets": list(rows["targets"]) + ["linear-floor"],
                    "conditions": len(rows["rows"]),
                    "cells": len(rows["rows"]) * len(SEEDS)},
        "analysis": {
            "primary": "per budget, paired by seed: test_loss differences GEO-RAW, HYBRID-RAW and "
                       "HYBRID-GEO with separate one-sided 95% exact-t bounds at each pair's df",
            "capability": "exam loss per learned parameter per family; exact counts and budget "
                          "mismatch reported beside every comparison",
            "costs": "train wall time (recorded per run), serialized model size (artifact bytes), "
                     "and inference cost measured BOTH as feature encoding and forward pass "
                     "(positions/second, python implementation, reported as measured)",
            "slices": "exam error by phase (opening/middlegame/endgame) and material bucket, "
                      "computed deterministically from the exam FENs",
            "overfit_gap": "loss on the training rows minus exam loss per model, labelled as a "
                           "distribution gap (there is no held-out validation split)",
            "semantic_ablation": "on the preregistered sentinel (HYBRID at the 12K budget, seed 0), "
                                 "zero one named geometry group's input columns at a time at "
                                 "INFERENCE on the frozen weights; report FULL - group loss change. "
                                 "No retraining, no combinatorial ablations",
            "no_pooling": "budgets are reported separately; no global winner score"},
        "stage_order": "GEO-LINEAR (the 43-parameter direct floor) is measured inside the same "
                       "frozen lattice, and the report distinguishes named-input observability "
                       "(linear) from hidden-unit opacity (crelu1)",
        "guardrails": ["teacher authority does not change inside S14", "no search or game claims",
                       "no auxiliary geometry prediction loss (that is S15)",
                       "no hand-crafted feature tuning after outcomes are visible",
                       "no silent feature dropping for missing values (the registry emits all "
                       "families for every position)",
                       "no copying of legacy semantic model code",
                       "negative and non-monotonic results are preserved"],
        "stop_conditions": ["the D0036 TargetSpec hash differs from X0008's sealed 4k arm",
                            "a lattice cell cannot train or evaluate",
                            "the geometry registry hash differs from the manifest's pin",
                            "membership or the frozen budget mapping fails"],
    }
    digest = write_artifact(PREREG, prereg)
    (S14 / "s14-preregistration.hash").write_text(digest + "\n", encoding="utf-8")
    save(preregistration={"file": "tools/s14/s14-preregistration.json", "hash": digest})
    print(f"preregistration {digest}", flush=True)
    return 0


def step_experiment() -> int:
    if "experiment" in state():
        print("[cached] experiment")
        return 0
    service = svc()
    sealed = json.loads((LAB / "tools" / "s12" / "s12-result.json").read_text(encoding="utf-8"))
    sealed_hash = sealed["instrument_contract"]["train_spec_by_arm"]["4k"][0]
    if CVS_SPEC_HASH != sealed_hash:
        raise LabError(f"the S14 teacher spec {CVS_SPEC_HASH} is not X0008's 4k arm {sealed_hash}")
    rows = ladder()["rows"]
    dataset = service.store.get(DATASET)
    control = service.store.get("T0004")
    base = {key: value for key, value in control.params.items() if key != "TARGET_SPEC"}
    spec = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                      producer="36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838",
                      budget={"nodeBudget": 4_000}, value_path=("scoreCpStm",), pov="stm",
                      k=256.0, lam=1.0)
    if spec.spec_hash() != sealed_hash:
        raise LabError("the reconstructed CVS-4k spec does not hash to the sealed value")
    recipe = service.create_training_recipe(
        name="s14-cvs-4k", params={**base, "EPOCHS": 1, "BATCH": BATCH, "MAX_UPDATES": MAX_UPDATES},
        target_spec=spec, description="S14: representation is the only variable; teacher CVS-4k")
    while True:
        candidate = service.store.next_id("X")
        if not [a for a in service.store.list("A", verify=True, kind=Ablation)
                if a.experiment_id == candidate]:
            xid = candidate
            break
    widths = sorted({row["H"] for row in rows})

    def unique_name(base: str) -> str:
        """Baseline names are unique per family-generation; a crashed attempt leaves orphans, so a
        corrected retry takes a fresh name rather than colliding with its own leftovers."""
        taken = {a.baseline_name for a in service.store.list("A", verify=True, kind=Ablation)
                 if a.baseline_name}
        if base not in taken:
            return base
        index = 1
        while f"{base}_R{index}" in taken:
            index += 1
        return f"{base}_R{index}"

    arms_payload, conditions = [], {}
    for family_input in FAMILIES:
        family_rows = [row for row in rows if row["input"] == family_input]
        model_config = {"INPUT": family_input, "H": family_rows[0]["H"]}
        baseline = service.register_baseline(
            name=unique_name(f"S14_{family_input}"), dataset_id=DATASET,
            training_recipe_id=recipe.id,
            eval_protocol_id=EXAM_PROTOCOL, model_config=model_config,
            supervision_divergence=DECLARATION, experiment_id=xid,
            notes=f"S14 {family_input}: representation arm, teacher CVS-4k unchanged")
        ablation_ids = {str(family_rows[0]["budget_target"]): baseline.id}
        for row in family_rows[1:]:
            overrides = {"H": row["H"]}
            if row.get("ARCH") == "linear":
                overrides = {"ARCH": "linear", "H": 1}
            ablation_ids[str(row["budget_target"])] = service.create_ablation(
                baseline_id=baseline.id, overrides=overrides, supervision_divergence=DECLARATION,
                experiment_id=xid,
                notes=f"S14 {family_input} budget {row['budget_target']} "
                      f"({row['learned_parameters']} params)").id
        split = next(entry for entry in dataset.splits if entry.name == "train")
        arms_payload.append({
            "arm_id": f"S14-{family_input}", "scale": SCALE_NAME, "scale_nodes": SCALE_NODES,
            "node_budget": NODE_BUDGET, "prefix_size": split.count, "realized_nodes": SCALE_NODES,
            "dataset_id": DATASET, "dataset_manifest_hash": dataset.manifest_hash,
            "training_recipe_id": recipe.id, "recipe_hash": recipe.recipe_hash,
            "train_target_spec_hash": spec.spec_hash(), "record_ids_hash": split.record_ids_hash,
            "ablations": list(ablation_ids.values())})
        conditions[family_input] = {str(row["budget_target"]): {
            "H": row["H"], "ARCH": row.get("ARCH", "crelu1"),
            "learned_parameters": row["learned_parameters"], "mismatch": row["mismatch"],
            "ablation_id": ablation_ids[str(row["budget_target"])]} for row in family_rows}
        print(f"  {family_input}: " + ", ".join(
            f"{key}={value['learned_parameters']}p(H={value['H']}"
            + (",linear)" if value["ARCH"] == "linear" else ")")
            for key, value in conditions[family_input].items()), flush=True)
    experiment = service.create_experiment(
        name="S14 representation: RAW vs GEO vs HYBRID at matched learned-parameter budgets",
        preregistration_hash=(S14 / "s14-preregistration.hash").read_text(encoding="utf-8").strip(),
        reference_unit_nodes=37_285_491, scales={SCALE_NAME: SCALE_NODES}, node_budgets=[NODE_BUDGET],
        arms=arms_payload, eval_protocol_id=EXAM_PROTOCOL, widths=widths, seeds=list(SEEDS),
        source_id="S0012", normalization_id="N0014",
        candidate_universe_hash=json.loads((LAB / "tools" / "s12" / "s12-state.json")
                                           .read_text(encoding="utf-8"))["universe"]["universe_hash"],
        order_seed=20260920,
        order_hash=json.loads((LAB / "tools" / "s12" / "s12-state.json")
                              .read_text(encoding="utf-8"))["universe"]["order_hash"],
        analysis={"design": "three input representations over one teacher/exam contract at four "
                            "matched learned-parameter budgets plus the GEO linear floor",
                  "primary": "per budget, paired by seed: GEO-RAW, HYBRID-RAW, HYBRID-GEO",
                  "capability": "loss per learned parameter; exact counts and mismatches reported",
                  "semantic_ablation": "named-group input lesions at inference on the HYBRID 12K "
                                       "seed-0 sentinel"},
        experiment_id=xid, notes="3 representation arms x 4 budgets (+GEO linear floor) x 20 seeds")
    expected = service.expected_membership(experiment)
    total = 13 * len(SEEDS)
    if len(expected) != total:
        print(f"STOP: expected membership {len(expected)} != {total}")
        return 3
    save(xid=xid, experiment={"id": experiment.id, "arms": len(experiment.arms),
                              "cells": len(expected)}, conditions=conditions)
    print(f"experiment {experiment.id}: {len(experiment.arms)} arms, {len(expected)} cells", flush=True)
    return 0


def step_run() -> int:
    st = state()
    service = svc()
    queued = st.get("queued")
    if not queued:
        experiment = service.store.get_as(st["xid"], Experiment)
        queued = {ablation_id: [run.id for run in service.queue_runs(ablation_id, seeds=SEEDS)]
                  for arm in experiment.arms for ablation_id in arm.ablations}
        total = sum(len(ids) for ids in queued.values())
        if total != 13 * len(SEEDS):
            raise LabError(f"queued {total} runs, expected {13 * len(SEEDS)}")
        save(queued=queued)
        print(f"queued {total} runs", flush=True)
    experiment = service.store.get_as(st["xid"], Experiment)
    report = service.membership_report(experiment)
    retried = []
    # a RUNNING run at driver start is an orphan: this driver is the only executor, so nothing
    # else can be working on it (the earlier driver was killed mid-run). It is marked INVALID
    # with its reason recorded, and the replacement semantics recover the cell.
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != st["xid"] or run.status.value != "RUNNING":
            continue
        run.status = __import__("cvslab.schemas", fromlist=["RunStatus"]).RunStatus.INVALID
        run.error = "orphaned RUNNING: the executing driver died; retried by a later driver"
        service.store.update_run(run)
        print(f"  orphaned {run.id} marked INVALID (dead driver)", flush=True)
    # cells whose only run is INVALID are retried: the crash stays recorded, the retry counts
    all_runs = [run for run in service.store.list("R", verify=True, kind=Run)
                if run.experiment_id == st["xid"]]
    invalid_only = {}
    by_cell: dict = {}
    for run in all_runs:
        by_cell.setdefault((run.ablation_id, run.seed), []).append(run)
    for (ablation_id, seed), runs in by_cell.items():
        if len(runs) == 1 and runs[0].status.value == "INVALID":
            invalid_only[(ablation_id, seed)] = runs[0]
    for (ablation_id, seed), crashed in sorted(invalid_only.items()):
        retry = service.queue_runs(ablation_id, seeds=[seed])[0]
        retried.append({"cell": [ablation_id, seed], "crashed": crashed.id, "retry": retry.id,
                        "error": crashed.error})
        print(f"  retry {ablation_id} seed {seed} (crash: {crashed.error})", flush=True)
    counts: dict[str, int] = {}
    for run in service.store.list("R", verify=True, kind=Run):
        if run.experiment_id != st["xid"]:
            continue
        if run.status.value in ("COMPLETED", "FAILED", "INVALID"):
            counts[run.status.value] = counts.get(run.status.value, 0) + 1
            continue
        finished = service.execute_run(run.id)
        counts[finished.status.value] = counts.get(finished.status.value, 0) + 1
        if finished.status.value != "COMPLETED":
            print(f"  {run.id} {finished.status.value}: "
                  f"{[c.detail for c in finished.integrity if c.status == 'fail'][:2]}", flush=True)
    save(run_statuses=counts, retries=retried)
    print("run statuses:", counts, f"({len(retried)} cells retried)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["manifest", "preregister", "experiment", "run"])
    args = parser.parse_args()
    return {"manifest": step_manifest, "preregister": step_preregister,
            "experiment": step_experiment, "run": step_run}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
