"""S7-P pilot: freeze the new arms, execute them, analyse the three preregistered contrasts.

Implements the merged preregistration (tools/s7/s7-preregistration.json, `pilot` section)
against EXISTING labels only:

    P1  94 rows x 400k   the twenty S6 UNIFORM cells (R0054..R0088), reused verbatim
    P2  94 rows x 16k    new D + T + A + 20 runs   (the depth contrast)
    P3  926 rows x 16k   new D + T + A + 20 runs   (the rows contrast)
    P4  926 rows x 2k    new D + T + A + 20 runs   (exploratory only, no decision authority)

No new labels are bought, no engine search runs, no source pool is generated, no priority
selection is used, and nothing historical is mutated. Every new arm declares its
train/eval TargetSpec divergence before its runs exist (P1), and every run must pass the
supervision authority check.

    python tools/s7/s7_pilot.py freeze     # datasets, recipes, ablations (idempotent)
    python tools/s7/s7_pilot.py run        # queue + execute the 60 new runs
    python tools/s7/s7_pilot.py analyze    # three contrasts per width, sealed result
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.funnel.campaign import paired_effect
from cvslab.schemas import TERMINAL_RUN_STATUSES, Run, TrainingRecipe
from cvslab.service import LabService, protocol_target_spec, recipe_target_spec
from cvslab.store import Store
from cvslab.targets import TargetSpec

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
TOOLS = LAB / "tools" / "s7"
STATE = TOOLS / "s7-pilot-state.json"
IDENTITIES = TOOLS / "s7-pilot-identities.json"
RESULT = TOOLS / "s7-pilot-result.json"
SUMMARY = TOOLS / "s7-pilot-result-summary.json"
PRODUCER = "36e9b1592858939cf88233bf99d367e3ea9c43a044f342f69792d49a17d09838"
INVENTORY = TOOLS / "s7-pilot-inventory.json"

WIDTHS = (1, 4, 16, 32)
SEEDS = (0, 1, 2, 3, 4)
RISK = "S7-P pilot: trained on its own budget's observation, examined on the frozen instrument E0004/D0010"
UNIFORM_RUNS = [54, 55, 56, 57, 58, 64, 65, 66, 67, 68, 74, 75, 76, 77, 78, 84, 85, 86, 87, 88]

SPECS = {
    "16k": TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow", producer=PRODUCER,
                      budget={"nodeBudget": 16000}, value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0),
    "2k": TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow", producer=PRODUCER,
                     budget={"nodeBudget": 2000}, value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0),
}
ARMS = {
    "P2": {"rows": 94, "spec": "16k", "name": "s7p-rows94-16k", "source": "D0009 record identities"},
    "P3": {"rows": 926, "spec": "16k", "name": "s7p-rows926-16k", "source": "the 926 train-eligible records"},
    "P4": {"rows": 926, "spec": "2k", "name": "s7p-rows926-2k", "source": "same 926 records, 2k observation"},
}


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save(**updates) -> None:
    STATE.write_text(json.dumps({**state(), **updates}, indent=1, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] saved {list(updates)}", flush=True)


def log_inventory() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def freeze() -> int:
    inventory = log_inventory()
    if "stop" in inventory:
        print("STOP: the inventory did not pass")
        return 3
    svc = LabService(Store(str(LAB / "labstore")))
    st = state()
    if "arms" in st:
        print("[cached] freeze already done")
        return 0

    normalization_id = inventory["corpus"]["normalization"]
    uniform_ids = inventory["uniform_record_ids"]
    identities: dict = {"normalization": normalization_id, "arms": {}, "recipes": {}, "ablations": {}}

    control_recipe_params = inventory["control_recipe"]["params"]      # EPOCHS/BATCH/LR/OPTIMIZER/INIT_STD/K/LAMBDA
    for label, spec in SPECS.items():
        recipe = svc.create_training_recipe(name=f"s7p-{label}",
                                            params={**control_recipe_params},
                                            target_spec=spec,
                                            description=f"S7-P pilot supervision at {label} nodes; identical to T0004 "
                                                        "except the frozen TargetSpec")
        identities["recipes"][label] = {"id": recipe.id, "hash": recipe.recipe_hash,
                                        "spec": recipe_target_spec(recipe).spec_hash()}
    print("recipes:", {k: v["id"] for k, v in identities["recipes"].items()}, flush=True)

    eligible_ids = inventory["eligible_record_ids"]
    for arm, definition in ARMS.items():
        if arm == "P2" and len(uniform_ids) != definition["rows"]:
            raise SystemExit(f"P2 must be the 94 D0009 rows; found {len(uniform_ids)}")
        record_ids = uniform_ids if arm == "P2" else eligible_ids
        if len(record_ids) != definition["rows"]:
            raise SystemExit(f"{arm} must have {definition['rows']} rows; found {len(record_ids)}")
        spec = SPECS[definition["spec"]]
        dataset = svc.freeze_dataset(normalization_id, name=definition["name"], record_ids=record_ids,
                                     required_labels=["search_shallow_cp"], target_spec=spec,
                                     fractions=(1.0, 0.0, 0.0),
                                     campaign={"purpose": "S7-P pilot arm", "arm": arm,
                                               "rows": definition["rows"], "record_ids_from": definition["source"],
                                               "node_budget": spec.budget["nodeBudget"],
                                               "target_spec": spec.spec_hash(),
                                               "inventory": "tools/s7/s7-pilot-inventory.json",
                                               "design": "docs/SUPERVISION_DENSITY_FRONTIER.md section 4"})
        recipe: TrainingRecipe = svc.store.get_as(identities["recipes"][definition["spec"]]["id"], TrainingRecipe)
        baseline = svc.register_baseline(name=f"S7P_{arm}_H16", dataset_id=dataset.id,
                                         training_recipe_id=recipe.id, eval_protocol_id="E0004",
                                         model_config={"INPUT": "RAW", "H": 16},
                                         notes=f"{arm}: {definition['rows']} rows with the "
                                               f"{spec.budget['nodeBudget']} observation, examined on E0004/D0010",
                                         supervision_divergence=RISK)
        ablations = {"16": baseline.id}
        for width in (1, 4, 32):
            ablation = svc.create_ablation(baseline_id=baseline.id, overrides={"H": width}, notes=f"{arm} H={width}",
                                           supervision_divergence=RISK)
            ablations[str(width)] = ablation.id
        identities["ablations"][arm] = ablations
        identities["arms"][arm] = {"dataset": dataset.id, "manifest": dataset.manifest_hash,
                                   "rows": dataset.splits[0].count, "recipe": recipe.id,
                                   "recipe_hash": recipe.recipe_hash, "protocol": "E0004",
                                   "spec": spec.spec_hash(), "ablations": ablations}
        print(f"{arm}: dataset {dataset.id} ({dataset.splits[0].count} rows), baseline {baseline.id}, "
              f"ablations {ablations}", flush=True)

    save(arms=identities["arms"], recipes=identities["recipes"])
    IDENTITIES.write_text(json.dumps({"control": {"dataset": "D0009", "recipe": "T0004", "protocol": "E0004",
                                                  "runs": [f"R{n:04d}" for n in UNIFORM_RUNS]},
                                      "inventory_hash": inventory.get("candidate_universe_hash"),
                                      "frozen": identities}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("written", IDENTITIES)
    return 0


def run() -> int:
    svc = LabService(Store(str(LAB / "labstore")))
    st = state()
    if "runs" not in st:
        queued: dict = {}
        for arm, arm_state in st["arms"].items():
            for width, ablation_id in arm_state["ablations"].items():
                runs = svc.queue_runs(ablation_id, seeds=SEEDS)
                queued[f"{arm}-H{width}"] = [r.id for r in runs]
        save(queued=queued)
        print("queued", sum(len(v) for v in queued.values()), "runs", flush=True)
    queued = state()["queued"]
    results: dict = {}
    for cell, run_ids in queued.items():
        for run_id in run_ids:
            run: Run = svc.store.get_as(run_id, Run)
            if run.status in TERMINAL_RUN_STATUSES:
                results[run_id] = run.status.value
                continue
            finished = svc.execute_run(run_id)
            results[run_id] = finished.status.value
            if finished.status.value != "COMPLETED":
                failed = [c.detail for c in finished.integrity if c.status == "fail"]
                print(f"  {run_id} {finished.status.value}: {failed[:3]}", flush=True)
    save(results=results)
    counts: dict = {}
    for value in results.values():
        counts[value] = counts.get(value, 0) + 1
    print("run statuses:", counts, flush=True)
    return 0 if counts.get("INVALID", 0) == 0 and counts.get("COMPLETED", 0) == 60 else 3


def collect(svc: LabService) -> dict:
    """test_loss per (arm, width, seed); the reused arm comes from the S6 UNIFORM cells."""
    out: dict = {}
    for number in UNIFORM_RUNS:
        run: Run = svc.store.get_as(f"R{number:04d}", Run)
        metric = next(m for m in run.metrics if m.name == "test_loss")
        out.setdefault("P1", {}).setdefault(str(run.effective_config["H"]), {})[run.seed] = metric.value
    for cell, run_ids in state()["queued"].items():
        arm, width = cell.split("-H")
        for run_id in run_ids:
            run = svc.store.get_as(run_id, Run)
            if run.status.value != "COMPLETED":
                continue
            metric = next(m for m in run.metrics if m.name == "test_loss")
            out.setdefault(arm, {}).setdefault(width, {})[run.seed] = metric.value
    return out


def analyze() -> int:
    svc = LabService(Store(str(LAB / "labstore")))
    values = collect(svc)
    inventory = log_inventory()

    def effect(a, b, width):
        """Paired seed difference: first named treatment minus second, t-CI over the five seeds."""
        return paired_effect(values.get(a, {}).get(str(width), {}), values.get(b, {}).get(str(width), {}),
                             seeds=SEEDS)

    contrasts = {"P_depth": ("P2", "P1"), "P_rows": ("P3", "P2"), "P_econ": ("P3", "P1")}
    exploratory = {"P_2k_vs_16k_same_rows": ("P4", "P3"), "P_2k_vs_400k": ("P4", "P1")}
    report: dict = {"unit": "paired seed difference; 95% CI, Student t, df=4, t=2.776",
                    "primary_endpoint": "test_loss on E0004/D0010",
                    "sign_convention": "negative favours the FIRST named treatment",
                    "values": values, "contrasts": {}, "exploratory": {}}
    for name, (a, b) in contrasts.items():
        report["contrasts"][name] = {str(width): {"a": a, "b": b, **effect(a, b, width)} for width in WIDTHS}
    for name, (a, b) in exploratory.items():
        report["exploratory"][name] = {str(width): {"a": a, "b": b, **effect(a, b, width)} for width in WIDTHS}

    econ = report["contrasts"]["P_econ"]
    if all(econ[str(w)]["ci_high"] < 0 for w in WIDTHS):
        reading = "BROAD_SHALLOW_BEATS_NARROW_DEEP_AT_40_PERCENT_COMPUTE"
    elif all(econ[str(w)]["ci_low"] > 0 for w in WIDTHS):
        reading = "DEPTH_IS_LOAD_BEARING_EVEN_AGAINST_9_9X_ROWS"
    else:
        reading = "NO_DETECTABLE_DIFFERENCE_AT_40_PERCENT_COMPUTE"
    depth = report["contrasts"]["P_depth"]
    rows = report["contrasts"]["P_rows"]
    report["reading"] = {
        "P_econ": reading,
        # P_depth = (16k) - (400k): a NEGATIVE difference means the shallower labels did better
        "P_depth_by_width": {str(w): ("shallower 16k better" if depth[str(w)]["ci_high"] < 0 else
                                      "deeper 400k better" if depth[str(w)]["ci_low"] > 0 else "no difference")
                             for w in WIDTHS},
        "P_rows_by_width": {str(w): ("more rows better" if rows[str(w)]["ci_high"] < 0 else
                                     "fewer rows better" if rows[str(w)]["ci_low"] > 0 else "no difference")
                            for w in WIDTHS},
    }

    labels = {"P1": {"rows": 94, "budget": 400000}, "P2": {"rows": 94, "budget": 16000},
              "P3": {"rows": 926, "budget": 16000}, "P4": {"rows": 926, "budget": 2000}}
    report["economics"] = {arm: {**spec, "teacher_nodes_estimate": spec["rows"] * spec["budget"],
                                 "relative_to_P1": round(spec["rows"] * spec["budget"] / (94 * 400000), 4)}
                           for arm, spec in labels.items()}
    report["new_labels_bought"] = 0
    report["reused_cells"] = [f"R{n:04d}" for n in UNIFORM_RUNS]
    report["inventory_candidate_universe_hash"] = inventory.get("candidate_universe_hash")
    report["scope"] = ("this pool, this teacher, these four arms; not matched compute; requires P1; "
                       "pilot results never enter the frontier decision rule")
    RESULT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    summary = {"pilot": "S7-P existing-label pilot",
               "design": "same positions different depth (P_depth); same depth more rows (P_rows); "
                         "the economic combination (P_econ, not matched compute)",
               "reading": report["reading"], "economics": report["economics"],
               "contrasts": {k: {w: {"effect": round(v["width_estimate"], 6),
                                     "ci": [round(v["ci_low"], 6), round(v["ci_high"], 6)], "n": v["n"]}
                                 for w, v in cells.items()} for k, cells in report["contrasts"].items()},
               "exploratory": {k: {w: {"effect": round(v["width_estimate"], 6),
                                       "ci": [round(v["ci_low"], 6), round(v["ci_high"], 6)], "n": v["n"]}
                                   for w, v in cells.items()} for k, cells in report["exploratory"].items()},
               "evidence": {"result": "tools/s7/s7-pilot-result.json",
                            "inventory": "tools/s7/s7-pilot-inventory.json",
                            "identities": "tools/s7/s7-pilot-identities.json",
                            "reused_cells": [f"R{n:04d}" for n in UNIFORM_RUNS]},
               "scope": report["scope"]}
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["reading"], indent=1))
    print("written", RESULT, "and", SUMMARY)
    return 0


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"freeze": freeze, "run": run, "analyze": analyze}[step]())
