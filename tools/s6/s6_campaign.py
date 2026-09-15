"""S6 campaign launch: materialize the 40 cells and execute them.

Launch-time invariants (refuse rather than weaken):
* exactly 4 widths x 2 arms x 5 shared seeds = 40 cells;
* PRIORITY uses the frozen arm dataset/protocol, UNIFORM the other; both must have real
  T####/E#### identities (None => INVALID, never runnable);
* the campaign pins universe/split/candidate/policy hashes and both dataset manifest
  hashes; any drift aborts;
* every cell is recorded COMPLETE or INVALID with cause; no retries under changed settings;
* after all cells finish, the PREREGISTERED paired analysis runs on the actual test_loss
  values (no hand-selection).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"F:\Github\chess-vision-studio-lab")
from cvslab.data import read_jsonl
from cvslab.funnel.campaign import (CampaignPlan, WIDTHS, DEFAULT_SEEDS, EXPECTED_PARAMS, decide,
                                    interaction_effects, paired_effect)
from cvslab.service import LabService
from cvslab.store import LabError, Store
from cvslab.schemas import RunStatus

OUT = Path(r"F:\Github\_parity_tmp\s6")
RESULT = OUT / "s6-campaign-result.json"
PROGRESS = json.loads((OUT / "progress.json").read_text())


def cached(name):
    entry = PROGRESS[name]
    return entry.get("cached", entry)


def main() -> int:
    pool, s4, imp, uni = cached("pool"), cached("s4run"), cached("import"), cached("universe")
    ds, proto = cached("datasets"), cached("protocols")
    svc = LabService(Store(r"F:\Github\chess-vision-studio-lab\labstore"))

    # ---- pinned campaign identity: abort on drift --------------------------
    pins = {"universe_hash": uni["universe_hash"], "split_policy_hash": uni["split_policy_hash"],
            "candidate_universe_hash": uni["candidate_universe_hash"], "policy_hash": imp["policy_hash"],
            "D_PRIORITY": ds["report"]["arms"]["PRIORITY"]["manifest_hash"],
            "D_UNIFORM": ds["report"]["arms"]["UNIFORM"]["manifest_hash"]}
    arm_binding = {
        "PRIORITY": {"dataset_id": ds["report"]["arms"]["PRIORITY"]["dataset_id"],
                     "eval_protocol_id": proto["ids"]["PRIORITY"], "recipe_id": ds["recipe"]},
        "UNIFORM": {"dataset_id": ds["report"]["arms"]["UNIFORM"]["dataset_id"],
                    "eval_protocol_id": proto["ids"]["UNIFORM"], "recipe_id": ds["recipe"]},
    }
    for arm, binding in arm_binding.items():
        if not all(binding.values()):
            raise LabError(f"{arm} binding has a None identity; cells would not be runnable")
    dataset = {arm: svc.store.get(binding["dataset_id"]) for arm, binding in arm_binding.items()}
    for arm, object_ in dataset.items():
        if object_.manifest_hash != pins[f"D_{arm}"]:
            raise LabError(f"{arm} dataset manifest drifted: {object_.manifest_hash} != {pins[f'D_{arm}']}")
        if object_.campaign["universe_hash"] != pins["universe_hash"]:
            raise LabError(f"{arm} dataset no longer pins the campaign universe")

    # ---- materialize ablations (baselines are the H=1 cells) ---------------
    plan = CampaignPlan()
    ablations: dict[tuple[str, int], str] = {}
    for arm in ("PRIORITY", "UNIFORM"):
        baseline = svc.register_baseline(name=f"S6-{arm}", dataset_id=arm_binding[arm]["dataset_id"],
                                         training_recipe_id=arm_binding[arm]["recipe_id"],
                                         eval_protocol_id=arm_binding[arm]["eval_protocol_id"],
                                         model_config={"INPUT": "RAW", "H": 1},
                                         notes=f"S6 {arm} arm, H=1")
        ablations[(arm, 1)] = baseline.id
        for width in WIDTHS:
            if width == 1:
                continue
            ablation = svc.create_ablation(baseline_id=baseline.id, overrides={"H": width},
                                           notes=f"S6 {arm} arm, H={width}")
            ablations[(arm, width)] = ablation.id

    cells = plan.expand(split_policy_hash=pins["split_policy_hash"], recipe_id=ds["recipe"],
                        recipe_hash=ds["recipe_hash"], eval_semantics_hash=proto["semantics_hash"],
                        eval_ids_by_arm=proto["ids"])
    assert len(cells) == 40

    # ---- execute every cell; no retries, no tuning -------------------------
    results: dict[str, dict] = {}
    for cell in cells:
        ablation_id = ablations[(cell.arm, cell.width)]
        ablation = svc.store.get(ablation_id)
        if (ablation.param_count != EXPECTED_PARAMS[cell.width]
                or ablation.dataset_id != arm_binding[cell.arm]["dataset_id"]
                or ablation.eval_protocol_id != arm_binding[cell.arm]["eval_protocol_id"]
                or ablation.training_recipe_id != arm_binding[cell.arm]["recipe_id"]):
            raise LabError(f"{cell.cell_id}: ablation identity mismatch before launch")
        if cell.recipe_id is None or cell.eval_protocol_id is None:
            results[cell.cell_id] = {"status": "INVALID", "cause": "planner-only identities"}
            continue
        [run] = svc.queue_runs(ablation_id, seeds=(cell.seed,))
        finished = svc.execute_run(run.id)
        entry = {"status": finished.status.value, "run_id": run.id, "ablation_id": ablation_id,
                 "width": cell.width, "arm": cell.arm, "seed": cell.seed,
                 "param_count": finished.param_count}
        if finished.status == RunStatus.COMPLETED:
            metric = next((m for m in finished.metrics if m.name == "test_loss"), None)
            if metric is None:
                entry.update(status="INVALID", cause="no test_loss metric recorded")
            else:
                entry["test_loss"] = metric.value
        else:
            entry["cause"] = "; ".join(f"{c.name}: {c.detail}" for c in finished.integrity
                                       if c.status == "fail") or (finished.error or "unknown")
        results[cell.cell_id] = entry
        print(f"{cell.cell_id}: {entry['status']} {entry.get('test_loss', '')}", flush=True)

    # ---- preregistered analysis over the ACTUAL values ---------------------
    invalid = {cid: entry for cid, entry in results.items() if entry["status"] != "COMPLETE"}
    losses = {cid: entry["test_loss"] for cid, entry in results.items() if entry["status"] == "COMPLETE"}
    effects = {}
    analysis_error = None
    if invalid:
        analysis_error = f"{len(invalid)} INVALID cells; preregistered analysis requires all 40"
    else:
        for width in WIDTHS:
            priority = {seed: losses[f"H{width:02d}-PRIORITY-s{seed}"] for seed in DEFAULT_SEEDS}
            uniform = {seed: losses[f"H{width:02d}-UNIFORM-s{seed}"] for seed in DEFAULT_SEEDS}
            effects[width] = paired_effect(priority, uniform)
    verdict = decide(effects) if effects else "INVALID"
    payload = {
        "campaign": "S6 PRIORITY vs UNIFORM",
        "pins": pins, "arm_binding": arm_binding,
        "cells_total": 40, "cells_complete": 40 - len(invalid), "invalid": invalid,
        "losses": losses, "effects_by_width": effects,
        "interaction": interaction_effects(effects) if len(effects) >= 2 else None,
        "decision": verdict, "analysis_error": analysis_error,
        "preregistration": "paired seed-level differences, t 95% CI, every width required",
    }
    RESULT.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print("DECISION:", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
