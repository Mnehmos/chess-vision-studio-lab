"""S12 analysis: the lower economic frontier 4k/2k/1k/512 at one teacher budget.

    python tools/s12/s12_analyse.py report   # reads X0008, writes s12-result.json
    python tools/s12/s12_analyse.py seal     # seals X0008 with the result (completeness gate)

Preregistered (tools/s12/s12-preregistration.json, sha256:72950623...):

  primary   d = test_loss(512) - test_loss(4k), paired by seed, per width;
            separate one-sided 95% bounds, exact Student-t at each width's actual df
  decision  SHALLOWER_FRONTIER iff upper < 0 at every width
            FRONTIER_EXHAUSTED iff lower > 0 at every width
            INCONCLUSIVE otherwise
  adjacent  2k-4k, 1k-2k, 512-1k (descriptive; the per-halving tax and its bend)
  economics realized node totals and rows per arm, reported separately from quality
  no_pooling widths are reported, never pooled
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from cvslab.funnel.campaign import paired_effect
from cvslab.schemas import Experiment, Run, RunStatus
from cvslab.service import LabService
from cvslab.store import Store

LAB = pathlib.Path(r"F:\Github\chess-vision-studio-lab")
S12 = LAB / "tools" / "s12"
STATE = S12 / "s12-state.json"
RESULT = S12 / "s12-result.json"
SUMMARY = S12 / "s12-result-summary.json"
XID = "X0008"
WIDTHS = (1, 4, 16, 32)
DEPTHS = (4_000, 2_000, 1_000, 512)
LABEL = {4_000: "4k", 2_000: "2k", 1_000: "1k", 512: "512"}
HALVINGS = (("2k-4k", "2k", "4k"), ("1k-2k", "1k", "2k"), ("512-1k", "512", "1k"))
PRIMARY = ("512", "4k")
SEEDS = tuple(range(20))


def service() -> LabService:
    return LabService(Store(str(LAB / "labstore")))


def collect():
    """test_loss per (arm, width, seed) for X0008, with the integrity of every run re-verified."""
    svc = service()
    experiment: Experiment = svc.store.get_as(XID, Experiment)
    arm_of = {ablation_id: arm.arm_id for arm in experiment.arms for ablation_id in arm.ablations}
    seeds_of = {arm.arm_id: tuple(arm.seeds) for arm in experiment.arms}
    values: dict = {}
    invalid, seen = [], {}
    contract: dict = {"run_ids": [], "train_spec_by_arm": {}, "eval_specs": set(),
                      "divergences": set(), "datasets_by_arm": {}}
    for run in svc.store.list("R", verify=True, kind=Run):
        if run.experiment_id != XID:
            continue
        arm = arm_of.get(run.ablation_id)
        if arm is None or arm not in ("S12-4k", "S12-2k", "S12-1k", "S12-512"):
            continue
        label = arm.replace("S12-", "")
        contract["run_ids"].append(run.id)
        contract["train_spec_by_arm"].setdefault(label, set()).add(run.train_target_spec_hash)
        contract["datasets_by_arm"].setdefault(label, set()).add(run.dataset_id)
        contract["eval_specs"].add(run.eval_target_spec_hash)
        contract["divergences"].add(run.supervision_divergence)
        seen[(label, run.effective_config["H"])] = seen.get((label, run.effective_config["H"]), 0) + 1
        if run.status != RunStatus.COMPLETED:
            invalid.append(run.id)
            continue
        metric = next((m for m in run.metrics if m.name == "test_loss"), None)
        if metric is None:
            invalid.append(run.id)
            continue
        values.setdefault((label, run.effective_config["H"]), {})[run.seed] = metric.value
    duplicates = {f"{k[0]}-H{k[1]}": n for k, n in seen.items() if n != 20}
    exam = {"shared_exam_hash": svc.assert_runs_comparable(contract["run_ids"]),
            "exam_identity": list(svc.instrument_identity(svc.store.get_as(contract["run_ids"][0], Run))),
            "train_spec_by_arm": {k: sorted(v) for k, v in contract["train_spec_by_arm"].items()},
            "datasets_by_arm": {k: sorted(v) for k, v in contract["datasets_by_arm"].items()},
            "eval_spec_hashes_seen": sorted(contract["eval_specs"]),
            "supervision_divergence": sorted(contract["divergences"]),
            "runs_comparable": len(contract["run_ids"])}
    return experiment, seeds_of, values, invalid, duplicates, exam


def contrast(values, a: str, b: str, width: int) -> dict:
    """d = test_loss(a) - test_loss(b), paired by seed; positive = the shallower arm (a) is worse."""
    two = paired_effect(values[(a, width)], values[(b, width)], seeds=SEEDS)
    one = paired_effect(values[(a, width)], values[(b, width)], seeds=SEEDS, one_sided=True)
    stderr = (two["ci_high"] - two["width_estimate"]) / two["critical"]
    lower, upper = one["ci_low"], one["ci_high"]
    verdict = "a_worse" if lower > 0 else ("a_better" if upper < 0 else "unresolved")
    return {"n": two["n"], "d": two["width_estimate"], "stderr": stderr,
            "t_one_sided": one["critical"], "df": two["n"] - 1,
            "one_sided_bounds": [lower, upper], "two_sided_ci": [two["ci_low"], two["ci_high"]],
            "verdict": verdict, "mean_a": statistics.mean(values[(a, width)].values()),
            "mean_b": statistics.mean(values[(b, width)].values())}


def report() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    experiment, seeds_of, values, invalid, duplicates, exam = collect()
    cells_found = sum(len(v) for v in values.values())
    print(f"{XID}: {cells_found}/{experiment.expected_run_count} cells, "
          f"{len(invalid)} invalid, duplicates={duplicates}", flush=True)
    if cells_found != experiment.expected_run_count or invalid:
        print("REFUSING to analyse an incomplete or invalid study")
        return 1
    for arm in ("S12-4k", "S12-2k", "S12-1k", "S12-512"):
        if tuple(sorted(seeds_of[arm] or experiment.seeds)) != SEEDS:
            print(f"REFUSING: {arm} seeds {seeds_of[arm] or experiment.seeds}")
            return 1

    per_width = {}
    for width in WIDTHS:
        primary = contrast(values, *PRIMARY, width)
        adjacent = {name: contrast(values, a, b, width) for name, a, b in HALVINGS}
        taxes = [adjacent[name]["d"] for name, _, _ in HALVINGS]
        per_width[str(width)] = {
            "primary_512_minus_4k": primary,
            "adjacent": adjacent,
            "per_halving_tax": {"2k-4k": taxes[0], "1k-2k": taxes[1], "512-1k": taxes[2]},
            "tax_accelerates": taxes[1] > taxes[0] and taxes[2] > taxes[1],
            "tax_at_512_step_exceeds_2k_step": taxes[2] > taxes[0],
        }

    primaries = {w: per_width[str(w)]["primary_512_minus_4k"] for w in WIDTHS}
    if all(p["one_sided_bounds"][1] < 0 for p in primaries.values()):
        decision = "SHALLOWER_FRONTIER"
    elif all(p["one_sided_bounds"][0] > 0 for p in primaries.values()):
        decision = "FRONTIER_EXHAUSTED"
    else:
        decision = "INCONCLUSIVE"

    worsening = [str(w) for w in WIDTHS
                 if per_width[str(w)]["primary_512_minus_4k"]["verdict"] == "a_worse"]
    unresolved = [str(w) for w in WIDTHS
                  if per_width[str(w)]["primary_512_minus_4k"]["verdict"] == "unresolved"]
    resolving_steps = {}
    for name, _, _ in HALVINGS:
        resolving_steps[name] = [str(w) for w in WIDTHS
                                 if per_width[str(w)]["adjacent"][name]["verdict"] != "unresolved"]

    arms_state = {a["arm_id"]: a for a in state["arms"]}
    economics = {
        "equal_teacher_budget_nodes": 74_570_982,
        "arms": {a["arm_id"]: {"rows": a["prefix_size"], "realized_nodes": a["realized_nodes"],
                               "nodes_per_row": a["realized_nodes"] / a["prefix_size"],
                               "fraction_of_target": a["realized_nodes"] / 74_570_982}
                 for a in state["arms"]},
        "coverage_ratio_512_over_4k": (arms_state["512"]["prefix_size"]
                                       / arms_state["4k"]["prefix_size"]),
        "per_row_cost_ratio_4k_over_512": ((arms_state["4k"]["realized_nodes"]
                                            / arms_state["4k"]["prefix_size"])
                                           / (arms_state["512"]["realized_nodes"]
                                              / arms_state["512"]["prefix_size"])),
        "labels_bought": state["labels"]["per_depth"],
        "note": ("teacher cost is held equal BY DESIGN: every arm spends the same node budget and "
                 "buys more rows as labels get cheaper; student compute is also equal (MAX_UPDATES "
                 "760 x batch 256 = 194,560 presentations, the S8 control)"),
    }

    levels = {label: {str(w): statistics.mean(values[(label, w)].values()) for w in WIDTHS}
              for label in ("4k", "2k", "1k", "512")}

    result = {
        "experiment": XID,
        "preregistration_hash": experiment.preregistration_hash,
        "question": ("how far can teacher search be halved (4k -> 2k -> 1k -> 512) before the "
                     "quality penalty accelerates enough that the savings stop paying?"),
        "cells": {"expected": experiment.expected_run_count, "analysed": cells_found,
                  "invalid_runs": invalid, "duplicate_cells": duplicates},
        "arm_seeds": {arm.replace("S12-", ""): list(seeds_of[arm] or experiment.seeds)
                      for arm in seeds_of},
        "primary_512_minus_4k": {str(w): per_width[str(w)]["primary_512_minus_4k"]
                                 for w in WIDTHS},
        "adjacent_contrasts": {name: {str(w): per_width[str(w)]["adjacent"][name]
                                      for w in WIDTHS} for name, _, _ in HALVINGS},
        "per_halving_tax_by_width": {str(w): per_width[str(w)]["per_halving_tax"] for w in WIDTHS},
        "acceleration": {str(w): {"tax_accelerates_monotonically":
                                  per_width[str(w)]["tax_accelerates"],
                                  "512_step_tax_exceeds_2k_step":
                                  per_width[str(w)]["tax_at_512_step_exceeds_2k_step"]}
                         for w in WIDTHS},
        "decision": decision,
        "decision_rule": ("SHALLOWER_FRONTIER iff the one-sided 95% upper bound of "
                          "test_loss(512)-test_loss(4k) is below zero at every width; "
                          "FRONTIER_EXHAUSTED iff the lower bound is above zero at every width; "
                          "otherwise INCONCLUSIVE"),
        "widths_with_512_worse": worsening,
        "widths_unresolved_512_vs_4k": unresolved,
        "first_halving_that_resolves_per_width": resolving_steps,
        "mean_test_loss": levels,
        "instrument_contract": exam,
        "economics": economics,
        "provenance": {
            "universe": state["universe"],
            "prefix_reproduction": (
                "the append-only order preserves the S7-S order minus 36 records that the new "
                "games bridged to the exam by transposition; 8 of them lay inside the old 4k "
                "prefix and 32 inside the old 2k prefix, so X0008's 4k/2k arms are NOT row-identical "
                "to S8/S9/X0002/X0003's arms (18,953 vs 18,961 rows at 4k). Exact cross-experiment "
                "value reproduction is therefore not claimed; the realized-node targets, which are "
                "what the preregistration fixed, are met at 1.0000 in all four arms."),
            "nesting": state["nesting"],
        },
        "evidence": {"result": "tools/s12/s12-result.json",
                     "preregistration": "tools/s12/s12-preregistration.json",
                     "state": "tools/s12/s12-state.json"},
    }
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n",
                      encoding="utf-8")

    summary = {
        "experiment": XID, "decision": decision,
        "primary_512_minus_4k": {str(w): {"d": round(primaries[w]["d"], 6),
                                          "one_sided_bounds": [round(v, 6) for v in primaries[w]["one_sided_bounds"]],
                                          "t_one_sided": primaries[w]["t_one_sided"],
                                          "verdict": primaries[w]["verdict"]}
                                 for w in WIDTHS},
        "per_halving_tax": {str(w): {k: round(v, 6)
                                     for k, v in per_width[str(w)]["per_halving_tax"].items()}
                            for w in WIDTHS},
        "acceleration": {str(w): per_width[str(w)]["tax_accelerates"] for w in WIDTHS},
        "mean_test_loss": {arm: {w: round(v, 6) for w, v in levels[arm].items()} for arm in levels},
        "economics": economics["arms"],
        "evidence": result["evidence"],
    }
    SUMMARY.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n",
                       encoding="utf-8")

    print(f"decision: {decision}", flush=True)
    for w in WIDTHS:
        p = primaries[w]
        print(f"  H{w}: 512-4k d={p['d']:+.6f} one-sided [{p['one_sided_bounds'][0]:+.6f}, "
              f"{p['one_sided_bounds'][1]:+.6f}]  {p['verdict']}", flush=True)
    print("  tax 2k-4k / 1k-2k / 512-1k per width:", flush=True)
    for w in WIDTHS:
        t = per_width[str(w)]["per_halving_tax"]
        print(f"    H{w}: {t['2k-4k']:+.6f} / {t['1k-2k']:+.6f} / {t['512-1k']:+.6f}"
              f"  accelerates={per_width[str(w)]['tax_accelerates']}", flush=True)
    return 0


def seal() -> int:
    svc = service()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    sealed = svc.seal_experiment(XID, result)
    print(f"{XID} sealed: {json.dumps(sealed.result.get('membership'), default=str)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["report", "seal"])
    args = parser.parse_args()
    return {"report": report, "seal": seal}[args.step]()


if __name__ == "__main__":
    raise SystemExit(main())
