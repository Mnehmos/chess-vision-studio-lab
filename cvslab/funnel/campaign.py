"""S6 (#12): the PRIORITY-vs-UNIFORM causal experiment machinery.

This module is deterministic planning and integrity code only. It builds:

* the **frozen experimental universe** — one S5 raw source, one canonical corpus, one
  pinned S4 funnel run + policy, one transposition-safe split assignment, one eligible
  record universe with the deep labels the experiment needs;
* **matched data arms** — PRIORITY (the frozen priority-v1 deep-selection arm) and
  UNIFORM (uniformly selected from the same eligible universe) with identical effective
  row counts and *measured* deep-label compute compared under a declared tolerance;
* the **reliability matrix** — H x arm x seed cells (40 primary runs), each cell carrying
  the same training recipe, evaluation suite, split identity and seed set, so width is the
  only model-capacity variable and arm the only data-selection variable.

Fail-closed rules: unequal arm sizes, mismatched split identity, missing deep labels,
mixed policy hashes, changed candidate universe, or compute parity outside tolerance all
refuse the experiment rather than weaken it. Nothing here tunes priority-v1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..hashing import canonical_json, hash_obj
from ..store import LabError

WIDTHS = (1, 4, 16, 32)
ARM_NAMES = ("PRIORITY", "UNIFORM")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
# exact learned parameters for the RAW-768 family: 768*H + H(b1) + H(w2) + 1(b2)
EXPECTED_PARAMS = {1: 771, 4: 3081, 16: 12321, 32: 24641}
# declared compute-parity tolerance between arms (measured deep nodes)
COMPUTE_PARITY_TOLERANCE = 0.02  # 2%


@dataclass(frozen=True)
class PlannedRun:
    """One cell of the reliability matrix; identity is (width, arm, seed)."""

    width: int
    arm: str
    seed: int
    param_count: int

    @property
    def cell_id(self) -> str:
        return f"H{self.width:02d}-{self.arm}-s{self.seed}"


@dataclass
class CampaignPlan:
    widths: tuple[int, ...] = WIDTHS
    arms: tuple[str, ...] = ARM_NAMES
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    # everything below is shared by EVERY cell — changing it per-arm is a bug
    recipe_params: dict = field(default_factory=lambda: {
        "OPTIMIZER": "adam", "EPOCHS": 40, "BATCH": 256, "LR": 0.003,
        "INIT_STD": 0.05, "LAMBDA": 1.0, "K": 256.0, "INPUT": "RAW", "OUT_SCALE_CP": 400.0,
    })
    eval_suite: str = "frozen-tiny-suite-v1"
    init_policy: str = "cold-start"  # never warm-start from legacy weights

    def expand(self) -> list[PlannedRun]:
        runs = [PlannedRun(width, arm, seed, EXPECTED_PARAMS[width])
                for width in self.widths for arm in self.arms for seed in self.seeds]
        cells = [run.cell_id for run in runs]
        if len(cells) != len(set(cells)):
            raise LabError("campaign matrix contains duplicate cells; refusing to plan")
        return runs

    def expected_total(self) -> int:
        return len(self.widths) * len(self.arms) * len(self.seeds)

    def identity(self) -> dict:
        """The plan's own hash: the same cells + controls must reproduce it."""
        return {
            "widths": list(self.widths), "arms": list(self.arms), "seeds": list(self.seeds),
            "recipe_params": dict(sorted(self.recipe_params.items())),
            "eval_suite": self.eval_suite, "init_policy": self.init_policy,
            "expected_params": {str(w): EXPECTED_PARAMS[w] for w in self.widths},
        }

    def plan_hash(self) -> str:
        return hash_obj(self.identity())


def assert_shared_controls(runs: list[PlannedRun], plan: CampaignPlan) -> None:
    """Changing the arm must never change recipe, eval suite, init seed or params formula."""
    by_arm: dict[str, set[tuple]] = {}
    for run in runs:
        by_arm.setdefault(run.arm, set()).add((run.width, run.seed, run.param_count))
    for arm, cells in by_arm.items():
        for other_arm, other_cells in by_arm.items():
            if arm == other_arm:
                continue
            missing = cells - other_cells
            if missing:
                raise LabError(f"arm {arm} and {other_arm} do not share the same cell set: {sorted(missing)}")
    for run in runs:
        if run.param_count != EXPECTED_PARAMS[run.width]:
            raise LabError(f"{run.cell_id}: parameter count {run.param_count} != "
                           f"{EXPECTED_PARAMS[run.width]} for the RAW-768 formula")


# ---------------------------------------------------------------------------
# Frozen experimental universe and matched arms
# ---------------------------------------------------------------------------


@dataclass
class Universe:
    """The one frozen universe every arm draws from. Nothing below may be resampled per arm."""

    source_id: str
    normalization_id: str
    funnel_run_id: str
    policy_version: str
    policy_hash: str
    eligible_ids: tuple[str, ...]          # canonical record ids with the required deep labels
    eligible_components: dict[str, str]    # record_id -> transposition component group
    deep_nodes_by_id: dict[str, int]       # measured deep-label nodes per eligible record
    split_by_component: dict[str, str]     # component group -> train|val|test (frozen once)

    def universe_hash(self) -> str:
        return hash_obj({
            "source": self.source_id, "normalization": self.normalization_id,
            "funnel_run": self.funnel_run_id, "policy_hash": self.policy_hash,
            "eligible_count": len(self.eligible_ids),
            "eligible_hash": hash_obj(sorted(self.eligible_ids)),
            "split_hash": hash_obj(sorted(self.split_by_component.items())),
            "deep_nodes_total": sum(self.deep_nodes_by_id.values()),
        })


def freeze_universe(*, source_id: str, normalization_id: str, funnel_run_id: str,
                    policy_version: str, policy_hash: str,
                    eligible: list[dict], split_seed: int,
                    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Universe:
    """Freeze the universe: eligible records (with their component group and measured deep
    nodes) and ONE split assignment over component groups, shared by every arm."""
    from ..data import _split_for

    if not eligible:
        raise LabError("eligible universe is empty; nothing to experiment on")
    eligible_ids, components, deep_nodes = [], {}, {}
    for record in eligible:
        record_id, component, nodes = record["record_id"], record["group"], int(record.get("deep_nodes", 0))
        if not component or component.endswith(":UNKNOWN"):
            raise LabError(f"{record_id}: eligible record has unknown grouping; splits would be unsafe")
        if nodes <= 0:
            raise LabError(f"{record_id}: eligible record carries no measured deep-label nodes")
        eligible_ids.append(record_id)
        components[record_id] = component
        deep_nodes[record_id] = nodes
    if len(eligible_ids) != len(set(eligible_ids)):
        raise LabError("eligible universe contains duplicate record identities")
    split_by_component = {component: _split_for(component, split_seed, fractions)
                          for component in sorted(set(components.values()))}
    return Universe(source_id=source_id, normalization_id=normalization_id, funnel_run_id=funnel_run_id,
                    policy_version=policy_version, policy_hash=policy_hash,
                    eligible_ids=tuple(sorted(eligible_ids)), eligible_components=components,
                    deep_nodes_by_id=deep_nodes, split_by_component=split_by_component)


def select_arms(universe: Universe, *, deep_selected_ids: list[str], selection_seed: int,
                target_rows: int) -> dict[str, list[str]]:
    """PRIORITY = the frozen deep-selection arm's records; UNIFORM = same count, uniformly
    sampled from the SAME eligible universe (seed-derived, deterministic)."""
    import random

    eligible = set(universe.eligible_ids)
    priority = sorted(set(deep_selected_ids))
    unknown = [record_id for record_id in priority if record_id not in eligible]
    if unknown:
        raise LabError(f"deep-selection arm names {len(unknown)} records outside the eligible "
                       f"universe (e.g. {unknown[0]}); the candidate universe changed")
    if target_rows and len(priority) != target_rows:
        raise LabError(f"PRIORITY arm has {len(priority)} rows but the plan requires {target_rows}")
    rng = random.Random(selection_seed)
    uniform = sorted(rng.sample(sorted(eligible), len(priority)))
    if len(uniform) != len(priority):
        raise LabError("arms are not row-matched; refusing the equal-rows experiment")
    return {"PRIORITY": priority, "UNIFORM": uniform}


def compute_parity(universe: Universe, arms: dict[str, list[str]],
                   *, tolerance: float = COMPUTE_PARITY_TOLERANCE) -> dict:
    """Measured deep-label compute per arm from actual evidence, with a declared tolerance."""
    totals = {arm: sum(universe.deep_nodes_by_id[record_id] for record_id in ids)
              for arm, ids in arms.items()}
    priority, uniform = totals["PRIORITY"], totals["UNIFORM"]
    baseline = max(priority, uniform) or 1
    relative_gap = abs(priority - uniform) / baseline
    if relative_gap > tolerance:
        raise LabError(
            f"compute parity outside tolerance: PRIORITY {priority:,} vs UNIFORM {uniform:,} deep nodes "
            f"({relative_gap:.2%} > {tolerance:.2%}); the equal-compute experiment is refused")
    return {"deep_nodes": totals, "relative_gap": round(relative_gap, 6), "tolerance": tolerance}


def split_identity(universe: Universe, arms: dict[str, list[str]]) -> dict:
    """The same canonical record must map to the same split in every arm (fail closed)."""
    per_arm = {arm: {record_id: universe.split_by_component[universe.eligible_components[record_id]]
                     for record_id in ids} for arm, ids in arms.items()}
    reference = per_arm["PRIORITY"]
    for arm, mapping in per_arm.items():
        for record_id, split in mapping.items():
            if reference.get(record_id, split) != split:
                raise LabError(f"{record_id}: split differs between arms ({arm}={split}); "
                               "datasets must never resample splits independently")
    shared = set().union(*per_arm.values())
    return {
        "split_assignment_hash": hash_obj(sorted(reference.items())),
        "shared_records": len(shared),
        "per_arm_split_hashes": {arm: hash_obj(sorted(mapping.items())) for arm, mapping in per_arm.items()},
    }
