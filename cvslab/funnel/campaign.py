"""S6 (#12): the PRIORITY-vs-UNIFORM causal experiment machinery.

The treatment and its control were already randomized and paid for by the frozen
policy: S3's triage produced `selection.deep` and `selection.uniform` from the same
pre-selection pool under the policy seed, and S4 deep-labeled them. **S6 compares
those recorded arms; it never re-randomizes after observing which records received
expensive labels.**

Deterministic planning and integrity code only:

* the **frozen experimental universe** — one candidate population (the full
  pre-selection, train-eligible triage pool), its transposition components, the
  measured Tier-3 deep-label evidence, the recorded PRIORITY/UNIFORM memberships,
  and ONE global component→split assignment;
* **matched arms** — the recorded `selection.deep` / `selection.uniform` ids, with
  equal row counts and measured deep-label compute compared under a declared,
  pre-registered tolerance;
* the **reliability matrix** — H x arm x seed cells (40 primary runs) where every
  cell carries a `controls_hash` over the exact T#### recipe, E#### suite, split
  policy, target semantics and cold-start policy, so paired cells cannot silently
  diverge.

Pre-registered statistics (before any result was seen) are in PREREGISTRATION.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from ..hashing import hash_obj
from ..store import LabError

WIDTHS = (1, 4, 16, 32)
ARM_NAMES = ("PRIORITY", "UNIFORM")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
# exact learned parameters for the RAW-768 family: 768*H + H(b1) + H(w2) + 1(b2)
EXPECTED_PARAMS = {1: 771, 4: 3081, 16: 12321, 32: 24641}
# declared compute-parity tolerance between the two recorded arms (measured deep nodes);
# part of the campaign identity because changing it changes admissibility
COMPUTE_PARITY_TOLERANCE = 0.02

TARGET_SEMANTICS = "eval_cp sigmoid target, LAMBDA=1.0, K=256 (RAW-768 family)"
COLD_START_POLICY = "cold-start; no legacy warm start"

PREREGISTRATION = {
    "primary_estimate": "per width H: mean over seeds of the paired difference "
                        "test_loss_PRIORITY(seed) - test_loss_UNIFORM(seed)",
    "uncertainty": "95% CI from the five paired differences (Student t, df=4, t=2.776)",
    "secondary": "CP MAE, sign agreement, phase/material/motif slices where supported",
    "interaction": "width x strategy = difference of paired effects between widths, blocked by seed",
    "decision": "SUPPORTED iff the paired CI upper bound < 0 at every width; "
                "REJECTED iff the paired CI lower bound > 0 at every width; else INCONCLUSIVE. "
                "A win at H=1 alone is not sufficient evidence.",
    "scope": "this candidate pool only; independent-pool replication is #16",
}


@dataclass(frozen=True)
class PlannedRun:
    """One cell of the reliability matrix; identity is (width, arm, seed)."""

    width: int
    arm: str
    seed: int
    param_count: int
    controls_hash: str

    @property
    def cell_id(self) -> str:
        return f"H{self.width:02d}-{self.arm}-s{self.seed}"


@dataclass
class CampaignPlan:
    widths: tuple[int, ...] = WIDTHS
    arms: tuple[str, ...] = ARM_NAMES
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    recipe_params: dict = field(default_factory=lambda: {
        "OPTIMIZER": "adam", "EPOCHS": 40, "BATCH": 256, "LR": 0.003,
        "INIT_STD": 0.05, "LAMBDA": 1.0, "K": 256.0, "INPUT": "RAW", "OUT_SCALE_CP": 400.0,
    })
    eval_suite: str = "frozen-tiny-suite-v1"
    init_policy: str = COLD_START_POLICY
    target_semantics: str = TARGET_SEMANTICS
    compute_parity_tolerance: float = COMPUTE_PARITY_TOLERANCE

    def controls_hash(self, split_policy_hash: str) -> str:
        """Identity of every control a cell must share with its paired cells."""
        return hash_obj({
            "recipe_params": dict(sorted(self.recipe_params.items())),
            "eval_suite": self.eval_suite,
            "split_policy_hash": split_policy_hash,
            "target_semantics": self.target_semantics,
            "init_policy": self.init_policy,
        })

    def expand(self, *, split_policy_hash: str) -> list[PlannedRun]:
        controls = self.controls_hash(split_policy_hash)
        runs = [PlannedRun(width, arm, seed, EXPECTED_PARAMS[width], controls)
                for width in self.widths for arm in self.arms for seed in self.seeds]
        cells = [run.cell_id for run in runs]
        if len(cells) != len(set(cells)):
            raise LabError("campaign matrix contains duplicate cells; refusing to plan")
        return runs

    def expected_total(self) -> int:
        return len(self.widths) * len(self.arms) * len(self.seeds)

    def identity(self) -> dict:
        return {
            "widths": list(self.widths), "arms": list(self.arms), "seeds": list(self.seeds),
            "recipe_params": dict(sorted(self.recipe_params.items())),
            "eval_suite": self.eval_suite, "init_policy": self.init_policy,
            "target_semantics": self.target_semantics,
            "compute_parity_tolerance": self.compute_parity_tolerance,
            "expected_params": {str(w): EXPECTED_PARAMS[w] for w in self.widths},
        }

    def plan_hash(self) -> str:
        return hash_obj(self.identity())


def assert_shared_controls(runs: list[PlannedRun], plan: CampaignPlan) -> None:
    """Every cell carries one common controls_hash; paired cells cannot diverge."""
    if not runs:
        raise LabError("no planned runs")
    hashes = {run.controls_hash for run in runs}
    if len(hashes) != 1:
        raise LabError(f"cells disagree on controls identity: {sorted(hashes)}")
    by_arm: dict[str, set[tuple]] = {}
    for run in runs:
        by_arm.setdefault(run.arm, set()).add((run.width, run.seed, run.param_count, run.controls_hash))
    for arm, cells in by_arm.items():
        for other_arm, other_cells in by_arm.items():
            if arm != other_arm and cells - other_cells:
                raise LabError(f"arm {arm} and {other_arm} do not share the same cell set")
    for run in runs:
        if run.param_count != EXPECTED_PARAMS[run.width]:
            raise LabError(f"{run.cell_id}: parameter count {run.param_count} != "
                           f"{EXPECTED_PARAMS[run.width]} for the RAW-768 formula")


# ---------------------------------------------------------------------------
# Frozen experimental universe: recorded arms, no resampling
# ---------------------------------------------------------------------------


@dataclass
class Universe:
    """Candidate population + recorded arms + one global split assignment."""

    source_id: str
    normalization_id: str
    funnel_run_id: str
    policy_version: str
    policy_hash: str
    candidate_ids: tuple[str, ...]              # full pre-selection, train-eligible triage pool
    components: dict[str, str]                  # record_id -> component group (all candidates)
    deep_nodes_by_id: dict[str, int]            # measured Tier-3 nodes (only where labels exist)
    arms: dict[str, tuple[str, ...]]            # recorded PRIORITY / UNIFORM memberships
    split_seed: int
    fractions: tuple[float, float, float]
    split_by_component: dict[str, str]          # component -> train|val|test (frozen once)

    def split_policy_hash(self) -> str:
        return hash_obj({"split_seed": self.split_seed, "fractions": list(self.fractions),
                         "assignments": sorted(self.split_by_component.items())})

    def candidate_universe_hash(self) -> str:
        return hash_obj(sorted(self.candidate_ids))

    def universe_hash(self) -> str:
        """Everything that would change the experiment if it changed: the full
        record→component mapping, the measured record→deep-nodes mapping, the policy
        identity, the split policy and the candidate-universe hash."""
        return hash_obj({
            "source": self.source_id, "normalization": self.normalization_id,
            "funnel_run": self.funnel_run_id,
            "policy_version": self.policy_version, "policy_hash": self.policy_hash,
            "candidate_universe_hash": self.candidate_universe_hash(),
            "components": sorted(self.components.items()),
            "deep_nodes": sorted(self.deep_nodes_by_id.items()),
            "arms": {arm: list(ids) for arm, ids in sorted(self.arms.items())},
            "split_policy_hash": self.split_policy_hash(),
        })


def freeze_universe(*, source_id: str, normalization_id: str, funnel_run_id: str,
                    policy_version: str, policy_hash: str,
                    candidates: list[dict], deep_nodes: dict[str, int],
                    selections: dict[str, list[str]], split_seed: int,
                    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Universe:
    """Freeze the universe from RECORDED evidence.

    * ``candidates``: the full pre-selection triage population
      (``{record_id, group, train_eligible}``) — not only what happened to get labels.
    * ``deep_nodes``: measured Tier-3 nodes for records that actually received deep labels.
    * ``selections``: the frozen ``selection.deep`` / ``selection.uniform`` id lists from
      the pinned run; no new randomization happens here.
    """
    from ..data import _split_for

    if not candidates:
        raise LabError("candidate universe is empty; nothing to experiment on")
    candidate_ids, components, eligibility = [], {}, {}
    for record in candidates:
        record_id, component = record["record_id"], record.get("group") or ""
        if not component or component.endswith(":UNKNOWN"):
            raise LabError(f"{record_id}: candidate has unknown grouping; splits would be unsafe")
        candidate_ids.append(record_id)
        components[record_id] = component
        eligibility[record_id] = bool(record.get("train_eligible", True))
    if len(candidate_ids) != len(set(candidate_ids)):
        raise LabError("candidate universe contains duplicate record identities")
    candidate_set = set(candidate_ids)

    arms: dict[str, tuple[str, ...]] = {}
    for arm, key in (("PRIORITY", "deep"), ("UNIFORM", "uniform")):
        ids = sorted(set(selections.get(key) or []))
        if not ids:
            raise LabError(f"recorded {key} selection is empty; the pinned run has no {arm} arm")
        outside = [record_id for record_id in ids if record_id not in candidate_set]
        if outside:
            raise LabError(f"{arm} names {len(outside)} records outside the candidate universe "
                           f"(e.g. {outside[0]}); the candidate population changed")
        ineligible = [record_id for record_id in ids if not eligibility.get(record_id, False)]
        if ineligible:
            raise LabError(f"{arm} names non-train-eligible records (e.g. {ineligible[0]})")
        missing = [record_id for record_id in ids if int(deep_nodes.get(record_id, 0)) <= 0]
        if missing:
            raise LabError(f"{arm} names {len(missing)} records without measured Tier-3 deep labels "
                           f"(e.g. {missing[0]})")
        arms[arm] = tuple(ids)
    if len(arms["PRIORITY"]) != len(arms["UNIFORM"]):
        raise LabError(f"recorded arms are not row-matched: PRIORITY {len(arms['PRIORITY'])} vs "
                       f"UNIFORM {len(arms['UNIFORM'])}; refusing the equal-rows experiment")

    split_by_component = {component: _split_for(component, split_seed, fractions)
                          for component in sorted(set(components.values()))}
    return Universe(source_id=source_id, normalization_id=normalization_id, funnel_run_id=funnel_run_id,
                    policy_version=policy_version, policy_hash=policy_hash,
                    candidate_ids=tuple(sorted(candidate_ids)), components=components,
                    deep_nodes_by_id={record_id: int(nodes) for record_id, nodes in deep_nodes.items()},
                    arms=arms, split_seed=split_seed, fractions=fractions,
                    split_by_component=split_by_component)


def compute_parity(universe: Universe, *, tolerance: Optional[float] = None) -> dict:
    """Measured deep-label compute of the RECORDED arms, under a pre-registered tolerance."""
    if tolerance is None:
        tolerance = COMPUTE_PARITY_TOLERANCE
    totals = {arm: sum(universe.deep_nodes_by_id[record_id] for record_id in ids)
              for arm, ids in universe.arms.items()}
    priority, uniform = totals["PRIORITY"], totals["UNIFORM"]
    relative_gap = abs(priority - uniform) / (max(priority, uniform) or 1)
    if relative_gap > tolerance:
        raise LabError(
            f"compute parity outside tolerance: PRIORITY {priority:,} vs UNIFORM {uniform:,} deep nodes "
            f"({relative_gap:.2%} > {tolerance:.2%}); the equal-compute experiment is refused")
    return {"deep_nodes": totals, "relative_gap": round(relative_gap, 6), "tolerance": tolerance}


def split_identity(universe: Universe) -> dict:
    """One global component split policy, referenced identically by both D#### arms."""
    per_arm = {arm: hash_obj(sorted((record_id, universe.split_by_component[universe.components[record_id]])
                                     for record_id in ids))
               for arm, ids in universe.arms.items()}
    return {"split_policy_hash": universe.split_policy_hash(),
            "component_count": len(universe.split_by_component),
            "per_arm_record_split_hashes": per_arm,
            "note": "record-level hashes differ because the selected rows differ; the component "
                    "split POLICY hash is what both arms must share"}


# ---------------------------------------------------------------------------
# Pre-registered statistics (declared before results were seen)
# ---------------------------------------------------------------------------

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def paired_effect(priority_by_seed: dict[int, float], uniform_by_seed: dict[int, float]) -> dict:
    """Per-width primary estimate: paired seed-level differences (same seeds both arms)."""
    import math

    shared = sorted(set(priority_by_seed) & set(uniform_by_seed))
    if len(shared) < 2:
        raise LabError("paired effect needs at least two shared seeds")
    differences = [priority_by_seed[seed] - uniform_by_seed[seed] for seed in shared]
    n = len(differences)
    mean = sum(differences) / n
    variance = sum((value - mean) ** 2 for value in differences) / (n - 1)
    standard_error = math.sqrt(variance / n)
    critical = _T95.get(n - 1, 1.96)
    return {"width_estimate": mean, "ci_low": mean - critical * standard_error,
            "ci_high": mean + critical * standard_error, "n": n, "seeds": shared,
            "differences": differences, "method": PREREGISTRATION["uncertainty"]}


def interaction_effects(effects_by_width: dict[int, dict]) -> dict:
    """Width x strategy: differences of paired effects between widths, blocked by seed."""
    widths = sorted(effects_by_width)
    if len(widths) < 2:
        raise LabError("interaction needs at least two widths")
    pairs = {}
    for index in range(1, len(widths)):
        low, high = widths[index - 1], widths[index]
        low_effect, high_effect = effects_by_width[low], effects_by_width[high]
        shared = sorted(set(low_effect["seeds"]) & set(high_effect["seeds"]))
        per_seed = {seed: high_effect["width_estimate"] - low_effect["width_estimate"] for seed in shared}
        pairs[f"H{low}->H{high}"] = {
            "delta": high_effect["width_estimate"] - low_effect["width_estimate"],
            "per_seed": per_seed, "n": len(shared),
        }
    return {"pairs": pairs, "method": PREREGISTRATION["interaction"]}


def decide(effects_by_width: dict[int, dict]) -> str:
    """The pre-registered decision rule; a win at one width is not sufficient."""
    if not effects_by_width:
        raise LabError("no width effects to decide on")
    if all(effect["ci_high"] < 0 for effect in effects_by_width.values()):
        return "SUPPORTED"
    if all(effect["ci_low"] > 0 for effect in effects_by_width.values()):
        return "REJECTED"
    return "INCONCLUSIVE"
