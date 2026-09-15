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

TARGET_SEMANTICS = ("frozen TargetSpec supervision — the completed S6 campaign used search_deep_cp targets (value.targets.scoreCpStm, stm POV, 400k-node budget) with sigmoid(cp/K), K=256, LAMBDA=1; the governing identity is the TargetSpec hash pinned in T#### and E####")
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
    recipe_id: Optional[str] = None
    eval_protocol_id: Optional[str] = None

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

    def controls_hash(self, split_policy_hash: str, *, recipe_id: Optional[str] = None,
                      recipe_hash: Optional[str] = None,
                      eval_semantics_hash: Optional[str] = None) -> str:
        """Identity of every control a cell must share with its paired cells.

        Pass the real lab identities when they exist: `recipe_id`/`recipe_hash` from the
        frozen T#### and `eval_semantics_hash` = the E#### protocol's semantics with the
        dataset binding removed (arms necessarily bind different D####). Without them the
        planner-level dictionary is used, which is weaker and only valid pre-freeze.
        """
        return hash_obj({
            "recipe": {"id": recipe_id, "hash": recipe_hash,
                       "params": dict(sorted(self.recipe_params.items()))},
            "eval": {"semantics_hash": eval_semantics_hash, "suite": self.eval_suite},
            "split_policy_hash": split_policy_hash,
            "target_semantics": self.target_semantics,
            "init_policy": self.init_policy,
        })

    def expand(self, *, split_policy_hash: str, recipe_id: Optional[str] = None,
               recipe_hash: Optional[str] = None, eval_semantics_hash: Optional[str] = None,
               eval_ids_by_arm: Optional[dict[str, str]] = None) -> list[PlannedRun]:
        """Cells carry the real T####/E#### identities when the freeze stage supplies them.

        A runnable campaign must not be generated from planner-only `None` identities:
        pass the frozen T recipe identity/hash and the arm-specific E protocols (verified
        equal on semantics) here.
        """
        if eval_ids_by_arm is not None:
            missing = [arm for arm in self.arms if not eval_ids_by_arm.get(arm)]
            if missing:
                raise LabError(f"eval_ids_by_arm is missing {missing}; every arm needs its own E####")
        controls = self.controls_hash(split_policy_hash, recipe_id=recipe_id,
                                      recipe_hash=recipe_hash, eval_semantics_hash=eval_semantics_hash)
        runs = [PlannedRun(width, arm, seed, EXPECTED_PARAMS[width], controls,
                           recipe_id=recipe_id,
                           eval_protocol_id=(eval_ids_by_arm or {}).get(arm))
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
        if not bool(record.get("train_eligible", True)):
            raise LabError(f"{record_id}: candidate universe must contain only train-eligible records; "
                           "holdouts cannot enter the candidate-universe hash")
        candidate_ids.append(record_id)
        components[record_id] = component
        eligibility[record_id] = True
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


def compute_parity(universe: Universe, *, tolerance: float) -> dict:
    """Measured deep-label compute of the RECORDED arms under the plan's declared tolerance.

    `tolerance` is mandatory so a campaign cannot declare one tolerance in its hashed
    identity and accidentally execute another; pass `plan.compute_parity_tolerance`.
    """
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


def paired_effect(priority_by_seed: dict[int, float], uniform_by_seed: dict[int, float],
                  *, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> dict:
    """Per-width primary estimate: paired differences over the PREREGISTERED seed set.

    Seeds are not intersected silently: a missing seed means a failed cell, which must
    be reported INVALID rather than shrinking the analysis.
    """
    import math

    missing = [seed for seed in seeds if seed not in priority_by_seed or seed not in uniform_by_seed]
    if missing:
        raise LabError(f"paired effect requires every preregistered seed; missing {missing} "
                       "(report those cells INVALID instead of dropping them)")
    shared = sorted(seeds)
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
    """Width x strategy interaction, genuinely blocked by seed.

    Each seed contributes its own contrast d_H_high(seed) - d_H_low(seed), taken from the
    seed-level paired differences; uncertainty is computed over those contrasts.
    """
    import math

    widths = sorted(effects_by_width)
    if len(widths) < 2:
        raise LabError("interaction needs at least two widths")
    pairs = {}
    for index in range(1, len(widths)):
        low, high = widths[index - 1], widths[index]
        low_effect, high_effect = effects_by_width[low], effects_by_width[high]
        if low_effect["seeds"] != high_effect["seeds"]:
            raise LabError(f"H{low} and H{high} were not evaluated on the same seed set")
        low_by_seed = dict(zip(low_effect["seeds"], low_effect["differences"]))
        high_by_seed = dict(zip(high_effect["seeds"], high_effect["differences"]))
        per_seed = {seed: high_by_seed[seed] - low_by_seed[seed] for seed in low_effect["seeds"]}
        contrasts = [per_seed[seed] for seed in low_effect["seeds"]]
        n = len(contrasts)
        mean = sum(contrasts) / n
        variance = sum((value - mean) ** 2 for value in contrasts) / (n - 1) if n > 1 else 0.0
        standard_error = math.sqrt(variance / n) if n > 1 else 0.0
        critical = _T95.get(n - 1, 1.96)
        pairs[f"H{low}->H{high}"] = {
            "delta": mean, "ci_low": mean - critical * standard_error,
            "ci_high": mean + critical * standard_error,
            "per_seed": per_seed, "n": n,
        }
    return {"pairs": pairs, "method": PREREGISTRATION["interaction"]}


def decide(effects_by_width: dict[int, dict], *, required_widths: tuple[int, ...] = WIDTHS) -> str:
    """The pre-registered decision rule over the FULL width set.

    A win at one width is not sufficient, and missing widths are not silently
    ignored: incomplete evidence is refused so the caller must mark those cells
    INVALID explicitly instead of letting them disappear.
    """
    missing = [width for width in required_widths if width not in effects_by_width]
    if missing:
        raise LabError(f"decision requires every preregistered width; missing {missing} "
                       "(failed cells must be reported INVALID, not omitted)")
    if all(effects_by_width[width]["ci_high"] < 0 for width in required_widths):
        return "SUPPORTED"
    if all(effects_by_width[width]["ci_low"] > 0 for width in required_widths):
        return "REJECTED"
    return "INCONCLUSIVE"


def eval_semantics_hash(protocol) -> str:
    """Evaluation semantics of an E#### protocol, excluding the dataset binding.

    Two arms must bind different D#### manifests, so the protocols cannot be literally the
    same E####; this hash proves split, metrics, params, bootstrap configuration and
    protocol version are identical apart from that binding.
    """
    from ..schemas import EvaluationProtocol
    if not isinstance(protocol, EvaluationProtocol):
        raise LabError(f"expected an EvaluationProtocol, got {type(protocol).__name__}")
    return hash_obj({
        "split": protocol.split, "metrics": [m.model_dump(mode="json") for m in protocol.metrics],
        "params": dict(sorted(protocol.params.items())),
        "bootstrap_samples": protocol.bootstrap_samples, "bootstrap_seed": protocol.bootstrap_seed,
        "protocol_version": protocol.protocol_version, "non_claims": list(protocol.non_claims),
    })


def verify_arm_protocols(protocols_by_arm: dict) -> str:
    """Arm-specific E#### protocols must differ only by their D#### binding."""
    hashes = {arm: eval_semantics_hash(protocol) for arm, protocol in protocols_by_arm.items()}
    if len(set(hashes.values())) != 1:
        raise LabError(f"arm evaluation protocols differ beyond the dataset binding: {hashes}")
    return next(iter(hashes.values()))


def freeze_arm_datasets(store, universe: Universe, plan: CampaignPlan, *, recipe,
                        deep_engine_seconds: Optional[dict[str, float]] = None,
                        name_prefix: str = "s6", target_spec=None,
                        training_fractions: Optional[tuple[float, float, float]] = None) -> dict:
    """Freeze the PRIORITY and UNIFORM D#### arms from the RECORDED memberships.

    Membership comes exclusively from ``universe.arms`` (the recorded
    ``selection.deep`` / ``selection.uniform`` fields) — never from the presence of a
    deep label, so audit/holdout labels cannot leak into an arm. Fails closed before
    returning: parity per the plan's declared tolerance, shared split policy, equal
    rows, and every provenance relationship recorded in the campaign block.
    """
    from ..schemas import TrainingRecipe
    from ..data import freeze_dataset

    if not isinstance(recipe, TrainingRecipe):
        raise LabError("freeze_arm_datasets needs the frozen T#### TrainingRecipe")
    parity = compute_parity(universe, tolerance=plan.compute_parity_tolerance)
    identity = split_identity(universe)
    datasets, report_rows = {}, {}
    for arm in ("PRIORITY", "UNIFORM"):
        record_ids = list(universe.arms[arm])
        dataset = freeze_dataset(
            store, universe.normalization_id, name=f"{name_prefix}-{arm.lower()}-arm",
            record_ids=record_ids, required_labels=["search_deep_cp"],
            target_spec=target_spec,
            split_seed=universe.split_seed,
            fractions=training_fractions or universe.fractions,
            campaign={
                "arm": arm,
                "membership_source": "selection.deep" if arm == "PRIORITY" else "selection.uniform",
                "source_id": universe.source_id, "normalization_id": universe.normalization_id,
                "funnel_run_id": universe.funnel_run_id,
                "policy_version": universe.policy_version, "policy_hash": universe.policy_hash,
                "selection_split_policy_hash": identity["split_policy_hash"],
                "training_layout": ("all-selected-records-train"
                                    if (training_fractions or universe.fractions) == (1.0, 0.0, 0.0)
                                    else "global-component-split"),
                "candidate_universe_hash": universe.candidate_universe_hash(),
                "universe_hash": universe.universe_hash(),
                "recipe_id": recipe.id, "recipe_hash": recipe.recipe_hash,
                "deep_nodes": parity["deep_nodes"][arm],
                "deep_engine_seconds": (deep_engine_seconds or {}).get(arm),
                "compute_parity_tolerance": plan.compute_parity_tolerance,
                **({"target_spec_hash": target_spec.spec_hash()} if target_spec is not None else {})})
        if dataset.counts["records"] != len(record_ids):
            raise LabError(f"{arm}: frozen {dataset.counts['records']} rows for {len(record_ids)} selected ids; "
                           "missing deep labels must fail closed")
        # the frozen D files must realise the Universe's split policy exactly: recompute the
        # record->split mapping from the actual split files and compare with the identity
        from ..data import read_jsonl as _read_jsonl
        realised = {}
        for manifest in dataset.splits:
            for row in _read_jsonl(store.abs(manifest.path)):
                realised[row["record_id"]] = manifest.name
        from ..hashing import hash_obj as _hash_obj
        layout = ("all-selected-records-train"
                  if (training_fractions or universe.fractions) == (1.0, 0.0, 0.0)
                  else "global-component-split")
        if layout == "all-selected-records-train":
            # amended design: the arm is a TREATMENT — every selected record trains, and the
            # global component split's role is the evaluation exclusion policy, not the arm layout
            if set(realised) != set(record_ids) or any(split != "train" for split in realised.values()):
                raise LabError(f"{arm}: all-train layout violated; realised {sorted(set(realised.values()))}")
        elif _hash_obj(sorted(realised.items())) != identity["per_arm_record_split_hashes"][arm]:
            raise LabError(f"{arm}: frozen dataset splits do not match the Universe split policy "
                           f"(seed {universe.split_seed}, fractions {list(universe.fractions)}); "
                           "the manifest would claim one split policy while the files follow another")
        datasets[arm] = dataset
        report_rows[arm] = {"dataset_id": dataset.id, "rows": dataset.counts["records"],
                            "manifest_hash": dataset.manifest_hash, "deep_nodes": parity["deep_nodes"][arm],
                            "deep_engine_seconds": (deep_engine_seconds or {}).get(arm)}
    if datasets["PRIORITY"].counts["records"] != datasets["UNIFORM"].counts["records"]:
        raise LabError("frozen arms are not row-matched")
    return {
        "source_id": universe.source_id, "normalization_id": universe.normalization_id,
        "funnel_run_id": universe.funnel_run_id, "policy_version": universe.policy_version,
        "policy_hash": universe.policy_hash,
        "split_policy_hash": identity["split_policy_hash"],
        "candidate_universe_hash": universe.candidate_universe_hash(),
        "universe_hash": universe.universe_hash(),
        "recipe_id": recipe.id, "recipe_hash": recipe.recipe_hash,
        "arms": report_rows, "parity": parity,
        "note": "arm-specific E#### protocols are verified separately via verify_arm_protocols(); "
                "the D#### campaign blocks record recipe/split/universe/parity provenance.",
    }
