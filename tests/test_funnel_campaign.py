import pytest

from cvslab.funnel.campaign import (
    ARM_NAMES,
    COMPUTE_PARITY_TOLERANCE,
    DEFAULT_SEEDS,
    EXPECTED_PARAMS,
    PREREGISTRATION,
    WIDTHS,
    CampaignPlan,
    assert_shared_controls,
    compute_parity,
    decide,
    freeze_universe,
    interaction_effects,
    paired_effect,
    split_identity,
)
from cvslab.store import LabError

POLICY_HASH = "sha256:" + "a" * 64
SPLIT_HASH = "sha256:" + "b" * 64


def make_inputs(groups: int = 12, per_group: int = 3, deep_groups: int = 6):
    """Candidates (full triage pool) + measured deep nodes for a subset + recorded arms."""
    candidates, deep_nodes = [], {}
    for group_index in range(groups):
        for index in range(per_group):
            record_id = f"pos_{group_index:02d}_{index}"
            candidates.append({"record_id": record_id, "group": f"S0001:g{group_index:04d}",
                               "train_eligible": True})
    for group_index in range(deep_groups):
        for index in range(per_group):
            deep_nodes[f"pos_{group_index:02d}_{index}"] = 400_000
    ordered = sorted(deep_nodes)
    return candidates, deep_nodes, {"deep": ordered[:6], "uniform": ordered[6:12]}


def make_universe(**overrides):
    candidates, deep_nodes, selections = make_inputs()
    params = dict(source_id="S0001", normalization_id="N0001", funnel_run_id="R0001",
                  policy_version="priority-v1", policy_hash=POLICY_HASH,
                  candidates=candidates, deep_nodes=deep_nodes, selections=selections, split_seed=0)
    params.update(overrides)
    return freeze_universe(**params)


# --- plan -------------------------------------------------------------------


def test_plan_expands_to_40_unique_cells_with_shared_controls():
    plan = CampaignPlan()
    runs = plan.expand(split_policy_hash=SPLIT_HASH)
    assert len(runs) == 40 == plan.expected_total()
    assert len({run.cell_id for run in runs}) == 40
    assert {(run.width, run.arm, run.seed) for run in runs} == {
        (w, a, s) for w in WIDTHS for a in ARM_NAMES for s in DEFAULT_SEEDS}
    assert all(run.param_count == EXPECTED_PARAMS[run.width] for run in runs)
    assert {run.controls_hash for run in runs} == {plan.controls_hash(SPLIT_HASH)}
    assert_shared_controls(runs, plan)


def test_paired_cells_cannot_diverge_and_controls_hash_covers_controls():
    plan = CampaignPlan()
    base_controls = plan.controls_hash(SPLIT_HASH)
    for change in ({"recipe_params": {**plan.recipe_params, "LR": 0.001}},
                   {"eval_suite": "other-suite"},
                   {"target_semantics": "different target"},
                   {"init_policy": "warm-start-legacy"}):
        tweaked = CampaignPlan(**{**{}, **change})
        assert tweaked.controls_hash(SPLIT_HASH) != base_controls
        assert tweaked.plan_hash() != plan.plan_hash()
    different_split = CampaignPlan().controls_hash("sha256:" + "c" * 64)
    assert different_split != base_controls

    # paired cells built from a divergent controls identity are refused
    from dataclasses import replace
    runs = plan.expand(split_policy_hash=SPLIT_HASH)
    broken = [replace(run, controls_hash="sha256:" + "d" * 64) if run.cell_id == "H16-UNIFORM-s0" else run
              for run in runs]
    with pytest.raises(LabError, match="cells disagree on controls identity"):
        assert_shared_controls(broken, plan)


def test_compute_parity_tolerance_is_part_of_campaign_identity():
    base, strict = CampaignPlan(), CampaignPlan(compute_parity_tolerance=0.01)
    assert base.plan_hash() != strict.plan_hash()
    assert base.identity()["compute_parity_tolerance"] == COMPUTE_PARITY_TOLERANCE


# --- universe: recorded arms, no resampling ---------------------------------


def test_universe_uses_recorded_arms_and_keeps_candidates_separate():
    universe = make_universe()
    assert len(universe.candidate_ids) == 36          # full triage pool
    assert len(universe.deep_nodes_by_id) == 18       # only what actually got deep labels
    assert len(universe.arms["PRIORITY"]) == len(universe.arms["UNIFORM"]) == 6
    assert set(universe.arms["PRIORITY"]) <= set(universe.candidate_ids)
    assert universe.split_policy_hash().startswith("sha256:")
    assert universe.candidate_universe_hash().startswith("sha256:")


def test_universe_refuses_changed_population_and_missing_evidence():
    candidates, deep_nodes, selections = make_inputs()
    with pytest.raises(LabError, match="outside the candidate universe"):
        freeze_universe(source_id="S0001", normalization_id="N0001", funnel_run_id="R0001",
                        policy_version="priority-v1", policy_hash=POLICY_HASH,
                        candidates=candidates, deep_nodes=deep_nodes,
                        selections={"deep": ["pos_nope"], "uniform": sorted(deep_nodes)[:1]}, split_seed=0)
    with pytest.raises(LabError, match="without measured Tier-3 deep labels"):
        freeze_universe(source_id="S0001", normalization_id="N0001", funnel_run_id="R0001",
                        policy_version="priority-v1", policy_hash=POLICY_HASH,
                        candidates=candidates, deep_nodes={},
                        selections=selections, split_seed=0)
    with pytest.raises(LabError, match="not row-matched"):
        freeze_universe(source_id="S0001", normalization_id="N0001", funnel_run_id="R0001",
                        policy_version="priority-v1", policy_hash=POLICY_HASH,
                        candidates=candidates, deep_nodes=deep_nodes,
                        selections={"deep": sorted(deep_nodes)[:6], "uniform": sorted(deep_nodes)[6:8]},
                        split_seed=0)
    with pytest.raises(LabError, match="non-train-eligible"):
        ineligible = [dict(record, train_eligible=False) if record["record_id"] == selections["deep"][0]
                      else record for record in candidates]
        freeze_universe(source_id="S0001", normalization_id="N0001", funnel_run_id="R0001",
                        policy_version="priority-v1", policy_hash=POLICY_HASH,
                        candidates=ineligible, deep_nodes=deep_nodes, selections=selections, split_seed=0)


def test_universe_hash_detects_node_redistribution():
    """Moving measured nodes between records (same total) must change the universe identity."""
    universe = make_universe()
    ids = sorted(universe.deep_nodes_by_id)
    skewed = dict(universe.deep_nodes_by_id)
    skewed[ids[0]] += 100_000
    skewed[ids[1]] -= 100_000
    from dataclasses import replace
    tampered = replace(universe, deep_nodes_by_id=skewed)
    assert sum(skewed.values()) == sum(universe.deep_nodes_by_id.values())
    assert tampered.universe_hash() != universe.universe_hash()


def test_compute_parity_from_recorded_arms_and_tolerance_refusal():
    universe = make_universe()
    parity = compute_parity(universe, tolerance=1.0)
    assert parity["deep_nodes"]["PRIORITY"] > 0
    from dataclasses import replace
    skewed = dict(universe.deep_nodes_by_id)
    for record_id in universe.arms["UNIFORM"]:
        skewed[record_id] //= 4
    with pytest.raises(LabError, match="compute parity outside tolerance"):
        compute_parity(replace(universe, deep_nodes_by_id=skewed), tolerance=0.02)


def test_split_identity_shares_one_policy_hash_across_arms():
    universe = make_universe()
    identity = split_identity(universe)
    assert identity["split_policy_hash"] == universe.split_policy_hash()
    assert identity["component_count"] == len(universe.split_by_component)
    assert identity["per_arm_record_split_hashes"]["PRIORITY"] != \
        identity["per_arm_record_split_hashes"]["UNIFORM"]  # rows differ; the policy is shared
    for record_id in universe.candidate_ids:
        assert universe.split_by_component[universe.components[record_id]] in ("train", "val", "test")


# --- pre-registered statistics ----------------------------------------------


def test_paired_effect_and_interaction_are_deterministic():
    priority = {seed: 0.010 - seed * 0.0001 for seed in range(5)}
    uniform = {seed: 0.011 - seed * 0.0001 for seed in range(5)}   # priority better at every seed
    effect = paired_effect(priority, uniform)
    assert effect["n"] == 5 and effect["width_estimate"] < 0
    assert effect["ci_high"] < 0  # consistent paired win
    assert paired_effect(priority, uniform) == effect

    effects = {1: paired_effect({s: 0.010 for s in range(5)}, {s: 0.011 for s in range(5)}),
               4: paired_effect({s: 0.010 for s in range(5)}, {s: 0.012 for s in range(5)})}
    interaction = interaction_effects(effects)
    assert interaction["pairs"]["H1->H4"]["delta"] == pytest.approx(-0.001)
    assert decide(effects) == "SUPPORTED"

    assert decide({1: {"ci_high": 0.001, "ci_low": -0.001, "width_estimate": 0.0, "seeds": [0]}}) == "INCONCLUSIVE"
    assert decide({1: {"ci_low": 0.001, "ci_high": 0.002, "width_estimate": 0.0015, "seeds": [0]}}) == "REJECTED"


def test_preregistration_is_declared():
    assert set(PREREGISTRATION) >= {"primary_estimate", "uncertainty", "interaction", "decision", "scope"}
    assert "every width" in PREREGISTRATION["decision"]
    assert "not sufficient" in PREREGISTRATION["decision"]
