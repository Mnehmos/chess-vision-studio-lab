import pytest

from cvslab.funnel.campaign import (
    ARM_NAMES,
    DEFAULT_SEEDS,
    EXPECTED_PARAMS,
    WIDTHS,
    CampaignPlan,
    assert_shared_controls,
    compute_parity,
    freeze_universe,
    select_arms,
    split_identity,
)
from cvslab.store import LabError


def eligible_record(record_id: str, component: str, nodes: int = 400_000) -> dict:
    return {"record_id": record_id, "group": component, "deep_nodes": nodes}


def make_universe(policy_hash: str = "sha256:" + "a" * 64, groups: int = 10, per_group: int = 4):
    records = []
    for group_index in range(groups):
        for index in range(per_group):
            records.append(eligible_record(f"pos_{group_index:02d}{index}", f"S0001:g{group_index:04d}"))
    return freeze_universe(source_id="S0001", normalization_id="N0001", funnel_run_id="R0001",
                           policy_version="priority-v1", policy_hash=policy_hash,
                           eligible=records, split_seed=0)


# ---------------------------------------------------------------------------
# The handoff's precondition: matrix expands to exactly 40 unique planned cells
# ---------------------------------------------------------------------------


def test_plan_expands_to_exactly_40_unique_cells():
    plan = CampaignPlan()
    runs = plan.expand()
    assert len(runs) == 40 == plan.expected_total()
    assert len({run.cell_id for run in runs}) == 40
    cells = {(run.width, run.arm, run.seed) for run in runs}
    assert cells == {(w, a, s) for w in WIDTHS for a in ARM_NAMES for s in DEFAULT_SEEDS}
    for run in runs:
        assert run.param_count == EXPECTED_PARAMS[run.width]
    assert_shared_controls(runs, plan)


def test_exact_parameter_counts_are_the_formula_values():
    assert EXPECTED_PARAMS == {1: 771, 4: 3081, 16: 12321, 32: 24641}
    for width, expected in EXPECTED_PARAMS.items():
        assert 768 * width + 2 * width + 1 == expected


def test_arm_cannot_silently_change_controls_or_plan_identity():
    base = CampaignPlan()
    assert base.plan_hash() == CampaignPlan().plan_hash()  # stable
    tweaked = CampaignPlan()
    tweaked.recipe_params = {**tweaked.recipe_params, "LR": 0.001}
    assert tweaked.plan_hash() != base.plan_hash()
    suite = CampaignPlan()
    suite.eval_suite = "other-suite"
    assert suite.plan_hash() != base.plan_hash()
    warmed = CampaignPlan()
    warmed.init_policy = "warm-start-legacy"
    assert warmed.plan_hash() != base.plan_hash()

    # a plan whose arms disagree on cells is refused
    runs = base.expand()
    broken = [run for run in runs if not (run.arm == "UNIFORM" and run.seed == 4)]
    with pytest.raises(LabError, match="do not share the same cell set"):
        assert_shared_controls(broken, base)


# ---------------------------------------------------------------------------
# Frozen universe + matched arms
# ---------------------------------------------------------------------------


def test_universe_freezes_one_split_assignment_over_components():
    universe = make_universe()
    assert len(universe.eligible_ids) == 40
    assert set(universe.split_by_component) == {f"S0001:g{index:04d}" for index in range(10)}
    assert universe.universe_hash().startswith("sha256:")
    # all records of one component share one split (transposition-safe unit)
    for record_id, component in universe.eligible_components.items():
        assert universe.split_by_component[component] in ("train", "val", "test")


def test_universe_refuses_unsafe_or_empty_inputs():
    with pytest.raises(LabError, match="unknown grouping"):
        freeze_universe(source_id="S0001", normalization_id="N0001", funnel_run_id="R0001",
                        policy_version="priority-v1", policy_hash="sha256:" + "a" * 64,
                        eligible=[eligible_record("pos_1", "S0001:UNKNOWN")], split_seed=0)
    with pytest.raises(LabError, match="no measured deep-label nodes"):
        freeze_universe(source_id="S0001", normalization_id="N0001", funnel_run_id="R0001",
                        policy_version="priority-v1", policy_hash="sha256:" + "a" * 64,
                        eligible=[eligible_record("pos_1", "S0001:g0001", nodes=0)], split_seed=0)
    with pytest.raises(LabError, match="duplicate record identities"):
        freeze_universe(source_id="S0001", normalization_id="N0001", funnel_run_id="R0001",
                        policy_version="priority-v1", policy_hash="sha256:" + "a" * 64,
                        eligible=[eligible_record("pos_1", "S0001:g0001"),
                                  eligible_record("pos_1", "S0001:g0002")], split_seed=0)
    with pytest.raises(LabError, match="empty"):
        freeze_universe(source_id="S0001", normalization_id="N0001", funnel_run_id="R0001",
                        policy_version="priority-v1", policy_hash="sha256:" + "a" * 64,
                        eligible=[], split_seed=0)


def test_arms_are_row_matched_and_seed_deterministic():
    universe = make_universe()
    deep_ids = list(universe.eligible_ids[:12])
    first = select_arms(universe, deep_selected_ids=deep_ids, selection_seed=7, target_rows=12)
    second = select_arms(universe, deep_selected_ids=deep_ids, selection_seed=7, target_rows=12)
    assert first["PRIORITY"] == sorted(deep_ids)
    assert len(first["UNIFORM"]) == len(first["PRIORITY"]) == 12
    assert first == second  # seed-derived, reproducible
    assert set(first["UNIFORM"]) <= set(universe.eligible_ids)  # same universe, no resampling
    assert select_arms(universe, deep_selected_ids=deep_ids, selection_seed=8,
                       target_rows=12)["UNIFORM"] != first["UNIFORM"]

    with pytest.raises(LabError, match="outside the eligible universe"):
        select_arms(universe, deep_selected_ids=["pos_nope"], selection_seed=0, target_rows=1)
    with pytest.raises(LabError, match="requires 12"):
        select_arms(universe, deep_selected_ids=deep_ids[:5], selection_seed=0, target_rows=12)


def test_compute_parity_uses_measured_nodes_and_refuses_outside_tolerance():
    universe = make_universe()
    deep_ids = list(universe.eligible_ids[:10])
    arms = select_arms(universe, deep_selected_ids=deep_ids, selection_seed=3, target_rows=10)
    parity = compute_parity(universe, arms, tolerance=1.0)
    assert parity["deep_nodes"]["PRIORITY"] == parity["deep_nodes"]["UNIFORM"]  # identical budgets per record
    # now make the gap real: UNIFORM's records carry half the measured nodes
    skewed = {record_id: (nodes if record_id in set(arms["PRIORITY"]) else nodes // 2)
              for record_id, nodes in universe.deep_nodes_by_id.items()}
    from dataclasses import replace
    skewed_universe = replace(universe, deep_nodes_by_id=skewed)
    with pytest.raises(LabError, match="compute parity outside tolerance"):
        compute_parity(skewed_universe, arms, tolerance=0.02)


def test_split_identity_is_shared_across_arms():
    universe = make_universe()
    deep_ids = list(universe.eligible_ids[:16])
    arms = select_arms(universe, deep_selected_ids=deep_ids, selection_seed=1, target_rows=16)
    identity = split_identity(universe, arms)
    assert identity["split_assignment_hash"].startswith("sha256:")
    assert identity["shared_records"] == len(set(arms["PRIORITY"]) | set(arms["UNIFORM"]))
    assert identity["per_arm_split_hashes"]["PRIORITY"].startswith("sha256:")
