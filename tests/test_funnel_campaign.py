import json

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
    # ANY ineligible candidate is refused (not only selected ones): holdouts must never
    # enter the candidate-universe hash
    ineligible = [dict(record, train_eligible=False) if record["record_id"] == candidates[-1]["record_id"]
                  else record for record in candidates]
    with pytest.raises(LabError, match="train-eligible records"):
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
    full = {width: effects[1] for width in (1, 4, 16, 32)}
    assert decide(full) == "SUPPORTED"  # the full preregistered width set

    indecisive = {"ci_high": 0.001, "ci_low": -0.001, "width_estimate": 0.0, "seeds": [0], "differences": [0.0]}
    assert decide({1: indecisive}, required_widths=(1,)) == "INCONCLUSIVE"
    losing = {"ci_low": 0.001, "ci_high": 0.002, "width_estimate": 0.0015, "seeds": [0], "differences": [0.0015]}
    assert decide({1: losing}, required_widths=(1,)) == "REJECTED"


def test_preregistration_is_declared():
    assert set(PREREGISTRATION) >= {"primary_estimate", "uncertainty", "interaction", "decision", "scope"}
    assert "every width" in PREREGISTRATION["decision"]
    assert "not sufficient" in PREREGISTRATION["decision"]


def test_decide_requires_the_full_preregistered_width_set():
    effects = {width: {"ci_high": -0.001, "ci_low": -0.002, "width_estimate": -0.0015, "seeds": list(range(5)),
                       "differences": [-0.0015] * 5} for width in (1, 4, 16, 32)}
    assert decide(effects) == "SUPPORTED"
    with pytest.raises(LabError, match="missing \[4, 16, 32\]"):
        decide({1: effects[1]})  # a win at H=1 alone is not evidence


def test_paired_effect_requires_every_preregistered_seed():
    complete = {seed: 0.01 for seed in range(5)}
    assert paired_effect(complete, {seed: 0.011 for seed in range(5)})["n"] == 5
    with pytest.raises(LabError, match="missing \[4\]"):
        paired_effect({seed: 0.01 for seed in range(4)}, {seed: 0.011 for seed in range(4)})


def test_interaction_uses_real_seed_level_contrasts():
    # H1: priority better by a constant; H4: the advantage shrinks per seed
    priority_h1 = {seed: 0.010 for seed in range(5)}
    uniform_h1 = {seed: 0.012 for seed in range(5)}          # d_H1 = -0.002 every seed
    priority_h4 = {seed: 0.010 for seed in range(5)}
    uniform_h4 = {seed: 0.011 + seed * 0.0001 for seed in range(5)}  # d_H4 varies by seed
    effects = {1: paired_effect(priority_h1, uniform_h1), 4: paired_effect(priority_h4, uniform_h4)}
    interaction = interaction_effects(effects)
    per_seed = interaction["pairs"]["H1->H4"]["per_seed"]
    expected = {seed: effects[4]["differences"][seed] - effects[1]["differences"][seed] for seed in range(5)}
    assert per_seed == expected
    assert len(set(per_seed.values())) > 1  # genuinely seed-level, not one aggregate contrast


def test_compute_parity_requires_an_explicit_tolerance():
    import inspect
    signature = inspect.signature(compute_parity)
    assert signature.parameters["tolerance"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        compute_parity(make_universe())  # type: ignore[call-arg]


def test_eval_semantics_hash_excludes_dataset_binding(lab, tmp_path):
    from cvslab.funnel.campaign import eval_semantics_hash
    source = lab.create_fixture_source(n_games=6, seed=1)
    normalization = lab.normalize([source.id], name="n")
    dataset_a = lab.freeze_dataset(normalization.id, name="a")
    dataset_b = lab.freeze_dataset(normalization.id, name="b")
    protocol_a = lab.create_eval_protocol(name="ea", dataset_id=dataset_a.id)
    protocol_b = lab.create_eval_protocol(name="eb", dataset_id=dataset_b.id)
    assert eval_semantics_hash(protocol_a) == eval_semantics_hash(protocol_b)  # same semantics
    assert protocol_a.id != protocol_b.id and protocol_a.dataset_id != protocol_b.dataset_id
    differing = lab.create_eval_protocol(name="ec", dataset_id=dataset_a.id, bootstrap_seed=7)
    assert eval_semantics_hash(differing) != eval_semantics_hash(protocol_a)


def test_expand_propagates_real_identities_and_requires_per_arm_protocols():
    plan = CampaignPlan()
    runs = plan.expand(split_policy_hash=SPLIT_HASH, recipe_id="T0001", recipe_hash="sha256:" + "e" * 64,
                       eval_semantics_hash="sha256:" + "f" * 64,
                       eval_ids_by_arm={"PRIORITY": "E0001", "UNIFORM": "E0002"})
    assert all(run.recipe_id == "T0001" for run in runs)
    assert {run.eval_protocol_id for run in runs if run.arm == "PRIORITY"} == {"E0001"}
    assert {run.eval_protocol_id for run in runs if run.arm == "UNIFORM"} == {"E0002"}
    assert len({run.controls_hash for run in runs}) == 1
    with pytest.raises(LabError, match="missing \\['UNIFORM'\\]"):
        plan.expand(split_policy_hash=SPLIT_HASH, eval_ids_by_arm={"PRIORITY": "E0001"})
    planner_only = CampaignPlan().expand(split_policy_hash=SPLIT_HASH)
    bound = plan.expand(split_policy_hash=SPLIT_HASH, recipe_id="T0001", recipe_hash="sha256:" + "e" * 64)
    assert planner_only[0].controls_hash != bound[0].controls_hash  # real identities change the hash


def test_freeze_arm_pair_records_full_provenance_and_excludes_non_members(lab):
    from cvslab.funnel.campaign import freeze_arm_datasets
    from cvslab.hashing import write_jsonl
    from cvslab.schemas import SourceSnapshot, utc_now

    # a small canonical corpus with explicit game grouping
    import chess
    rows = []
    for game in range(4):
        board = chess.Board()
        for ply, move in enumerate(("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"), start=1):
            board.push_uci(move)
        rows.append({"fen": board.fen(), "game": f"g{game}", "ply": 6, "res": 1.0})
    source_id = lab.store.next_id("S")
    rel = f"sources/{source_id}/positions.jsonl"
    write_jsonl(lab.store.abs(rel), rows)
    lab.store.make_readonly(lab.store.abs(rel))
    from cvslab.hashing import sha256_file
    lab.store.create(SourceSnapshot(id=source_id, name="corpus", locality="local", source_origin="test",
                                    generator={"name": "t"}, license="t", importer="t", importer_version=1,
                                    path=rel, content_hash=sha256_file(lab.store.abs(rel)), row_count=len(rows),
                                    game_count=4, label_authorities=[], compute={}, created_at=utc_now()))
    normalization = lab.normalize([source_id], name="n")
    records = [json.loads(line) for line in
               lab.store.abs(normalization.path).read_text(encoding="utf-8").splitlines()]
    # only ONE record survives (all four games transpose to the same position) -> use two distinct positions
    assert len(records) == 1
    record_id = records[0]["record_id"]
    # the arms must be row-matched and deep-labeled; duplicate the corpus to get two distinct positions
    from cvslab.data import _load_canonical
    subset = _load_canonical(lab.store, lab.store.get(normalization.id))
    # the arms are frozen from deep-labeled evidence: register the Tier-3 labels the
    # real S4 run would have produced (this fixture corpus has none of its own)
    lab.register_labels(normalization.id, family="search_deep_cp", producer="legacy.cvs.search.deep",
                        authority="legacy.cvs.search.deep", pov="stm",
                        rows=[{"record_id": record["record_id"],
                               "value": {"targets": {}, "deep": {}, "shallowToDeep": {}},
                               "budget": {"nodeBudget": 400000}} for record in subset])
    universe = freeze_universe(
        source_id=source_id, normalization_id=normalization.id, funnel_run_id="R0001",
        policy_version="priority-v1", policy_hash="sha256:" + "a" * 64,
        candidates=[{"record_id": r["record_id"], "group": r["group"], "train_eligible": True} for r in subset],
        deep_nodes={r["record_id"]: 400_000 for r in subset},
        selections={"deep": [record_id], "uniform": [record_id]}, split_seed=0)
    recipe = lab.create_training_recipe(name="s6-recipe", params={"EPOCHS": 2})
    report = freeze_arm_datasets(lab.store, universe, CampaignPlan(), recipe=recipe,
                                 deep_engine_seconds={"PRIORITY": 12.5, "UNIFORM": 12.4})
    for key in ("source_id", "normalization_id", "funnel_run_id", "policy_hash", "split_policy_hash",
                "candidate_universe_hash", "universe_hash", "recipe_id", "recipe_hash", "parity"):
        assert key in report
    for arm in ("PRIORITY", "UNIFORM"):
        entry = report["arms"][arm]
        assert entry["rows"] == 1 and entry["manifest_hash"].startswith("sha256:")
        dataset = lab.store.get(entry["dataset_id"])
        assert dataset.campaign["arm"] == arm
        assert dataset.campaign["membership_source"] in ("selection.deep", "selection.uniform")
        assert dataset.campaign["deep_nodes"] == 400_000
        assert dataset.campaign["universe_hash"] == universe.universe_hash()
    assert report["parity"]["relative_gap"] == 0.0
