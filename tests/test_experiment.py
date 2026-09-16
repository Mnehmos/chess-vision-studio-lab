"""X#### acceptance tests — the experiment contract, not the prefix list.

Required cases (PR #33 review): arm identity, lattice shape, per-arm widths, frozen seeds,
the X<->A back-reference, non-arm queueing, corrupted contracts, differing exams, membership
completeness at analysis and at seal, a complete miniature matrix end to end, a SEALED study
refusing more runs, and historical identities being unaffected.

The miniature matrix keeps the shape of the 320-cell study (16 arms x 4 widths x 5 seeds):
two scales x one supervision depth = two arms, four widths each, two seeds => 8 cells.
"""
import json
import os
import stat

import pytest

from cvslab.schemas import Run
from cvslab.service import protocol_target_spec, recipe_target_spec
from cvslab.store import ImmutableError, LabError, payload_hash
from cvslab.targets import TargetSpec

SPEC_16K = TargetSpec(family="search_shallow_cp", authority="legacy.cvs.search.shallow",
                      producer="a" * 64, budget={"nodeBudget": 16000},
                      value_path=("scoreCpStm",), pov="stm", k=256.0, lam=1.0)
SPEC_400K = TargetSpec(family="search_deep_cp", authority="legacy.cvs.search.deep",
                       producer="a" * 64, budget={"nodeBudget": 400000},
                       value_path=("targets", "scoreCpStm"), pov="stm", k=256.0, lam=1.0)
WIDTHS = (1, 4, 16, 32)
SEEDS = (0, 1)
SCALES = {"2x": 74_570_982, "5x": 186_427_455}
DECLARATION = "miniature X#### matrix: trained on 16k, examined on the frozen 400k instrument"


def build_lab(lab):
    from cvslab.data import _load_canonical
    source = lab.create_fixture_source(n_games=6, seed=41)
    normalization = lab.normalize([source.id], name="n")
    evaluated = sorted(r["record_id"] for r in _load_canonical(lab.store, lab.store.get(normalization.id)))
    lab.register_labels(normalization.id, family="search_shallow_cp", producer="a" * 64,
                        authority="legacy.cvs.search.shallow", pov="stm",
                        rows=[{"record_id": r, "value": {"scoreCpStm": 30 + i, "bestMove": "e2e4", "nodes": 16000},
                               "budget": {"nodeBudget": 16000, "nodes": 16000}} for i, r in enumerate(evaluated)])
    lab.register_labels(normalization.id, family="search_deep_cp", producer="a" * 64,
                        authority="legacy.cvs.search.deep", pov="stm",
                        rows=[{"record_id": r, "value": {"targets": {"scoreCpStm": -80 + 10 * i}},
                               "budget": {"nodeBudget": 400000, "nodes": 400000}} for i, r in enumerate(evaluated)])
    train = lab.freeze_dataset(normalization.id, name="d-train", required_labels=["search_shallow_cp"],
                               target_spec=SPEC_16K, fractions=(1.0, 0.0, 0.0))
    exam = lab.freeze_dataset(normalization.id, name="d-exam", required_labels=["search_deep_cp"],
                              target_spec=SPEC_400K, fractions=(1.0, 0.0, 0.0))
    recipe = lab.create_training_recipe(name="t-16k", params={"K": 256.0, "LAMBDA": 1.0, "EPOCHS": 1},
                                        target_spec=SPEC_16K)
    protocol = lab.create_eval_protocol(name="e-400k", dataset_id=exam.id, split="train", k=256.0, lam=1.0,
                                        target_spec=SPEC_400K)
    return normalization, train, exam, recipe, protocol


def arm_material(lab, normalization, recipe, protocol, xid, *, scale, budget=16000, suffix="A",
                 widths=WIDTHS):
    """One lattice cell: its own D####, one baseline, and one ablation per declared width."""
    dataset = lab.freeze_dataset(normalization.id, name=f"d-arm-{suffix}", required_labels=["search_shallow_cp"],
                                 target_spec=SPEC_16K, fractions=(1.0, 0.0, 0.0),
                                 campaign={"arm": f"{scale}-D{budget}", "target_spec": SPEC_16K.spec_hash()})
    baseline = lab.register_baseline(name=f"ARM_{suffix}", dataset_id=dataset.id, training_recipe_id=recipe.id,
                                     eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 16},
                                     supervision_divergence=DECLARATION, experiment_id=xid)
    ablations = [baseline.id] + [lab.create_ablation(baseline_id=baseline.id, overrides={"H": width},
                                                     supervision_divergence=DECLARATION,
                                                     experiment_id=xid).id for width in widths if width != 16]
    split = next(entry for entry in dataset.splits if entry.name == "train")
    arm = {"arm_id": f"{scale}-D{budget}", "scale": scale, "scale_nodes": SCALES.get(scale, 1),
           "node_budget": budget, "prefix_size": split.count, "realized_nodes": budget * split.count,
           "dataset_id": dataset.id, "dataset_manifest_hash": dataset.manifest_hash,
           "training_recipe_id": recipe.id, "recipe_hash": recipe.recipe_hash,
           "train_target_spec_hash": recipe_target_spec(recipe).spec_hash(),
           "record_ids_hash": split.record_ids_hash, "ablations": ablations}
    return arm


def make_experiment(lab, arms, protocol, xid, *, seeds=SEEDS, widths=WIDTHS, name="S-mini"):
    return lab.create_experiment(
        name=name, preregistration_hash="sha256:" + "p" * 64, reference_unit_nodes=37_285_491,
        scales=dict(SCALES), node_budgets=[16000], arms=arms, eval_protocol_id=protocol.id,
        widths=list(widths), seeds=list(seeds), source_id="S0001", normalization_id="N0001",
        candidate_universe_hash="sha256:" + "u" * 64, order_seed=20260920, order_hash="sha256:" + "o" * 64,
        analysis={"primary": "A4-A1 per scale"}, experiment_id=xid, notes="miniature matrix")


def miniature(lab, *, widths=WIDTHS):
    """A complete two-cell lattice ready to queue."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    xid = lab.store.next_id("X")
    arms = [arm_material(lab, normalization, recipe, protocol, xid, scale="2x", suffix="A", widths=widths),
            arm_material(lab, normalization, recipe, protocol, xid, scale="5x", suffix="B", widths=widths)]
    experiment = make_experiment(lab, arms, protocol, xid, widths=widths)
    return normalization, train, exam, recipe, protocol, xid, arms, experiment


def _rewrite_run(lab, run_id, **changes):
    path = lab.store.object_path(run_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    payload["record_hash"] = payload_hash({k: v for k, v in payload.items() if k != "record_hash"})
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
        handle.write("\n")


def test_arm_rows_and_record_hash_are_proved_against_the_dataset(lab):
    """1. the arm's declared rows and record ids must match its frozen D####."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    xid = lab.store.next_id("X")
    good = arm_material(lab, normalization, recipe, protocol, xid, scale="2x", suffix="A")
    other = arm_material(lab, normalization, recipe, protocol, xid, scale="5x", suffix="B")

    with pytest.raises(LabError, match="prefix_size"):
        make_experiment(lab, [dict(good, prefix_size=good["prefix_size"] + 1), other], protocol, xid)
    with pytest.raises(LabError, match="record_ids_hash"):
        make_experiment(lab, [dict(good, record_ids_hash="sha256:" + "0" * 64), other], protocol, xid)


def test_the_design_must_be_exactly_the_declared_lattice(lab):
    """2. every declared scale x depth cell, exactly once, and nothing else."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    xid = lab.store.next_id("X")
    first = arm_material(lab, normalization, recipe, protocol, xid, scale="2x", suffix="A")
    with pytest.raises(LabError, match="exactly the declared lattice"):
        make_experiment(lab, [first], protocol, xid)          # a cell is missing
    second = arm_material(lab, normalization, recipe, protocol, xid, scale="5x", suffix="B")
    # an undeclared scale is refused (and refused cleanly, not with a KeyError)
    with pytest.raises(LabError, match="does not declare"):
        make_experiment(lab, [first, dict(second, scale="9x")], protocol, xid)
    # arms may PARTITION one cell: two arms of the same (scale, depth) split the width ladder.
    # The cell is then incomplete, so the coverage rule refuses it.
    half_a = arm_material(lab, normalization, recipe, protocol, xid, scale="2x", suffix="P", widths=(1, 4))
    half_a = dict(half_a, arm_id="2x-D16000-partition")     # a distinct id for the same cell
    # alone in its cell it presents only half the ladder, so the cell is incomplete and refused
    with pytest.raises(LabError, match="presents widths"):
        make_experiment(lab, [half_a, second], protocol, xid)


def test_the_scale_name_must_carry_its_numeric_target(lab):
    """2b. a cell called "20x" may not carry the 2x budget."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    xid = lab.store.next_id("X")
    first = arm_material(lab, normalization, recipe, protocol, xid, scale="2x", suffix="A")
    partner = arm_material(lab, normalization, recipe, protocol, xid, scale="5x", suffix="B")
    with pytest.raises(LabError, match="declares scale_nodes"):
        make_experiment(lab, [dict(first, scale_nodes=SCALES["2x"] * 10), partner], protocol, xid)


def test_the_design_must_present_the_declared_width_ladder(lab):
    """3. no declared width may vanish from the design.

    One arm may carry all four widths (S7-S/S9/S10) or the widths may be distributed across arms
    of the same supervision depth (S11's targeted design, one width per arm). What is forbidden is
    a design in which a declared width is missing at that depth.
    """
    normalization, train, exam, recipe, protocol = build_lab(lab)
    xid = lab.store.next_id("X")
    short = arm_material(lab, normalization, recipe, protocol, xid, scale="2x", suffix="A",
                         widths=(1, 16, 32))                       # H4 missing at this depth
    partner = arm_material(lab, normalization, recipe, protocol, xid, scale="5x", suffix="B",
                           widths=(1, 16, 32))                     # and here too
    with pytest.raises(LabError, match="presents widths"):
        make_experiment(lab, [short, partner], protocol, xid, widths=(1, 4, 16, 32))


def test_seeds_outside_the_frozen_list_are_refused(lab):
    """4. an experiment's cells are its design: no extra seeds."""
    normalization, train, exam, recipe, protocol, xid, arms, experiment = miniature(lab)
    with pytest.raises(LabError, match="outside .*frozen seed list"):
        lab.queue_runs(arms[0]["ablations"][0], seeds=(0, 7))
    assert experiment.id


def test_ablation_back_reference_is_required(lab):
    """5. an ablation that does not carry this experiment's id cannot be frozen into it."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    xid = lab.store.next_id("X")
    dataset = lab.freeze_dataset(normalization.id, name="d-unbound", required_labels=["search_shallow_cp"],
                                 target_spec=SPEC_16K, fractions=(1.0, 0.0, 0.0))
    baseline = lab.register_baseline(name="UNBOUND", dataset_id=dataset.id, training_recipe_id=recipe.id,
                                     eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 16},
                                     supervision_divergence=DECLARATION)      # no experiment_id
    unbound = [baseline.id] + [lab.create_ablation(baseline_id=baseline.id, overrides={"H": width},
                                                   supervision_divergence=DECLARATION).id
                               for width in WIDTHS if width != 16]
    split = next(entry for entry in dataset.splits if entry.name == "train")
    arm = {"arm_id": "2x-D16k", "scale": "2x", "scale_nodes": SCALES["2x"], "node_budget": 16000,
           "prefix_size": split.count, "realized_nodes": 1, "dataset_id": dataset.id,
           "dataset_manifest_hash": dataset.manifest_hash, "training_recipe_id": recipe.id,
           "recipe_hash": recipe.recipe_hash,
           "train_target_spec_hash": recipe_target_spec(recipe).spec_hash(),
           "record_ids_hash": split.record_ids_hash, "ablations": unbound}
    partner = arm_material(lab, normalization, recipe, protocol, xid, scale="5x", suffix="P")
    with pytest.raises(LabError, match="not bound to this experiment"):
        make_experiment(lab, [arm, partner], protocol, xid)


def test_queueing_a_non_arm_is_refused(lab):
    """6. a run cannot join an experiment it was not frozen into."""
    normalization, train, exam, recipe, protocol, xid, arms, experiment = miniature(lab)
    outsider = lab.register_baseline(name="OUTSIDER", dataset_id=arms[0]["dataset_id"],
                                     training_recipe_id=recipe.id, eval_protocol_id=protocol.id,
                                     model_config={"INPUT": "RAW", "H": 4}, supervision_divergence=DECLARATION,
                                     experiment_id=experiment.id)     # claims the study, not listed in it
    with pytest.raises(LabError, match="not an arm of"):
        lab.queue_runs(outsider.id, seeds=(0,))
    plain = lab.register_baseline(name="PLAIN", dataset_id=arms[0]["dataset_id"],
                                  training_recipe_id=recipe.id, eval_protocol_id=protocol.id,
                                  model_config={"INPUT": "RAW", "H": 4}, supervision_divergence=DECLARATION)
    [run] = lab.queue_runs(plain.id, seeds=(0,))
    assert run.experiment_id is None


def test_execution_detects_a_corrupted_contract(lab):
    """7. a run whose recorded arm identity was changed fails closed before training."""
    normalization, train, exam, recipe, protocol, xid, arms, experiment = miniature(lab)
    [run] = lab.queue_runs(arms[0]["ablations"][0], seeds=(0,))
    _rewrite_run(lab, run.id, recipe_hash="sha256:" + "9" * 64)
    finished = lab.execute_run(run.id)
    assert finished.status.value == "INVALID"
    assert finished.model_id is None
    assert "recipe_hash" in {c.name: c for c in finished.integrity}["experiment_contract"].detail


def test_a_different_exam_cannot_be_compared(lab):
    """8. one exam per study; a run recording another protocol is refused."""
    normalization, train, exam, recipe, protocol, xid, arms, experiment = miniature(lab)
    second_exam = lab.freeze_dataset(normalization.id, name="d-exam-16k", required_labels=["search_shallow_cp"],
                                     target_spec=SPEC_16K, fractions=(1.0, 0.0, 0.0))
    other_protocol = lab.create_eval_protocol(name="e-16k", dataset_id=second_exam.id, split="train",
                                              k=256.0, lam=1.0, target_spec=SPEC_16K)
    [run] = lab.queue_runs(arms[0]["ablations"][0], seeds=(0,))
    _rewrite_run(lab, run.id, eval_protocol_id=other_protocol.id, protocol_hash=other_protocol.protocol_hash)
    finished = lab.execute_run(run.id)
    assert finished.status.value == "INVALID"
    assert "evaluation protocol" in {c.name: c for c in finished.integrity}["experiment_contract"].detail


def test_incomplete_membership_is_refused(lab):
    """9. missing, duplicate and partial membership block analysis and sealing."""
    normalization, train, exam, recipe, protocol, xid, arms, experiment = miniature(lab)
    with pytest.raises(LabError, match="have no run"):
        lab.assert_experiment_complete(experiment)
    with pytest.raises(LabError, match="have no run"):
        lab.seal_experiment(experiment.id, {"decision": "early"})

    for arm in arms:
        for ablation_id in arm["ablations"]:
            for run in lab.queue_runs(ablation_id, seeds=SEEDS):
                assert lab.execute_run(run.id).status.value == "COMPLETED"
    expected = 2 * 4 * len(SEEDS)
    report = lab.assert_experiment_complete(experiment)
    assert report["expected_cells"] == expected and report["completed_cells"] == expected

    lab.queue_runs(arms[0]["ablations"][0], seeds=(0,))      # a duplicate cell
    with pytest.raises(LabError, match="more than one run"):
        lab.assert_experiment_complete(experiment)
    with pytest.raises(LabError, match="more than one run"):
        lab.seal_experiment(experiment.id, {"decision": "early"})


def test_complete_miniature_matrix_is_accepted_and_then_sealed(lab):
    """10 + 11. a complete matrix analyses, seals, and a SEALED study refuses more runs."""
    normalization, train, exam, recipe, protocol, xid, arms, experiment = miniature(lab)
    for arm in arms:
        for ablation_id in arm["ablations"]:
            for run in lab.queue_runs(ablation_id, seeds=SEEDS):
                assert lab.execute_run(run.id).status.value == "COMPLETED"
    run_ids = [r.id for r in lab.store.list("R", verify=True, kind=Run) if r.experiment_id == experiment.id]
    assert lab.assert_experiment_runs(run_ids) == experiment.id
    sealed = lab.seal_experiment(experiment.id, {"decision": "miniature"})
    assert sealed.status == "SEALED" and sealed.result["membership"]["expected_cells"] == 2 * 4 * len(SEEDS)
    assert sealed.record_hash
    with pytest.raises(LabError, match="SEALED"):
        lab.queue_runs(arms[0]["ablations"][0], seeds=(9,))
    with pytest.raises(ImmutableError):
        lab.store.update_experiment(sealed.model_copy(update={"name": "renamed"}))


def test_historical_identities_need_no_experiment(lab):
    """12. the X capability is additive: runs without an experiment keep working."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    baseline = lab.register_baseline(name="LEGACY", dataset_id=train.id, training_recipe_id=recipe.id,
                                     eval_protocol_id=protocol.id, model_config={"INPUT": "RAW", "H": 4},
                                     supervision_divergence=DECLARATION)
    [run] = lab.queue_runs(baseline.id, seeds=(0,))
    assert run.experiment_id is None
    finished = lab.execute_run(run.id)
    assert finished.status.value == "COMPLETED"
    assert "experiment_contract" not in {c.name for c in finished.integrity}


def test_an_arm_may_declare_its_own_seed_subset(lab):
    """A precision study spends extra seeds only where uncertainty limits."""
    normalization, train, exam, recipe, protocol = build_lab(lab)
    xid = lab.store.next_id("X")
    first = arm_material(lab, normalization, recipe, protocol, xid, scale="2x", suffix="A")
    second = arm_material(lab, normalization, recipe, protocol, xid, scale="5x", suffix="B")
    decision = {"primary": "2k-16k", "margin": 0.001}
    experiment = lab.create_experiment(
        name="S-precision", preregistration_hash="sha256:" + "p" * 64, reference_unit_nodes=1,
        scales=dict(SCALES), node_budgets=[16000],
        arms=[dict(first, seeds=[0, 1, 2]), dict(second, seeds=[0])],
        eval_protocol_id=protocol.id, widths=list(WIDTHS), seeds=[0, 1, 2],
        source_id="S0001", normalization_id="N0001", candidate_universe_hash="sha256:" + "u" * 64,
        order_seed=1, order_hash="sha256:" + "o" * 64, analysis=decision, experiment_id=xid)
    expected = lab.expected_membership(experiment)
    assert len(expected) == 4 * 3 + 4 * 1          # arm A: 4 widths x 3 seeds; arm B: 4 widths x 1
    with pytest.raises(LabError, match="outside arm"):
        lab.queue_runs(first["ablations"][0], seeds=(0, 1, 2, 9))
    [run] = lab.queue_runs(second["ablations"][0], seeds=(0,))
    assert run.seed == 0
    # a seed outside the experiment's own declared list is refused at creation: fresh ablations
    # bound to the new experiment, so the binding check does not fire first
    other_xid = lab.store.next_id("X")
    bad_first = arm_material(lab, normalization, recipe, protocol, other_xid, scale="2x", suffix="C")
    bad_second = arm_material(lab, normalization, recipe, protocol, other_xid, scale="5x", suffix="D")
    with pytest.raises(LabError, match="outside the experiment's seed list"):
        lab.create_experiment(
            name="S-bad", preregistration_hash="sha256:" + "p" * 64, reference_unit_nodes=1,
            scales=dict(SCALES), node_budgets=[16000],
            arms=[dict(bad_first, seeds=[0, 7]), dict(bad_second, seeds=[0])],
            eval_protocol_id=protocol.id, widths=list(WIDTHS), seeds=[0, 1, 2],
            source_id="S0001", normalization_id="N0001", candidate_universe_hash="sha256:" + "u" * 64,
            order_seed=1, order_hash="sha256:" + "o" * 64, analysis=decision, experiment_id=other_xid)
