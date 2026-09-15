import json
import os
import stat
from datetime import datetime

import pytest

from cvslab.schemas import EvidenceState, Finding, Run, RunStatus
from cvslab.store import ImmutableError, TamperError


def test_phase0_proof_completes_end_to_end(seeded_lab):
    result = seeded_lab.result
    assert set(result["runs"].values()) == {"COMPLETED"}
    finding: Finding = seeded_lab.store.get(result["finding"])
    assert finding.result in (EvidenceState.SUPPORTED, EvidenceState.REJECTED, EvidenceState.INCONCLUSIVE)
    assert finding.promotion_state.value == "NOT_ELIGIBLE"
    assert {g.name for g in finding.promotion_gates} == {"fixed_node_benchmark", "fixed_time_benchmark", "game_gate_sprt"}
    assert finding.record_hash  # sealed
    assert finding.intervention.param_count == 768 * 8 + 16 + 1
    assert finding.control.param_count == 771
    assert finding.effect is not None and finding.effect.n > 0
    assert finding.decision_rule.startswith("Effect = intervention - control")

    run: Run = seeded_lab.store.get(next(iter(result["runs"])))
    assert run.compute.train_examples_seen > 0
    assert run.environment is not None
    assert any(c.name == "param_count_serialized" and c.status == "pass" for c in run.integrity)
    assert run.model_id and seeded_lab.store.get(run.model_id).param_count == run.param_count
    assert run.log_path and "finished COMPLETED" in seeded_lab.run_log(run.id)


def test_findings_stay_searchable_including_negative(seeded_lab):
    finding_id = seeded_lab.result["finding"]
    state = seeded_lab.result["result"]
    assert [f.id for f in seeded_lab.list_findings()] == [finding_id]
    assert [f.id for f in seeded_lab.list_findings(result=state)] == [finding_id]
    if state != "SUPPORTED":
        assert seeded_lab.list_findings(result="SUPPORTED") == []
    assert [f.id for f in seeded_lab.list_findings(query="width")] == [finding_id]


def test_rerunning_creates_new_run_ids_not_edits(seeded_lab):
    [new_run] = seeded_lab.queue_runs(seeded_lab.result["baseline"], seeds=(9,))
    assert new_run.id not in seeded_lab.result["runs"]
    finished = seeded_lab.execute_run(new_run.id)
    assert finished.status == RunStatus.COMPLETED
    with pytest.raises(ImmutableError):
        seeded_lab.store.update_run(finished)


def test_tampered_run_record_raises_and_alerts(seeded_lab):
    run_id = next(iter(seeded_lab.result["runs"]))
    path = seeded_lab.store.object_path(run_id)
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["seed"] = 42
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(TamperError):
        seeded_lab.store.get(run_id)
    overview = seeded_lab.overview()
    assert any(run_id in alert and "record hash mismatch" in alert for alert in overview.integrity_alerts)


def test_preview_flags_noops_multivariable_and_duplicates(seeded_lab):
    baseline_id = seeded_lab.result["baseline"]
    preview = seeded_lab.preview_ablation(baseline_id, {"H": 1})
    assert preview.can_create is False
    assert any("equals the baseline" in w for w in preview.warnings)

    preview = seeded_lab.preview_ablation(baseline_id, {"H": 2, "LR": 0.001})
    assert len(preview.diff) == 2
    assert any("Multi-variable" in w for w in preview.warnings)
    assert any("independent axes" in w for w in preview.warnings)

    reused_width = seeded_lab.store.get(seeded_lab.result["ablation"]).effective_config["H"]
    duplicate = seeded_lab.preview_ablation(baseline_id, {"H": reused_width})
    assert duplicate.duplicate_of == seeded_lab.result["ablation"]
    assert duplicate.can_create is False

    with pytest.raises(Exception, match="unregistered"):
        seeded_lab.preview_ablation(baseline_id, {"SECRET_SWITCH": 1})


def test_run_fails_closed_to_invalid_when_evidence_corrupted(lab):
    lab.result = lab.run_phase0_proof(n_games=25, epochs=2, seeds=(0,), control_width=1, intervention_width=4)
    dataset = lab.store.get(lab.result["dataset"])
    path = lab.store.abs(dataset.splits[0].path)
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    path.write_bytes(path.read_bytes() + b'{"record_id": "pos_forged"}\n')

    ablation = lab.create_ablation(baseline_id=lab.result["baseline"], overrides={"H": 3})
    [run] = lab.queue_runs(ablation.id, seeds=(0,))
    finished = lab.execute_run(run.id)
    assert finished.status == RunStatus.INVALID
    assert any(c.name == "dataset_files" and c.status == "fail" for c in finished.integrity)
    assert any(c.name == "preflight" and c.status == "fail" for c in finished.integrity)

    # a comparison whose intervention arm has only invalid evidence must itself be INVALID
    hypothesis = lab.create_hypothesis(title="invalid arm", statement="s", metric="test_loss")
    finding = lab.draft_finding(hypothesis_id=hypothesis.id, control_ablation_id=lab.result["baseline"],
                                intervention_ablation_id=ablation.id)
    assert finding.result == EvidenceState.INVALID
    assert any(c.status == "fail" for c in finding.integrity)


def test_baseline_names_unique_per_generation(lab):
    lab.result = lab.run_phase0_proof(n_games=10, epochs=1, seeds=(0,), control_width=1, intervention_width=2)
    with pytest.raises(Exception, match="already frozen"):
        lab.register_baseline(name="raw", dataset_id=lab.result["dataset"],
                              training_recipe_id=lab.result["recipe"], eval_protocol_id=lab.result["protocol"])


def test_matrix_and_scaling_views_render(lab):
    lab.result = lab.run_phase0_proof(n_games=10, epochs=1, seeds=(0,), control_width=1, intervention_width=2)
    matrix = lab.matrix()
    assert [row.ablation_id for row in matrix.rows] == [lab.result["baseline"], lab.result["ablation"]]
    baseline_row, ablation_row = matrix.rows
    assert baseline_row.is_baseline and not any(cell.changed for cell in baseline_row.cells)
    assert [cell.key for cell in ablation_row.cells if cell.changed] == ["H"]
    assert ablation_row.metric_mean is not None
    assert ablation_row.finding_ids == [lab.result["finding"]]

    scaling = lab.scaling()
    assert len(scaling.points) == 2
    assert scaling.lower_is_better is True
    assert {p.param_count for p in scaling.points} == {771, 768 * 2 + 4 + 1}
    assert scaling.legacy_models == []

    running = lab.matrix(state="RUNNING")
    assert [row.ablation_id for row in running.rows] == [lab.result["baseline"]]  # control stays RUNNING


def test_overview_reports_compute_and_generation(seeded_lab):
    overview = seeded_lab.overview()
    assert overview.active_generation == "G01"
    assert overview.run_counts["COMPLETED"] == 2
    assert overview.compute.train_examples_seen > 0
    assert overview.legacy.models == 0  # nothing imported in the clean lab
    assert overview.object_counts["R"] == 2
    assert len(overview.recent_findings) == 1
    assert any("engine commit" in alert for alert in overview.integrity_alerts)  # warn, not silent
    assert not any("INVALID" in alert for alert in overview.integrity_alerts)
    assert datetime.fromisoformat(overview.recent_runs[0].finished_at.replace("Z", "+00:00"))
