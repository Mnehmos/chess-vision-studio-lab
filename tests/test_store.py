import json
import os
import stat

import pytest

from cvslab.schemas import EvidenceState, Hypothesis, Run, RunStatus, TrainingRecipe, utc_now
from cvslab.store import ImmutableError, Store, TamperError, payload_hash


def make_hypothesis(store, number=1):
    return store.create(Hypothesis(
        id=f"H{number:04d}", title="t", statement="s", family="NNUE", generation="G01", metric="test_loss",
        predicted_direction="decrease", min_effect=0.0, created_at=utc_now(),
    ))


def test_identities_are_never_reused(lab):
    make_hypothesis(lab.store, 1)
    with pytest.raises(ImmutableError):
        make_hypothesis(lab.store, 1)


def test_next_id_sequences_per_prefix(lab):
    assert lab.store.next_id("H") == "H0001"
    assert lab.store.next_id("H") == "H0002"
    assert lab.store.next_id("T") == "T0001"


def test_only_hypotheses_and_ablations_change_state(lab):
    hypothesis = make_hypothesis(lab.store, 1)
    updated = lab.store.update_state(hypothesis.id, EvidenceState.SUPPORTED, "test")
    assert updated.state == EvidenceState.SUPPORTED
    assert len(updated.state_history) == 1  # created directly with empty history

    recipe = lab.store.create(TrainingRecipe(
        id="T0001", name="r", family="NNUE", generation="G01", trainer="x", trainer_version=1, params={},
        recipe_hash="sha256:x", created_at=utc_now(),
    ))
    with pytest.raises(ImmutableError):
        lab.store.update_state(recipe.id, EvidenceState.SUPPORTED, "test")


def test_completed_run_is_sealed_and_tamper_evident(lab):
    run = lab.store.create(Run(
        id="R0001", ablation_id="A0001", hypothesis_id=None, display_label="x", seed=0,
        status=RunStatus.COMPLETED, effective_config={}, config_hash="sha256:c", identity_hash="sha256:i",
        param_count=1, dataset_id="D0001", dataset_manifest_hash="sha256:d", training_recipe_id="T0001",
        recipe_hash="sha256:r", eval_protocol_id="E0001", protocol_hash="sha256:p", queued_at=utc_now(),
    ))
    assert run.record_hash

    with pytest.raises(ImmutableError):
        lab.store.update_run(run.model_copy(deep=True))

    path = lab.store.object_path("R0001")
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    payload = path.read_bytes().replace(b'"seed": 0', b'"seed": 7')
    path.write_bytes(payload)
    with pytest.raises(TamperError):
        lab.store.get("R0001")


def test_run_identity_fields_are_fixed_at_queue_time(lab):
    run = lab.store.create(Run(
        id="R0001", ablation_id="A0001", hypothesis_id=None, display_label="x", seed=0,
        status=RunStatus.QUEUED, effective_config={}, config_hash="sha256:c", identity_hash="sha256:i",
        param_count=1, dataset_id="D0001", dataset_manifest_hash="sha256:d", training_recipe_id="T0001",
        recipe_hash="sha256:r", eval_protocol_id="E0001", protocol_hash="sha256:p", queued_at=utc_now(),
    ))
    run.seed = 5  # execution may not silently redefine what was queued
    with pytest.raises(ImmutableError):
        lab.store.update_run(run)


def test_artifacts_written_once(lab):
    rel, digest = lab.store.write_artifact("R0001", "model.json", b"{}")
    assert rel == "artifacts/R0001/model.json"
    assert digest.startswith("sha256:")
    with pytest.raises(ImmutableError):
        lab.store.write_artifact("R0001", "model.json", b"{}")


def test_path_escape_refused(lab):
    with pytest.raises(Exception):
        lab.store.abs("../outside.txt")


def test_list_filters_by_object_class(lab, tmp_path):
    make_hypothesis(lab.store, 1)
    assert len(lab.store.list("H")) == 1
    assert lab.store.list("R") == []


def make_run(store, *, status=RunStatus.COMPLETED):
    return store.create(Run(
        id="R0001", ablation_id="A0001", hypothesis_id=None, display_label="x", seed=0, status=status,
        effective_config={}, config_hash="sha256:c", identity_hash="sha256:i", param_count=1,
        dataset_id="D0001", dataset_manifest_hash="sha256:d", training_recipe_id="T0001",
        recipe_hash="sha256:r", eval_protocol_id="E0001", protocol_hash="sha256:p", queued_at=utc_now(),
    ))


def rewrite(path, payload):
    """Write a payload back exactly as another schema version would have recorded it."""
    path.chmod(0o644)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def test_record_seal_covers_the_recording_not_the_current_schema(lab):
    """A later schema addition must never turn historical evidence into apparent tampering.

    Simulated exactly as it happens in practice: an object written before a field existed
    keeps its recorded payload, and loading it under the current schema must still verify.
    """
    make_run(lab.store)
    path = lab.store.object_path("R0001")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = {key: value for key, value in payload.items() if key != "train_target_spec_hash"}
    assert len(recorded) < len(payload)  # the field exists in today's schema, not in the recording
    recorded = dict(recorded, record_hash=payload_hash(recorded))
    rewrite(path, recorded)

    loaded = lab.store.get("R0001")  # must not raise
    assert loaded.train_target_spec_hash is None  # the schema default fills the gap
    assert payload_hash(recorded) == loaded.record_hash


def test_record_seal_still_detects_tampering_with_recorded_values(lab):
    """The repair must not weaken the seal: changing anything recorded is still detected."""
    make_run(lab.store)
    path = lab.store.object_path("R0001")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["seed"] = 7  # recorded value changed, hash left as written
    rewrite(path, payload)
    with pytest.raises(TamperError):
        lab.store.get("R0001")
