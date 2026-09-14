import pytest
from fastapi.testclient import TestClient

from cvslab.api import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "labstore", worker=False)
    app.state.service.run_phase0_proof(n_games=20, epochs=2, seeds=(0,), control_width=1, intervention_width=4)
    with TestClient(app) as test_client:
        yield test_client, app.state.service


def test_overview_endpoint(client):
    http, _ = client
    body = http.get("/api/overview").json()
    assert body["active_generation"] == "G01"
    assert body["run_counts"]["COMPLETED"] == 2
    assert http.get("/api/overview").status_code == 200


def test_matrix_scaling_findings_and_objects(client):
    http, service = client
    matrix = http.get("/api/matrix").json()
    assert len(matrix["rows"]) == 2
    assert [sw["key"] for sw in matrix["switches"]][0] == "INPUT"

    scaling = http.get("/api/scaling", params={"metric": "cp_mae"}).json()
    assert scaling["metric"] == "cp_mae" and len(scaling["points"]) == 2

    findings = http.get("/api/findings").json()
    assert len(findings) == 1 and findings[0]["promotion_state"] == "NOT_ELIGIBLE"
    finding_id = findings[0]["id"]
    assert http.get(f"/api/findings/{finding_id}").status_code == 200

    run_id = findings[0]["control"]["run_ids"][0]
    assert http.get(f"/api/runs/{run_id}/log").status_code == 200
    assert "epoch" in http.get(f"/api/runs/{run_id}/log").text

    assert http.get("/api/object/D0001").status_code == 200
    assert http.get("/api/object/R9999").status_code == 404
    assert http.get("/api/objects/A").status_code == 200
    assert http.get("/api/objects/X").status_code == 400


def test_builder_flow_over_http(client):
    http, service = client
    hypothesis = http.post("/api/hypotheses", json={
        "title": "wider is better", "statement": "H=2 beats H=1 on held-out loss", "min_effect": 0.0,
    }).json()
    assert hypothesis["id"] == "H0002"

    preview = http.post("/api/ablations/preview", json={"baseline_id": "A0001", "overrides": {"H": 2}}).json()
    assert preview["can_create"] is True
    assert preview["param_count"] == 768 * 2 + 4 + 1
    assert preview["control_param_count"] == 771
    assert [d["key"] for d in preview["diff"]] == ["H"]

    ablation = http.post("/api/ablations", json={
        "baseline_id": "A0001", "overrides": {"H": 2}, "hypothesis_id": hypothesis["id"],
    }).json()
    assert ablation["display_label"].startswith("A0003 NNUE-G01-P1.5K @ RAW : H=2")

    queued = http.post("/api/runs", json={"ablation_id": ablation["id"], "seeds": [3]}).json()
    assert queued[0]["status"] == "QUEUED"
    finished = http.post(f"/api/runs/{queued[0]['id']}/execute").json()
    assert finished["status"] == "COMPLETED"
    assert any(m["name"] == "test_loss" for m in finished["metrics"])

    finding = http.post("/api/findings", json={
        "hypothesis_id": hypothesis["id"], "control_ablation_id": "A0001", "intervention_ablation_id": ablation["id"],
    }).json()
    assert finding["result"] in ("SUPPORTED", "REJECTED", "INCONCLUSIVE")
    assert finding["intervention"]["seeds"] == [3]

    # error paths surface as structured errors, not crashes
    assert http.post("/api/ablations/preview", json={"baseline_id": "A0001", "overrides": {"NOPE": 1}}).status_code == 400
    assert http.post("/api/ablations", json={"baseline_id": "A0002", "overrides": {"H": 4}}).status_code == 400
    assert http.get("/api/runs", params={"status": "bogus"}).status_code == 400


def test_baseline_registration_over_http(client):
    http, service = client
    created = http.post("/api/baselines", json={
        "name": "BIG", "dataset_id": "D0001", "training_recipe_id": "T0001", "eval_protocol_id": "E0001",
        "overrides": {"H": 32},
    })
    assert created.status_code == 201
    assert created.json()["param_count"] == 768 * 32 + 64 + 1

    conflict = http.post("/api/baselines", json={
        "name": "RAW", "dataset_id": "D0001", "training_recipe_id": "T0001", "eval_protocol_id": "E0001",
    })
    assert conflict.status_code == 400
    assert "already frozen" in conflict.json()["error"]


def test_switches_and_backlog_endpoints(client):
    http, _ = client
    registry = http.get("/api/switches").json()
    assert {sw["key"] for sw in registry} >= {"INPUT", "H", "LAMBDA", "EPOCHS"}
    assert http.get("/api/switches", params={"family": "NOPE"}).status_code == 400
    backlog = http.get("/api/backlog").json()
    assert backlog["intake_id"] is None and backlog["switches"] == []
