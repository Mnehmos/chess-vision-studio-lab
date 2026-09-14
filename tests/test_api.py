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


def test_data_workbench_endpoints(client):
    http, service = client
    # catalog with facet counts and dataset membership
    catalog = http.get("/api/catalog", params={"normalization_id": "N0001", "phase": "opening"}).json()
    assert catalog["total_matching"] > 0
    assert catalog["facets"]["phase"]["opening"] == catalog["total_matching"]
    assert "D0001" in catalog["facets"]["dataset"]
    assert http.get("/api/catalog", params={"normalization_id": "N0001", "vibes": "x"}).status_code == 400

    # append a second label authority, then see it in the catalog's tier facet
    record_id = catalog["page"][0]["record_id"]
    created = http.post("/api/labels", json={
        "normalization_id": "N0001", "family": "oracle_cp", "producer": "sf-16",
        "authority": "oracle.sf16", "rows": [{"record_id": record_id, "value": 12, "budget": {"depth": 20}}],
    })
    assert created.status_code == 201
    assert created.json()["families"] == {"oracle_cp": 1}
    assert http.post("/api/labels", json={
        "normalization_id": "N0001", "family": "vibes", "producer": "x", "authority": "a",
        "rows": [{"record_id": record_id, "value": 1}],
    }).status_code == 400
    tiered = http.get("/api/catalog", params={"normalization_id": "N0001", "tier": "oracle"}).json()
    assert tiered["total_matching"] == 1
    assert http.get("/api/normalizations/N0001/labels").json()[0]["rows"] == 1

    # stack preview -> freeze -> manifest
    arms = [{"name": "mid", "filter": {"phase": "middlegame"}, "policy": "fraction", "fraction": 0.5, "seed": 1}]
    preview = http.post("/api/stacks/preview", json={"normalization_id": "N0001", "arms": arms}).json()
    assert preview["arms"][0]["effective"] < preview["arms"][0]["available"]
    frozen = http.post("/api/datasets/freeze", json={
        "normalization_id": "N0001", "name": "mid-half", "arms": arms, "required_labels": ["eval_cp"],
    })
    assert frozen.status_code == 201
    assert frozen.json()["coverage"]["stack"] == {"mid": preview["arms"][0]["effective"]}

    # migrate -> diff -> rebuild descendant
    migrated = http.post("/api/normalizations", json={
        "from_normalization_id": "N0001", "name": "v2",
        "settings_overrides": {"phase_middlegame_min_material": 4000},
    })
    assert migrated.status_code == 201
    diff = http.get("/api/migration-diff", params={"from_id": "N0001", "to_id": "N0002"}).json()
    assert diff["changed"] > 0
    assert diff["affected_dataset_ids"] == ["D0001", "D0002"]  # every dataset frozen on the old standard
    rebuilt = http.post("/api/datasets/D0001/rebuild", json={"new_normalization_id": "N0002"})
    assert rebuilt.status_code == 201
    assert rebuilt.json()["parent_id"] == "D0001"
    assert http.post("/api/datasets/D0001/rebuild", json={"new_normalization_id": "N0002"}).status_code == 400
    # historical manifests unchanged
    original = http.get("/api/object/D0001").json()
    assert original["parent_id"] is None
