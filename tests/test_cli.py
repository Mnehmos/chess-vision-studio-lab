import json

import cvslab.cli as cli


def run_cli(home, *args):
    assert cli.main(["--home", str(home), *args]) == 0


def out_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_cli_full_tiny_flow(tmp_path, capsys):
    home = tmp_path / "labstore"
    run_cli(home, "demo", "--n-games", "10", "--epochs", "1", "--seeds", "0",
            "--control-width", "1", "--intervention-width", "4")
    result = out_json(capsys)
    assert set(result["runs"].values()) == {"COMPLETED"}

    overview = (run_cli(home, "overview"), out_json(capsys))[1]
    assert overview["run_counts"]["COMPLETED"] == 2

    matrix = (run_cli(home, "matrix"), out_json(capsys))[1]
    assert len(matrix["rows"]) == 2

    scaling = (run_cli(home, "scaling"), out_json(capsys))[1]
    assert len(scaling["points"]) == 2

    findings = (run_cli(home, "findings"), out_json(capsys))[1]
    assert len(findings) == 1 and findings[0]["promotion_state"] == "NOT_ELIGIBLE"

    ablations = (run_cli(home, "ls", "A"), out_json(capsys))[1]
    assert [a["id"] for a in ablations] == [result["baseline"], result["ablation"]]

    finding = (run_cli(home, "show", result["finding"]), out_json(capsys))[1]
    assert finding["hypothesis_id"] == result["hypothesis"]

    dataset = (run_cli(home, "show", result["dataset"]), out_json(capsys))[1]
    assert dataset["manifest_hash"].startswith("sha256:")

    backlog = (run_cli(home, "backlog"), out_json(capsys))[1]
    assert backlog["switches"] == []

    run_cli(home, "log", next(iter(result["runs"])))
    assert "epoch" in capsys.readouterr().out


def test_cli_schema_export(tmp_path, capsys):
    out_file = tmp_path / "schema.json"
    run_cli(tmp_path / "labstore", "schema", "--out", str(out_file))
    document = json.loads(out_file.read_text(encoding="utf-8"))
    assert document["title"] == "CVSLab" and "$defs" in document
    assert "Finding" in document["$defs"]


def test_cli_reports_errors_and_exits_nonzero(tmp_path, capsys):
    assert cli.main(["--home", str(tmp_path / "labstore"), "show", "R9999"]) == 1
    assert "not found" in capsys.readouterr().err
    import pytest
    with pytest.raises(SystemExit) as exc:  # argparse rejects the prefix choice before main returns
        cli.main(["--home", str(tmp_path / "labstore"), "ls", "Z"])  # Z is not an identity prefix
    assert exc.value.code == 2
