import pytest

from cvslab import families
from cvslab.store import LabError


def test_exact_parameter_counts():
    # 768*H + H (b1) + H (w2) + 1 (b2)
    assert families.param_count("NNUE", {"INPUT": "RAW", "H": 1}) == 771
    assert families.param_count("NNUE", {"INPUT": "RAW", "H": 16}) == 768 * 16 + 32 + 1
    assert families.parameter_shapes("NNUE", {"INPUT": "RAW", "H": 2}) == {
        "w1": [768, 2], "b1": [2], "w2": [2], "b2": []}


def test_coerce_rejects_out_of_range_and_wrong_types():
    assert families.coerce(families.switch_map("NNUE")["H"], "3") == 3
    with pytest.raises(LabError):
        families.coerce(families.switch_map("NNUE")["H"], 0)
    with pytest.raises(LabError):
        families.coerce(families.switch_map("NNUE")["H"], 1.5)
    with pytest.raises(LabError):
        families.coerce(families.switch_map("NNUE")["INPUT"], "GEO")
    with pytest.raises(LabError):
        families.coerce(families.switch_map("NNUE")["LR"], "not-a-number")


def test_unregistered_switches_are_refused():
    with pytest.raises(LabError, match="unregistered"):
        families.check_registered("NNUE", ["NOT_A_SWITCH"])


def test_merge_config_makes_defaults_explicit():
    merged = families.merge_config("NNUE", {}, {})
    assert set(merged) == {sw.key for sw in families.switches("NNUE")}
    assert merged["H"] == 1 and merged["INPUT"] == "RAW"


def test_diff_reports_only_deviations():
    control = families.merge_config("NNUE", {}, {})
    intervention = families.merge_config("NNUE", control, {"H": 8})
    diff = families.diff_configs("NNUE", control, intervention)
    assert [(d.key, d.control, d.value, d.axis) for d in diff] == [("H", 1, 8, "capacity")]


def test_display_label_shape():
    label = families.display_label("A0042", "NNUE", "G01", 12321, "RAW", [("H", 16)])
    assert label == "A0042 NNUE-G01-P12K @ RAW : H=16"
    assert families.display_label("A0001", "NNUE", "G01", 771, "RAW", []) == "A0001 NNUE-G01-P771 @ RAW"


def test_format_params():
    assert families.format_params(771) == "771"
    assert families.format_params(999) == "999"
    assert families.format_params(12321) == "12K"
    assert families.format_params(999_999) == "1000K"
    assert families.format_params(1_234_567) == "1.2M"
