"""S13 report metadata is DERIVED, not inherited from another contract.

X0010's sealed result cited `SF-DEEP (E0005/D0040)` and `s13-preregistration.json` because those
were literals in the warm report code, while the cold study bound `E0006/D0043` and the v2
preregistration. The sealed study is preserved with an erratum; these tests are the regression
guard so a future contract cannot inherit another study's exam ids or preregistration path.
"""
from __future__ import annotations

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "s13"))

import s13_analyse  # noqa: E402

MODULE = pathlib.Path(s13_analyse.__file__).read_text(encoding="utf-8")


def test_exam_labels_follow_the_bound_ids():
    labels = s13_analyse.exam_labels("E0004", "D0010", "E0006", "D0043")
    assert labels == ["CVS-DEEP (E0004/D0010)", "SF-DEEP (E0006/D0043)"]
    other = s13_analyse.exam_labels("E0004", "D0010", "E0009", "D0099")
    assert other[1] == "SF-DEEP (E0009/D0099)"


def _string_literals_outside_docstrings() -> list[str]:
    tree = ast.parse(MODULE)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value not in docstrings]


def test_the_report_code_carries_no_other_contracts_identifiers():
    """The warm identifiers that leaked into X0010 must not exist as literals in the code.

    Docstrings may narrate the leak (that is the record); code may not repeat it.
    """
    literals = _string_literals_outside_docstrings()
    offenders = [value for value in literals
                 if "E0005" in value or "D0040" in value
                 or "s13-preregistration.json" in value]
    assert offenders == [], f"hard-coded warm identifiers are back: {offenders}"


def test_the_preregistration_path_and_exam_key_are_contract_derived():
    assert s13_analyse.PREREG_KEY in ("preregistration", "preregistration_cold")
    assert s13_analyse.EXAM_KEY in ("exam", "exam_cold")
    assert s13_analyse.CONTRACT in ("cold", "warm")
    # the cold contract must point at v2 and its own exam record, never the warm ones
    if s13_analyse.CONTRACT == "cold":
        assert s13_analyse.EXAM_KEY == "exam_cold" and s13_analyse.PREREG_KEY == "preregistration_cold"
