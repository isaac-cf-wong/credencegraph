"""Reading rubric files: the example rubric, and every rule of the format."""

from __future__ import annotations

import datetime
import tomllib

import pytest
from _ingested import EXAMPLE_RUBRIC_PATH, RUBRIC

from credencegraph.core import Beta, Point
from credencegraph.rubric import Calibration, Check, Premises, RubricError, load_rubric, loads_rubric, rubric_from_dict

MINIMAL = """
[rubric]
format = 1
name = "tiny"
version = "1"

[types.result]
"""


def test_example_rubric():
    """Test that the documented example rubric loads with its types, fields, rules and parameters."""
    assert RUBRIC.identity() == {"name": "methods-paper", "version": "2026.2"}
    assert list(RUBRIC.types)[:3] == ["reference", "method", "result"]
    claim = RUBRIC.types["claim"]
    assert claim.assess
    assert claim.base == Beta(5, 5)
    assert claim.form["shape"].required
    assert claim.form["shape"].values == ("universal", "existential", "comparative", "bound")
    assert claim.rests_on.relation == "requires"
    assert claim.rests_on.min == 1
    assert claim.rests_on.strength == Beta(9, 1)
    assert "literature_claim" in claim.rests_on.types
    assert RUBRIC.types["assumption"].origins == ("document", "analyst")
    assert RUBRIC.types["result"].origins == ("document",)
    assert RUBRIC.parameters["sample_rate"].domain == (0.0, None)
    assert RUBRIC.initial == {"reference": "reference"}


def test_example_rubric_format_2_tables():
    """Test that the example rubric's calibration, checks, premises and hedges load."""
    assert RUBRIC.calibration == Calibration(
        "example", "log", "nothing: the values are illustrative", datetime.date(2026, 10, 10)
    )
    assert list(RUBRIC.checks) == ["recompute-numbers", "citation-support"]
    recompute = RUBRIC.checks["recompute-numbers"]
    assert (recompute.targets, recompute.sensitivity, recompute.false_pass, recompute.models) == (
        ("result", "claim"),
        Beta(19, 1),
        Beta(1, 9),
        (),
    )
    assert RUBRIC.checks["citation-support"].models == ("example-model-2026-01-01",)
    assert RUBRIC.premises == Premises(Beta(2, 2), Beta(9, 1), ("claim",), 3)
    assert dict(RUBRIC.hedges) == {"suggests": Beta(6, 4), "may": Beta(5, 5), "we show": Beta(9, 1)}


def test_check_names_are_evidence_types():
    """Test that each check is also a node type of the rubric, for evidence nodes only, after the declared types."""
    assert list(RUBRIC.types)[-2:] == ["recompute-numbers", "citation-support"]
    check_type = RUBRIC.types["citation-support"]
    assert (check_type.description, check_type.origins) == (
        RUBRIC.checks["citation-support"].description,
        ("evidence",),
    )


def test_minimal_rubric_defaults():
    """Test the defaults of a type that declares nothing."""
    result = loads_rubric(MINIMAL).types["result"]
    assert (result.description, result.origins, result.assess, result.base, dict(result.form), result.rests_on) == (
        "",
        ("document",),
        False,
        None,
        {},
        None,
    )


def test_probability_base():
    """Test that a base may be a bare probability."""
    assert loads_rubric(MINIMAL + "base = 0.3\n").types["result"].base == Point(0.3)


def _data(**changes) -> dict:
    """The minimal rubric as data, with top-level tables replaced."""
    data = tomllib.loads(MINIMAL)
    data.update(changes)
    return data


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("colour = 'red'\n", "unknown keys \\['colour'\\]"),
        ("origins = ['reader']\n", "'reader' is not one of"),
        ("origins = []\n", "non-empty array"),
        ("assess = 'yes'\n", "must be true or false"),
        ("base = 'beta:0,1'\n", "Beta.alpha must be > 0"),
        ("base = true\n", "must be a probability"),
        ("form.x = { type = 'text' }\n", "must be one of"),
        ("form.x = { type = 'enum' }\n", "an enum field needs values"),
        ("form.x = { type = 'number', values = ['a'] }\n", "an enum field needs values"),
        ("form.x = { type = 'number', optional = true }\n", "unknown keys \\['optional'\\]"),
        ("form.x = { type = 'enum', values = ['a', 'a'] }\n", "lists a value twice"),
        ("rests_on = { types = ['result'], relation = 'requires' }\n", "missing \\['strength'\\]"),
        (
            "rests_on = { types = ['result'], relation = 'refutes', strength = 0.9 }\n",
            "must be 'requires' or 'supports'",
        ),
        ("rests_on = { types = ['result'], relation = 'requires', strength = 0.9, min = 0 }\n", "integer of 1 or more"),
        (
            "rests_on = { types = ['finding'], relation = 'requires', strength = 0.9 }\n",
            "'finding', which is not a declared type",
        ),
    ],
)
def test_type_errors(extra, message):
    """Test that each malformed type declaration is refused with a message naming it."""
    with pytest.raises(RubricError, match=message):
        loads_rubric(MINIMAL + extra)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rubric": {"format": 3, "name": "n", "version": "1"}}, "not a known rubric format"),
        ({"rubric": {"format": True, "name": "n", "version": "1"}}, "not a known rubric format"),
        ({"rubric": {"format": 1, "name": "", "version": "1"}}, "rubric.name must be a non-empty string"),
        ({"rubric": {"format": 1, "name": "n"}}, "missing \\['version'\\]"),
        ({"rules": {}}, "unknown keys \\['rules'\\]"),
        ({"types": {"compound": {}}}, "built in"),
        ({"types": {"unassigned": {}}}, "built in"),
        ({"parameters": {"rate": {"domain": [0, 1]}}}, "missing \\['unit'\\]"),
        ({"parameters": {"rate": {"unit": "Hz", "domain": [1, 0]}}}, "low 1 above high 0"),
        ({"parameters": {"rate": {"unit": "Hz", "domain": [0]}}}, "must be \\[low, high\\]"),
        ({"parameters": {"rate": {"unit": "Hz", "domain": [0, "open"]}}}, "or -inf or inf"),
        ({"initial": {"sentence": "claim"}}, "not a declared type"),
        ({"initial": {"sentence": ["result"]}}, "not a declared type"),
    ],
)
def test_rubric_errors(changes, message):
    """Test that each malformed rubric is refused with a message naming the problem."""
    with pytest.raises(RubricError, match=message):
        rubric_from_dict(_data(**changes))


def test_initial_type_must_allow_document_nodes():
    """Test that a unit's initial type must be one a document node may have."""
    text = MINIMAL + "origins = ['analyst']\n[initial]\nsentence = 'result'\n"
    with pytest.raises(RubricError, match="allowed for document nodes"):
        loads_rubric(text)


def test_open_domain_ends():
    """Test that -inf and inf are open ends of a parameter's domain."""
    data = _data(parameters={"shift": {"unit": "s", "domain": [float("-inf"), float("inf")]}})
    assert rubric_from_dict(data).parameters["shift"].domain == (None, None)


def test_not_toml():
    """Test that a file that is not TOML is a rubric error, not a parser exception."""
    with pytest.raises(RubricError, match="not valid TOML"):
        loads_rubric("[rubric\n")


def test_load_from_path():
    """Test that a rubric loads from a path."""
    assert load_rubric(EXAMPLE_RUBRIC_PATH) == RUBRIC


def test_documented_rubric_is_the_example_file():
    """Test that the rubric printed in the format's documentation parses to the same TOML as the example file.

    The comparison is on the parsed tables, not the bytes, so reformatting either copy does not break it.
    """
    page = (EXAMPLE_RUBRIC_PATH.parents[1] / "verbatim-ingestion.md").read_text(encoding="utf-8")
    printed = page.split("```toml\n", 1)[1].split("```", 1)[0]
    assert tomllib.loads(printed) == tomllib.loads(EXAMPLE_RUBRIC_PATH.read_text(encoding="utf-8"))


FORMAT_2 = """
[rubric]
format = 2
name = "tiny"
version = "2"

[types.result]
[types.claim]
"""

CHECK = {"targets": ["result"], "sensitivity": "beta:9,1", "false_pass": "beta:1,9", "models": []}
CALIBRATION = {"status": "example", "scoring": "brier", "against": "nothing", "date": datetime.date(2026, 1, 1)}
PREMISES = {"base": "beta:2,2", "strength": "beta:9,1", "for": ["claim"], "max_per_conclusion": 2}


def _format_2(**changes) -> dict:
    """The minimal format-2 rubric as data, with top-level tables replaced."""
    data = tomllib.loads(FORMAT_2)
    data.update(changes)
    return data


def test_minimal_format_2_rubric():
    """Test that a format-2 rubric without the new tables loads with none of them."""
    rubric = loads_rubric(FORMAT_2)
    assert (rubric.calibration, dict(rubric.checks), rubric.premises, dict(rubric.hedges)) == (None, {}, None, {})
    assert list(rubric.types) == ["result", "claim"]


def test_format_2_tables_load():
    """Test that each format-2 table loads into the rubric."""
    rubric = rubric_from_dict(
        _format_2(
            calibration=CALIBRATION,
            checks={"rerun": {**CHECK, "description": "Rerun it.", "models": ["m-1"]}},
            premises=PREMISES,
            hedges={"may": 0.5},
        )
    )
    assert rubric.calibration == Calibration("example", "brier", "nothing", datetime.date(2026, 1, 1))
    assert rubric.checks["rerun"] == Check("rerun", ("result",), Beta(9, 1), Beta(1, 9), ("m-1",), "Rerun it.")
    assert rubric.premises == Premises(Beta(2, 2), Beta(9, 1), ("claim",), 2)
    assert rubric.hedge("may") == Point(0.5)


@pytest.mark.parametrize(
    "table",
    [
        {"calibration": CALIBRATION},
        {"checks": {"rerun": CHECK}},
        {"premises": PREMISES},
        {"hedges": {"may": 0.5}},
    ],
)
def test_format_1_refuses_format_2_tables(table):
    """Test that a format-1 rubric may not use a format-2 table, and the same table loads in format 2."""
    with pytest.raises(RubricError, match=f"unknown keys \\['{next(iter(table))}'\\]"):
        rubric_from_dict(_data(**table))
    rubric_from_dict(_format_2(**table))


def test_hedges_are_compared_after_normalisation_and_case_folding():
    """Test that a hedge is found whatever its case and whitespace, and stored under its normalised key."""
    rubric = rubric_from_dict(_format_2(hedges={"We  Show": "beta:9,1", "STRASSE": 0.6}))
    assert dict(rubric.hedges) == {"we show": Beta(9, 1), "strasse": Point(0.6)}
    assert rubric.hedge(" we\nSHOW ") == Beta(9, 1)
    assert rubric.hedge("Straße") == Point(0.6)
    assert rubric.hedge("we showed") is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        # A check name is an evidence node type, so it may not also be a declared or built-in type.
        ({"checks": {"result": CHECK}}, "checks.result: a check name is an evidence node type"),
        ({"checks": {"compound": CHECK}}, "checks.compound: a check name is an evidence node type"),
        # E[sensitivity] > E[false_pass], strictly.
        (
            {"checks": {"rerun": {**CHECK, "sensitivity": "beta:1,9", "false_pass": "beta:9,1"}}},
            "the mean of sensitivity .* must exceed the mean of false_pass",
        ),
        (
            {"checks": {"rerun": {**CHECK, "sensitivity": "beta:5,5", "false_pass": 0.5}}},
            "the mean of sensitivity .* must exceed the mean of false_pass",
        ),
        ({"checks": {"rerun": {**CHECK, "sensitivity": "beta:0,1"}}}, "checks.rerun.sensitivity: .*Beta.alpha"),
        ({"checks": {"rerun": {**CHECK, "false_pass": True}}}, "checks.rerun.false_pass must be a probability"),
        # targets and premises.for name declared types, not check types.
        ({"checks": {"rerun": {**CHECK, "targets": ["finding"]}}}, "checks.rerun.targets names 'finding'"),
        (
            {"checks": {"rerun": CHECK, "audit": {**CHECK, "targets": ["rerun"]}}},
            "checks.audit.targets names 'rerun', which is not a declared type",
        ),
        ({"checks": {"rerun": {**CHECK, "targets": []}}}, "checks.rerun.targets must be a non-empty array"),
        ({"premises": {**PREMISES, "for": ["finding"]}}, "premises.for names 'finding', which is not a declared type"),
        # Unknown and missing keys.
        ({"checks": {"rerun": {**CHECK, "model": []}}}, "checks.rerun has unknown keys \\['model'\\]"),
        ({"checks": {"rerun": {k: v for k, v in CHECK.items() if k != "models"}}}, "missing \\['models'\\]"),
        ({"checks": {"rerun": {**CHECK, "models": "m-1"}}}, "checks.rerun.models must be an array"),
        ({"checks": {"rerun": {**CHECK, "models": ["m-1", "m-1"]}}}, "lists a value twice"),
        ({"checks": {"rerun": "beta:9,1"}}, "checks.rerun must be a table"),
        ({"calibration": {**CALIBRATION, "fitted": True}}, "calibration has unknown keys \\['fitted'\\]"),
        ({"calibration": {"status": "example"}}, "calibration is missing"),
        ({"calibration": {**CALIBRATION, "status": "draft"}}, "calibration.status must be one of"),
        ({"calibration": {**CALIBRATION, "scoring": "spherical"}}, "calibration.scoring must be one of"),
        ({"calibration": {**CALIBRATION, "against": ""}}, "calibration.against must be a non-empty string"),
        ({"calibration": {**CALIBRATION, "date": "2026-01-01"}}, "calibration.date must be a date"),
        (
            {"calibration": {**CALIBRATION, "date": datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)}},
            "calibration.date must be a date",
        ),
        ({"premises": {**PREMISES, "maximum": 1}}, "premises has unknown keys \\['maximum'\\]"),
        ({"premises": {"base": "beta:2,2"}}, "premises is missing"),
        ({"premises": {**PREMISES, "max_per_conclusion": 0}}, "integer of 1 or more"),
        ({"premises": {**PREMISES, "max_per_conclusion": True}}, "integer of 1 or more"),
        ({"premises": {**PREMISES, "strength": "strong"}}, "premises.strength"),
        ({"rules": {}}, "unknown keys \\['rules'\\]"),
        # Hedge keys are compared after normalisation and case-folding.
        ({"hedges": {"We show": 0.9, "we  SHOW": 0.8}}, "'We show' and 'we  SHOW' are the same phrase"),
        ({"hedges": {" ": 0.5}}, "empty once normalised"),
        ({"hedges": {"may": "maybe"}}, "hedges.may"),
    ],
)
def test_format_2_errors(changes, message):
    """Test that each breach of a format-2 rule is refused with a message naming it."""
    with pytest.raises(RubricError, match=message):
        rubric_from_dict(_format_2(**changes))
