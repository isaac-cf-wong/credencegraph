"""Reading rubric files: the example rubric, and every rule of the format."""

from __future__ import annotations

import tomllib

import pytest
from _ingested import EXAMPLE_RUBRIC_PATH, RUBRIC

from credencegraph.core import Beta, Point
from credencegraph.rubric import RubricError, load_rubric, loads_rubric, rubric_from_dict

MINIMAL = """
[rubric]
format = 1
name = "tiny"
version = "1"

[types.result]
"""


def test_example_rubric():
    """Test that the documented example rubric loads with its types, fields, rules and parameters."""
    assert RUBRIC.identity() == {"name": "methods-paper", "version": "2026.1"}
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
        ({"rubric": {"format": 2, "name": "n", "version": "1"}}, "not a known rubric format"),
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
    """Test that the rubric printed in the format's documentation is the example file, word for word."""
    page = (EXAMPLE_RUBRIC_PATH.parents[1] / "verbatim-ingestion.md").read_text(encoding="utf-8")
    printed = page.split("```toml\n", 1)[1].split("```", 1)[0]
    assert printed == EXAMPLE_RUBRIC_PATH.read_text(encoding="utf-8")
