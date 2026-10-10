"""Recognising numerals, units and equation references in a node's text."""

from __future__ import annotations

import pytest

from credencegraph.rubric import has_eqref, has_numeral, has_unit, numerals


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("between 10 and 100 Hz.", {10.0, 100.0}),
        ("below 5%", {5.0, 0.05}),
        ("below 5\\% of the time", {5.0, 0.05}),
        ("a rate of 1.1%", {1.1, 0.011}),
        ("a step of 1.2e-3", {0.0012}),
        ("a step of 1.2E+3", {1200.0}),
        ("a step of 1.2\N{MULTIPLICATION SIGN}10^-3", {0.0012}),
        ("a step of $1.2 \\times 10^{-3}$", {0.0012}),
        ("1,000 events", {1000.0}),
        ("1\N{THIN SPACE}000 events", {1000.0}),
        ("1\\,000 events", {1000.0}),
        ("a shift of -3 and \N{MINUS SIGN}4", {-3.0, -4.0}),
        ("x = +2", {2.0}),
        ("about .5 of them", {0.5}),
        ("no numbers here", set()),
    ],
)
def test_numerals(text, expected):
    """Test the numeral forms the contract names, each parsed to its value."""
    assert numerals(text) == expected


def test_hyphen_between_numbers_is_not_a_sign():
    """Test that a hyphen after a word character, as in a range or an id, does not make a negative number."""
    assert numerals("pages 5-7") == {5.0, 7.0}


def test_percent_value_is_exact():
    """Test that a percentage divided by 100 is the float nearest the decimal, not a rounded division."""
    assert 1.1 / 100 != 0.011
    assert has_numeral("1.1%", 0.011)


def test_has_numeral_compares_values():
    """Test that a form number matches a numeral with the same value, whatever its spelling."""
    assert has_numeral("between 10 and 100 Hz", 10)
    assert has_numeral("a value of 10.0", 10)
    assert has_numeral("below 5%", 0.05)
    assert not has_numeral("below 5%", 0.04)
    assert not has_numeral("above 0.1", 0.01)


@pytest.mark.parametrize(
    ("text", "unit", "expected"),
    [
        ("between 10 and 100 Hz.", "Hz", True),
        ("100Hz", "Hz", False),
        ("a 3 kHz tone", "Hz", False),
        ("at $100\\,\\mathrm{Hz}$", "Hz", True),
        ("at \\si{Hz}", "Hz", True),
        ("at 3 m/s", "m/s", True),
        ("at 3 m/s", "m", True),
        ("no unit", "s", False),
        ("a proportion of 1.1%", "%", True),
        ("a proportion of 1.1\\%", "%", True),
        ("a proportion of 1.1 %", "%", True),
        ("a proportion of 1.1", "%", False),
        ("at 20\N{DEGREE SIGN}C", "\N{DEGREE SIGN}C", True),
        ("at 20\N{DEGREE SIGN}Cx", "\N{DEGREE SIGN}C", False),
    ],
)
def test_has_unit(text, unit, expected):
    """Test that a unit matches a token delimited by whitespace or punctuation, including LaTeX arguments."""
    assert has_unit(text, unit) is expected


@pytest.mark.parametrize(
    ("text", "label", "expected"),
    [
        ("as \\ref{eq:a} shows", "eq:a", True),
        ("as \\eqref{eq:a} shows", "eq:a", True),
        ("as \\cref{eq:a} shows", "eq:a", True),
        ("as \\Cref{eq:a} shows", "eq:a", True),
        ("as Eq. (3) shows", "3", True),
        ("as Eq. (3) shows", "eq:a", False),
        ("as \\eqref{eq:ab} shows", "eq:a", False),
        ("in 3 steps", "3", False),
    ],
)
def test_has_eqref(text, label, expected):
    """Test that an equation reference matches the referencing commands or a printed number in parentheses."""
    assert has_eqref(text, label) is expected
