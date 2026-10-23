"""Reading the dimension of a unit, and telling whether two units agree."""

from __future__ import annotations

import pytest

from credencegraph.rubric.units import agree, dimension


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Hz", "1/s"),
        ("Hz", "kHz"),
        ("Hz", "s^-1"),
        ("m/s", "m s^-1"),
        ("m/s", "km/h"),
        ("J", "kg m^2/s^2"),
        ("J", "kg·m^2·s^-2"),
        ("J", "eV"),
        ("N", "kg*m/s^2"),
        ("W", "J/s"),
        ("V", "W/A"),
        ("Ω", "V/A"),
        ("T", "Wb/m^2"),
        ("%", "1"),
        ("ms", "min"),
        ("Gyr", "s"),
        ("kpc", "au"),
        ("µs", "μs"),
        ("us", "ns"),
        ("dam", "m"),
        ("mmol", "mol"),
        ("hPa", "Pa"),
    ],
)
def test_units_of_one_dimension_agree(first, second):
    """Test that prefixed, compound and derived units agree with others of their dimension."""
    assert dimension(first) is not None
    assert agree(first, second)
    assert agree(second, first)


@pytest.mark.parametrize(
    ("first", "second"),
    [("Hz", "s"), ("m", "s"), ("J", "W"), ("kg", "g m"), ("m^2", "m"), ("%", "s"), ("Pa", "N")],
)
def test_units_of_different_dimensions_disagree(first, second):
    """Test that units of different dimensions, including the Hz and s of the same timescale, disagree."""
    assert not agree(first, second)
    assert not agree(second, first)


@pytest.mark.parametrize("unit", ["bananas", "", " ", "m/s/s", "Hz^x", "kHzz", "m//s", "/s", "m^"])
def test_unreadable_unit_has_no_dimension(unit):
    """Test that a unit outside the registry, or written in a way it does not read, has no dimension."""
    assert dimension(unit) is None


def test_unreadable_units_agree_only_with_themselves():
    """Test that an unreadable unit agrees with the same text and with nothing else."""
    assert agree("furlong", "furlong")
    assert not agree("furlong", "m")
    assert not agree("m", "furlong")
    assert not agree("furlong", "fortnight")
