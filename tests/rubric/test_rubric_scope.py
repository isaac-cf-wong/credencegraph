"""Intervals with open ends: intersection, emptiness, containment and how they are written."""

from __future__ import annotations

import pytest

from credencegraph.rubric.scope import UNBOUNDED, contains, intersection, is_empty, show


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((10, 100), (50, 200), (50, 100)),
        ((10, None), (None, 100), (10, 100)),
        (UNBOUNDED, (1, 2), (1, 2)),
        (UNBOUNDED, UNBOUNDED, UNBOUNDED),
        ((10, 20), (30, 40), (30, 20)),
        ((10, 20), (20, 30), (20, 20)),
    ],
)
def test_intersection(first, second, expected):
    """Test that an intersection takes the tighter end on each side, open ends included, and is symmetric."""
    assert intersection(first, second) == expected
    assert intersection(second, first) == expected


@pytest.mark.parametrize(
    ("interval", "empty"),
    [((30, 20), True), ((20, 20), False), ((None, 20), False), ((30, None), False), (UNBOUNDED, False)],
)
def test_is_empty(interval, empty):
    """Test that only a bounded interval whose low end is above its high end is empty; a point is not."""
    assert is_empty(interval) is empty


@pytest.mark.parametrize(
    ("outer", "inner", "inside"),
    [
        ((10, 100), (10, 100), True),
        ((10, 100), (50, 50), True),
        ((10, 100), (9, 50), False),
        ((10, 100), (50, 101), False),
        ((10, None), (20, None), True),
        ((10, None), (5, None), False),
        ((10, 100), (20, None), False),
        ((None, 100), (None, 50), True),
        ((0, 100), (None, 50), False),
        (UNBOUNDED, UNBOUNDED, True),
        ((10, 100), UNBOUNDED, False),
        ((30, 20), (25, 25), False),
    ],
)
def test_contains(outer, inner, inside):
    """Test containment: bounded ends are included, an open end fits only in an open end, and empty holds nothing."""
    assert contains(outer, inner) is inside


@pytest.mark.parametrize(
    ("interval", "text"),
    [((10, 100), "[10, 100]"), ((1.5, None), "[1.5, inf)"), ((None, 0), "(-inf, 0]"), ((3, 1), "empty")],
)
def test_show(interval, text):
    """Test how an interval is written in a message."""
    assert show(interval) == text
