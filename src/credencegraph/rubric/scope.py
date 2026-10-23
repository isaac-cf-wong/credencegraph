"""Scopes: closed intervals over the rubric's parameters, with open ends allowed.

An interval is ``(low, high)``, with ``None`` for an open end, so ``(10, None)`` is every value of
10 or more. Bounded ends are included. An intersection may be empty, which shows as ``low > high``;
a single value ``(v, v)`` is not empty.
"""

from __future__ import annotations

Interval = tuple[float | None, float | None]

UNBOUNDED: Interval = (None, None)


def intersection(first: Interval, second: Interval) -> Interval:
    """Return the intersection of two intervals, which is empty when they do not overlap.

    Args:
        first: An interval.
        second: Another interval.

    Returns:
        The tighter low end and the tighter high end.
    """
    lows = [end for end in (first[0], second[0]) if end is not None]
    highs = [end for end in (first[1], second[1]) if end is not None]
    return (max(lows) if lows else None, min(highs) if highs else None)


def is_empty(interval: Interval) -> bool:
    """Tell whether an interval holds no value.

    Args:
        interval: The interval.

    Returns:
        Whether both ends are bounded and the low end is above the high end.
    """
    low, high = interval
    return low is not None and high is not None and low > high


def contains(outer: Interval, inner: Interval) -> bool:
    """Tell whether every value of one interval lies in another.

    Args:
        outer: The interval that should hold the other.
        inner: The interval that should lie in it.

    Returns:
        Whether ``inner`` lies in ``outer``. An empty ``inner`` has no value to lie outside, so it
        lies in every interval; an empty ``outer`` holds nothing else, and an open end of ``inner``
        lies only within an open end of ``outer``.
    """
    if is_empty(inner):
        return True
    if is_empty(outer):
        return False
    low_ok = outer[0] is None or (inner[0] is not None and inner[0] >= outer[0])
    high_ok = outer[1] is None or (inner[1] is not None and inner[1] <= outer[1])
    return low_ok and high_ok


def show(interval: Interval) -> str:
    """Write an interval for a message, as ``[10, 100]``, ``[1, inf)`` or ``empty``.

    Args:
        interval: The interval.

    Returns:
        The interval in mathematical notation.
    """
    if is_empty(interval):
        return "empty"
    low, high = interval
    left = "(-inf" if low is None else f"[{low:g}"
    right = "inf)" if high is None else f"{high:g}]"
    return f"{left}, {right}"
