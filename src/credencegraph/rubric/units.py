"""Units: the physical dimension of a unit written as text, for telling whether two units agree.

A unit is one or more symbols, each with an optional SI prefix and an optional integer power,
joined by ``*``, ``·``, ``⋅``, the multiplication sign or a space, with ``/`` dividing by everything after it: ``Hz``,
``kHz``, ``m/s``, ``m s^-1``, ``kg·m^2/s^2``. Its dimension is a vector of powers of the seven SI
base dimensions. Only the dimension is computed, not the scale, so ``Hz`` and ``kHz`` agree and
``Hz`` and ``s`` do not. The registry is deliberately small; a unit it cannot read has no dimension.
"""

from __future__ import annotations

import re

Dimension = tuple[int, int, int, int, int, int, int]

# Powers of length, mass, time, current, temperature, amount and luminous intensity.
_NONE: Dimension = (0, 0, 0, 0, 0, 0, 0)
_L: Dimension = (1, 0, 0, 0, 0, 0, 0)
_M: Dimension = (0, 1, 0, 0, 0, 0, 0)
_T: Dimension = (0, 0, 1, 0, 0, 0, 0)
_I: Dimension = (0, 0, 0, 1, 0, 0, 0)


def _times(*factors: tuple[Dimension, int]) -> Dimension:
    """Return the product of dimensions, each raised to a power."""
    return tuple(sum(dimension[i] * power for dimension, power in factors) for i in range(7))  # type: ignore[return-value]


_ENERGY = _times((_M, 1), (_L, 2), (_T, -2))
_POWER = _times((_ENERGY, 1), (_T, -1))
_VOLT = _times((_POWER, 1), (_I, -1))

# Symbols that take an SI prefix.
_PREFIXABLE: dict[str, Dimension] = {
    "m": _L,
    "g": _M,
    "s": _T,
    "A": _I,
    "K": (0, 0, 0, 0, 1, 0, 0),
    "mol": (0, 0, 0, 0, 0, 1, 0),
    "cd": (0, 0, 0, 0, 0, 0, 1),
    "Hz": _times((_T, -1)),
    "N": _times((_M, 1), (_L, 1), (_T, -2)),
    "Pa": _times((_M, 1), (_L, -1), (_T, -2)),
    "J": _ENERGY,
    "eV": _ENERGY,
    "W": _POWER,
    "C": _times((_I, 1), (_T, 1)),
    "V": _VOLT,
    "Ω": _times((_VOLT, 1), (_I, -1)),
    "ohm": _times((_VOLT, 1), (_I, -1)),
    "S": _times((_VOLT, -1), (_I, 1)),
    "F": _times((_I, 1), (_T, 1), (_VOLT, -1)),
    "Wb": _times((_VOLT, 1), (_T, 1)),
    "T": _times((_VOLT, 1), (_T, 1), (_L, -2)),
    "H": _times((_VOLT, 1), (_T, 1), (_I, -1)),
    "L": _times((_L, 3)),
    "pc": _L,
    "yr": _T,
}

# Symbols that do not, matched before any prefix is tried.
_PLAIN: dict[str, Dimension] = {
    "1": _NONE,
    "%": _NONE,
    "rad": _NONE,
    "sr": _NONE,
    "min": _T,
    "h": _T,
    "d": _T,
    "au": _L,
    "ly": _L,
}

# SI prefixes, two-letter ones first so that ``da`` is not read as ``d``.
_PREFIXES = ("da", "Q", "R", "Y", "Z", "E", "P", "T", "G", "M", "k", "h", "d", "c", "m", "µ", "μ", "u", "n", "p", "f")
_PREFIXES += ("a", "z", "y", "r", "q")

_FACTOR = re.compile(r"(?P<symbol>[^\W\d_]+|%|1)(?:\^(?P<power>[+-]?\d+))?")
_JOINERS = re.compile("[*\N{MIDDLE DOT}\N{DOT OPERATOR}\N{MULTIPLICATION SIGN}\\s]+")


def _symbol(symbol: str) -> Dimension | None:
    """Return the dimension of one symbol, with or without an SI prefix."""
    if symbol in _PLAIN:
        return _PLAIN[symbol]
    if symbol in _PREFIXABLE:
        return _PREFIXABLE[symbol]
    for prefix in _PREFIXES:
        if symbol.startswith(prefix) and symbol[len(prefix) :] in _PREFIXABLE:
            return _PREFIXABLE[symbol[len(prefix) :]]
    return None


def dimension(unit: str) -> Dimension | None:
    """Return the dimension of a unit written as text.

    Args:
        unit: The unit, such as ``Hz``, ``m/s`` or ``kg m^2 s^-2``.

    Returns:
        The powers of the seven SI base dimensions, or ``None`` if the unit cannot be read.
    """
    numerator, *denominators = unit.strip().split("/")
    if len(denominators) > 1:
        return None
    factors: list[tuple[Dimension, int]] = []
    for sign, part in ((1, numerator), *((-1, denominator) for denominator in denominators)):
        terms = [term for term in _JOINERS.split(part.strip()) if term]
        if not terms:
            return None
        for term in terms:
            match = _FACTOR.fullmatch(term)
            found = None if match is None else _symbol(match["symbol"])
            if found is None:
                return None
            factors.append((found, sign * int(match["power"] or 1)))  # type: ignore[index]
    return _times(*factors)


def agree(first: str, second: str) -> bool:
    """Tell whether two units measure the same thing.

    Args:
        first: A unit.
        second: Another unit.

    Returns:
        Whether they are the same text, or both have the same dimension. Two different units the
        registry cannot read do not agree, since nothing shows that they do.
    """
    if first == second:
        return True
    found = dimension(first)
    return found is not None and found == dimension(second)
