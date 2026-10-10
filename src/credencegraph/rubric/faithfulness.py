"""Faithfulness: whether a form value is found in the text of the node it describes.

A form restates the text in a structured way; it may not add what the text does not say. Numbers,
quantities and equation references must therefore appear in the text, and a ``verbatim`` value
must be a substring of it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal

from credencegraph.rubric.model import EQREF, NUMBER, QUANTITY, VERBATIM

_MINUS = "\N{MINUS SIGN}"
_SIGNS = "+-" + _MINUS
_GROUP = "(?:,|\N{THIN SPACE}|\N{NARROW NO-BREAK SPACE}|\\\\,)"
_TIMES = "\N{MULTIPLICATION SIGN}"

# A numeral: an optional sign that does not follow a word character or a dot, digits in groups of
# three or plain, an optional fraction, an optional exponent (1.2e-3, 1.2 x 10^-3 written with a
# multiplication sign, 1.2 \times 10^{-3}) and an optional percent sign.
_NUMERAL = re.compile(
    rf"""
    (?P<sign>(?<![\w.])[{_SIGNS}])?
    (?<![\w.])
    (?P<digits>\d{{1,3}}(?:{_GROUP}\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    (?:
        [eE](?P<e>[{_SIGNS}]?\d+)
        |\s*(?:{_TIMES}|\\times)\s*10\s*\^\s*(?:\{{\s*(?P<braced>[{_SIGNS}]?\d+)\s*\}}|(?P<bare>[{_SIGNS}]?\d+))
    )?
    (?P<percent>\s*\\?%)?
    """,
    re.VERBOSE,
)


def numerals(text: str) -> set[float]:
    """Return the values of every numeral in a text.

    A numeral followed by ``%`` contributes both its value and its value divided by 100. Values are
    computed in decimal arithmetic and rounded once, so ``1.1%`` gives exactly the float ``0.011``.

    Args:
        text: The text.

    Returns:
        The values.
    """
    values: set[float] = set()
    for match in _NUMERAL.finditer(text):
        digits = re.sub(_GROUP, "", match["digits"])
        value = Decimal(digits)
        exponent = match["e"] or match["braced"] or match["bare"]
        if exponent is not None:
            value = value.scaleb(int(exponent.replace(_MINUS, "-")))
        if match["sign"] in ("-", _MINUS):
            value = -value
        values.add(float(value))
        if match["percent"]:
            values.add(float(value / 100))
    return values


def has_numeral(text: str, number: float) -> bool:
    """Tell whether a number appears as a numeral in a text.

    Args:
        text: The text.
        number: The number.

    Returns:
        Whether some numeral of the text parses to the same float.
    """
    return float(number) in numerals(text)


def has_unit(text: str, unit: str) -> bool:
    r"""Tell whether a unit appears in a text as a token, delimited by whitespace or punctuation.

    The argument of ``\mathrm{…}``, ``\text{…}`` and ``\si{…}`` is delimited by its braces, so it
    counts as a token too.

    Args:
        text: The text.
        unit: The unit, such as ``Hz``.

    Returns:
        Whether the unit is a token of the text.
    """
    return re.search(rf"(?<![^\W_]){re.escape(unit)}(?![^\W_])", text) is not None


def has_eqref(text: str, label: str) -> bool:
    r"""Tell whether a text references an equation.

    Args:
        text: The text.
        label: An equation label, matched by ``\ref``, ``\eqref``, ``\cref`` or ``\Cref``, or a
            printed number, matched as ``(label)``.

    Returns:
        Whether the text references it.
    """
    commands = ("ref", "eqref", "cref", "Cref")
    return any(f"\\{command}{{{label}}}" in text for command in commands) or f"({label})" in text


def unfaithful(field_type: str, value: object, text: str) -> str | None:
    """Explain why a form value is not found in a text, or return ``None`` if it is.

    The value must already have its field type. Field types without a faithfulness rule
    (``enum``, ``bool`` and ``scope``) always pass.

    Args:
        field_type: The field type.
        value: The value.
        text: The node's text.

    Returns:
        What is missing from the text, or ``None``.
    """
    if field_type == VERBATIM:
        return None if value in text else f"{value!r} is not a substring of the text"
    if field_type == NUMBER:
        return None if has_numeral(text, value) else f"no numeral in the text equals {value!r}"  # type: ignore[arg-type]
    if field_type == QUANTITY:
        assert isinstance(value, Mapping)  # noqa: S101 - checked against the field type first
        missing = []
        if not has_numeral(text, value["value"]):
            missing.append(f"no numeral in the text equals {value['value']!r}")
        if not has_unit(text, value["unit"]):
            missing.append(f"the unit {value['unit']!r} is not a token of the text")
        return "; ".join(missing) or None
    if field_type == EQREF:
        return None if has_eqref(text, value) else f"the text does not reference equation {value!r}"  # type: ignore[arg-type]
    return None
