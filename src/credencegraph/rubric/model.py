"""Rubrics: the node types of a graph built from a document, their forms and their edge rules.

A rubric is a TOML file, read with the standard library. Every key is checked, so a misspelt rule
is an error rather than a rule silently ignored.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from credencegraph.core.credence import Credence, as_credence, parse_credence
from credencegraph.core.errors import CredenceGraphError, ValidationError
from credencegraph.core.relation import REQUIRES, SUPPORTS

RUBRIC_FORMAT = 1

# The origins of a node: a chunk of the document, an implicit premise, or evidence about another node.
DOCUMENT = "document"
ANALYST = "analyst"
EVIDENCE = "evidence"
ORIGINS = (DOCUMENT, ANALYST, EVIDENCE)

# The built-in types, which a rubric may not declare.
UNASSIGNED = "unassigned"
COMPOUND = "compound"
BUILTIN_TYPES = (UNASSIGNED, COMPOUND)

# The field types of a form.
VERBATIM = "verbatim"
NUMBER = "number"
QUANTITY = "quantity"
EQREF = "eqref"
ENUM = "enum"
BOOL = "bool"
SCOPE = "scope"
FIELD_TYPES = (VERBATIM, NUMBER, QUANTITY, EQREF, ENUM, BOOL, SCOPE)


class RubricError(CredenceGraphError):
    """A rubric file is malformed, of an unknown format, or breaks a rule of the rubric format."""


@dataclass(frozen=True, slots=True)
class Parameter:
    """A parameter that ``scope`` fields may range over.

    Attributes:
        name: The parameter's name.
        unit: Its unit, for readers.
        domain: ``(low, high)``; ``None`` is an open end.
    """

    name: str
    unit: str
    domain: tuple[float | None, float | None] = (None, None)


@dataclass(frozen=True, slots=True)
class FormField:
    """A field of a type's form.

    Attributes:
        name: The field's name.
        type: One of ``FIELD_TYPES``.
        required: Whether a typed node must have it.
        values: The allowed strings of an ``enum`` field; empty otherwise.
    """

    name: str
    type: str
    required: bool = False
    values: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RestsOn:
    """A type's edge rule: what a node of the type must rest on.

    Attributes:
        types: The types a supporting node may have.
        min: How many supporting relations the node needs at least.
        relation: ``requires`` or ``supports``: the type of those relations, from the supporting node.
        strength: The strength ``annotate --rests-on`` gives a relation it creates.
    """

    types: tuple[str, ...]
    relation: str
    strength: Credence
    min: int = 1


@dataclass(frozen=True, slots=True)
class NodeType:
    """A node type declared by a rubric.

    Attributes:
        name: The type's name, stored as a node's ``kind``.
        description: What the type is for, shown in reports.
        origins: The origins the type may be used with.
        assess: Whether a node of the type needs an assessment at the ``assessed`` level.
        base: The base ``annotate`` gives a node it types, when the node has none.
        form: The form's fields by name, in declaration order.
        rests_on: The edge rule, if any.
    """

    name: str
    description: str = ""
    origins: tuple[str, ...] = (DOCUMENT,)
    assess: bool = False
    base: Credence | None = None
    form: Mapping[str, FormField] = field(default_factory=dict)
    rests_on: RestsOn | None = None


@dataclass(frozen=True, slots=True)
class Rubric:
    """A rubric: node types, their forms and edge rules, and the parameters scopes range over.

    Attributes:
        name: Identifies the rubric in reports.
        version: The rubric's own version, chosen by its author.
        types: The declared types by name, in declaration order.
        parameters: The parameters by name.
        initial: The type each chunk unit starts with, by unit.
    """

    name: str
    version: str
    types: Mapping[str, NodeType]
    parameters: Mapping[str, Parameter] = field(default_factory=dict)
    initial: Mapping[str, str] = field(default_factory=dict)

    def identity(self) -> dict[str, str]:
        """Return the rubric's name and version, as echoed in every report.

        Returns:
            ``{"name", "version"}``.
        """
        return {"name": self.name, "version": self.version}


def _table(
    value: Any, path: str, allowed: set[str] | None = None, required: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Check that ``value`` is a table with only ``allowed`` keys (any, if ``None``) and every ``required`` one."""
    if not isinstance(value, dict):
        msg = f"{path} must be a table, got {type(value).__name__}"
        raise RubricError(msg)
    unknown = [] if allowed is None else sorted(set(value) - allowed)
    if unknown:
        msg = f"{path} has unknown keys {unknown}; the allowed keys are {sorted(allowed)}"
        raise RubricError(msg)
    missing = sorted(required - set(value))
    if missing:
        msg = f"{path} is missing {missing}"
        raise RubricError(msg)
    return value


def _string(value: Any, path: str) -> str:
    """Check that ``value`` is a non-empty string."""
    if not isinstance(value, str) or not value:
        msg = f"{path} must be a non-empty string, got {value!r}"
        raise RubricError(msg)
    return value


def _strings(value: Any, path: str) -> tuple[str, ...]:
    """Check that ``value`` is a non-empty array of distinct non-empty strings."""
    if not isinstance(value, list) or not value:
        msg = f"{path} must be a non-empty array of strings, got {value!r}"
        raise RubricError(msg)
    items = tuple(_string(item, f"{path}[{i}]") for i, item in enumerate(value))
    if len(set(items)) != len(items):
        msg = f"{path} lists a value twice: {list(items)}"
        raise RubricError(msg)
    return items


def _bool(value: Any, path: str) -> bool:
    """Check that ``value`` is a boolean."""
    if not isinstance(value, bool):
        msg = f"{path} must be true or false, got {value!r}"
        raise RubricError(msg)
    return value


def _credence(value: Any, path: str) -> Credence:
    """Read a credence written as a probability or as text such as ``beta:8,2``."""
    try:
        if isinstance(value, str):
            return parse_credence(value)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return as_credence(float(value))
    except ValidationError as error:
        msg = f"{path}: {error}"
        raise RubricError(msg) from None
    msg = f"{path} must be a probability such as 0.3 or a string such as 'beta:8,2', got {value!r}"
    raise RubricError(msg)


def _domain(value: Any, path: str) -> tuple[float | None, float | None]:
    """Read a parameter domain ``[low, high]``, where ``-inf`` and ``inf`` are open ends."""
    if not isinstance(value, list) or len(value) != 2:  # noqa: PLR2004 - low and high
        msg = f"{path} must be [low, high], got {value!r}"
        raise RubricError(msg)
    ends: list[float | None] = []
    for end, item in zip(("low", "high"), value, strict=True):
        if isinstance(item, bool) or not isinstance(item, int | float) or math.isnan(item):
            msg = f"{path} {end} must be a number, or -inf or inf for an open end, got {item!r}"
            raise RubricError(msg)
        ends.append(None if math.isinf(item) else float(item))
    if value[0] > value[1]:
        msg = f"{path} has low {value[0]!r} above high {value[1]!r}"
        raise RubricError(msg)
    return ends[0], ends[1]


def _form_field(name: str, value: Any, path: str) -> FormField:
    """Read one form field."""
    table = _table(value, path, {"type", "required", "values"}, frozenset({"type"}))
    field_type = table["type"]
    if field_type not in FIELD_TYPES:
        msg = f"{path}.type must be one of {list(FIELD_TYPES)}, got {field_type!r}"
        raise RubricError(msg)
    if (field_type == ENUM) != ("values" in table):
        msg = f"{path}: an enum field needs values, and only an enum field takes them"
        raise RubricError(msg)
    values = _strings(table["values"], f"{path}.values") if "values" in table else ()
    return FormField(name, field_type, _bool(table.get("required", False), f"{path}.required"), values)


def _rests_on(value: Any, path: str) -> RestsOn:
    """Read an edge rule; the types it names are checked once every type is known."""
    table = _table(value, path, {"types", "min", "relation", "strength"}, frozenset({"types", "relation", "strength"}))
    minimum = table.get("min", 1)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        msg = f"{path}.min must be an integer of 1 or more, got {minimum!r}"
        raise RubricError(msg)
    relation = table["relation"]
    if relation not in (REQUIRES, SUPPORTS):
        msg = f"{path}.relation must be {REQUIRES!r} or {SUPPORTS!r}, got {relation!r}"
        raise RubricError(msg)
    return RestsOn(
        _strings(table["types"], f"{path}.types"), relation, _credence(table["strength"], f"{path}.strength"), minimum
    )


def _node_type(name: str, value: Any) -> NodeType:
    """Read one type declaration."""
    path = f"types.{name}"
    if name in BUILTIN_TYPES:
        msg = f"{path}: {name!r} is built in and may not be declared"
        raise RubricError(msg)
    table = _table(value, path, {"description", "origins", "assess", "base", "form", "rests_on"})
    origins = _strings(table.get("origins", [DOCUMENT]), f"{path}.origins")
    for origin in origins:
        if origin not in ORIGINS:
            msg = f"{path}.origins: {origin!r} is not one of {list(ORIGINS)}"
            raise RubricError(msg)
    form_table = _table(table.get("form", {}), f"{path}.form")
    form = {key: _form_field(key, item, f"{path}.form.{key}") for key, item in form_table.items()}
    return NodeType(
        name,
        _string(table["description"], f"{path}.description") if "description" in table else "",
        origins,
        _bool(table.get("assess", False), f"{path}.assess"),
        _credence(table["base"], f"{path}.base") if "base" in table else None,
        MappingProxyType(form),
        _rests_on(table["rests_on"], f"{path}.rests_on") if "rests_on" in table else None,
    )


def rubric_from_dict(data: Any) -> Rubric:
    """Build a rubric from the data of a rubric file.

    Args:
        data: The parsed TOML document.

    Returns:
        The rubric.

    Raises:
        RubricError: If the data breaks a rule of the rubric format, or its format is unknown.
    """
    top = _table(data, "the rubric", {"rubric", "parameters", "initial", "types"}, frozenset({"rubric", "types"}))
    header = _table(top["rubric"], "rubric", {"format", "name", "version"}, frozenset({"format", "name", "version"}))
    if type(header["format"]) is not int or header["format"] != RUBRIC_FORMAT:
        msg = f"rubric.format {header['format']!r} is not a known rubric format; this version reads {RUBRIC_FORMAT}"
        raise RubricError(msg)
    parameters = {}
    for name, value in _table(top.get("parameters", {}), "parameters").items():
        table = _table(value, f"parameters.{name}", {"unit", "domain"}, frozenset({"unit"}))
        domain = _domain(table["domain"], f"parameters.{name}.domain") if "domain" in table else (None, None)
        parameters[name] = Parameter(name, _string(table["unit"], f"parameters.{name}.unit"), domain)
    types_table = _table(top["types"], "types")
    types = {name: _node_type(name, value) for name, value in types_table.items()}
    for node_type in types.values():
        rule = node_type.rests_on
        for name in rule.types if rule else ():
            if name not in types:
                msg = f"types.{node_type.name}.rests_on.types names {name!r}, which is not a declared type"
                raise RubricError(msg)
    initial = {}
    for unit, name in _table(top.get("initial", {}), "initial").items():
        if not isinstance(name, str) or name not in types or DOCUMENT not in types[name].origins:
            msg = f"initial.{unit} names {name!r}, which is not a declared type allowed for document nodes"
            raise RubricError(msg)
        initial[unit] = name
    return Rubric(
        _string(header["name"], "rubric.name"),
        _string(header["version"], "rubric.version"),
        MappingProxyType(types),
        MappingProxyType(parameters),
        MappingProxyType(initial),
    )


def loads_rubric(text: str) -> Rubric:
    """Read a rubric from TOML text.

    Args:
        text: The TOML document.

    Returns:
        The rubric.

    Raises:
        RubricError: If the text is not TOML or breaks a rule of the rubric format.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        msg = f"not valid TOML: {error}"
        raise RubricError(msg) from None
    return rubric_from_dict(data)


def load_rubric(path: str | Path) -> Rubric:
    """Read a rubric file.

    Args:
        path: The TOML file.

    Returns:
        The rubric.

    Raises:
        RubricError: If the file is not TOML or breaks a rule of the rubric format.
        OSError: If the file cannot be read.
    """
    return loads_rubric(Path(path).read_text(encoding="utf-8"))
