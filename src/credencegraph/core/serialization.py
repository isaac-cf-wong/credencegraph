"""Versioned JSON serialisation of graphs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from credencegraph.core.anchor import SourceAnchor
from credencegraph.core.attributes import copy_json
from credencegraph.core.credence import Beta, Credence, Point
from credencegraph.core.errors import CycleError, ValidationError
from credencegraph.core.graph import Graph
from credencegraph.core.node import DEFAULT_KIND, Node
from credencegraph.core.relation import Relation

FORMAT = "credencegraph"
VERSION = 1
SCHEMA_RESOURCE = "graph-1.schema.json"

_ANCHOR_FIELDS = {"document", "locator", "quote", "digest"}
_NODE_FIELDS = {"id", "kind", "statement", "sources", "base", "stated", "attributes"}
_RELATION_FIELDS = {"id", "type", "source", "target", "strength", "attributes"}
_GRAPH_FIELDS = {"format", "version", "nodes", "relations"}


def credence_to_json(credence: Credence) -> float | dict[str, float]:
    """Encode a credence: a ``Point`` as a bare number, a ``Beta`` as ``{"alpha", "beta"}``.

    Args:
        credence: The credence.

    Returns:
        The JSON-ready value.
    """
    if isinstance(credence, Point):
        return credence.p
    return {"alpha": credence.alpha, "beta": credence.beta}


def graph_to_dict(graph: Graph) -> dict[str, Any]:
    """Convert a graph to a JSON-ready dictionary in the current format version.

    Args:
        graph: The graph.

    Returns:
        A dictionary with ``format``, ``version``, ``nodes`` and ``relations``.
    """
    return {
        "format": FORMAT,
        "version": VERSION,
        "nodes": [_node_to_dict(node) for node in graph.nodes.values()],
        "relations": [_relation_to_dict(rel) for rel in graph.relations.values()],
    }


def _optional_credence(credence: Credence | None) -> float | dict[str, float] | None:
    return None if credence is None else credence_to_json(credence)


def _node_to_dict(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "kind": node.kind,
        "statement": node.statement,
        "sources": [
            {"document": a.document, "locator": a.locator, "quote": a.quote, "digest": a.digest} for a in node.sources
        ],
        "base": _optional_credence(node.base),
        "stated": _optional_credence(node.stated),
        "attributes": copy_json(node.attributes, "attributes"),
    }


def _relation_to_dict(rel: Relation) -> dict[str, Any]:
    return {
        "id": rel.id,
        "type": rel.type,
        "source": rel.source,
        "target": rel.target,
        "strength": _optional_credence(rel.strength),
        "attributes": copy_json(rel.attributes, "attributes"),
    }


def _mapping(value: Any, path: str, allowed: set[str], required: set[str]) -> Mapping[str, Any]:
    """Check that ``value`` is an object with only known keys, and all the required ones."""
    if not isinstance(value, dict):
        msg = f"{path} must be an object, got {type(value).__name__}"
        raise ValidationError(msg)
    unknown = sorted(set(value) - allowed, key=str)
    if unknown:
        msg = f"{path} has unknown field(s) {unknown}"
        raise ValidationError(msg)
    missing = sorted(required - set(value))
    if missing:
        msg = f"{path} is missing required field(s) {missing}"
        raise ValidationError(msg)
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        msg = f"{path} must be an array, got {type(value).__name__}"
        raise ValidationError(msg)
    return value


def _credence_from_json(value: Any, path: str) -> Credence | None:
    if value is None:
        return None
    if isinstance(value, dict):
        fields = _mapping(value, path, {"alpha", "beta"}, {"alpha", "beta"})
        return Beta(fields["alpha"], fields["beta"])
    return Point(value)


def _build(path: str, build: Callable[[], Any]) -> Any:
    """Run ``build``, prefixing any validation error with the location of the offending item."""
    try:
        return build()
    except CycleError:
        raise
    except ValidationError as exc:
        if str(exc).startswith(path):
            raise
        msg = f"{path}: {exc}"
        raise ValidationError(msg) from exc


def _anchor_from_dict(value: Any, path: str) -> SourceAnchor:
    fields = _mapping(value, path, _ANCHOR_FIELDS, {"document"})
    return SourceAnchor(**fields)


def _node_from_dict(value: Any, path: str) -> Node:
    fields = _mapping(value, path, _NODE_FIELDS, {"id"})
    sources = tuple(
        _anchor_from_dict(item, f"{path}.sources[{i}]")
        for i, item in enumerate(_array(fields.get("sources", []), f"{path}.sources"))
    )
    return Node(
        id=fields["id"],
        kind=fields.get("kind", DEFAULT_KIND),
        statement=fields.get("statement"),
        sources=sources,
        base=_credence_from_json(fields.get("base"), f"{path}.base"),
        stated=_credence_from_json(fields.get("stated"), f"{path}.stated"),
        attributes=fields.get("attributes", {}),
    )


def _relation_from_dict(value: Any, path: str) -> Relation:
    fields = _mapping(value, path, _RELATION_FIELDS, {"id", "type", "source", "target"})
    return Relation(
        id=fields["id"],
        type=fields["type"],
        source=fields["source"],
        target=fields["target"],
        strength=_credence_from_json(fields.get("strength"), f"{path}.strength"),
        attributes=fields.get("attributes", {}),
    )


def graph_from_dict(data: Any) -> Graph:
    """Build a graph from a dictionary in the ``credencegraph`` JSON format.

    Nodes are added before relations, in document order, through the same validated writes as
    any other graph, so a cycle in the file is rejected and named.

    Args:
        data: The decoded JSON document.

    Returns:
        The graph.

    Raises:
        ValidationError: If the document is malformed, has the wrong format or an unsupported version, or
            breaks a rule of the data model. The message locates the offending item.
        CycleError: If the relations form a cycle among ``requires``, ``supports`` and ``refutes``.
    """
    document = _mapping(data, "document", _GRAPH_FIELDS, {"format", "version", "nodes", "relations"})
    if document["format"] != FORMAT:
        msg = f"document format must be {FORMAT!r}, got {document['format']!r}"
        raise ValidationError(msg)
    version = document["version"]
    if isinstance(version, bool) or version != VERSION:
        msg = f"unsupported {FORMAT} version {version!r}; this release reads version {VERSION}"
        raise ValidationError(msg)
    graph = Graph()
    for i, item in enumerate(_array(document["nodes"], "nodes")):
        path = f"nodes[{i}]"
        _build(path, lambda item=item, path=path: graph.add_node(_node_from_dict(item, path)))
    for i, item in enumerate(_array(document["relations"], "relations")):
        path = f"relations[{i}]"
        _build(path, lambda item=item, path=path: graph.add_relation(_relation_from_dict(item, path)))
    return graph


def dumps(graph: Graph, *, indent: int | None = 2) -> str:
    """Serialise a graph to a JSON string.

    Args:
        graph: The graph.
        indent: JSON indentation, or ``None`` for a single line.

    Returns:
        The JSON text.
    """
    return json.dumps(graph_to_dict(graph), indent=indent, allow_nan=False)


def loads(text: str) -> Graph:
    """Parse a graph from a JSON string.

    Args:
        text: The JSON text.

    Returns:
        The graph.

    Raises:
        ValidationError: If the text is not valid JSON, or as ``graph_from_dict``.
        CycleError: As ``graph_from_dict``.
    """
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        msg = f"not valid JSON: {exc}"
        raise ValidationError(msg) from exc
    return graph_from_dict(data)


def _reject_constant(name: str) -> None:
    msg = f"not valid JSON: {name} is not allowed"
    raise ValidationError(msg)


def dump(graph: Graph, path: str | Path, *, indent: int | None = 2) -> None:
    """Write a graph to a JSON file.

    Args:
        graph: The graph.
        path: Destination file.
        indent: JSON indentation, or ``None`` for a single line.
    """
    Path(path).write_text(dumps(graph, indent=indent) + "\n", encoding="utf-8")


def load(path: str | Path) -> Graph:
    """Read a graph from a JSON file.

    Args:
        path: Source file.

    Returns:
        The graph.

    Raises:
        ValidationError: As ``loads``.
        CycleError: As ``loads``.
    """
    return loads(Path(path).read_text(encoding="utf-8"))


def json_schema() -> dict[str, Any]:
    """Return the JSON Schema for the current format version.

    The schema covers structure and types. Rules it cannot express, namely unique ids, relations
    naming existing nodes, acyclicity, and the ban on an ``equivalent`` or ``exclusive`` relation
    joining a node to itself, are enforced when a graph is loaded.

    Returns:
        The schema as a dictionary.
    """
    text = resources.files("credencegraph.core").joinpath("schema", SCHEMA_RESOURCE).read_text(encoding="utf-8")
    return json.loads(text)
