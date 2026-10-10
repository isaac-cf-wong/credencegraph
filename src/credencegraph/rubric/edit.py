"""Edits that classify a graph built from a document: splitting a chunk, and annotating a node.

Each edit returns a new graph and leaves the one it was given untouched, so a caller writes the
result only when every step succeeded.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from itertools import count, product
from string import ascii_lowercase
from typing import Any

from credencegraph.core.anchor import SourceAnchor
from credencegraph.core.attributes import copy_json
from credencegraph.core.errors import ValidationError
from credencegraph.core.graph import Graph
from credencegraph.core.node import Node
from credencegraph.core.relation import Relation
from credencegraph.rubric.model import COMPOUND, DOCUMENT, UNASSIGNED, Rubric
from credencegraph.rubric.text import text_digest
from credencegraph.rubric.validate import (
    CITES,
    FAILS,
    HOLDS,
    NOT_ASSESSED,
    PART_OF,
    UNDETERMINED,
    form_problems,
    node_text,
    origin_of,
)

_CONTENT = ("form", "assessment")


def _suffixes() -> Iterator[str]:
    """Yield ``a``, ``b``, …, ``z``, ``aa``, ``ab``, …: the suffixes of child ids, in order."""
    for length in count(1):
        for letters in product(ascii_lowercase, repeat=length):
            yield "".join(letters)


def _rebuilt(
    graph: Graph,
    nodes: Mapping[str, Node],
    after: str | None = None,
    new: Sequence[Node] = (),
    relations: Iterable[Relation] = (),
) -> Graph:
    """Copy a graph, replacing some nodes, inserting ``new`` nodes after ``after`` and adding relations.

    Args:
        graph: The graph, left unchanged.
        nodes: Replacement nodes by id.
        after: The id after which ``new`` is inserted.
        new: Nodes to insert.
        relations: Relations to append.

    Returns:
        The new graph.
    """
    out = Graph()
    for node in graph:
        out.add_node(nodes.get(node.id, node))
        if node.id == after:
            for child in new:
                out.add_node(child)
    for relation in [*graph.relations.values(), *relations]:
        out.add_relation(relation)
    return out


def _attributes(node: Node, **changes: Any) -> dict[str, Any]:
    """Return a node's attributes as plain JSON, with keys set, or removed where the value is ``None``."""
    attributes = copy_json(node.attributes, f"Node {node.id!r} attributes")
    for key, value in changes.items():
        if value is None:
            attributes.pop(key, None)
        else:
            attributes[key] = value
    return attributes


def _occurrences(text: str, clause: str) -> list[int]:
    """Return every start index of ``clause`` in ``text``, overlapping ones included."""
    starts = []
    start = text.find(clause)
    while start != -1:
        starts.append(start)
        start = text.find(clause, start + 1)
    return starts


def _check_splittable(graph: Graph, node: Node) -> None:
    """Refuse to split a node that cannot become a compound.

    Raises:
        ValidationError: If the node is not a top-level document chunk with a statement and one
            source with a locator, carries a form, an assessment or a base, or is an endpoint of a
            relation a compound may not have.
    """
    node_id = node.id
    if origin_of(node) != DOCUMENT:
        msg = f"node {node_id!r} is not a document node; only a chunk of the document can be split"
        raise ValidationError(msg)
    if any(relation.type == PART_OF and relation.source == node_id for relation in graph.relations.values()):
        msg = f"node {node_id!r} is a child of a compound, and a child is never itself a compound"
        raise ValidationError(msg)
    carried = [name for name in _CONTENT if name in node.attributes] + (["base"] if node.base is not None else [])
    if carried:
        msg = (
            f"node {node_id!r} carries {', '.join(carried)}, which a compound may not; "
            f"'annotate {node_id} --type unassigned' removes them"
        )
        raise ValidationError(msg)
    stray = [
        relation.id
        for relation in graph.relations.values()
        if (relation.target == node_id and relation.type != PART_OF)
        or (relation.source == node_id and relation.type != CITES)
    ]
    if stray:
        msg = f"node {node_id!r} is an endpoint of relations a compound may not have: {stray}"
        raise ValidationError(msg)
    if len(node.sources) != 1 or node.sources[0].locator is None or node.statement is None:
        msg = f"node {node_id!r} needs a statement and exactly one source with a locator to be split"
        raise ValidationError(msg)


def split_node(
    graph: Graph, node_id: str, clauses: Sequence[str] = (), *, fields: bool = False
) -> tuple[Graph, list[str]]:
    """Make a document chunk a compound, with one span child per clause or one field child.

    A span child's text is a clause of the chunk, its locator the chunk's plus ``;chars=<start>-<end>``.
    A field child has no text, a copy of the chunk's anchor, and gets its content from a form later.
    Children are ``<node id>a``, ``<node id>b``, … in the order created, joined to the chunk by a
    ``part-of`` relation, and start ``unassigned``. Splitting a compound adds children to it.

    Args:
        graph: The graph, left unchanged.
        node_id: The chunk to split.
        clauses: The clauses, each occurring exactly once in the chunk's text.
        fields: Add one field child instead of span children.

    Returns:
        The new graph, and the ids of the children added.

    Raises:
        ValidationError: If the node is not a top-level document chunk with a locator, carries a form,
            an assessment or a base, takes part in a relation a compound may not, or a clause does not
            occur exactly once in its text; or if both or neither of ``clauses`` and ``fields`` is given.
    """
    if bool(clauses) == fields:
        msg = "split needs either clauses or fields, not both"
        raise ValidationError(msg)
    node = graph.nodes[node_id]
    _check_splittable(graph, node)
    anchor = node.sources[0]
    spans = []
    for clause in clauses:
        starts = _occurrences(node.statement, clause) if clause else []
        if len(starts) != 1:
            msg = f"clause {clause!r} occurs {len(starts)} times in the text of {node_id!r}, not exactly once"
            raise ValidationError(msg)
        spans.append((starts[0], clause))
    unit = node.attributes.get("unit")
    base_attributes = {"origin": DOCUMENT, **({"unit": unit} if unit is not None else {})}
    suffixes = (suffix for suffix in _suffixes() if f"{node_id}{suffix}" not in graph)
    children = []
    for start, clause in spans or [(None, None)]:
        child_id = f"{node_id}{next(suffixes)}"
        if start is None:
            child = Node(child_id, kind=UNASSIGNED, sources=node.sources, attributes=dict(base_attributes))
        else:
            locator = f"{anchor.locator};chars={start}-{start + len(clause)}"
            source = SourceAnchor(anchor.document, locator, clause, text_digest(clause))
            child = Node(
                child_id, kind=UNASSIGNED, statement=clause, sources=(source,), attributes=dict(base_attributes)
            )
        children.append(child)
    relations = [Relation(f"{child.id}-{PART_OF}-{node_id}", PART_OF, child.id, node_id) for child in children]
    for relation in relations:
        if relation.id in graph.relations:
            msg = f"a relation with id {relation.id!r} already exists"
            raise ValidationError(msg)
    existing = [
        relation.source
        for relation in graph.relations.values()
        if relation.type == PART_OF and relation.target == node_id
    ]
    order = list(graph.nodes)
    after = max([node_id, *existing], key=order.index)
    compound = replace(node, kind=COMPOUND)
    return _rebuilt(graph, {node_id: compound}, after, children, relations), [child.id for child in children]


def parse_value(text: str) -> Any:
    """Read a form value from the command line: as JSON if it parses, else as the string itself.

    Args:
        text: The value as typed.

    Returns:
        The value.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def annotate_node(  # noqa: PLR0913 - one argument per kind of annotation
    graph: Graph,
    rubric: Rubric,
    node_id: str,
    *,
    node_type: str | None = None,
    settings: Mapping[str, Any] | None = None,
    rests_on: Sequence[str] = (),
    verdict: str | None = None,
    reason: str | None = None,
) -> Graph:
    """Type, fill in, link and assess a node, validating each write against the rubric.

    The steps run in that order, so a type given here is the one the form and links are checked
    against. Required form fields may be left out: they are filled one at a time, and
    ``check_graph`` at ``typed`` reports the ones still missing.

    Args:
        graph: The graph, left unchanged.
        rubric: The rubric.
        node_id: The node.
        node_type: A declared type allowed for the node's origin, or ``unassigned``, which removes
            the type with the node's form, assessment and base. Assigning a type with a ``base``
            sets it when the node has none.
        settings: Form fields to set, by name; each must be declared by the node's type and its value
            must have the field type and be found in the node's text.
        rests_on: Nodes to add the type's edge-rule relation from, with the rule's strength.
        verdict: ``holds``, ``fails`` or ``undetermined``.
        reason: Why the node was not assessed; sets the verdict ``not_assessed``.

    Returns:
        The new graph.

    Raises:
        ValidationError: If any write breaks the rubric, or a relation would close a cycle.
    """
    node = graph.nodes[node_id]
    origin = origin_of(node)
    if origin is None:
        msg = f"node {node_id!r} has no valid attributes.origin, so the rubric cannot apply to it"
        raise ValidationError(msg)
    if node.kind == COMPOUND:
        msg = f"node {node_id!r} is a compound; annotate its children instead"
        raise ValidationError(msg)
    if verdict is not None and reason is not None:
        msg = "give a verdict or a reason for not assessing, not both"
        raise ValidationError(msg)
    if node_type is not None:
        node = _typed(rubric, node, origin, node_type)
    if settings:
        node = _filled(graph, rubric, node, origin, settings)
    if verdict is not None or reason is not None:
        node = _assessed(rubric, node, verdict, reason)
    relations = _links(graph, rubric, node, rests_on)
    return _rebuilt(graph, {node_id: node}, relations=relations)


def _typed(rubric: Rubric, node: Node, origin: str, node_type: str) -> Node:
    """Give a node a type, or remove it with ``unassigned``."""
    if node_type == UNASSIGNED:
        if origin != DOCUMENT:
            msg = f"only document nodes may be unassigned; node {node.id!r} is {origin}"
            raise ValidationError(msg)
        return replace(node, kind=UNASSIGNED, base=None, attributes=_attributes(node, form=None, assessment=None))
    spec = rubric.types.get(node_type)
    if spec is None:
        msg = f"type {node_type!r} is not declared by rubric {rubric.name!r}; it declares {list(rubric.types)}"
        raise ValidationError(msg)
    if origin not in spec.origins:
        msg = f"type {node_type!r} allows origins {list(spec.origins)}, and node {node.id!r} is {origin}"
        raise ValidationError(msg)
    return replace(node, kind=node_type, base=node.base if node.base is not None else spec.base)


def _filled(graph: Graph, rubric: Rubric, node: Node, origin: str, settings: Mapping[str, Any]) -> Node:
    """Set form fields, each checked against its field type and the node's text."""
    if node.kind not in rubric.types:
        msg = f"node {node.id!r} has type {node.kind!r}; give it a declared type before filling in its form"
        raise ValidationError(msg)
    if origin != DOCUMENT:
        msg = f"only document nodes carry a form; node {node.id!r} is {origin}"
        raise ValidationError(msg)
    form = dict(copy_json(node.attributes.get("form", {}), "form"))
    form.update(settings)
    _, invalid, unfound = form_problems(rubric, node.kind, dict(settings), node_text(graph, node))
    problems = invalid + unfound
    if problems:
        msg = f"node {node.id!r}: {'; '.join(problems)}"
        raise ValidationError(msg)
    return replace(node, attributes=_attributes(node, form=form))


def _assessed(rubric: Rubric, node: Node, verdict: str | None, reason: str | None) -> Node:
    """Set a node's assessment."""
    if node.kind not in rubric.types:
        msg = f"node {node.id!r} has type {node.kind!r}; give it a declared type before assessing it"
        raise ValidationError(msg)
    if reason is not None:
        if not reason.strip():
            msg = "the reason for not assessing a node must not be empty"
            raise ValidationError(msg)
        assessment = {"verdict": NOT_ASSESSED, "reason": reason}
    elif verdict in (HOLDS, FAILS, UNDETERMINED):
        assessment = {"verdict": verdict}
    else:
        msg = f"verdict must be {HOLDS!r}, {FAILS!r} or {UNDETERMINED!r}, got {verdict!r}"
        raise ValidationError(msg)
    return replace(node, attributes=_attributes(node, assessment=assessment))


def _links(graph: Graph, rubric: Rubric, node: Node, sources: Sequence[str]) -> list[Relation]:
    """Build the edge-rule relations from each of ``sources`` into the node."""
    if not sources:
        return []
    spec = rubric.types.get(node.kind)
    rule = spec.rests_on if spec is not None else None
    if rule is None:
        msg = f"type {node.kind!r} of node {node.id!r} has no edge rule, so it rests on nothing"
        raise ValidationError(msg)
    relations = []
    for source in dict.fromkeys(sources):
        kind = graph.nodes[source].kind
        if kind not in rule.types:
            msg = f"node {source!r} has type {kind!r}; type {node.kind!r} rests on {list(rule.types)}"
            raise ValidationError(msg)
        relation_id = f"{source}-{rule.relation}-{node.id}"
        if relation_id in graph.relations:
            msg = f"a relation with id {relation_id!r} already exists"
            raise ValidationError(msg)
        relations.append(Relation(relation_id, rule.relation, source, node.id, strength=rule.strength))
    return relations
