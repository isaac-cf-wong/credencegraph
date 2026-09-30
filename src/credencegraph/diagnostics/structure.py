"""Structural diagnostics: checks on the graph itself, which need no inference."""

from __future__ import annotations

from credencegraph.core.graph import Graph
from credencegraph.core.relation import STRENGTH_TYPES
from credencegraph.diagnostics.records import MISSING_PARAMETER, UNANCHORED, Finding
from credencegraph.semantics.compiler import inference_sets


def _describe(ids: list[str]) -> str:
    """Name a set of nodes merged by ``equivalent`` relations.

    Args:
        ids: The node ids, representative first.

    Returns:
        ``'a'``, or ``'a' (merged with 'b', 'c')``.
    """
    if len(ids) == 1:
        return repr(ids[0])
    return f"{ids[0]!r} (merged with {', '.join(repr(i) for i in ids[1:])})"


def missing_parameters(graph: Graph) -> list[Finding]:
    """Find inference variables without a ``base`` and inferential relations without a ``strength``.

    Nodes merged by ``equivalent`` relations form one variable, which needs a base on at least one of
    them. A graph with any such finding cannot be compiled: there is no default credence.

    ``Relation`` already refuses to be built without a strength when its type needs one, so the
    second check only fires on a graph whose relations were altered after validation.

    Args:
        graph: The graph.

    Returns:
        One finding per variable without a base, then one per relation without a strength.
    """
    findings: list[Finding] = []
    for root, ids in inference_sets(graph).items():
        if any(graph.nodes[node_id].base is not None for node_id in ids):
            continue
        findings.append(
            Finding(
                id=f"{MISSING_PARAMETER}:base:{root}",
                diagnostic=MISSING_PARAMETER,
                message=f"node {_describe(ids)} is an inference variable without a base; there is no default credence",
                nodes=tuple(ids),
            )
        )
    for relation in graph.relations.values():
        if relation.type in STRENGTH_TYPES and relation.strength is None:
            findings.append(
                Finding(
                    id=f"{MISSING_PARAMETER}:strength:{relation.id}",
                    diagnostic=MISSING_PARAMETER,
                    message=f"{relation.type} relation {relation.id!r} has no strength",
                    nodes=(relation.source, relation.target),
                    relations=(relation.id,),
                )
            )
    return findings


def unanchored(graph: Graph) -> list[Finding]:
    """Find inference variables with no source and no inferential parents.

    Such a variable is an assumption nobody wrote down: nothing in the graph says where its credence
    comes from. Nodes merged by ``equivalent`` relations are one variable, anchored if any of them
    has a source or is the target of a ``requires``, ``supports`` or ``refutes`` relation.

    Args:
        graph: The graph.

    Returns:
        One finding per unanchored variable, in insertion order.
    """
    has_parents = {relation.target for relation in graph.relations.values() if relation.type in STRENGTH_TYPES}
    findings: list[Finding] = []
    for root, ids in inference_sets(graph).items():
        if any(graph.nodes[node_id].sources or node_id in has_parents for node_id in ids):
            continue
        findings.append(
            Finding(
                id=f"{UNANCHORED}:{root}",
                diagnostic=UNANCHORED,
                message=f"node {_describe(ids)} has no source and no inferential parents: an assumption nobody wrote down",
                nodes=tuple(ids),
            )
        )
    return findings
