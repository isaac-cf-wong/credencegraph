"""Structural diagnostics: checks on the graph itself, which need no inference."""

from __future__ import annotations

from credencegraph.core.errors import ValidationError
from credencegraph.core.graph import Graph
from credencegraph.core.relation import STRENGTH_TYPES, SUPPORTS
from credencegraph.diagnostics.records import CORRELATED_SUPPORT, MISSING_PARAMETER, UNANCHORED, Finding
from credencegraph.semantics.compiler import inference_sets

# Annotation types that name where a node's content came from: two nodes that point at the same node
# through one of them share a source, as two nodes anchored in the same document do.
PROVENANCE_TYPES = ("authored_by", "derived_from")

# The smallest group of supports reported by default: any two that share a source.
DEFAULT_MIN_SUPPORTS = 2


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


def _min_supports(value: object) -> int:
    """Validate the smallest group of supports worth reporting.

    Args:
        value: The candidate.

    Returns:
        The value.

    Raises:
        ValidationError: If ``value`` is not an integer >= 2.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:  # noqa: PLR2004 - a group is two or more
        msg = f"min_supports: must be an integer >= 2, got {value!r}"
        raise ValidationError(msg)
    return value


def _render(key: tuple[str, str]) -> str:
    """Write a shared source for a finding's message.

    Args:
        key: ``("document", <document>)``, or a provenance annotation type and the node it points at.

    Returns:
        ``document 'doi:10.1/x'`` or ``derived_from 'dataset'``.
    """
    return f"{key[0]} {key[1]!r}"


def _provenance(
    graph: Graph, members: dict[str, list[str]], root: dict[str, str]
) -> dict[str, dict[tuple[str, str], None]]:
    """Collect the sources each inference variable names.

    Args:
        graph: The graph.
        members: The node ids of each variable, by representative.
        root: The representative of each node id that is an inference variable.

    Returns:
        For each representative, its sources as keys in first-seen order: ``("document", <document>)``
        for each anchor of a member, then ``(<annotation type>, <node>)`` for each provenance annotation
        from a member, the node named by its representative when it is an inference variable.
    """
    provenance: dict[str, dict[tuple[str, str], None]] = {representative: {} for representative in members}
    for representative, ids in members.items():
        for node_id in ids:
            for anchor in graph.nodes[node_id].sources:
                provenance[representative][("document", anchor.document)] = None
    for relation in graph.relations.values():
        if relation.type in PROVENANCE_TYPES and relation.source in root:
            key = (relation.type, root.get(relation.target, relation.target))
            provenance[root[relation.source]][key] = None
    return provenance


def correlated_support(graph: Graph, min_supports: int = DEFAULT_MIN_SUPPORTS) -> list[Finding]:
    """Find supports of one node that share a source but no common parent.

    The ``supports`` relations into a node combine as a noisy OR of independent reasons: ten of
    strength 0.2 into a node of base 0 give it 1 - 0.8^10, about 0.893. Reasons that come from one
    source are rarely independent, and nothing in the graph says so unless they share a parent. Two
    supporting variables share a source when a node of each is anchored in the same document, or
    points at the same node through an ``authored_by`` or ``derived_from`` annotation. A group of at
    least ``min_supports`` supporting variables of one node that share a source is reported when no
    variable is a parent, through ``requires``, ``supports`` or ``refutes``, of every one of them. A
    relation of strength exactly 0 does not make a parent: it leaves the child's probability the same
    whatever the parent's value.
    The usual fix is that common parent: a proposition such as "this source's unverified reasoning
    is sound", which each of them requires with strength 1, so that they fail together. A node of
    base b with no other parents then gets at most b + (1 - b) c from the group, c the credence of
    the proposition, however large the group.

    Nodes merged by ``equivalent`` relations are one variable, named after the earliest of them, with
    the sources of all of them. A group that shares several sources is reported once, under the first.

    Args:
        graph: The graph.
        min_supports: The smallest group reported, at least 2; the default, 2, reports any two.

    Returns:
        One finding per node and group, the nodes in insertion order. Its id is
        ``correlated-support:<node>:<kind>:<source>``, where the kind is ``document`` or the annotation
        type. Its nodes are the supported node, then the group; its relations are the group's
        ``supports`` relations into the node. It has no value; its details are ``supports``, the size
        of the group, and ``min_supports``.

    Raises:
        ValidationError: If ``min_supports`` is not an integer >= 2.
    """
    count = _min_supports(min_supports)
    members = inference_sets(graph)
    root = {node_id: representative for representative, ids in members.items() for node_id in ids}
    provenance = _provenance(graph, members, root)
    parents: dict[str, set[str]] = {}
    supports: dict[str, dict[str, list[str]]] = {}
    for relation in graph.relations.values():
        if relation.type in STRENGTH_TYPES:
            parent, child = root[relation.source], root[relation.target]
            # A term of strength 0 multiplies the child's table by 1 - 0 = 1 at both of the parent's
            # values, so it makes the parent no cause of the child; a Beta strength has a positive mean.
            if relation.strength is not None and relation.strength.mean > 0.0:
                parents.setdefault(child, set()).add(parent)
            if relation.type == SUPPORTS:
                supports.setdefault(child, {}).setdefault(parent, []).append(relation.id)
    findings: list[Finding] = []
    for target in members:
        by_parent = supports.get(target, {})
        sharing: dict[tuple[str, str], list[str]] = {}
        for parent in by_parent:
            for key in provenance[parent]:
                sharing.setdefault(key, []).append(parent)
        groups: dict[tuple[str, ...], list[tuple[str, str]]] = {}
        for key, group in sharing.items():
            if len(group) >= count and not set.intersection(*(parents.get(parent, set()) for parent in group)):
                groups.setdefault(tuple(group), []).append(key)
        for group, keys in groups.items():
            names = ", ".join(repr(parent) for parent in group)
            findings.append(
                Finding(
                    id=f"{CORRELATED_SUPPORT}:{target}:{keys[0][0]}:{keys[0][1]}",
                    diagnostic=CORRELATED_SUPPORT,
                    message=(
                        f"node {target!r} has {len(group)} supports, {names}, that share "
                        f"{' and '.join(_render(key) for key in keys)} but no common parent: inference counts "
                        "them as independent reasons; if they rest on one source's reasoning, add a proposition "
                        "for it that each of them requires"
                    ),
                    nodes=(target, *group),
                    relations=tuple(relation for parent in group for relation in by_parent[parent]),
                    details={"supports": float(len(group)), "min_supports": float(count)},
                )
            )
    return findings
