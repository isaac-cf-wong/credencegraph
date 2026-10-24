"""Structural diagnostics: checks on the graph itself, which need no inference."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from credencegraph.core.credence import Credence
from credencegraph.core.errors import ValidationError
from credencegraph.core.graph import Graph
from credencegraph.core.relation import ACYCLIC_TYPES, OBSERVES, STRENGTH_TYPES, SUPPORTS
from credencegraph.diagnostics.records import CORRELATED_SUPPORT, MISSING_PARAMETER, UNANCHORED, Finding
from credencegraph.semantics.compiler import inference_sets
from credencegraph.semantics.cpt import Term, true_probability

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
    them. An observation, the target of an ``observes`` relation, takes its probability from that
    relation and needs no base. A graph with any such finding cannot be compiled: there is no
    default credence.

    ``Relation`` already refuses to be built without a strength when its type needs one, so the
    second check only fires on a graph whose relations were altered after validation.

    Args:
        graph: The graph.

    Returns:
        One finding per variable without a base, then one per relation without a strength.
    """
    observed = {relation.target for relation in graph.relations.values() if relation.type == OBSERVES}
    findings: list[Finding] = []
    for root, ids in inference_sets(graph).items():
        if any(graph.nodes[node_id].base is not None or node_id in observed for node_id in ids):
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
    has a source or is the target of a ``requires``, ``supports``, ``refutes`` or ``observes``
    relation.

    Args:
        graph: The graph.

    Returns:
        One finding per unanchored variable, in insertion order.
    """
    has_parents = {relation.target for relation in graph.relations.values() if relation.type in ACYCLIC_TYPES}
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


def _lineage(node: str, parents: dict[str, set[str]], cache: dict[str, set[str]]) -> set[str]:
    """Return ``node`` and every variable it descends from through ``parents``, cached in ``cache``."""
    if node not in cache:
        seen = {node}
        stack = [node]
        while stack:
            for parent in parents.get(stack.pop(), ()):
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        cache[node] = seen
    return cache[node]


def _constant(base: Credence | None, terms: list[tuple[str, str, Fraction]], fixed: dict[str, int]) -> int | None:
    """Evaluate a variable that might always take the same value.

    Args:
        base: The variable's base, or None if it has none.
        terms: The type, parent and strength of each relation of nonzero strength into it.
        fixed: The value of each fixed variable.

    Returns:
        The variable's value, 0 or 1, if its table gives exactly that probability at every value of
        its parents that are not in ``fixed``, with those in ``fixed`` at their value; otherwise None.
    """
    if base is None:
        return None
    prior = Fraction(base.mean)
    axes = list(dict.fromkeys(parent for _, parent, _ in terms))
    rows = [Term(axes.index(parent), rtype, strength) for rtype, parent, strength in terms]
    known = {parent: fixed[parent] for parent in axes if parent in fixed}
    free = [parent for parent in axes if parent not in known]
    seen = None
    for assignment in product((0, 1), repeat=len(free)):
        known.update(zip(free, assignment, strict=True))
        value = true_probability(prior, rows, [known[parent] for parent in axes])
        if value not in (0, 1) or (seen is not None and value != seen):
            return None
        seen = value
    return int(seen)  # product yields at least the empty assignment, so seen is set


def _fixed_points(
    graph: Graph,
    members: dict[str, list[str]],
    parents: dict[str, set[str]],
    terms: dict[str, list[tuple[str, str, Fraction]]],
) -> dict[str, int]:
    """Find the variables that always take the same value, and that value.

    A variable is fixed when its table gives one probability, exactly 0 or 1, at every value of its
    parents that are not fixed, with the fixed ones at their value. A variable with no parents is
    fixed at a base of mean 0 or 1, and so is, for example, one of base 1 whose parents only support
    it: it is true with probability b P + 1 - P = 1 whatever they are. Parameters are taken at their
    means, in ``Fraction`` arithmetic, so no rounding makes or unmakes a fixed point. A variable on a
    cycle of relations, or below one, is not counted.

    Args:
        graph: The graph.
        members: The node ids of each variable, by representative.
        parents: The parents of each variable through relations of nonzero strength.
        terms: The type, parent and strength of each of those relations, by child.

    Returns:
        The value of each fixed variable, by representative.
    """
    children: dict[str, list[str]] = {}
    for child, own in parents.items():
        for parent in own:
            children.setdefault(parent, []).append(child)
    waiting = {representative: len(parents.get(representative, ())) for representative in members}
    ready = [representative for representative, count in waiting.items() if count == 0]
    fixed: dict[str, int] = {}
    while ready:
        representative = ready.pop()
        base = next((graph.nodes[i].base for i in members[representative] if graph.nodes[i].base is not None), None)
        value = _constant(base, terms.get(representative, []), fixed)
        if value is not None:
            fixed[representative] = value
        for child in children.get(representative, ()):
            waiting[child] -= 1
            if waiting[child] == 0:
                ready.append(child)
    return fixed


def correlated_support(graph: Graph, min_supports: int = DEFAULT_MIN_SUPPORTS) -> list[Finding]:
    """Find supports of one node that share a source but no common ancestor.

    The ``supports`` relations into a node combine as a noisy OR of independent reasons: ten of
    strength 0.2 into a node of base 0 give it 1 - 0.8^10, about 0.893. Reasons that come from one
    source are rarely independent, and nothing in the graph says so unless they share an ancestor. Two
    supporting variables share a source when a node of each is anchored in the same document, or
    points at the same node through an ``authored_by`` or ``derived_from`` annotation. A group of at
    least ``min_supports`` supporting variables of one node that share a source is reported when no
    variable is a common ancestor of them: an ancestor, through chains of ``requires``, ``supports``
    or ``refutes`` relations, of every one of them, or one of them that is an ancestor of all the
    others, as in ``p0 supports p1``, ``p0 supports claim``, ``p1 supports claim``. Either way the
    graph already makes them depend on one variable, so inference does not count them as independent.
    A relation of strength exactly 0 does not make a parent, nor a support: it leaves the child's
    probability the same whatever the parent's value. Nor is a fixed point a common ancestor, nor an
    ancestor through it: a variable that always takes the same value, whether it has no parents and
    a base of mean exactly 0 or 1 or has a table that gives it that value whatever its parents are,
    as with a base of 1 and only ``supports`` parents, correlates nothing and passes nothing on.
    The usual fix is a common parent: a proposition such as "this source's unverified reasoning
    is sound", which each of them requires with strength 1, so that they fail together. A node of
    base b with no other parents then gets at most b + (1 - b) c from the group, c the credence of
    the proposition, however large the group.

    Nodes merged by ``equivalent`` relations are one variable, named after the earliest of them, with
    the sources of all of them. A group that shares several sources is reported once, under the first.

    Several ``supports`` relations from one variable into the same node, for example one from each
    of two merged nodes, are separate terms of the noisy OR too: two of strength 0.2 lift a node of
    base 0 to 0.36 where one gives 0.2. At least ``min_supports`` of them of nonzero strength are
    reported whatever the variable's parents, since no common parent can correlate terms that
    already share their only input; the fix is one relation in their place.

    Args:
        graph: The graph.
        min_supports: The smallest group reported, at least 2; the default, 2, reports any two.

    Returns:
        One finding per node and group, the nodes in insertion order, and for each node its groups
        before its repeated supports. Its id is ``correlated-support:<node>:<kind>:<source>``, where
        the kind is ``document`` or the annotation type, or ``variable`` with the supporting variable
        as the source for supports repeated from one variable. Its nodes are the supported node, then
        the group or the variable; its relations are their ``supports`` relations of nonzero strength
        into the node. It has no value; its details are
        ``supports``, the size of the group or the number of relations, and ``min_supports``.

    Raises:
        ValidationError: If ``min_supports`` is not an integer >= 2.
    """
    count = _min_supports(min_supports)
    members = inference_sets(graph)
    root = {node_id: representative for representative, ids in members.items() for node_id in ids}
    provenance = _provenance(graph, members, root)
    parents: dict[str, set[str]] = {}
    lineage: dict[str, set[str]] = {}
    supports: dict[str, dict[str, list[str]]] = {}
    terms: dict[str, list[tuple[str, str, Fraction]]] = {}
    for relation in graph.relations.values():
        if relation.type in STRENGTH_TYPES:
            parent, child = root[relation.source], root[relation.target]
            # A term of strength 0 multiplies the child's table by 1 - 0 = 1 at both of the parent's
            # values, so it makes the parent no cause of the child; a Beta strength has a positive mean.
            if relation.strength is not None and relation.strength.mean > 0.0:
                parents.setdefault(child, set()).add(parent)
                terms.setdefault(child, []).append((relation.type, parent, Fraction(relation.strength.mean)))
                if relation.type == SUPPORTS:
                    supports.setdefault(child, {}).setdefault(parent, []).append(relation.id)
        elif relation.type == OBSERVES and relation.sensitivity.mean != relation.false_pass.mean:  # type: ignore[union-attr]
            # At the means, an observation depends on its claim unless it is as likely whichever value
            # the claim takes. It has no base, so it is never a fixed point and needs no terms.
            parents.setdefault(root[relation.target], set()).add(root[relation.source])
    fixed = _fixed_points(graph, members, parents, terms)
    # A fixed variable takes its value whatever its parents are, so nothing above it reaches its descendants.
    effective = {child: own for child, own in parents.items() if child not in fixed}
    findings: list[Finding] = []
    for target in members:
        by_parent = supports.get(target, {})
        sharing: dict[tuple[str, str], list[str]] = {}
        for parent in by_parent:
            for key in provenance[parent]:
                sharing.setdefault(key, []).append(parent)
        groups: dict[tuple[str, ...], list[tuple[str, str]]] = {}
        for key, group in sharing.items():
            common = set.intersection(*(_lineage(parent, effective, lineage) for parent in group))
            if len(group) >= count and not common - fixed.keys():
                groups.setdefault(tuple(group), []).append(key)
        for group, keys in groups.items():
            names = ", ".join(repr(parent) for parent in group)
            findings.append(
                Finding(
                    id=f"{CORRELATED_SUPPORT}:{target}:{keys[0][0]}:{keys[0][1]}",
                    diagnostic=CORRELATED_SUPPORT,
                    message=(
                        f"node {target!r} has {len(group)} supports, {names}, that share "
                        f"{' and '.join(_render(key) for key in keys)} but no common ancestor: inference counts "
                        "them as independent reasons; if they rest on one source's reasoning, add a proposition "
                        "for it that each of them requires"
                    ),
                    nodes=(target, *group),
                    relations=tuple(relation for parent in group for relation in by_parent[parent]),
                    details={"supports": float(len(group)), "min_supports": float(count)},
                )
            )
        findings.extend(
            Finding(
                id=f"{CORRELATED_SUPPORT}:{target}:variable:{parent}",
                diagnostic=CORRELATED_SUPPORT,
                message=(
                    f"node {target!r} has {len(relations)} supports from one variable, "
                    f"{_describe(members[parent])}: inference counts them as independent reasons; "
                    "if they are one reason, replace them with a single supports relation"
                ),
                nodes=(target, parent),
                relations=tuple(relations),
                details={"supports": float(len(relations)), "min_supports": float(count)},
            )
            for parent, relations in by_parent.items()
            if len(relations) >= count
        )
    return findings
