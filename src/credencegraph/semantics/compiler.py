"""Compile a graph into a network."""

from __future__ import annotations

from collections.abc import Iterable

from credencegraph.core.credence import Credence
from credencegraph.core.graph import Graph
from credencegraph.core.relation import EQUIVALENT, EXCLUSIVE, STRENGTH_TYPES, is_inferential
from credencegraph.semantics.errors import CompileError
from credencegraph.semantics.network import CONSTRAINT, PROPOSITION, Link, Network, ParameterKey, Variable


class _UnionFind:
    """Disjoint sets of node ids, each represented by its earliest member."""

    def __init__(self, items: Iterable[str]) -> None:
        """Start with every item in a set of its own.

        Args:
            items: The items, in the order that decides representatives.
        """
        self._parent: dict[str, str] = {}
        self._order: dict[str, int] = {}
        for position, item in enumerate(items):
            self._parent[item] = item
            self._order[item] = position

    def find(self, item: str) -> str:
        """Return the representative of ``item``'s set.

        Args:
            item: An item.

        Returns:
            The earliest member of its set.
        """
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        """Merge the sets of ``a`` and ``b``.

        Args:
            a: An item.
            b: Another item.
        """
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self._order[root_b] < self._order[root_a]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a


def _inference_nodes(graph: Graph) -> list[str]:
    """List the inference variables of a graph, in insertion order.

    Args:
        graph: The graph.

    Returns:
        Ids of the nodes that have a base or are an endpoint of an inferential relation.
    """
    ids = {node.id for node in graph if node.base is not None}
    for relation in graph.relations.values():
        if is_inferential(relation.type):
            ids.update((relation.source, relation.target))
    return [node.id for node in graph if node.id in ids]


def _merge(graph: Graph, node_ids: list[str]) -> tuple[_UnionFind, dict[str, list[str]]]:
    """Merge nodes joined by ``equivalent`` relations.

    Args:
        graph: The graph.
        node_ids: Its inference variables.

    Returns:
        The union-find structure and the members of each set by representative, both in insertion order.
    """
    sets = _UnionFind(node_ids)
    for relation in graph.relations.values():
        if relation.type == EQUIVALENT:
            sets.union(relation.source, relation.target)
    members: dict[str, list[str]] = {}
    for node_id in node_ids:
        members.setdefault(sets.find(node_id), []).append(node_id)
    return sets, members


def _bases(graph: Graph, members: dict[str, list[str]]) -> dict[str, str]:
    """Pick the node whose base each merged set uses.

    Args:
        graph: The graph.
        members: Members of each set by representative.

    Returns:
        For each representative, the id of the earliest member that has a base.

    Raises:
        CompileError: If a set has no base at all, or two of its members have different bases.
    """
    chosen: dict[str, str] = {}
    missing: list[str] = []
    for root, ids in members.items():
        carriers = [node_id for node_id in ids if graph.nodes[node_id].base is not None]
        if not carriers:
            missing.append(ids[0] if len(ids) == 1 else f"{ids[0]} (merged with {', '.join(ids[1:])})")
            continue
        first: Credence | None = graph.nodes[carriers[0]].base
        for other in carriers[1:]:
            if graph.nodes[other].base != first:
                msg = (
                    f"equivalent nodes {carriers[0]!r} and {other!r} have conflicting bases "
                    f"{first!r} and {graph.nodes[other].base!r}"
                )
                raise CompileError(msg)
        chosen[root] = carriers[0]
    if missing:
        msg = "inference variables without a base (there is no default credence): " + ", ".join(missing)
        raise CompileError(msg)
    return chosen


def _topological(roots: list[str], edges: dict[str, list[tuple[str, str]]]) -> list[str]:
    """Order the merged sets so that every parent precedes its children.

    Args:
        roots: The representatives, in insertion order.
        edges: For each representative, its outgoing ``(relation id, child representative)`` pairs.

    Returns:
        The representatives in a topological order, ties broken by insertion order.

    Raises:
        CompileError: If the relations form a cycle once ``equivalent`` nodes are merged.
    """
    indegree = dict.fromkeys(roots, 0)
    for out in edges.values():
        for _, child in out:
            indegree[child] += 1
    rank = {root: position for position, root in enumerate(roots)}
    ready = sorted((root for root in roots if indegree[root] == 0), key=rank.__getitem__)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for _, child in edges.get(current, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=rank.__getitem__)
    if len(order) == len(roots):
        return order
    raise CompileError(_describe_cycle(roots, {root for root in roots if indegree[root] > 0}, edges))


def _describe_cycle(roots: list[str], stuck: set[str], edges: dict[str, list[tuple[str, str]]]) -> str:
    """Name one cycle among the sets a topological sort could not order.

    Args:
        roots: The representatives, in insertion order.
        stuck: Representatives left with unresolved parents: each lies on a cycle or downstream of one.
        edges: Outgoing ``(relation id, child representative)`` pairs.

    Returns:
        A message naming the cycle's nodes and relations.
    """
    # Drop stuck sets that feed no other stuck set; what is left lies on a cycle or leads into one,
    # so walking forward inside it must eventually repeat a set.
    on_path = set(stuck)
    changed = True
    while changed:
        changed = False
        for root in list(on_path):
            if not any(child in on_path for _, child in edges.get(root, ())):
                on_path.discard(root)
                changed = True
    current = next(root for root in roots if root in on_path)
    path: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    while current not in seen:
        seen[current] = len(path)
        relation, child = next((r, c) for r, c in edges[current] if c in on_path)
        path.append((current, relation))
        current = child
    cycle = path[seen[current] :]
    rendered = cycle[0][0]
    for position, (_, relation) in enumerate(cycle):
        rendered += f" --[{relation}]--> {cycle[(position + 1) % len(cycle)][0]}"
    return f"relations form a cycle once equivalent nodes are merged: {rendered}"


def compile_graph(graph: Graph) -> Network:
    """Compile a graph into a network.

    Nodes joined by ``equivalent`` relations are merged into one variable, named after the earliest
    of them; their bases must agree. Each ``exclusive`` relation becomes a constraint variable that
    is false only when both of its nodes are true, and that every query observes as true. Nodes that
    are not inference variables, and annotation relations, are left out.

    Args:
        graph: The graph.

    Returns:
        The network, with every parameter at its credence mean.

    Raises:
        CompileError: If an inference variable has no base, merged nodes have conflicting bases, or
            the relations form a cycle once ``equivalent`` nodes are merged.
    """
    node_ids = _inference_nodes(graph)
    sets, members = _merge(graph, node_ids)
    bases = _bases(graph, members)

    edges: dict[str, list[tuple[str, str]]] = {}
    links: dict[str, list[tuple[str, str, str]]] = {}
    parameters: dict[ParameterKey, Credence] = {}
    for node_id in bases.values():
        parameters[ParameterKey("base", node_id)] = graph.nodes[node_id].base  # type: ignore[assignment]
    for relation in graph.relations.values():
        if relation.type not in STRENGTH_TYPES:
            continue
        parent, child = sets.find(relation.source), sets.find(relation.target)
        edges.setdefault(parent, []).append((relation.id, child))
        links.setdefault(child, []).append((relation.id, relation.type, parent))
        parameters[ParameterKey("strength", relation.id)] = relation.strength  # type: ignore[assignment]

    order = _topological(list(members), edges)
    index = {root: position for position, root in enumerate(order)}
    variables: list[Variable] = []
    for root in order:
        own = links.get(root, [])
        variables.append(
            Variable(
                index=index[root],
                name=root,
                kind=PROPOSITION,
                members=tuple(members[root]),
                parents=tuple(sorted({index[parent] for _, _, parent in own})),
                base=bases[root],
                links=tuple(Link(relation, rtype, index[parent]) for relation, rtype, parent in own),
            )
        )
    for relation in graph.relations.values():
        if relation.type != EXCLUSIVE:
            continue
        parents = tuple(sorted({index[sets.find(relation.source)], index[sets.find(relation.target)]}))
        variables.append(
            Variable(
                index=len(variables),
                name=f"exclusive[{relation.id}]",
                kind=CONSTRAINT,
                members=(),
                parents=parents,
                relation=relation.id,
            )
        )
    carried = frozenset(graph.nodes) - frozenset(node_ids)
    return Network(tuple(variables), parameters, carried)
