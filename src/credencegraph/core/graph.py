"""The graph container."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType

from credencegraph.core.errors import CycleError, ValidationError
from credencegraph.core.node import Node
from credencegraph.core.relation import ACYCLIC_TYPES, Relation


class Graph:
    """Nodes and relations keyed by stable string ids.

    Every write is validated and leaves the graph unchanged when it is rejected. Relations must name
    existing nodes, and the subgraph of ``requires``, ``supports`` and ``refutes`` relations must
    stay acyclic. Annotation relations and ``equivalent``/``exclusive`` relations do not take part in
    the cycle check.
    """

    def __init__(self) -> None:
        """Create an empty graph."""
        self._nodes: dict[str, Node] = {}
        self._relations: dict[str, Relation] = {}
        # node id -> [(relation id, relation type, target id)], for the acyclic subgraph only
        self._acyclic_out: dict[str, list[tuple[str, str, str]]] = {}

    @property
    def nodes(self) -> Mapping[str, Node]:
        """Read-only view of the nodes by id, in insertion order."""
        return MappingProxyType(self._nodes)

    @property
    def relations(self) -> Mapping[str, Relation]:
        """Read-only view of the relations by id, in insertion order."""
        return MappingProxyType(self._relations)

    def __len__(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        """Iterate over the nodes in insertion order."""
        return iter(self._nodes.values())

    def __contains__(self, node_id: object) -> bool:
        """Tell whether a node with this id exists."""
        return node_id in self._nodes

    def __eq__(self, other: object) -> bool:
        """Compare graphs by their nodes and relations, ignoring insertion order."""
        if not isinstance(other, Graph):
            return NotImplemented
        return self._nodes == other._nodes and self._relations == other._relations

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Summarise the graph."""
        return f"Graph(nodes={len(self._nodes)}, relations={len(self._relations)})"

    def add_node(self, node: Node) -> Node:
        """Add a node.

        Args:
            node: The node to add.

        Returns:
            The node.

        Raises:
            ValidationError: If ``node`` is not a ``Node`` or its id is already taken.
        """
        if not isinstance(node, Node):
            msg = f"add_node expects a Node, got {type(node).__name__}"
            raise ValidationError(msg)
        if node.id in self._nodes:
            msg = f"duplicate node id {node.id!r}"
            raise ValidationError(msg)
        self._nodes[node.id] = node
        return node

    def add_relation(self, relation: Relation) -> Relation:
        """Add a relation.

        Args:
            relation: The relation to add.

        Returns:
            The relation.

        Raises:
            ValidationError: If ``relation`` is not a ``Relation``, its id is already taken, or an endpoint
                is not a node of the graph.
            CycleError: If a ``requires``, ``supports`` or ``refutes`` relation would close a cycle among
                such relations. The message names the cycle.
        """
        if not isinstance(relation, Relation):
            msg = f"add_relation expects a Relation, got {type(relation).__name__}"
            raise ValidationError(msg)
        if relation.id in self._relations:
            msg = f"duplicate relation id {relation.id!r}"
            raise ValidationError(msg)
        for role, node_id in (("source", relation.source), ("target", relation.target)):
            if node_id not in self._nodes:
                msg = f"relation {relation.id!r} has {role} {node_id!r}, which is not a node of the graph"
                raise ValidationError(msg)
        if relation.type in ACYCLIC_TYPES:
            self._reject_cycle(relation)
            self._acyclic_out.setdefault(relation.source, []).append((relation.id, relation.type, relation.target))
        self._relations[relation.id] = relation
        return relation

    def _reject_cycle(self, relation: Relation) -> None:
        """Raise ``CycleError`` if adding ``relation`` would close a cycle.

        A cycle exists exactly when a path already leads from the relation's target back to its source.

        Args:
            relation: The candidate relation, of an acyclic type.

        Raises:
            CycleError: If a path leads from ``relation.target`` to ``relation.source``.
        """
        path = self._find_path(relation.target, relation.source)
        if path is None:
            return
        edges = [(relation.id, relation.type, relation.target), *path]
        nodes = [relation.source, *(target for _, _, target in edges)]
        rendered = nodes[0]
        for (rel_id, rel_type, _), node in zip(edges, nodes[1:], strict=True):
            rendered += f" --{rel_type}[{rel_id}]--> {node}"
        msg = f"relation {relation.id!r} would create a cycle: {rendered}"
        raise CycleError(msg, tuple(nodes), tuple(rel_id for rel_id, _, _ in edges))

    def _find_path(self, start: str, goal: str) -> list[tuple[str, str, str]] | None:
        """Find a path along acyclic-type relations from ``start`` to ``goal``.

        Args:
            start: Node id to start from.
            goal: Node id to reach.

        Returns:
            The edges of a path as ``(relation id, relation type, target id)``, empty when
            ``start == goal``, or ``None`` when ``goal`` is unreachable.
        """
        if start == goal:
            return []
        came_from: dict[str, tuple[str, tuple[str, str, str]]] = {}
        stack = [start]
        seen = {start}
        while stack:
            current = stack.pop()
            for edge in self._acyclic_out.get(current, ()):
                nxt = edge[2]
                if nxt in seen:
                    continue
                seen.add(nxt)
                came_from[nxt] = (current, edge)
                if nxt == goal:
                    path = []
                    node = goal
                    while node != start:
                        node, step = came_from[node][0], came_from[node][1]
                        path.append(step)
                    return path[::-1]
                stack.append(nxt)
        return None
