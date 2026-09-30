"""Random graphs for comparing engines and checking properties."""

from __future__ import annotations

import numpy as np

from credencegraph.core import Beta, Graph, Node, Relation

TYPES = ("requires", "supports", "refutes")


def _credence(rng: np.random.Generator, extremes: bool) -> Beta | float:
    """Draw a credence: a point value or a Beta, rarely exactly 0 or 1 when ``extremes`` is set.

    Args:
        rng: The random generator.
        extremes: Whether exact 0 and 1 may be drawn.

    Returns:
        The credence.
    """
    if extremes and rng.random() < 0.05:
        return float(rng.integers(0, 2))
    if rng.random() < 0.5:
        return float(rng.uniform(0.01, 0.99))
    return Beta(float(rng.uniform(0.5, 20.0)), float(rng.uniform(0.5, 20.0)))


def random_graph(rng: np.random.Generator, max_variables: int = 12, extremes: bool = True) -> Graph:
    """Build a random graph that compiles to at most ``max_variables`` variables.

    Nodes are grouped into equivalence classes first; inferential relations only run from an earlier
    class to a later one, so the graph stays acyclic after merging. Only the first node of a class
    carries a base. Exclusive relations, an annotation node and an annotation relation are added too.

    Args:
        rng: The random generator.
        max_variables: The largest number of network variables, constraints included.
        extremes: Whether bases and strengths may be exactly 0 or 1.

    Returns:
        The graph.
    """
    n_classes = int(rng.integers(1, max_variables + 1))
    n_exclusive = int(rng.integers(0, max_variables - n_classes + 1)) if rng.random() < 0.4 else 0
    graph = Graph()
    classes: list[list[str]] = []
    for c in range(n_classes):
        size = 1 + int(rng.random() < 0.15) + int(rng.random() < 0.05)
        ids = [f"n{c}" if k == 0 else f"n{c}_{k}" for k in range(size)]
        classes.append(ids)
        for k, node_id in enumerate(ids):
            graph.add_node(Node(node_id, base=_credence(rng, extremes) if k == 0 else None))
        for k in range(1, size):
            graph.add_relation(Relation(f"eq{c}_{k}", "equivalent", ids[int(rng.integers(0, k))], ids[k]))
    density = rng.uniform(0.1, 0.6)
    count = 0
    for child in range(n_classes):
        for parent in range(child):
            if rng.random() < density:
                source = classes[parent][int(rng.integers(0, len(classes[parent])))]
                target = classes[child][int(rng.integers(0, len(classes[child])))]
                rtype = TYPES[int(rng.integers(0, 3))]
                graph.add_relation(Relation(f"r{count}", rtype, source, target, strength=_credence(rng, extremes)))
                count += 1
    everyone = [node_id for ids in classes for node_id in ids]
    for k in range(n_exclusive):
        a, b = rng.choice(len(everyone), size=2, replace=len(everyone) < 2)
        if everyone[a] == everyone[b]:
            continue
        graph.add_relation(Relation(f"x{k}", "exclusive", everyone[a], everyone[b]))
    graph.add_node(Node("person", kind="person"))
    graph.add_relation(Relation("auth", "authored_by", everyone[0], "person"))
    return graph


def proposition_ids(graph: Graph) -> list[str]:
    """List the ids of the proposition nodes of a random graph.

    Args:
        graph: A graph from ``random_graph``.

    Returns:
        Every node id except the annotation node.
    """
    return [node.id for node in graph if node.kind == "proposition"]


def random_assignment(rng: np.random.Generator, ids: list[str], max_size: int) -> dict[str, bool]:
    """Draw a random partial assignment of node ids.

    Args:
        rng: The random generator.
        ids: Candidate node ids.
        max_size: The largest number of nodes assigned.

    Returns:
        The assignment.
    """
    size = int(rng.integers(0, min(max_size, len(ids)) + 1))
    chosen = rng.choice(len(ids), size=size, replace=False)
    return {ids[i]: bool(rng.integers(0, 2)) for i in chosen}
