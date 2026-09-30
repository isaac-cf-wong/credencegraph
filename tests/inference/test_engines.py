"""Variable elimination against exhaustive enumeration, and the engines' limits."""

from __future__ import annotations

import numpy as np
import pytest
from _graphs import proposition_ids, random_assignment, random_graph

from credencegraph.core import Graph, Node, Relation, ValidationError
from credencegraph.inference import (
    Enumeration,
    ProblemTooLargeError,
    Query,
    VariableElimination,
    ZeroProbabilityError,
    min_fill_order,
)
from credencegraph.semantics import compile_graph

ENUMERATION = Enumeration()
ELIMINATION = VariableElimination()


def _both(network, query):
    """Answer a query with both engines, recording the error type when an engine rejects it.

    Random queries can hold impossible evidence or set merged nodes to different values.
    """
    answers = []
    for engine in (ENUMERATION, ELIMINATION):
        try:
            answers.append(engine.query(network, query))
        except (ZeroProbabilityError, ValidationError) as error:
            answers.append(type(error))
    return answers


@pytest.mark.parametrize("seed", range(300))
def test_elimination_matches_enumeration(seed):
    """Test that both engines agree on marginals, joints, conditionals and interventions."""
    rng = np.random.default_rng(seed)
    graph = random_graph(rng)
    network = compile_graph(graph)
    assert len(network) <= 12
    ids = proposition_ids(graph)
    queries = [Query({node_id: True}) for node_id in ids]
    for _ in range(8):
        queries.append(
            Query(
                random_assignment(rng, ids, 3) or {ids[0]: False},
                random_assignment(rng, ids, 3),
                random_assignment(rng, ids, 2),
            )
        )
    for query in queries:
        reference, answer = _both(network, query)
        if isinstance(reference, type):
            assert answer is reference
            continue
        np.testing.assert_allclose(answer, reference, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("seed", range(50))
def test_probability_of_nothing_is_one(seed):
    """Test that summing every table out gives total probability 1 without constraints."""
    rng = np.random.default_rng(seed)
    network = compile_graph(random_graph(rng))
    if network.constraints:
        return
    for engine in (ENUMERATION, ELIMINATION):
        np.testing.assert_allclose(engine.probability(network, {}), 1.0, rtol=1e-12, atol=0.0)


def _star(n_parents):
    """Build a node with ``n_parents`` supporting root parents."""
    graph = Graph()
    graph.add_node(Node("x", base=0.1))
    for i in range(n_parents):
        graph.add_node(Node(f"p{i}", base=0.5))
        graph.add_relation(Relation(f"r{i}", "supports", f"p{i}", "x", strength=0.5))
    return compile_graph(graph)


class TestLimits:
    """Explicit errors instead of silent fallbacks."""

    def test_factor_size_limit(self):
        """Test that an intermediate factor above the limit raises and names the limit."""
        network = _star(3)
        # With x fixed by the query, eliminating the first parent joins a table over the three parents.
        engine = VariableElimination(max_factor_size=7)
        with pytest.raises(ProblemTooLargeError, match=r"limit of 7 entries \(max_factor_size\)") as info:
            engine.query(network, Query({"x": True}))
        assert info.value.required == 8
        assert info.value.limit == 7
        assert "sampling engine" in str(info.value)
        assert "does not fall back" in str(info.value)

    def test_factor_size_at_limit(self):
        """Test that a factor exactly at the limit is allowed."""
        network = _star(3)
        answer = VariableElimination(max_factor_size=8).query(network, Query({"x": True}))
        np.testing.assert_allclose(answer, 1 - 0.9 * 0.75**3, rtol=1e-12, atol=0.0)

    def test_limit_counts_evidence(self):
        """Test that observed variables do not count towards the factor size."""
        network = _star(3)
        engine = VariableElimination(max_factor_size=2)
        answer = engine.query(network, Query({"x": True}, {"p0": True, "p1": False, "p2": True}))
        np.testing.assert_allclose(answer, 1 - 0.9 * 0.5 * 0.5, rtol=1e-12, atol=0.0)

    def test_default_limit(self):
        """Test that the default limit is 2**22 entries."""
        assert VariableElimination().max_factor_size == 2**22

    def test_enumeration_limit(self):
        """Test that enumeration refuses to sum over too many free variables."""
        network = _star(3)
        with pytest.raises(ProblemTooLargeError, match=r"4 free variables, above the limit of 3") as info:
            Enumeration(max_free_variables=3).probability(network, {})
        assert (info.value.required, info.value.limit) == (4, 3)
        assert Enumeration(max_free_variables=3).probability(network, {0: 1}) > 0


class TestMinFill:
    """The elimination order."""

    def test_chain_eliminates_from_the_ends(self):
        """Test that a chain is eliminated without fill-in, starting from an end."""
        order = min_fill_order([(0,), (0, 1), (1, 2), (2, 3)])
        assert sorted(order) == [0, 1, 2, 3]
        assert order[0] in {0, 3}

    def test_prefers_no_fill(self):
        """Test that a variable whose neighbours are already connected goes before one needing fill."""
        # 0 is the hub of a star over 1, 2, 3: eliminating it would join the leaves, so leaves go first
        # until only one edge is left, where the tie goes to the lower index.
        order = min_fill_order([(0, 1), (0, 2), (0, 3)])
        assert order == [1, 2, 0, 3]

    def test_covers_every_variable(self):
        """Test that the order lists each variable once, isolated ones included."""
        assert sorted(min_fill_order([(5,), (1, 2, 3), (3, 4)])) == [1, 2, 3, 4, 5]
