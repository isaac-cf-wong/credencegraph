"""Exact sensitivities against central finite differences, and the constrained case by hand.

A central difference ``(P(θ + h) - P(θ - h)) / 2h`` of a function that is linear in θ has no
truncation error, so without ``exclusive`` constraints the only disagreement is rounding, of order
``eps / h``. With a constraint ``P(T)`` is a ratio of two linear functions of θ and the central
difference carries a truncation error of order ``h**2``. With ``h = 1e-6`` both are far below the
tolerance used here.
"""

from __future__ import annotations

import numpy as np
import pytest

from credencegraph.core import Beta, Graph, Node, Relation
from credencegraph.diagnostics import derivatives
from credencegraph.inference import Enumeration, VariableElimination, ZeroProbabilityError, marginal
from credencegraph.semantics import ParameterKey, compile_graph

STEP = 1e-6
# Measured worst case over the graphs below is 1.4e-10 (9.9e-11 without constraints), from rounding in the two evaluations
# divided by 2h; 1e-8 leaves two orders of magnitude of margin and still rejects any real error in
# a derivative of order 0.01 or more.
TOLERANCE = 1e-8
TYPES = ("requires", "supports", "refutes")


def central_difference(network, target, key):
    """Estimate dP(target)/dθ by a central difference of step ``STEP``."""
    theta = network.values[key]
    up = marginal(network.with_parameters({key: theta + STEP}), target, draws=0).point
    down = marginal(network.with_parameters({key: theta - STEP}), target, draws=0).point
    return (up - down) / (2 * STEP)


def interior_graph(rng, n_nodes, with_constraints):
    """Build a random acyclic graph whose parameters all lie in (0.05, 0.95).

    Nodes run in a fixed order and relations only go forward, so the graph is acyclic. The last
    node is the target. Some pairs are merged by ``equivalent`` and, if asked, some are made
    ``exclusive``.
    """
    graph = Graph()
    for k in range(n_nodes):
        if rng.random() < 0.5:
            base = float(rng.uniform(0.05, 0.95))
        else:
            base = Beta.from_mean_concentration(float(rng.uniform(0.05, 0.95)), float(rng.uniform(2, 20)))
        graph.add_node(Node(f"n{k}", base=base))
    count = 0
    for child in range(1, n_nodes):
        for parent in range(child):
            if rng.random() < 0.5:
                rtype = TYPES[int(rng.integers(0, 3))]
                strength = float(rng.uniform(0.05, 0.95))
                graph.add_relation(Relation(f"r{count}", rtype, f"n{parent}", f"n{child}", strength=strength))
                count += 1
    graph.add_node(Node("twin"))
    graph.add_relation(Relation("eq", "equivalent", f"n{int(rng.integers(0, n_nodes - 1))}", "twin"))
    if with_constraints:
        for k in range(int(rng.integers(1, 3))):
            first, second = rng.choice(n_nodes, size=2, replace=False)
            graph.add_relation(Relation(f"x{k}", "exclusive", f"n{first}", f"n{second}"))
    return graph, f"n{n_nodes - 1}"


@pytest.mark.parametrize("with_constraints", [False, True], ids=["unconstrained", "exclusive"])
@pytest.mark.parametrize("seed", range(8))
def test_against_central_differences(seed, with_constraints):
    """Test every parameter's exact derivative against a central difference on a random graph."""
    rng = np.random.default_rng(seed)
    graph, target = interior_graph(rng, int(rng.integers(3, 7)), with_constraints)
    network = compile_graph(graph)
    assert bool(network.constraints) == with_constraints
    exact = derivatives(network, target)
    for key, slope in exact.items():
        np.testing.assert_allclose(slope, central_difference(network, target, key), rtol=0.0, atol=TOLERANCE)


@pytest.mark.parametrize("seed", range(4))
def test_engines_agree(seed):
    """Test that both exact engines give the same derivatives on a constrained random graph."""
    graph, target = interior_graph(np.random.default_rng(100 + seed), 6, with_constraints=True)
    network = compile_graph(graph)
    by_elimination = derivatives(network, target, engine=VariableElimination())
    by_enumeration = derivatives(network, target, engine=Enumeration())
    for key, slope in by_elimination.items():
        np.testing.assert_allclose(slope, by_enumeration[key], rtol=1e-12, atol=1e-15)


class TestExclusive:
    """Roots A and B with exclusive(A, B): P(A) = a (1 - b) / (1 - a b), a ratio in both parameters."""

    a, b = 0.3, 0.6

    @pytest.fixture
    def network(self):
        """The two exclusive roots."""
        graph = Graph()
        graph.add_node(Node("A", base=self.a))
        graph.add_node(Node("B", base=self.b))
        graph.add_relation(Relation("x", "exclusive", "A", "B"))
        return compile_graph(graph)

    def test_quotient_rule(self, network):
        """Test dP/da = (1 - b) / (1 - ab)^2 and dP/db = -a (1 - a) / (1 - ab)^2."""
        slopes = derivatives(network, "A")
        denominator = (1 - self.a * self.b) ** 2
        np.testing.assert_allclose(slopes[ParameterKey("base", "A")], (1 - self.b) / denominator, rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            slopes[ParameterKey("base", "B")], -self.a * (1 - self.a) / denominator, rtol=1e-12, atol=0
        )

    def test_two_point_difference_is_not_the_derivative(self, network):
        """Test that P(A | a = 1) - P(A | a = 0) = 1 here, far from the true slope of 0.595."""
        key = ParameterKey("base", "A")
        high = marginal(network.with_parameters({key: 1.0}), "A", draws=0).point
        low = marginal(network.with_parameters({key: 0.0}), "A", draws=0).point
        assert high - low == 1.0
        assert abs(derivatives(network, "A")[key] - 1.0) > 0.4


def test_impossible_constraints():
    """Test that constraints of probability zero are refused: two certain nodes that exclude each other."""
    graph = Graph()
    graph.add_node(Node("A", base=1.0))
    graph.add_node(Node("B", base=1.0))
    graph.add_relation(Relation("x", "exclusive", "A", "B"))
    with pytest.raises(ZeroProbabilityError, match="probability zero"):
        derivatives(compile_graph(graph), "A")
