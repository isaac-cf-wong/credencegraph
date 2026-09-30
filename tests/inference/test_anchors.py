"""Analytic anchors: small networks whose answers have closed forms derived by hand.

Each expected value below is worked out on paper from the relation semantics (base b, strengths
r, s, f, and independent root priors), not by calling the library.
"""

from __future__ import annotations

import numpy as np
import pytest

from credencegraph.core import Beta, Graph, Node, Relation
from credencegraph.inference import Enumeration, VariableElimination, conditional, intervene, joint, marginal
from credencegraph.semantics import compile_graph

ENGINES = [pytest.param(VariableElimination(), id="elimination"), pytest.param(Enumeration(), id="enumeration")]


def close(actual, expected):
    """Assert agreement to 1e-12 relative, with no absolute slack."""
    np.testing.assert_allclose(actual.point, expected, rtol=1e-12, atol=0.0)


def graph_of(nodes, relations=()):
    """Build a graph from ``(id, base)`` pairs and ``(id, type, source, target, strength)`` tuples."""
    graph = Graph()
    for node_id, base in nodes:
        graph.add_node(Node(node_id, base=base))
    for rel_id, rtype, source, target, strength in relations:
        graph.add_relation(Relation(rel_id, rtype, source, target, strength=strength))
    return graph


@pytest.mark.parametrize("engine", ENGINES)
class TestSingleRoot:
    """A node with no parents: its probability is its base."""

    def test_point_base(self, engine):
        """Test that P(X) = b for a point base."""
        network = compile_graph(graph_of([("x", 0.37)]))
        close(marginal(network, "x", engine=engine), 0.37)
        close(marginal(network, "x", False, engine=engine), 0.63)

    def test_beta_base_uses_the_mean(self, engine):
        """Test that a Beta(2, 5) base gives the point answer 2/7."""
        network = compile_graph(graph_of([("x", Beta(2, 5))]))
        close(marginal(network, "x", engine=engine), 2 / 7)

    def test_independent_roots(self, engine):
        """Test that unrelated roots are independent: P(X and not Y) = x (1 - y)."""
        network = compile_graph(graph_of([("x", 0.2), ("y", 0.7)]))
        close(joint(network, {"x": True, "y": False}, engine=engine), 0.2 * 0.3)
        close(conditional(network, "x", {"y": True}, engine=engine), 0.2)


@pytest.mark.parametrize("engine", ENGINES)
class TestChain:
    """A -> B -> C by ``supports``, with a = 0.3, b_B = 0.1, b_C = 0.2, s_AB = 0.6, s_BC = 0.5."""

    a, b_b, b_c, s_ab, s_bc = 0.3, 0.1, 0.2, 0.6, 0.5

    @pytest.fixture
    def network(self):
        """The chain."""
        return compile_graph(
            graph_of(
                [("A", self.a), ("B", self.b_b), ("C", self.b_c)],
                [("ab", "supports", "A", "B", self.s_ab), ("bc", "supports", "B", "C", self.s_bc)],
            )
        )

    def b_given(self, a_true):
        """P(B = 1 | A)."""
        return 1 - (1 - self.b_b) * (1 - self.s_ab) if a_true else self.b_b

    def c_given(self, b_true):
        """P(C = 1 | B)."""
        return 1 - (1 - self.b_c) * (1 - self.s_bc) if b_true else self.b_c

    def test_marginals(self, network, engine):
        """Test P(B) and P(C) by the law of total probability along the chain."""
        p_b = self.a * self.b_given(True) + (1 - self.a) * self.b_given(False)
        # By hand: 0.3 * 0.64 + 0.7 * 0.1 = 0.262.
        np.testing.assert_allclose(p_b, 0.262, rtol=1e-12, atol=0.0)
        p_c = p_b * self.c_given(True) + (1 - p_b) * self.c_given(False)
        close(marginal(network, "B", engine=engine), p_b)
        close(marginal(network, "C", engine=engine), p_c)

    def test_diagnostic_conditional(self, network, engine):
        """Test P(A | C) by Bayes' rule over the hidden B."""

        def c_given_a(a_true):
            p_b = self.b_given(a_true)
            return p_b * self.c_given(True) + (1 - p_b) * self.c_given(False)

        numerator = self.a * c_given_a(True)
        expected = numerator / (numerator + (1 - self.a) * c_given_a(False))
        close(conditional(network, "A", {"C": True}, engine=engine), expected)

    def test_intervention_differs_from_conditioning(self, network, engine):
        """Test that do(B = 0) leaves A at its prior while conditioning on B = 0 lowers it."""
        close(intervene(network, "A", {"B": False}, engine=engine), self.a)
        close(intervene(network, "C", {"B": False}, engine=engine), self.b_c)
        expected = self.a * (1 - self.b_given(True))
        expected /= expected + (1 - self.a) * (1 - self.b_given(False))
        close(conditional(network, "A", {"B": False}, engine=engine), expected)
        assert expected < self.a

    def test_requires_chain(self, engine):
        """Test A -> B by ``requires`` with r = 1: B needs A, so P(B) = a b_B."""
        network = compile_graph(graph_of([("A", 0.4), ("B", 0.9)], [("ab", "requires", "A", "B", 1.0)]))
        close(marginal(network, "B", engine=engine), 0.4 * 0.9)
        close(conditional(network, "A", {"B": True}, engine=engine), 1.0)


@pytest.mark.parametrize("engine", ENGINES)
class TestSingleGate:
    """One child with independent root parents of one relation type."""

    priors = (0.2, 0.5, 0.9)
    strengths = (0.7, 0.4, 0.25)
    base = 0.15

    def build(self, rtype):
        """Build a child ``x`` with three root parents of type ``rtype``."""
        nodes = [("x", self.base)] + [(f"p{i}", p) for i, p in enumerate(self.priors)]
        relations = [(f"r{i}", rtype, f"p{i}", "x", s) for i, s in enumerate(self.strengths)]
        return compile_graph(graph_of(nodes, relations))

    def test_pure_noisy_or(self, engine):
        """Test P(X) = 1 - (1 - b) prod(1 - s_i p_i) for supports."""
        expected = 1 - (1 - self.base) * (1 - 0.7 * 0.2) * (1 - 0.4 * 0.5) * (1 - 0.25 * 0.9)
        close(marginal(self.build("supports"), "x", engine=engine), expected)

    def test_pure_noisy_and(self, engine):
        """Test P(X) = b prod(1 - r_i (1 - p_i)) for requires."""
        expected = self.base * (1 - 0.7 * 0.8) * (1 - 0.4 * 0.5) * (1 - 0.25 * 0.1)
        close(marginal(self.build("requires"), "x", engine=engine), expected)

    def test_refuter_only(self, engine):
        """Test P(X) = b prod(1 - f_k p_k) for refutes."""
        expected = self.base * (1 - 0.7 * 0.2) * (1 - 0.4 * 0.5) * (1 - 0.25 * 0.9)
        close(marginal(self.build("refutes"), "x", engine=engine), expected)

    def test_noisy_or_all_supports_true(self, engine):
        """Test P(X | every support true) = 1 - (1 - b) prod(1 - s_i)."""
        given = {f"p{i}": True for i in range(3)}
        expected = 1 - (1 - self.base) * 0.3 * 0.6 * 0.75
        close(conditional(self.build("supports"), "x", given, engine=engine), expected)

    def test_strict_requirement_false(self, engine):
        """Test that X is false whenever a strictly required premise is false."""
        network = compile_graph(graph_of([("x", 0.8), ("p", 0.5)], [("r", "requires", "p", "x", 1.0)]))
        close(conditional(network, "x", {"p": False}, engine=engine), 0.0)
        close(conditional(network, "x", {"p": True}, engine=engine), 0.8)

    def test_one_of_each(self, engine):
        """Test P(X) = (1 - r (1 - p_R)) (1 - (1 - b)(1 - s p_S)) (1 - f p_F): the three gates factorise."""
        network = compile_graph(
            graph_of(
                [("x", 0.3), ("R", 0.6), ("S", 0.4), ("F", 0.2)],
                [
                    ("rr", "requires", "R", "x", 0.9),
                    ("rs", "supports", "S", "x", 0.5),
                    ("rf", "refutes", "F", "x", 0.75),
                ],
            )
        )
        expected = (1 - 0.9 * 0.4) * (1 - 0.7 * (1 - 0.5 * 0.4)) * (1 - 0.75 * 0.2)
        close(marginal(network, "x", engine=engine), expected)

    def test_veto(self, engine):
        """Test the veto asymmetry: b = 0.1, a support of 0.9 and a refuter of 0.9 give 0.091."""
        network = compile_graph(
            graph_of(
                [("x", 0.1), ("S", 0.5), ("F", 0.5)],
                [("rs", "supports", "S", "x", 0.9), ("rf", "refutes", "F", "x", 0.9)],
            )
        )
        # 1 - 0.9 * 0.1 = 0.91 with the support alone; the refuter keeps a tenth of it.
        close(conditional(network, "x", {"S": True, "F": True}, engine=engine), 0.091)


@pytest.mark.parametrize("engine", ENGINES)
class TestEquivalentAndExclusive:
    """Merged nodes and exclusion constraints."""

    def test_exclusive_renormalises(self, engine):
        """Test P(A | not both) = a (1 - b) / (1 - a b) and P(A and B) = 0."""
        graph = graph_of([("A", 0.6), ("B", 0.5)])
        graph.add_relation(Relation("x", "exclusive", "A", "B"))
        network = compile_graph(graph)
        close(marginal(network, "A", engine=engine), 0.6 * 0.5 / (1 - 0.3))
        close(joint(network, ["A", "B"], engine=engine), 0.0)

    def test_equivalent_nodes_share_one_variable(self, engine):
        """Test that A = A' are one proposition: P(A and not A') = 0, P(A | A') = 1."""
        graph = graph_of([("A", 0.25)])
        graph.add_node(Node("A2"))
        graph.add_relation(Relation("e", "equivalent", "A", "A2"))
        network = compile_graph(graph)
        close(marginal(network, "A2", engine=engine), 0.25)
        close(joint(network, {"A": True, "A2": False}, engine=engine), 0.0)
        close(conditional(network, "A", {"A2": True}, engine=engine), 1.0)

    def test_relations_from_merged_nodes_combine(self, engine):
        """Test that supports from A and its equivalent A' act as two noisy-OR mechanisms."""
        graph = graph_of([("A", 0.4), ("C", 0.1)])
        graph.add_node(Node("A2"))
        graph.add_relation(Relation("e", "equivalent", "A2", "A"))
        graph.add_relation(Relation("s1", "supports", "A", "C", strength=0.5))
        graph.add_relation(Relation("s2", "supports", "A2", "C", strength=0.2))
        network = compile_graph(graph)
        expected = 0.4 * (1 - 0.9 * 0.5 * 0.8) + 0.6 * 0.1
        close(marginal(network, "C", engine=engine), expected)

    def test_exclusive_with_its_own_equivalent(self, engine):
        """Test that exclusive(A, A') for equivalent A, A' forces A false."""
        graph = graph_of([("A", 0.7), ("B", 0.3)], [("s", "supports", "A", "B", 0.5)])
        graph.add_node(Node("A2"))
        graph.add_relation(Relation("e", "equivalent", "A", "A2"))
        graph.add_relation(Relation("x", "exclusive", "A", "A2"))
        network = compile_graph(graph)
        close(marginal(network, "A", engine=engine), 0.0)
        close(marginal(network, "B", engine=engine), 0.3)
