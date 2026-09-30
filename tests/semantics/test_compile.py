"""Tests for compiling a graph into a network."""

from __future__ import annotations

import numpy as np
import pytest

from credencegraph.core import Beta, Graph, Node, Relation, ValidationError
from credencegraph.semantics import CONSTRAINT, PROPOSITION, CompileError, Link, ParameterKey, compile_graph


def graph_of(*nodes, relations=()):
    """Build a graph from nodes and ``(id, type, source, target, strength)`` tuples."""
    graph = Graph()
    for node in nodes:
        graph.add_node(node)
    for rel_id, rtype, source, target, strength in relations:
        graph.add_relation(Relation(rel_id, rtype, source, target, strength=strength))
    return graph


class TestVariables:
    """Which nodes become variables, and how they are ordered."""

    def test_annotations_are_ignored(self):
        """Test that nodes joined only by annotations are carried, not compiled."""
        graph = graph_of(
            Node("h", base=0.4),
            Node("alice", kind="person"),
            Node("doc", kind="document"),
            relations=[("a", "authored_by", "h", "alice", None), ("c", "cites", "doc", "h", None)],
        )
        network = compile_graph(graph)
        assert [v.name for v in network.variables] == ["h"]
        with pytest.raises(ValidationError, match="'alice' is not an inference variable"):
            network.index("alice")
        with pytest.raises(ValidationError, match="no node 'ghost'"):
            network.index("ghost")

    def test_topological_order(self):
        """Test that parents come before children, ties in insertion order."""
        graph = graph_of(
            Node("c", base=0.1),
            Node("b", base=0.2),
            Node("a", base=0.3),
            relations=[("ab", "supports", "a", "b", 0.5), ("bc", "requires", "b", "c", 0.5)],
        )
        network = compile_graph(graph)
        assert [v.name for v in network.variables] == ["a", "b", "c"]
        c = network.variables[network.index("c")]
        assert c.parents == (network.index("b"),)
        assert c.links == (Link("bc", "requires", network.index("b")),)
        assert network.factors[c.index].scope == (network.index("b"), c.index)

    def test_parameters_and_means(self):
        """Test that every base and strength is a parameter, valued at its mean."""
        graph = graph_of(
            Node("a", base=Beta(1, 3)),
            Node("b", base=0.5),
            relations=[("ab", "supports", "a", "b", Beta(3, 1))],
        )
        network = compile_graph(graph)
        assert dict(network.parameters) == {
            ParameterKey("base", "a"): Beta(1, 3),
            ParameterKey("base", "b"): graph.nodes["b"].base,
            ParameterKey("strength", "ab"): Beta(3, 1),
        }
        assert dict(network.values) == {
            ParameterKey("base", "a"): 0.25,
            ParameterKey("base", "b"): 0.5,
            ParameterKey("strength", "ab"): 0.75,
        }

    def test_tables_are_read_only(self):
        """Test that a compiled network cannot be changed through its tables."""
        network = compile_graph(graph_of(Node("a", base=0.5)))
        with pytest.raises(ValueError, match="read-only"):
            network.factors[0].table[0] = 1.0

    def test_empty_graph(self):
        """Test that a graph without inference variables compiles to an empty network."""
        network = compile_graph(graph_of(Node("x")))
        assert len(network) == 0
        assert repr(network) == "Network(propositions=0, constraints=0)"


class TestErrors:
    """Graphs that cannot be compiled."""

    def test_missing_base(self):
        """Test that every inference variable without a base is named, with no default credence."""
        graph = graph_of(
            Node("a"),
            Node("b", base=0.5),
            Node("c"),
            relations=[("ab", "supports", "a", "b", 0.5), ("bc", "supports", "b", "c", 0.5)],
        )
        with pytest.raises(CompileError, match=r"without a base \(there is no default credence\): a, c$"):
            compile_graph(graph)

    def test_missing_base_names_merged_nodes(self):
        """Test that a merged set without a base lists its members."""
        graph = graph_of(Node("a"), Node("b"), relations=[("e", "equivalent", "a", "b", None)])
        with pytest.raises(CompileError, match=r"a \(merged with b\)"):
            compile_graph(graph)

    def test_conflicting_bases(self):
        """Test that equivalent nodes with different bases are rejected."""
        graph = graph_of(Node("a", base=0.5), Node("b", base=0.6), relations=[("e", "equivalent", "a", "b", None)])
        with pytest.raises(CompileError, match="'a' and 'b' have conflicting bases"):
            compile_graph(graph)

    def test_cycle_after_merging(self):
        """Test that a cycle that only closes once equivalent nodes merge is rejected and named."""
        graph = graph_of(
            *(Node(n, base=0.5) for n in ("a", "b", "c")),
            Node("d"),
            relations=[
                ("ab", "supports", "a", "b", 0.5),
                ("cd", "supports", "c", "d", 0.5),
                ("e1", "equivalent", "b", "c", None),
                ("e2", "equivalent", "d", "a", None),
            ],
        )
        with pytest.raises(
            CompileError, match=r"cycle once equivalent nodes are merged: a --\[ab\]--> b --\[cd\]--> a"
        ):
            compile_graph(graph)

    def test_relation_within_a_merged_set(self):
        """Test that a strength relation between two equivalent nodes is a cycle."""
        graph = graph_of(
            Node("a", base=0.5),
            Node("b"),
            relations=[("s", "supports", "a", "b", 0.5), ("e", "equivalent", "a", "b", None)],
        )
        with pytest.raises(CompileError, match=r"a --\[s\]--> a"):
            compile_graph(graph)

    def test_cycle_downstream_node_is_not_reported(self):
        """Test that a node fed by a cycle, though it cannot be ordered either, is left out of the report."""
        graph = graph_of(
            Node("w", base=0.5),
            Node("a", base=0.5),
            Node("c", base=0.5),
            Node("d"),
            relations=[
                ("cw", "supports", "c", "w", 0.5),
                ("ac", "supports", "a", "c", 0.5),
                ("cd", "supports", "c", "d", 0.5),
                ("e", "equivalent", "d", "a", None),
            ],
        )
        with pytest.raises(CompileError, match=r"merged: a --\[ac\]--> c --\[cd\]--> a$"):
            compile_graph(graph)


class TestEquivalentAndExclusive:
    """Merging and constraint variables."""

    def test_merge_keeps_every_member(self):
        """Test that a merged variable lists all its nodes and is named after the earliest."""
        graph = graph_of(
            Node("x", base=0.5),
            Node("y"),
            Node("z"),
            relations=[("e1", "equivalent", "z", "y", None), ("e2", "equivalent", "y", "x", None)],
        )
        network = compile_graph(graph)
        (variable,) = network.variables
        assert variable.name == "x"
        assert variable.members == ("x", "y", "z")
        assert variable.base == "x"
        assert network.index("y") == network.index("z") == 0

    def test_equal_bases_merge(self):
        """Test that equivalent nodes may repeat the same base; the earliest one is the parameter."""
        graph = graph_of(
            Node("a"), Node("b", base=Beta(2, 2)), Node("c", base=Beta(2, 2)),
            relations=[("e1", "equivalent", "a", "b", None), ("e2", "equivalent", "b", "c", None)],
        )  # fmt: skip
        network = compile_graph(graph)
        assert list(network.parameters) == [ParameterKey("base", "b")]

    def test_exclusive_becomes_an_observed_constraint(self):
        """Test that exclusive(a, b) adds a constraint variable over a and b, observed true."""
        graph = graph_of(Node("a", base=0.5), Node("b", base=0.5), relations=[("x", "exclusive", "a", "b", None)])
        network = compile_graph(graph)
        constraint = network.variables[2]
        assert (constraint.kind, constraint.name, constraint.relation) == (CONSTRAINT, "exclusive[x]", "x")
        assert constraint.parents == (0, 1)
        assert dict(network.constraints) == {2: True}
        np.testing.assert_array_equal(network.factors[2].table[..., 1], [[1.0, 1.0], [1.0, 0.0]])
        assert repr(network) == "Network(propositions=2, constraints=1)"
        assert network.variables[0].kind == PROPOSITION


class TestNetworkCopies:
    """``with_parameters`` and ``intervene`` return new networks."""

    @pytest.fixture
    def network(self):
        """A root supporting a child."""
        graph = graph_of(Node("a", base=0.5), Node("b", base=0.2), relations=[("ab", "supports", "a", "b", 0.5)])
        return compile_graph(graph)

    def test_with_parameters(self, network):
        """Test that new values rebuild the tables and leave the original alone."""
        changed = network.with_parameters({ParameterKey("strength", "ab"): 1.0})
        assert changed.factors[1].table[1, 1] == 1.0
        np.testing.assert_allclose(network.factors[1].table[1, 1], 0.6, rtol=1e-12, atol=0.0)
        assert changed.values[ParameterKey("base", "a")] == 0.5

    @pytest.mark.parametrize(
        ("values", "match"),
        [
            ({ParameterKey("base", "zz"): 0.5}, "no parameter"),
            ({ParameterKey("base", "a"): 1.5}, "parameter base of 'a'"),
            ({ParameterKey("base", "a"): True}, "real number"),
        ],
    )
    def test_with_parameters_rejects(self, network, values, match):
        """Test that unknown parameters and non-probabilities are rejected."""
        with pytest.raises(ValidationError, match=match):
            network.with_parameters(values)

    def test_intervene(self, network):
        """Test that intervening replaces the table by a point mass and cuts the parents."""
        cut = network.intervene({"b": False})
        assert cut.factors[1].scope == (1,)
        np.testing.assert_array_equal(cut.factors[1].table, [1.0, 0.0])
        assert dict(cut.interventions) == {1: False}
        assert network.interventions == {}
        assert cut.intervene({"b": True}).interventions == {1: True}

    def test_intervene_rejects(self, network):
        """Test that non-bool values and contradictory merged settings are rejected."""
        with pytest.raises(ValidationError, match="must be True or False"):
            network.intervene({"a": 1})
        with pytest.raises(ValidationError, match="no node 'q'"):
            network.intervene({"q": True})
        merged = compile_graph(
            graph_of(Node("a", base=0.5), Node("a2"), relations=[("e", "equivalent", "a", "a2", None)])
        )
        with pytest.raises(ValidationError, match="sets 'a' both true and false"):
            merged.intervene({"a": True, "a2": False})
