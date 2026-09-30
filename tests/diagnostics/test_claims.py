"""Overclaim and underclaim: stated credences against the ones their premises deliver.

The graph is the chain A -> B by ``supports``: P(A) = a and P(B) = b + (1 - b) s a.
"""

from __future__ import annotations

import pytest

from credencegraph.core import Beta, Graph, Node, Relation, ValidationError
from credencegraph.diagnostics import OVERCLAIM, UNDERCLAIM, claims
from credencegraph.semantics import compile_graph

A, B, S = 0.5, 0.2, 0.6
P_B = B + (1 - B) * S * A  # 0.44


def chain(stated_b=None, stated_a=None):
    """Build the chain with optional stated credences on A and B."""
    graph = Graph()
    graph.add_node(Node("A", base=A, stated=stated_a))
    graph.add_node(Node("B", base=B, stated=stated_b))
    graph.add_relation(Relation("ab", "supports", "A", "B", strength=S))
    return graph


def test_overclaim():
    """Test that stating 0.9 against a computed 0.44 is an overclaim of 0.46."""
    (finding,) = claims(chain(stated_b=0.9))
    assert finding.id == "overclaim:B"
    assert finding.diagnostic == OVERCLAIM
    assert finding.nodes == ("B",)
    assert finding.value == pytest.approx(0.9 - P_B, rel=1e-12, abs=0.0)
    assert finding.details["computed"] == pytest.approx(P_B, rel=1e-12, abs=0.0)
    assert finding.details["stated"] == 0.9
    assert "more confidence" in finding.message


def test_underclaim():
    """Test that stating 0.2 against a computed 0.44 is an underclaim of -0.24."""
    (finding,) = claims(chain(stated_b=0.2))
    assert finding.id == "underclaim:B"
    assert finding.diagnostic == UNDERCLAIM
    assert finding.value == pytest.approx(0.2 - P_B, rel=1e-12, abs=0.0)
    assert "less confidence" in finding.message


def test_within_threshold_is_silent():
    """Test that a gap of 0.06 is below the default threshold of 0.1 and a threshold of 0.05 reports it."""
    graph = chain(stated_b=0.5)
    assert claims(graph) == []
    assert [f.id for f in claims(graph, threshold=0.05)] == ["overclaim:B"]


def test_beta_stated_uses_its_mean():
    """Test that a Beta(9, 1) stated credence is compared through its mean, 0.9."""
    (finding,) = claims(chain(stated_b=Beta(9, 1)))
    assert finding.value == pytest.approx(0.9 - P_B, rel=1e-12, abs=0.0)


def test_root_compares_with_its_base():
    """Test that a root's computed credence is its base."""
    (finding,) = claims(chain(stated_a=0.95))
    assert finding.id == "overclaim:A"
    assert finding.value == pytest.approx(0.45, rel=1e-12, abs=0.0)


def test_precompiled_network_is_used():
    """Test that a network passed in is used as given, here with the strength raised to 1."""
    graph = chain(stated_b=0.9)
    network = compile_graph(graph).with_parameters({("strength", "ab"): 1.0})
    (finding,) = claims(graph, network)
    assert finding.details["computed"] == pytest.approx(B + (1 - B) * A, rel=1e-12, abs=0.0)


def test_stated_on_a_carried_node_is_skipped():
    """Test that a stated credence on a node outside inference has nothing to compare with."""
    graph = chain()
    graph.add_node(Node("note", kind="note", stated=0.9))
    assert claims(graph) == []


def test_bad_threshold():
    """Test that a threshold outside [0, 1] is rejected."""
    with pytest.raises(ValidationError, match="threshold"):
        claims(chain(), threshold=1.5)


def test_gap_equal_to_threshold_is_not_reported():
    """Test that a gap of 0.1 on paper, 0.2 - (1 - 0.9) = 0.10000000000000003 in floating point, is not reported."""
    graph = Graph()
    graph.add_node(Node("x", base=0.1, stated=0.2))
    assert 0.2 - (1 - 0.9) > 0.1
    assert claims(graph, threshold=0.1) == []
    assert [f.id for f in claims(graph, threshold=0.0999)] == ["overclaim:x"]
