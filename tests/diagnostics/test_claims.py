"""Overclaim and underclaim: stated credences against the ones their premises deliver.

The graph is the chain A -> B by ``supports``: P(A) = a and P(B) = b + (1 - b) s a.
"""

from __future__ import annotations

import math

import pytest

from credencegraph.core import Beta, Graph, Node, Relation, ValidationError
from credencegraph.diagnostics import DEFAULT_CLAIM_THRESHOLD, OVERCLAIM, UNDERCLAIM, claims
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
    """Test that 0.5 against 0.44, 0.24 apart in log-odds, is below the default and a threshold of 0.05 reports it."""
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


@pytest.mark.parametrize("threshold", [-0.1, math.inf, math.nan, True, "0.1"])
def test_bad_threshold(threshold):
    """Test that a threshold that is not a finite number >= 0 is rejected."""
    with pytest.raises(ValidationError, match="threshold"):
        claims(chain(), threshold=threshold)


def test_threshold_above_one_is_accepted():
    """Test that a log-odds threshold is not capped at 1: ln(20) silences 0.9 against 0.44, ln(10) does not."""
    assert claims(chain(stated_b=0.9), threshold=math.log(20)) == []
    assert [f.id for f in claims(chain(stated_b=0.9), threshold=math.log(10))] == ["overclaim:B"]


def root(base, stated):
    """A single root ``x``, whose computed credence is its base."""
    graph = Graph()
    graph.add_node(Node("x", base=base, stated=stated))
    return graph


def test_default_is_the_log_odds_gap_across_the_window_centred_on_even_odds():
    """Test that the default is logit(0.55) - logit(0.45) = 2 ln(11/9)."""
    assert math.isclose(DEFAULT_CLAIM_THRESHOLD, 2 * math.log(11 / 9), rel_tol=1e-15, abs_tol=0.0)


def test_gap_equal_to_threshold_is_not_reported():
    """Test that 0.55 against 0.45, a gap of exactly the default on paper, is not reported."""
    graph = root(0.45, 0.55)
    assert claims(graph) == []
    assert [f.id for f in claims(graph, threshold=DEFAULT_CLAIM_THRESHOLD - 1e-6)] == ["overclaim:x"]


@pytest.mark.parametrize(
    ("base", "stated", "expected"),
    [(0.001, 0.07, "overclaim:x"), (0.07, 0.001, "underclaim:x"), (0.999, 0.93, "underclaim:x")],
)
def test_order_of_magnitude_on_a_small_probability_is_reported(base, stated, expected):
    """Test that a 70x error in a small probability, only 0.069 apart, is reported, and its mirror near 1."""
    (finding,) = claims(root(base, stated))
    assert finding.id == expected
    assert finding.value == pytest.approx(stated - base, rel=1e-12, abs=0.0)


def test_every_gap_above_a_tenth_in_probability_is_reported():
    """Test that the default reports every pair 0.1000001 apart, either way round, for a base every 0.0005.

    The windows that come closest to the threshold are those centred on 0.5, and the sweep includes
    the closest, 0.45 against 0.5500001.
    """
    for i in range(1800):
        low = i / 2000
        high = low + 0.1000001
        assert [f.diagnostic for f in claims(root(low, high))] == [OVERCLAIM], (low, high)
        assert [f.diagnostic for f in claims(root(high, low))] == [UNDERCLAIM], (high, low)


def test_rounding_allowance_at_the_default():
    """Test the 1e-12 allowance at the window [0.45, 0.55], where the default is tightest.

    A pair more than 0.1 + 1e-12 apart, which an absolute threshold of 0.1 reports, is reported, and
    still is with the threshold raised by 3e-12. A pair 1e-13 beyond 0.1 is within the allowance and
    is not reported. A pair that is reported, with a log-odds gap 1.25e-12 above the default, is
    dropped by a raise of 5e-13: the allowance moves with the threshold, so a raise smaller than the
    allowance can still drop a pair.
    """
    beyond = root(0.45, 0.55 + 1.1e-12)
    assert 0.55 + 1.1e-12 - 0.45 > 0.1 + 1e-12
    assert [f.id for f in claims(beyond)] == ["overclaim:x"]
    assert [f.id for f in claims(beyond, threshold=DEFAULT_CLAIM_THRESHOLD + 3e-12)] == ["overclaim:x"]
    assert claims(root(0.45, 0.55 + 1e-13)) == []
    edge = root(0.45, 0.5500000000003095)
    assert [f.id for f in claims(edge)] == ["overclaim:x"]
    assert claims(edge, threshold=DEFAULT_CLAIM_THRESHOLD + 5e-13) == []


@pytest.mark.parametrize(("base", "stated"), [(0.45, 0.5500001), (0.44, 0.5405)])
def test_gap_just_above_a_tenth_near_even_odds_is_reported(base, stated):
    """Test that pairs just over 0.1 apart near 0.5, which a threshold of ln(1.5) would drop, are reported."""
    assert abs(stated - base) > 0.1
    assert claims(root(base, stated), threshold=math.log(1.5)) == []
    assert [f.id for f in claims(root(base, stated))] == ["overclaim:x"]


@pytest.mark.parametrize(("base", "stated"), [(0.99, 1.0), (0.0, 0.01), (1.0, 0.99)])
def test_certainty_on_one_side_is_reported_with_a_finite_value(base, stated):
    """Test that a gap to exactly 0 or 1 is infinite in log-odds, reported, and its value stays finite."""
    (finding,) = claims(root(base, stated))
    assert finding.value == pytest.approx(stated - base, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("p", [0.0, 0.3, 1.0])
def test_equal_stated_and_computed_is_silent_at_any_threshold(p):
    """Test that equal values, certain ones included, are never reported, even at threshold 0."""
    assert claims(root(p, p), threshold=0.0) == []
