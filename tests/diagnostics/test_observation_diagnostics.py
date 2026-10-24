"""Diagnostics on graphs with ``observes`` relations."""

from __future__ import annotations

import numpy as np
import pytest

from credencegraph.core import Beta, Graph, Node, Relation, SourceAnchor
from credencegraph.diagnostics import (
    correlated_support,
    crux,
    derivatives,
    missing_parameters,
    sensitivity,
    unanchored,
)
from credencegraph.inference import Enumeration, VariableElimination
from credencegraph.semantics import ParameterKey, compile_graph


def _checked():
    """Build a claim ``h`` with one check ``o`` of it, through the ``observes`` relation ``r``."""
    graph = Graph()
    graph.add_node(Node("h", base=Beta(3, 7), sources=[SourceAnchor("doi:10.0000/example")]))
    graph.add_node(Node("o"))
    graph.add_relation(Relation("r", "observes", "h", "o", sensitivity=Beta(18, 2), false_pass=Beta(2, 8)))
    return graph


def test_an_observation_needs_no_base():
    """Test that an observation is not reported as missing a base."""
    assert missing_parameters(_checked()) == []


def test_an_observation_is_anchored_by_its_claim():
    """Test that an observation is not reported as unanchored: its ``observes`` relation anchors it."""
    assert unanchored(_checked()) == []


def test_derivatives_match_bayes():
    """Test dP(H | pass)/ds and dP(H | pass)/df against the derivative of s h / (s h + f (1 - h)).

    d/ds = h (1 - h) f / D^2 and d/df = -h (1 - h) s / D^2, with D = s h + f (1 - h), at the means.
    """
    network = compile_graph(_checked())
    h, s, f = 0.3, 0.9, 0.2
    slopes = derivatives(network, "h", evidence={"o": True})
    denominator = (s * h + f * (1 - h)) ** 2
    np.testing.assert_allclose(slopes[ParameterKey("sensitivity", "r")], h * (1 - h) * f / denominator, rtol=1e-12)
    np.testing.assert_allclose(slopes[ParameterKey("false_pass", "r")], -h * (1 - h) * s / denominator, rtol=1e-12)


def test_an_unrun_check_does_not_bear_on_the_claim():
    """Test that with the observation unobserved, the claim's sensitivity to s and f is exactly zero."""
    network = compile_graph(_checked())
    values = {finding.id: finding.value for finding in sensitivity(network, "h") + crux(network, "h")}
    for kind in ("sensitivity", "false_pass"):
        assert values[f"sensitivity:{kind}:r"] == 0.0
        assert values[f"crux:{kind}:r"] == 0.0


def test_a_run_check_is_named_as_a_weak_point():
    """Test that once observed, s and f are reported and named as an observes relation's parameters."""
    network = compile_graph(_checked())
    findings = {finding.id: finding for finding in sensitivity(network, "h", evidence={"o": True})}
    finding = findings["sensitivity:sensitivity:r"]
    assert "the sensitivity of observes relation 'r'" in finding.message
    assert finding.nodes == ("h", "o")
    assert finding.relations == ("r",)
    assert "the false-pass rate of observes relation 'r'" in findings["sensitivity:false_pass:r"].message


def _claim_with_consequence():
    """Add to ``_checked`` a node ``c`` that ``h`` supports, so that ``h`` can be observed with ``c`` the target."""
    graph = _checked()
    graph.add_node(Node("c", base=Beta(1, 9), sources=[SourceAnchor("doi:10.0000/other")]))
    graph.add_relation(Relation("hc", "supports", "h", "c", strength=Beta(6, 4)))
    return graph


def test_a_run_check_moves_what_its_claim_supports():
    """Test that the check's rates bear on a consequence of the claim once the check is observed."""
    values = {f.id: f.value for f in crux(compile_graph(_claim_with_consequence()), "c", evidence={"o": True})}
    assert values["crux:sensitivity:r"] > 0.0
    assert values["crux:false_pass:r"] > 0.0


@pytest.mark.parametrize("engine", [VariableElimination(), Enumeration()], ids=type)
def test_a_check_of_an_observed_claim_cancels(engine):
    """Test that with the claim itself observed, the check's rates cancel out of its consequence exactly.

    The rates are the only uncertain parameters, so the cancellation decides the verdict: with the
    claim free they move the consequence and are ranked, and with it observed crux is undefined.
    """
    graph = Graph()
    graph.add_node(Node("h", base=0.3, sources=[SourceAnchor("doi:10.0000/example")]))
    graph.add_node(Node("c", base=0.1))
    graph.add_node(Node("o"))
    graph.add_relation(Relation("hc", "supports", "h", "c", strength=0.6))
    graph.add_relation(Relation("r", "observes", "h", "o", sensitivity=Beta(18, 2), false_pass=Beta(2, 8)))
    network = compile_graph(graph)
    free = {f.id for f in crux(network, "c", evidence={"o": True}, engine=engine)}
    assert {"crux:sensitivity:r", "crux:false_pass:r"} <= free
    fixed = [(f.id, f.value) for f in crux(network, "c", evidence={"o": True, "h": True}, engine=engine)]
    assert fixed == [("crux:c", None)]


def _two_checks_supporting(sensitivity, false_pass):
    """Two checks of ``h``, ``o1`` and ``o2``, from one document, each supporting ``t``."""
    graph = Graph()
    graph.add_node(Node("h", base=0.5, sources=[SourceAnchor("doi:10.0000/a")]))
    graph.add_node(Node("t", base=0.1, sources=[SourceAnchor("doi:10.0000/b")]))
    for name in ("o1", "o2"):
        graph.add_node(Node(name, sources=[SourceAnchor("doi:10.0000/shared")]))
        graph.add_relation(Relation(f"h{name}", "observes", "h", name, sensitivity=sensitivity, false_pass=false_pass))
        graph.add_relation(Relation(f"{name}t", "supports", name, "t", strength=0.5))
    return graph


def test_checks_of_one_claim_share_it_as_a_common_ancestor():
    """Test that two supports that are checks of one claim are not reported: the claim correlates them."""
    assert correlated_support(_two_checks_supporting(0.9, 0.2)) == []


def test_uninformative_checks_do_not_share_their_claim():
    """Test that checks with s = f do not depend on the claim, so their shared source is reported."""
    (finding,) = correlated_support(_two_checks_supporting(0.5, 0.5))
    assert finding.nodes == ("t", "o1", "o2")
