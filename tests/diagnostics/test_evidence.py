"""The diagnostics under evidence, against Bayes' theorem worked by hand.

Most tests use one graph: a hypothesis H with base h, and two observations of it, o1 and o2, each
with base b_i and a ``supports`` relation from H with strength s_i. From the relation semantics

    P(o_i | H) = L_i = 1 - (1 - b_i)(1 - s_i)    and    P(o_i | not H) = b_i,

so observing o1 true updates H by Bayes' theorem,

    P(H | o1) = h L1 / D,   D = h L1 + (1 - h) b1.

Differentiating that ratio directly, with dL1/ds1 = 1 - b1 and dL1/db1 = 1 - s1,

    dP/dh  = L1 b1 / D^2
    dP/ds1 = h (1 - h) b1 (1 - b1) / D^2
    dP/db1 = -h (1 - h) s1 / D^2

and o2, unobserved, sums out of both sides, so its parameters do not move the answer. Every expected
value below follows from these lines, without the engine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from credencegraph.core import Beta, Graph, Node, Relation, ValidationError
from credencegraph.diagnostics import (
    claims,
    crux,
    derivatives,
    diagnose,
    failure_impact,
    sensitivity,
    single_points_of_failure,
    value_of_information,
)
from credencegraph.inference import Enumeration, VariableElimination, ZeroProbabilityError
from credencegraph.semantics import ParameterKey, compile_graph

ENGINES = [pytest.param(VariableElimination(), id="elimination"), pytest.param(Enumeration(), id="enumeration")]

h, b1, s1, b2, s2 = 0.3, 0.2, 0.75, 0.2, 0.75
L1 = 1 - (1 - b1) * (1 - s1)  # 0.8
L2 = 1 - (1 - b2) * (1 - s2)  # 0.8
D = h * L1 + (1 - h) * b1  # 0.38
POSTERIOR = h * L1 / D  # 0.6316
SD_H = math.sqrt(h * (1 - h) / (3 + 7 + 1))  # Beta(3, 7)
SD_S1 = math.sqrt(s1 * (1 - s1) / (3 + 1 + 1))  # Beta(3, 1)

BASE_H, BASE_O1, BASE_O2 = ParameterKey("base", "H"), ParameterKey("base", "o1"), ParameterKey("base", "o2")
STRENGTH_1, STRENGTH_2 = ParameterKey("strength", "H1"), ParameterKey("strength", "H2")


def close(actual, expected):
    """Assert agreement to 1e-12 relative, with no absolute slack."""
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def binary_entropy(p):
    """Entropy of a Bernoulli(p) variable in bits."""
    return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def by_id(findings):
    """Index findings by id."""
    return {finding.id: finding for finding in findings}


def observations(stated=None, stated_o1=None):
    """The hypothesis and its two observations, with H and the first strength uncertain."""
    graph = Graph()
    graph.add_node(Node("H", base=Beta(3, 7), stated=stated))
    graph.add_node(Node("o1", base=b1, stated=stated_o1))
    graph.add_node(Node("o2", base=b2))
    graph.add_relation(Relation("H1", "supports", "H", "o1", strength=Beta(3, 1)))
    graph.add_relation(Relation("H2", "supports", "H", "o2", strength=s2))
    return graph


@pytest.fixture
def network():
    """The compiled observation graph."""
    return compile_graph(observations())


@pytest.mark.parametrize("engine", ENGINES)
class TestSensitivity:
    """dP(H | o1)/dθ by the quotient rule, against the derivative of Bayes' theorem."""

    def test_derivatives(self, network, engine):
        """Test each derivative against its closed form."""
        slopes = derivatives(network, "H", evidence={"o1": True}, engine=engine)
        assert list(slopes) == list(network.parameters)
        close(slopes[BASE_H], L1 * b1 / D**2)
        close(slopes[STRENGTH_1], h * (1 - h) * b1 * (1 - b1) / D**2)
        close(slopes[BASE_O1], -h * (1 - h) * s1 / D**2)
        for unobserved in (BASE_O2, STRENGTH_2):
            assert abs(slopes[unobserved]) <= 1e-15

    def test_two_point_difference_is_not_the_derivative(self, network, engine):
        """Test that under evidence the answer is not a line in θ, so the two-point difference is wrong.

        P(H | o1) is 0 at h = 0 and 1 at h = 1, a difference of 1, while the derivative at h = 0.3 is
        L1 b1 / D^2 = 1.108.
        """
        slope = derivatives(network, "H", evidence={"o1": True}, engine=engine)[BASE_H]
        assert abs(slope - 1.0) > 0.1
        close(slope, L1 * b1 / D**2)

    def test_negative_observation(self, network, engine):
        """Test o1 false: P(H | not o1) = h (1 - L1) / (h (1 - L1) + (1 - h)(1 - b1)), by h."""
        slope = derivatives(network, "H", evidence={"o1": False}, engine=engine)[BASE_H]
        close(slope, (1 - L1) * (1 - b1) / (h * (1 - L1) + (1 - h) * (1 - b1)) ** 2)

    def test_findings_name_the_evidence(self, network, engine):
        """Test that the messages say which probability moved."""
        records = by_id(sensitivity(network, "H", evidence={"o1": True}, engine=engine))
        assert "P('H' | o1=true) changes by" in records["sensitivity:base:H"].message


@pytest.mark.parametrize("engine", ENGINES)
def test_crux(network, engine):
    """Test crux = |dP(H | o1)/dθ| sd(θ), with sd the parameter's own, not updated on the evidence."""
    records = by_id(crux(network, "H", evidence={"o1": True}, engine=engine))
    close(records["crux:base:H"].value, L1 * b1 / D**2 * SD_H)
    close(records["crux:strength:H1"].value, h * (1 - h) * b1 * (1 - b1) / D**2 * SD_S1)
    assert records["crux:base:o1"].value == 0.0
    assert "P('H' | o1=true) moves" in records["crux:base:H"].message


@pytest.mark.parametrize("engine", ENGINES)
class TestValueOfInformation:
    """I(H; o2 | o1), from the posterior after o1."""

    def test_against_entropies(self, network, engine):
        """Test the bits against H(o2 | o1) - H(o2 | H, o1), and the details against Bayes."""
        p_o2 = POSTERIOR * L2 + (1 - POSTERIOR) * b2
        bits = binary_entropy(p_o2) - (POSTERIOR * binary_entropy(L2) + (1 - POSTERIOR) * binary_entropy(b2))
        (finding,) = value_of_information(network, "H", evidence={"o1": True}, engine=engine)
        assert finding.id == "value-of-information:o2"
        close(finding.value, bits)
        close(finding.details["p_target"], POSTERIOR)
        close(finding.details["p_target_given_true"], POSTERIOR * L2 / p_o2)
        close(finding.details["p_target_given_false"], POSTERIOR * (1 - L2) / (1 - p_o2))
        assert finding.message.endswith("about 'H' given o1=true")

    def test_conditioning_changes_the_value(self, network, engine):
        """Test that the bits differ from the unconditional I(H; o2), which uses the prior h."""
        unconditional = binary_entropy(h * L2 + (1 - h) * b2) - (h * binary_entropy(L2) + (1 - h) * binary_entropy(b2))
        (before,) = [f for f in value_of_information(network, "H", engine=engine) if f.id.endswith("o2")]
        (after,) = value_of_information(network, "H", evidence={"o1": True}, engine=engine)
        close(before.value, unconditional)
        assert abs(after.value - before.value) > 1e-3


def premise_and_observation():
    """T requires premise A (strength 0.95) and is observed through o (supports, strength 0.75).

    P(T) = (a + (1 - a)(1 - r)) t = 0.724, and P(T | do(A = false)) = (1 - r) t = 0.04.
    """
    graph = Graph()
    graph.add_node(Node("A", base=0.9))
    graph.add_node(Node("T", base=0.8))
    graph.add_node(Node("o", base=0.2))
    graph.add_relation(Relation("AT", "requires", "A", "T", strength=0.95))
    graph.add_relation(Relation("To", "supports", "T", "o", strength=0.75))
    return compile_graph(graph)


@pytest.mark.parametrize("engine", ENGINES)
class TestSinglePointOfFailure:
    """P(T | do(A = false), o) against P(T | o), both by Bayes on the prior P(T) and on (1 - r) t."""

    @staticmethod
    def posterior(prior):
        """P(T | o) for a prior P(T), with L = 0.8 and b = 0.2."""
        return prior * 0.8 / (prior * 0.8 + (1 - prior) * 0.2)

    def test_evidence_changes_the_verdict(self, engine):
        """Test that A is a point of failure before o is seen and not after.

        Before: 0.04 / 0.724 = 0.055 of P(T), under the default 0.1. After: 0.143 / 0.913 = 0.156.
        """
        network = premise_and_observation()
        before = single_points_of_failure(network, "T", engine=engine)
        assert [f.id for f in before] == ["single-point-of-failure:A"]
        assert single_points_of_failure(network, "T", evidence={"o": True}, engine=engine) == []

    def test_values(self, engine):
        """Test the value and baseline of the finding under evidence against Bayes."""
        network = premise_and_observation()
        (finding,) = single_points_of_failure(network, "T", evidence={"o": True}, threshold=0.2, engine=engine)
        close(finding.value, self.posterior(0.05 * 0.8))
        close(finding.details["baseline"], self.posterior((0.9 + 0.1 * 0.05) * 0.8))
        assert "P('T' | o=true) falls from" in finding.message

    def test_failure_ruled_out_by_the_evidence_is_skipped(self, engine):
        """Test that a premise whose failure makes the evidence impossible is not a point of failure.

        A requires B with strength 1, so B is false whenever A is; observing B true rules out A failing.
        """
        graph = Graph()
        graph.add_node(Node("A", base=0.5))
        graph.add_node(Node("B", base=0.9))
        graph.add_node(Node("T", base=0.1))
        graph.add_relation(Relation("AB", "requires", "A", "B", strength=1.0))
        graph.add_relation(Relation("BT", "supports", "B", "T", strength=0.9))
        network = compile_graph(graph)
        assert single_points_of_failure(network, "T", evidence={"B": True}, threshold=1.0, engine=engine) == []

    def test_impossible_constraints_still_raise(self, engine):
        """Test that a failure under which the exclusive constraints alone are impossible still raises.

        A and B are certain unless Y vetoes A, and are exclusive: failing Y makes both true.
        """
        graph = Graph()
        graph.add_node(Node("Y", base=0.5))
        graph.add_node(Node("A", base=1.0))
        graph.add_node(Node("B", base=1.0))
        graph.add_node(Node("Z", base=0.5))
        graph.add_relation(Relation("YA", "refutes", "Y", "A", strength=1.0))
        graph.add_relation(Relation("AB", "exclusive", "A", "B"))
        network = compile_graph(graph)
        with pytest.raises(ZeroProbabilityError):
            single_points_of_failure(network, "B", evidence={"Z": True}, engine=engine)


class TestClaims:
    """Stated credences against P(X | evidence)."""

    def test_posterior_settles_an_overclaim(self):
        """Test that H stated at 0.9 overclaims the prior 0.3 but not the posterior after two observations.

        logit(0.9) - logit(0.8727) = 0.27, under the default 0.40.
        """
        graph = observations(stated=0.9)
        assert [f.id for f in claims(graph)] == ["overclaim:H"]
        assert claims(graph, evidence={"o1": True, "o2": True}) == []

    def test_computed_is_the_posterior(self):
        """Test the computed value under one observation, and that the message names the evidence."""
        (finding,) = claims(observations(stated=0.99), evidence={"o1": True})
        close(finding.details["computed"], POSTERIOR)
        assert "give 0.632 given o1=true" in finding.message

    def test_observed_node_is_skipped(self):
        """Test that a stated credence for an observed node is not compared with the observation.

        Unobserved, o1 is computed at h L1 + (1 - h) b1 = 0.38 and stated 0.5 is a gap of 0.49 in log-odds.
        """
        graph = observations(stated_o1=0.5)
        assert [f.id for f in claims(graph)] == ["overclaim:o1"]
        assert claims(graph, evidence={"o1": True}) == []

    def test_impossible_evidence_raises(self):
        """Test that evidence of probability zero raises even when no node is compared."""
        graph = Graph()
        graph.add_node(Node("A", base=0.0))
        with pytest.raises(ZeroProbabilityError):
            claims(graph, evidence={"A": True})


class TestValidation:
    """Evidence the diagnostics refuse."""

    @pytest.mark.parametrize(
        "diagnostic", [derivatives, sensitivity, crux, single_points_of_failure, value_of_information]
    )
    def test_observed_target(self, network, diagnostic):
        """Test that observing the target is refused: every diagnostic of it would be trivial."""
        with pytest.raises(ValidationError, match="target 'H' is fixed"):
            diagnostic(network, "H", evidence={"H": True})

    def test_observed_equivalent_of_the_target(self):
        """Test that observing a node merged with the target is refused too."""
        graph = observations()
        graph.add_node(Node("Hc"))
        graph.add_relation(Relation("same", "equivalent", "H", "Hc"))
        with pytest.raises(ValidationError, match="'Hc', which is equivalent to it,"):
            diagnose(graph, "H", evidence={"Hc": False})

    def test_unknown_evidence_node(self, network):
        """Test that evidence on a node that is not an inference variable is refused."""
        with pytest.raises(ValidationError):
            derivatives(network, "H", evidence={"nope": True})

    def test_non_bool_value(self, network):
        """Test that an observed value must be True or False."""
        with pytest.raises(ValidationError, match="must be True or False"):
            derivatives(network, "H", evidence={"o1": 1})

    def test_conflicting_equivalent_evidence(self):
        """Test that merged nodes observed at different values have probability zero."""
        graph = observations()
        graph.add_node(Node("o1b"))
        graph.add_relation(Relation("same", "equivalent", "o1", "o1b"))
        network = compile_graph(graph)
        with pytest.raises(ZeroProbabilityError, match="equivalent nodes different values"):
            derivatives(network, "H", evidence={"o1": True, "o1b": False})

    def test_impossible_evidence(self):
        """Test that evidence the graph rules out raises before any derivative is taken."""
        graph = Graph()
        graph.add_node(Node("A", base=0.0))
        graph.add_node(Node("T", base=0.5))
        network = compile_graph(graph)
        with pytest.raises(ZeroProbabilityError):
            derivatives(network, "T", evidence={"A": True})


class TestReport:
    """``diagnose`` with and without evidence."""

    def test_without_evidence_unchanged(self):
        """Test that no evidence, ``None`` and ``{}`` give the same report as before evidence existed."""
        graph = observations(stated=0.9)
        default = diagnose(graph, "H")
        assert diagnose(graph, "H", evidence=None) == default
        assert diagnose(graph, "H", evidence={}) == default
        assert "sensitivity:base:o1" in by_id(default)
        assert "value-of-information:o1" in by_id(default)

    def test_every_diagnostic_is_conditioned(self):
        """Test that each group of the report matches its diagnostic under the same evidence."""
        graph = observations(stated=0.9)
        network = compile_graph(graph)
        evidence = {"o1": True}
        findings = diagnose(graph, "H", evidence=evidence, failure_threshold=1.0)
        expected = [
            *claims(graph, network, evidence=evidence),
            *sensitivity(network, "H", evidence=evidence),
            *crux(network, "H", evidence=evidence),
            *single_points_of_failure(network, "H", evidence=evidence, threshold=1.0),
            *failure_impact(network, "H", evidence=evidence, threshold=1.0),
            *value_of_information(network, "H", evidence=evidence),
        ]
        assert [f for f in findings if f.diagnostic != "unanchored"] == expected
        assert "value-of-information:o1" not in by_id(findings)
