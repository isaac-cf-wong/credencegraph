"""Sensitivity, crux, single points of failure and value of information, against hand-derived values.

Most tests use one graph: roots A and B, and a target T with base t, where A ``requires`` T with
strength r and B ``supports`` T with strength s. From the relation semantics,

    P(T | A, B) = N * O,   N = 1 if A else 1 - r,   O = 1 - (1 - t) (1 - s)^B

and since A and B are independent roots, P(T) = G * H with

    G = a + (1 - a)(1 - r)    and    H = t + b s (1 - t).

Every expected value below follows from these two lines by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from credencegraph.core import Beta, Graph, Node, Relation, ValidationError
from credencegraph.diagnostics import (
    CRUX,
    SENSITIVITY,
    SINGLE_POINT_OF_FAILURE,
    VALUE_OF_INFORMATION,
    crux,
    derivatives,
    sensitivity,
    single_points_of_failure,
    value_of_information,
)
from credencegraph.inference import Engine, Enumeration, Query, VariableElimination, ZeroProbabilityError, marginal
from credencegraph.semantics import ParameterKey, compile_graph

ENGINES = [pytest.param(VariableElimination(), id="elimination"), pytest.param(Enumeration(), id="enumeration")]

a, b, t, r, s = 0.6, 0.5, 0.3, 0.8, 0.5
G = a + (1 - a) * (1 - r)  # 0.68
H = t + b * s * (1 - t)  # 0.475
SD_A = math.sqrt(a * (1 - a) / (6 + 4 + 1))  # Beta(6, 4)
SD_S = math.sqrt(s * (1 - s) / (2 + 2 + 1))  # Beta(2, 2)

BASE_A, BASE_B, BASE_T = ParameterKey("base", "A"), ParameterKey("base", "B"), ParameterKey("base", "T")
STRENGTH_R, STRENGTH_S = ParameterKey("strength", "AT"), ParameterKey("strength", "BT")


def close(actual, expected):
    """Assert agreement to 1e-12 relative, with no absolute slack."""
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


@pytest.fixture
def network():
    """The two-premise graph, with A and the support strength uncertain."""
    graph = Graph()
    graph.add_node(Node("A", base=Beta(6, 4)))
    graph.add_node(Node("B", base=b))
    graph.add_node(Node("T", base=t))
    graph.add_relation(Relation("AT", "requires", "A", "T", strength=r))
    graph.add_relation(Relation("BT", "supports", "B", "T", strength=Beta(2, 2)))
    return compile_graph(graph)


def by_id(findings):
    """Index findings by id."""
    return {finding.id: finding for finding in findings}


def binary_entropy(p):
    """Entropy of a Bernoulli(p) variable in bits."""
    return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@pytest.mark.parametrize("engine", ENGINES)
def test_marginal_matches_the_closed_form(network, engine):
    """Test the premise every other expected value rests on: P(T) = G H."""
    close(marginal(network, "T", draws=0, engine=engine).point, G * H)


@pytest.mark.parametrize("engine", ENGINES)
class TestSensitivity:
    """dP(T)/dθ for each of the five parameters."""

    def test_derivatives(self, network, engine):
        """Test each derivative against its closed form."""
        slopes = derivatives(network, "T", engine=engine)
        assert list(slopes) == list(network.parameters)
        close(slopes[BASE_A], r * H)
        close(slopes[STRENGTH_R], -(1 - a) * H)
        close(slopes[BASE_B], G * s * (1 - t))
        close(slopes[STRENGTH_S], G * b * (1 - t))
        close(slopes[BASE_T], G * (1 - b * s))

    def test_findings(self, network, engine):
        """Test the records: ids, subjects, signs and ranking by magnitude."""
        findings = sensitivity(network, "T", engine=engine)
        assert [f.diagnostic for f in findings] == [SENSITIVITY] * 5
        magnitudes = [abs(f.value) for f in findings]
        assert magnitudes == sorted(magnitudes, reverse=True)
        records = by_id(findings)
        strength = records["sensitivity:strength:AT"]
        assert strength.nodes == ("A", "T")
        assert strength.relations == ("AT",)
        close(strength.value, -(1 - a) * H)
        assert "requires relation 'AT'" in strength.message
        assert records["sensitivity:base:T"].nodes == ("T",)
        assert records["sensitivity:base:T"].details["parameter_value"] == t


@pytest.mark.parametrize("engine", ENGINES)
def test_crux(network, engine):
    """Test crux = |dP/dθ| sd(θ): only the two Beta parameters score, A first."""
    findings = crux(network, "T", engine=engine)
    assert [f.id for f in findings[:2]] == ["crux:base:A", "crux:strength:BT"]
    records = by_id(findings)
    close(records["crux:base:A"].value, r * H * SD_A)
    close(records["crux:strength:BT"].value, G * b * (1 - t) * SD_S)
    close(records["crux:base:A"].details["sd"], SD_A)
    for point_parameter in ("crux:base:B", "crux:base:T", "crux:strength:AT"):
        assert records[point_parameter].value == 0.0
    assert all(f.diagnostic == CRUX for f in findings)


@pytest.mark.parametrize("engine", ENGINES)
class TestSinglePointOfFailure:
    """P(T | do(A = 0)) = (1 - r) H and P(T | do(B = 0)) = G t, against P(T) = G H.

    As fractions of P(T) these are (1 - r) / G = 0.294 and t / H = 0.632.
    """

    def test_default_threshold(self, network, engine):
        """Test that neither failure takes T below a tenth of P(T), the default."""
        assert single_points_of_failure(network, "T", engine=engine) == []

    def test_higher_threshold(self, network, engine):
        """Test that a threshold of 0.5 catches A, 0.7 also B, and the list runs lowest first."""
        (finding,) = single_points_of_failure(network, "T", threshold=0.5, engine=engine)
        assert finding.id == "single-point-of-failure:A"
        assert finding.diagnostic == SINGLE_POINT_OF_FAILURE
        assert finding.nodes == ("A",)
        close(finding.value, (1 - r) * H)
        close(finding.details["baseline"], G * H)
        assert finding.details["threshold"] == 0.5
        assert "falls from 0.323 to 0.095" in finding.message
        findings = single_points_of_failure(network, "T", threshold=0.7, engine=engine)
        assert [f.id for f in findings] == ["single-point-of-failure:A", "single-point-of-failure:B"]
        close(findings[1].value, G * t)

    def test_already_low_target_is_ranked(self, engine):
        """Test that with P(T) = 0.095, below 0.1, the default reports the premise that sinks T only.

        X ``requires`` T with strength 1, W ``supports`` T with strength 0.2, and T -> D by
        ``supports``: P(T) = x (1 - (1 - t)(1 - w s)) = 0.5 (1 - 0.9 * 0.9) = 0.095. do(X = 0) takes
        it to 0; do(W = 0) to x t = 0.05, a dent of about half, which is below 0.1 but not below a
        tenth of P(T); do(D = 0) leaves it where it was.
        """
        graph = Graph()
        graph.add_node(Node("X", base=0.5))
        graph.add_node(Node("W", base=0.5))
        graph.add_node(Node("T", base=0.1))
        graph.add_node(Node("D", base=0.2))
        graph.add_relation(Relation("XT", "requires", "X", "T", strength=1.0))
        graph.add_relation(Relation("WT", "supports", "W", "T", strength=0.2))
        graph.add_relation(Relation("TD", "supports", "T", "D", strength=0.9))
        network = compile_graph(graph)
        close(marginal(network, "T", draws=0, engine=engine).point, 0.095)
        (finding,) = single_points_of_failure(network, "T", engine=engine)
        assert finding.id == "single-point-of-failure:X"
        assert finding.value == 0.0
        findings = single_points_of_failure(network, "T", threshold=1.0, engine=engine)
        assert [f.id for f in findings] == ["single-point-of-failure:X", "single-point-of-failure:W"]
        close(findings[1].value, 0.05)

    def test_failure_of_an_improbable_target_is_reported(self, engine):
        """Test that a premise taking P(T) = 5e-16 to 0 is reported: the rounding slack scales with P(T).

        X (base 0.5) ``requires`` T with strength 1 and T's base is 1e-15, so P(T) = 5e-16 and do(X = 0)
        leaves exactly 0. An absolute slack of 1e-12 would put the cutoff below zero and hide it.
        """
        graph = Graph()
        graph.add_node(Node("X", base=0.5))
        graph.add_node(Node("T", base=1e-15))
        graph.add_relation(Relation("XT", "requires", "X", "T", strength=1.0))
        (finding,) = single_points_of_failure(compile_graph(graph), "T", engine=engine)
        assert finding.id == "single-point-of-failure:X"
        assert finding.value == 0.0
        close(finding.details["baseline"], 5e-16)

    def test_probability_equal_to_threshold_is_not_reported(self, engine):
        """Test that falling to exactly half of P(T) is not below half of it.

        X (base 0.5) supports T with strength 0.5 and T's base is 0.2, so P(T) = 1 - 0.8 * 0.75 = 0.4
        and do(X = 0) leaves exactly the base, half of it; both come out as 0.2 in floating point. It
        is flagged once the threshold is raised.
        """
        graph = Graph()
        graph.add_node(Node("X", base=0.5))
        graph.add_node(Node("T", base=0.2))
        graph.add_relation(Relation("XT", "supports", "X", "T", strength=0.5))
        network = compile_graph(graph)
        assert single_points_of_failure(network, "T", threshold=0.5, engine=engine) == []
        (finding,) = single_points_of_failure(network, "T", threshold=0.5001, engine=engine)
        assert finding.id == "single-point-of-failure:X"
        close(finding.value, 0.2)

    def test_rounding_slack_is_relative(self, engine):
        """Test that a failure within 1e-12 of t P(T), relatively, is not reported and one beyond it is.

        The engine is wrapped so that do(X = 0) returns t P(T) scaled by 1 - 5e-13 or by 1 - 5e-12,
        whatever the network's own numbers, so the cutoff is checked without relying on rounding.
        """
        graph = Graph()
        graph.add_node(Node("X", base=0.5))
        graph.add_node(Node("T", base=0.2))
        graph.add_relation(Relation("XT", "supports", "X", "T", strength=0.5))
        network = compile_graph(graph)
        baseline = engine.query(network, Query({"T": True}))

        class Shifted(Engine):
            def __init__(self, factor):
                self.factor = factor

            def probability(self, network, assignment):
                return engine.probability(network, assignment)

            def query(self, queried, query):
                if queried is network:
                    return baseline
                return 0.5 * baseline * self.factor

        assert single_points_of_failure(network, "T", threshold=0.5, engine=Shifted(1 - 5e-13)) == []
        (finding,) = single_points_of_failure(network, "T", threshold=0.5, engine=Shifted(1 - 5e-12))
        assert finding.id == "single-point-of-failure:X"

    def test_impossible_constraints_raise(self, engine):
        """Test that a failure under which the exclusive constraints are impossible raises, with no evidence.

        A and B are certain unless Y vetoes A, and are exclusive. Y true leaves only B, so P(B) = 1 and
        the baseline is defined; failing Y makes both true, which the constraint rules out.
        """
        graph = Graph()
        graph.add_node(Node("Y", base=0.5))
        graph.add_node(Node("A", base=1.0))
        graph.add_node(Node("B", base=1.0))
        graph.add_relation(Relation("YA", "refutes", "Y", "A", strength=1.0))
        graph.add_relation(Relation("AB", "exclusive", "A", "B"))
        network = compile_graph(graph)
        close(engine.query(network, Query({"B": True})), 1.0)
        with pytest.raises(ZeroProbabilityError):
            single_points_of_failure(network, "B", engine=engine)

    def test_bad_threshold(self, network, engine):
        """Test that a threshold outside [0, 1] is rejected."""
        with pytest.raises(ValidationError, match="threshold"):
            single_points_of_failure(network, "T", threshold=-0.1, engine=engine)


@pytest.mark.parametrize("engine", ENGINES)
class TestValueOfInformation:
    """I(T; Y) = H(T) - sum over y of P(Y = y) H(T | Y = y), with binary entropies in bits."""

    def test_against_entropies(self, network, engine):
        """Test both premises against the entropy decomposition, A ranked first."""
        p_t = G * H
        info_a = binary_entropy(p_t) - a * binary_entropy(H) - (1 - a) * binary_entropy((1 - r) * H)
        t_given_b = G * (1 - (1 - t) * (1 - s))
        info_b = binary_entropy(p_t) - b * binary_entropy(t_given_b) - (1 - b) * binary_entropy(G * t)
        findings = value_of_information(network, "T", engine=engine)
        assert [f.id for f in findings] == ["value-of-information:A", "value-of-information:B"]
        assert all(f.diagnostic == VALUE_OF_INFORMATION for f in findings)
        np.testing.assert_allclose(findings[0].value, info_a, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(findings[1].value, info_b, rtol=1e-10, atol=0.0)
        close(findings[0].details["p_target_given_true"], H)
        close(findings[0].details["p_target_given_false"], (1 - r) * H)
        close(findings[0].details["p_target"], p_t)

    def test_independent_and_determining(self, engine):
        """Test that an unrelated root gives 0 bits and a copy of a fair coin gives exactly 1 bit.

        C is a fair coin and T = C exactly: base 0, and C supports T with strength 1, so T is true
        exactly when C is. U is an unrelated root.
        """
        graph = Graph()
        graph.add_node(Node("C", base=0.5))
        graph.add_node(Node("U", base=0.3))
        graph.add_node(Node("T", base=0.0))
        graph.add_relation(Relation("CT", "supports", "C", "T", strength=1.0))
        records = by_id(value_of_information(compile_graph(graph), "T", engine=engine))
        close(records["value-of-information:C"].value, 1.0)
        assert records["value-of-information:U"].value == 0.0
        assert "p_target_given_true" in records["value-of-information:U"].details

    def test_certain_premise_omits_undefined_conditional(self, engine):
        """Test that a premise that is never false has no P(T | Y = false) and gives 0 bits."""
        graph = Graph()
        graph.add_node(Node("Y", base=1.0))
        graph.add_node(Node("T", base=0.2))
        graph.add_relation(Relation("YT", "supports", "Y", "T", strength=0.5))
        (finding,) = value_of_information(compile_graph(graph), "T", engine=engine)
        assert finding.value == 0.0
        assert "p_target_given_false" not in finding.details
        close(finding.details["p_target_given_true"], 0.6)

    def test_independent_premise_is_exactly_zero_bits(self, engine):
        """Test that an unrelated root gets exactly 0 bits where rounding would give a negative value.

        With A = 0.86 supporting T (base 0.81, strength 0.51) and U = 0.35 unrelated, the four joint
        probabilities of T and U factorise only to within rounding, and the sum defining I(T; U)
        comes to about -3e-16 before it is clipped. Mutual information is never negative.
        """
        graph = Graph()
        graph.add_node(Node("A", base=0.86))
        graph.add_node(Node("U", base=0.35))
        graph.add_node(Node("T", base=0.81))
        graph.add_relation(Relation("AT", "supports", "A", "T", strength=0.51))
        records = by_id(value_of_information(compile_graph(graph), "T", engine=engine))
        assert records["value-of-information:U"].value == 0.0

    def test_merged_target_is_not_its_own_premise(self, engine):
        """Test that a node merged with the target is the target, not another variable."""
        graph = Graph()
        graph.add_node(Node("A", base=0.5))
        graph.add_node(Node("T", base=0.2))
        graph.add_node(Node("T2"))
        graph.add_relation(Relation("eq", "equivalent", "T", "T2"))
        graph.add_relation(Relation("AT", "supports", "A", "T2", strength=0.5))
        network = compile_graph(graph)
        assert [f.id for f in value_of_information(network, "T2", engine=engine)] == ["value-of-information:A"]
        assert [f.id for f in single_points_of_failure(network, "T", threshold=1.0, engine=engine)] == [
            "single-point-of-failure:A"
        ]


def test_unknown_target(network):
    """Test that every weak-point diagnostic rejects a target that is not a node."""
    for diagnostic in (sensitivity, crux, single_points_of_failure, value_of_information):
        with pytest.raises(ValidationError, match="no node 'nope'"):
            diagnostic(network, "nope")
