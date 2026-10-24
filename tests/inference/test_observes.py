"""The ``observes`` relation: evidence in likelihood direction, through a sensitivity and a false-pass rate.

Every expected value here is written from Bayes' rule, independently of the engine: a claim H of
prior h, observed through checks of sensitivity s = P(pass | H) and false-pass rate f = P(pass | not H).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from credencegraph.core import Beta, CycleError, Graph, Node, Point, Relation
from credencegraph.inference import (
    Enumeration,
    VariableElimination,
    conditional,
    marginal,
    sample_parameters,
)
from credencegraph.semantics import CompileError, ParameterKey, compile_graph

ENGINES = [Enumeration(), VariableElimination()]


def _checked(prior, checks):
    """Build a claim ``h`` observed by one observation node per ``(sensitivity, false_pass)`` pair.

    The observations are ``o0``, ``o1``, ..., each the target of an ``observes`` relation ``r0``,
    ``r1``, ... from ``h``.
    """
    graph = Graph()
    graph.add_node(Node("h", base=prior))
    for k, (sensitivity, false_pass) in enumerate(checks):
        graph.add_node(Node(f"o{k}"))
        graph.add_relation(Relation(f"r{k}", "observes", "h", f"o{k}", sensitivity=sensitivity, false_pass=false_pass))
    return graph


def _posterior(prior, likelihoods):
    """Bayes' rule for H given independent observations, each with its P(obs | H) and P(obs | not H)."""
    true = prior * np.prod([given_true for given_true, _ in likelihoods], axis=0)
    false = (1 - prior) * np.prod([given_false for _, given_false in likelihoods], axis=0)
    return true / (true + false)


# (s, f): a confirming check, a check whose pass disconfirms (s < f), an uninformative one, and the
# exact corners where a pass or a fail is conclusive.
CHECKS = [(0.9, 0.2), (0.2, 0.9), (0.5, 0.5), (1.0, 0.3), (0.7, 0.0)]


class TestBayes:
    """Conditioning on observations reproduces Bayes' rule."""

    @pytest.mark.parametrize("engine", ENGINES, ids=type)
    @pytest.mark.parametrize(("s", "f"), CHECKS)
    @pytest.mark.parametrize("prior", [0.3, 0.85])
    def test_pass(self, engine, s, f, prior):
        """Test P(H | pass) = s h / (s h + f (1 - h))."""
        answer = conditional(compile_graph(_checked(prior, [(s, f)])), "h", {"o0": True}, engine=engine).point
        np.testing.assert_allclose(answer, _posterior(prior, [(s, f)]), rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("engine", ENGINES, ids=type)
    @pytest.mark.parametrize(("s", "f"), CHECKS)
    @pytest.mark.parametrize("prior", [0.3, 0.85])
    def test_fail(self, engine, s, f, prior):
        """Test P(H | fail) = (1 - s) h / ((1 - s) h + (1 - f) (1 - h))."""
        answer = conditional(compile_graph(_checked(prior, [(s, f)])), "h", {"o0": False}, engine=engine).point
        np.testing.assert_allclose(answer, _posterior(prior, [(1 - s, 1 - f)]), rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("engine", ENGINES, ids=type)
    def test_observation_marginal(self, engine):
        """Test P(pass) = s h + f (1 - h)."""
        answer = marginal(compile_graph(_checked(0.3, [(0.9, 0.2)])), "o0", engine=engine).point
        np.testing.assert_allclose(answer, 0.9 * 0.3 + 0.2 * 0.7, rtol=1e-12, atol=0.0)

    def test_the_sign_follows_the_parameters(self):
        """Test that a pass raises the claim when s > f, lowers it when s < f, and leaves it when s = f.

        One relation type covers all three: no choice between ``supports`` and ``refutes`` is made.
        """
        passed = {
            (s, f): conditional(compile_graph(_checked(0.3, [(s, f)])), "h", {"o0": True}).point
            for s, f in [(0.9, 0.2), (0.2, 0.9), (0.5, 0.5)]
        }
        assert passed[0.9, 0.2] > 0.3
        assert passed[0.2, 0.9] < 0.3
        np.testing.assert_allclose(passed[0.5, 0.5], 0.3, rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("engine", ENGINES, ids=type)
    def test_independent_checks_multiply_likelihoods(self, engine):
        """Test that two checks, one passed and one failed, combine as a product of likelihoods."""
        checks = [(0.9, 0.2), (0.6, 0.3), (0.8, 0.1)]
        network = compile_graph(_checked(0.3, checks))
        answer = conditional(network, "h", {"o0": True, "o1": False}, engine=engine).point
        expected = _posterior(0.3, [(0.9, 0.2), (1 - 0.6, 1 - 0.3)])
        np.testing.assert_allclose(answer, expected, rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("engine", ENGINES, ids=type)
    def test_check_not_run_leaves_the_claim(self, engine):
        """Test that an observation with no given value is summed out and leaves every answer unchanged."""
        graph = Graph()
        graph.add_node(Node("h", base=0.3))
        graph.add_node(Node("c", base=0.1))
        graph.add_relation(Relation("hc", "supports", "h", "c", strength=0.7))
        bare = compile_graph(graph)
        graph.add_node(Node("o"))
        graph.add_relation(Relation("ho", "observes", "h", "o", sensitivity=0.9, false_pass=0.2))
        checked = compile_graph(graph)
        for evidence in ({}, {"c": True}, {"c": False}):
            np.testing.assert_allclose(
                conditional(checked, "h", evidence, engine=engine).point,
                conditional(bare, "h", evidence, engine=engine).point,
                rtol=1e-12,
                atol=0.0,
            )


class TestMultilinearity:
    """Each of s and f enters every joint probability with degree at most one."""

    @settings(max_examples=200, deadline=None)
    @given(
        prior=st.floats(0, 1),
        s=st.floats(0, 1),
        f=st.floats(0, 1),
        t=st.floats(0, 1),
        kind=st.sampled_from(["sensitivity", "false_pass"]),
        claim=st.integers(0, 1),
        observed=st.integers(0, 1),
    )
    def test_joint_is_a_line_in_each_parameter(self, prior, s, f, t, kind, claim, observed):  # noqa: PLR0913, PLR0917 - one argument per drawn value
        """Test P(H = h, O = o) at θ = t equals (1 - t) P(θ = 0) + t P(θ = 1) for θ = s and θ = f."""
        network = compile_graph(_checked(prior, [(s, f)]))
        key = ParameterKey(kind, "r0")
        engine = VariableElimination()
        assignment = {network.index("h"): claim, network.index("o0"): observed}
        low, high, mid = (engine.probability(network.with_parameters({key: v}), assignment) for v in (0.0, 1.0, t))
        np.testing.assert_allclose(mid, (1 - t) * low + t * high, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("seed", range(20))
    def test_point_answer_is_the_predictive_probability(self, seed):
        """Test that E[P(H, O)] / E[P(O)] over independent two-point s and f is P(H | O) at their means.

        This is the property that makes the point answer exact, checked on the parameters of two
        observations at once, with the prior varied too.
        """
        rng = np.random.default_rng(seed)
        network = compile_graph(_checked(0.4, [(0.8, 0.3), (0.6, 0.5)]))
        keys = list(network.parameters)
        lows, highs = rng.uniform(0, 1, len(keys)), rng.uniform(0, 1, len(keys))
        evidence = {network.index("o0"): int(rng.integers(0, 2)), network.index("o1"): int(rng.integers(0, 2))}
        engine = VariableElimination()
        numerator = denominator = 0.0
        for pick in itertools.product([0, 1], repeat=len(keys)):
            weight = np.prod([2 / 3 if bit else 1 / 3 for bit in pick])
            values = {key: high if bit else low for key, low, high, bit in zip(keys, lows, highs, pick, strict=True)}
            drawn = network.with_parameters(values)
            numerator += weight * engine.probability(drawn, {**evidence, network.index("h"): 1})
            denominator += weight * engine.probability(drawn, evidence)
        means = {key: (low + 2 * high) / 3 for key, low, high in zip(keys, lows, highs, strict=True)}
        named = {network.variables[i].name: bool(v) for i, v in evidence.items()}
        point = conditional(network.with_parameters(means), "h", named, draws=0).point
        np.testing.assert_allclose(numerator / denominator, point, rtol=1e-12, atol=1e-15)


# One kind of check, used twice: each edge gets its own copy of the calibration.
SENSITIVITY = Beta(18, 2)
FALSE_PASS = Beta(2, 8)
PRIOR = Beta(3, 7)


class TestBand:
    """The band draws s and f of every edge independently from their own credences."""

    def test_parameters_are_the_edges_own(self):
        """Test that each edge contributes its own sensitivity and false-pass parameters."""
        network = compile_graph(_checked(PRIOR, [(SENSITIVITY, FALSE_PASS)] * 2))
        assert list(network.parameters) == [
            ParameterKey("base", "h"),
            ParameterKey("sensitivity", "r0"),
            ParameterKey("false_pass", "r0"),
            ParameterKey("sensitivity", "r1"),
            ParameterKey("false_pass", "r1"),
        ]
        assert network.parameters[ParameterKey("sensitivity", "r1")] == SENSITIVITY

    def test_edges_of_one_kind_are_drawn_independently(self):
        """Test that two edges with the same credences get uncorrelated draws, not shared ones."""
        network = compile_graph(_checked(PRIOR, [(SENSITIVITY, FALSE_PASS)] * 2))
        samples = sample_parameters(network, 20000, rng=1)
        for kind in ("sensitivity", "false_pass"):
            first, second = samples[ParameterKey(kind, "r0")], samples[ParameterKey(kind, "r1")]
            # Uncorrelated draws give r ~ N(0, 1/20000): 0.03 is more than four standard deviations.
            assert abs(np.corrcoef(first, second)[0, 1]) < 0.03

    def test_band_is_the_closed_form_at_the_same_draws(self):
        """Test the band and mean over draws against Bayes' rule evaluated at each parameter draw."""
        network = compile_graph(_checked(PRIOR, [(SENSITIVITY, FALSE_PASS)] * 2))
        answer = conditional(network, "h", {"o0": True, "o1": False}, draws=4000, rng=7)
        drawn = sample_parameters(network, 4000, rng=7)
        prior = drawn[ParameterKey("base", "h")]
        s0, f0 = drawn[ParameterKey("sensitivity", "r0")], drawn[ParameterKey("false_pass", "r0")]
        s1, f1 = drawn[ParameterKey("sensitivity", "r1")], drawn[ParameterKey("false_pass", "r1")]
        true, false = prior * s0 * (1 - s1), (1 - prior) * f0 * (1 - f1)
        per_draw = true / (true + false)
        expected = np.quantile(per_draw, (0.05, 0.5, 0.95))
        np.testing.assert_allclose([answer.band.q05, answer.band.q50, answer.band.q95], expected, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(answer.mean_over_draws, per_draw.mean(), rtol=1e-12, atol=0.0)

    def test_band_against_an_independent_monte_carlo(self):
        """Test the band's quantiles against a Monte Carlo over (s, f) written without the engine.

        The reference draws the prior and every edge's s and f independently, with its own generator,
        and evaluates Bayes' rule. Each quantile the engine reports must sit where the reference's
        distribution function puts that probability, within four binomial standard errors of both
        samples. Sharing one draw of s and f between the two edges, which is what reusing a check
        kind's calibration would mean, widens the band and fails this test.
        """
        engine_draws, reference_draws = 4000, 400000
        network = compile_graph(_checked(PRIOR, [(SENSITIVITY, FALSE_PASS)] * 2))
        band = conditional(network, "h", {"o0": True, "o1": True}, draws=engine_draws, rng=11).band

        rng = np.random.default_rng(12345)
        prior = rng.beta(3, 7, reference_draws)
        s0, s1 = rng.beta(18, 2, reference_draws), rng.beta(18, 2, reference_draws)
        f0, f1 = rng.beta(2, 8, reference_draws), rng.beta(2, 8, reference_draws)
        reference = np.sort(_posterior(prior, [(s0, f0), (s1, f1)]))

        for q, value in ((0.05, band.q05), (0.5, band.q50), (0.95, band.q95)):
            at = np.searchsorted(reference, value) / reference_draws
            allowance = 4 * np.sqrt(q * (1 - q)) * (1 / np.sqrt(engine_draws) + 1 / np.sqrt(reference_draws))
            assert abs(at - q) <= allowance, (q, at)

    def test_check_not_run_leaves_the_band(self):
        """Test that an unobserved observation, uncertain s and f included, leaves the band of the claim."""
        bare = compile_graph(_checked(PRIOR, []))
        checked = compile_graph(_checked(PRIOR, [(SENSITIVITY, FALSE_PASS)]))
        # The prior is the first parameter of both networks, so one seed gives it the same draws.
        one, other = marginal(bare, "h", draws=2000, rng=3), marginal(checked, "h", draws=2000, rng=3)
        np.testing.assert_allclose(other.point, one.point, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            [other.band.q05, other.band.q50, other.band.q95], [one.band.q05, one.band.q50, one.band.q95], rtol=1e-12
        )


class TestCompile:
    """How an ``observes`` relation compiles, and what an observation may not have."""

    def test_observation_table(self):
        """Test that the observation's table is P(O = o | H = h) = [[1 - f, f], [1 - s, s]]."""
        network = compile_graph(_checked(0.3, [(0.9, 0.2)]))
        variable = network.variables[network.index("o0")]
        assert variable.base is None
        assert [link.type for link in variable.links] == ["observes"]
        factor = network.factors[variable.index]
        assert factor.scope == (network.index("h"), variable.index)
        np.testing.assert_array_equal(factor.table, [[1 - 0.2, 0.2], [1 - 0.9, 0.9]])

    def test_observation_may_be_a_parent(self):
        """Test that an observation can itself support another node."""
        graph = _checked(0.3, [(0.9, 0.2)])
        graph.add_node(Node("x", base=0.1))
        graph.add_relation(Relation("ox", "supports", "o0", "x", strength=0.5))
        network = compile_graph(graph)
        p_pass = 0.9 * 0.3 + 0.2 * 0.7
        expected = p_pass * (1 - 0.9 * 0.5) + (1 - p_pass) * 0.1
        np.testing.assert_allclose(marginal(network, "x").point, expected, rtol=1e-12, atol=0.0)

    def test_observation_with_a_base(self):
        """Test that an observation may not have a base: its relation sets its probability."""
        graph = Graph()
        graph.add_node(Node("h", base=0.3))
        graph.add_node(Node("o", base=0.5))
        graph.add_relation(Relation("ho", "observes", "h", "o", sensitivity=0.9, false_pass=0.2))
        with pytest.raises(CompileError, match=r"observation 'o' .* node 'o' gives it a base"):
            compile_graph(graph)

    def test_observation_with_a_base_through_an_equivalent(self):
        """Test that a base reaching an observation through a merged node is refused too."""
        graph = _checked(0.3, [(0.9, 0.2)])
        graph.add_node(Node("twin", base=0.5))
        graph.add_relation(Relation("eq", "equivalent", "o0", "twin"))
        with pytest.raises(CompileError, match="node 'twin' gives it a base"):
            compile_graph(graph)

    @pytest.mark.parametrize(
        "extra",
        [
            Relation("x", "supports", "c", "o0", strength=0.5),
            Relation("x", "observes", "c", "o0", sensitivity=0.5, false_pass=0.1),
        ],
        ids=["supports", "observes"],
    )
    def test_observation_with_another_parent(self, extra):
        """Test that an observation takes no other relation that makes a parent."""
        graph = _checked(0.3, [(0.9, 0.2)])
        graph.add_node(Node("c", base=0.5))
        graph.add_relation(extra)
        with pytest.raises(CompileError, match=r"relations \['x'\] also point into it"):
            compile_graph(graph)

    def test_cycle_through_observes(self):
        """Test that an ``observes`` relation takes part in the acyclicity rule."""
        graph = _checked(0.3, [(0.9, 0.2)])
        with pytest.raises(CycleError):
            graph.add_relation(Relation("back", "supports", "o0", "h", strength=0.5))

    def test_point_credences_are_not_drawn(self):
        """Test that s and f given as bare floats are ``Point`` parameters, held in every draw."""
        network = compile_graph(_checked(PRIOR, [(0.9, 0.2)]))
        assert network.parameters[ParameterKey("sensitivity", "r0")] == Point(0.9)
        assert set(sample_parameters(network, 3, rng=0)) == {ParameterKey("base", "h")}
