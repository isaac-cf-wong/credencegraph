"""Tests for parameter uncertainty: Monte Carlo bands, the mean over draws, and the point answer."""

from __future__ import annotations

import numpy as np
import pytest
from _graphs import proposition_ids, random_assignment, random_graph

from credencegraph.core import Beta, Graph, Node, Relation, ValidationError
from credencegraph.inference import (
    DEFAULT_DRAWS,
    Answer,
    Band,
    Query,
    VariableElimination,
    ZeroProbabilityError,
    answers_over_draws,
    conditional,
    has_spread,
    intervene,
    joint,
    marginal,
    sample_parameters,
)
from credencegraph.semantics import PROPOSITION, ParameterKey, compile_graph

ENGINE = VariableElimination()

# A four-node network that uses all three strength-carrying relation types:
#   A --requires(r)--> C,   B --supports(s)--> D,   C --refutes(f)--> D,
# with bases a, b, c, d. The spreads are wide on purpose, so that the mean of P(A | D) over draws is
# far from the point answer.
CREDENCES = {
    "a": Beta(1, 3),
    "b": Beta(3, 2),
    "c": Beta(4, 2),
    "d": Beta(1, 4),
    "r": Beta(6, 2),
    "s": Beta(2, 2),
    "f": Beta(3, 2),
}
KEYS = {
    "a": ParameterKey("base", "A"),
    "b": ParameterKey("base", "B"),
    "c": ParameterKey("base", "C"),
    "d": ParameterKey("base", "D"),
    "r": ParameterKey("strength", "r"),
    "s": ParameterKey("strength", "s"),
    "f": ParameterKey("strength", "f"),
}

MEANS = {name: credence.mean for name, credence in CREDENCES.items()}


@pytest.fixture
def four_nodes():
    """The four-node network described above."""
    graph = Graph()
    for name in "ABCD":
        graph.add_node(Node(name, base=CREDENCES[name.lower()]))
    graph.add_relation(Relation("r", "requires", "A", "C", strength=CREDENCES["r"]))
    graph.add_relation(Relation("s", "supports", "B", "D", strength=CREDENCES["s"]))
    graph.add_relation(Relation("f", "refutes", "C", "D", strength=CREDENCES["f"]))
    return compile_graph(graph)


def closed_form(theta):
    """Return P(A, D) and P(D) for the four-node network, worked out by hand.

    B and the sufficiency of D are independent of A and C, so P(D) factorises into the chance that
    D's sufficiency fires, ``b (1 - (1 - d)(1 - s)) + (1 - b) d``, times the chance that the refuter
    C does not veto it, which is ``1 - c f`` when A holds and ``1 - (1 - r) c f`` when it does not.
    ``theta`` maps the parameter names above to values, or to arrays of draws evaluated element-wise.
    """
    a, b, c, d, r, s, f = (theta[name] for name in "abcdrsf")
    sufficiency = b * (1 - (1 - d) * (1 - s)) + (1 - b) * d
    no_veto_given_a = 1 - c * f
    no_veto_given_not_a = 1 - (1 - r) * c * f
    p_ad = a * sufficiency * no_veto_given_a
    p_d = sufficiency * (a * no_veto_given_a + (1 - a) * no_veto_given_not_a)
    return p_ad, p_d


def ratio_and_error(numerator, denominator):
    """Return the ratio of means of paired draws and its delta-method standard error."""
    n = len(numerator)
    ratio = numerator.mean() / denominator.mean()
    cov = np.cov(numerator, denominator)
    variance = (cov[0, 0] - 2 * ratio * cov[0, 1] + ratio**2 * cov[1, 1]) / n
    return ratio, np.sqrt(max(variance, 0.0)) / denominator.mean()


class TestFourNodeTable:
    """At means, the Monte Carlo predictive value and the mean over draws, on the four-node network."""

    DRAWS = 400_000

    def closed_form_draws(self, seed):
        """Draw the seven parameters independently and evaluate the closed form at every draw."""
        rng = np.random.default_rng(seed)
        theta = {name: rng.beta(c.alpha, c.beta, self.DRAWS) for name, c in CREDENCES.items()}
        return closed_form(theta)

    def test_point_answers_match_the_closed_form(self, four_nodes):
        """Test the engine's point answers against the hand-derived formula at the means."""
        p_ad, p_d = closed_form(MEANS)
        np.testing.assert_allclose(marginal(four_nodes, "D", draws=0).point, p_d, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            conditional(four_nodes, "A", {"D": True}, draws=0).point, p_ad / p_d, rtol=1e-12, atol=0.0
        )

    def test_point_answer_is_the_monte_carlo_predictive(self, four_nodes):
        """Test that the point answer equals E[P(A, D)] / E[P(D)] within Monte Carlo error.

        With these draws the predictive estimates sit 0.7 and 2.2 standard errors from the point
        answers; over 40 other seeds the standardised gap had mean 0.07 and standard deviation 1.08.
        """
        p_ad, p_d = self.closed_form_draws(seed=1)
        point_d = marginal(four_nodes, "D", draws=0).point
        point_a_given_d = conditional(four_nodes, "A", {"D": True}, draws=0).point

        assert abs(p_d.mean() - point_d) < 4 * p_d.std() / np.sqrt(self.DRAWS)
        predictive, error = ratio_and_error(p_ad, p_d)
        assert abs(predictive - point_a_given_d) < 4 * error

    def test_mean_of_the_conditional_is_a_different_number(self, four_nodes):
        """Test that the average of P(A | D, theta) over draws is many standard errors from the point answer.

        This is why the two are reported under separate names. Measured gap: about 31 standard errors.
        """
        p_ad, p_d = self.closed_form_draws(seed=1)
        per_draw = p_ad / p_d
        point = conditional(four_nodes, "A", {"D": True}, draws=0).point
        assert per_draw.mean() - point > 20 * per_draw.std() / np.sqrt(self.DRAWS)

    def test_band_and_mean_are_the_per_draw_answers(self, four_nodes):
        """Test the band and mean_over_draws against the closed form evaluated on the very same draws."""
        answer = conditional(four_nodes, "A", {"D": True}, draws=500, rng=7)
        samples = sample_parameters(four_nodes, 500, rng=7)
        p_ad, p_d = closed_form({name: samples[key] for name, key in KEYS.items()})
        per_draw = p_ad / p_d

        np.testing.assert_allclose(answer.mean_over_draws, per_draw.mean(), rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(
            [answer.band.q05, answer.band.q50, answer.band.q95],
            np.quantile(per_draw, [0.05, 0.5, 0.95]),
            rtol=1e-12,
            atol=0.0,
        )
        assert answer.draws == 500

    def test_mean_over_draws_is_reported_separately(self, four_nodes):
        """Test that mean_over_draws is its own quantity: not the point answer, and several errors from it.

        With these draws the gap is 8.3 standard errors of mean_over_draws.
        """
        answer = conditional(four_nodes, "A", {"D": True}, draws=40_000, rng=0)
        samples = sample_parameters(four_nodes, 40_000, rng=0)
        p_ad, p_d = closed_form({name: samples[key] for name, key in KEYS.items()})
        per_draw = p_ad / p_d
        error = per_draw.std() / np.sqrt(len(per_draw))
        at_means = closed_form(MEANS)

        assert answer.point == conditional(four_nodes, "A", {"D": True}, draws=0).point
        np.testing.assert_allclose(answer.point, at_means[0] / at_means[1], rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(answer.mean_over_draws, per_draw.mean(), rtol=1e-12, atol=0.0)
        assert answer.mean_over_draws - answer.point > 4 * error


def _random_case(seed):
    """Build a random network and a satisfiable query on it: a node with parents given one or two others."""
    rng = np.random.default_rng(seed)
    graph = random_graph(rng, max_variables=6, extremes=False)
    network = compile_graph(graph)
    propositions = [v for v in network.variables if v.kind == PROPOSITION]
    children = [v for v in propositions if v.parents] or propositions
    target_id = children[int(rng.integers(0, len(children)))].name
    others = [node_id for node_id in proposition_ids(graph) if network.index(node_id) != network.index(target_id)]
    target = {target_id: bool(rng.integers(0, 2))}
    evidence = random_assignment(rng, others, 2) if others else {}
    try:
        conditional(network, target, evidence, draws=0)
    except ZeroProbabilityError:
        evidence = {}
    return network, target, evidence


def _predictive(network, target, evidence, samples):
    """Estimate E[P(A, E)] and E[P(E)] from parameter draws, observing every exclusive constraint."""
    constraints = dict.fromkeys(network.constraints, 1)
    given = {**{network.index(k): int(v) for k, v in evidence.items()}, **constraints}
    asked = {network.index(k): int(v) for k, v in target.items()}
    count = len(next(iter(samples.values())))
    numerator, denominator = np.empty(count), np.empty(count)
    for draw in range(count):
        drawn = network.with_parameters({key: float(values[draw]) for key, values in samples.items()})
        denominator[draw] = ENGINE.probability(drawn, given)
        numerator[draw] = ENGINE.probability(drawn, {**given, **asked})
    return numerator, denominator


def test_point_answer_is_the_predictive_probability_on_random_networks():
    """Test that the point answer equals the Monte Carlo predictive E[P(A, E)] / E[P(E)] on random networks.

    For each of 30 fixed seeds, ``E[P(A, E)]`` and ``E[P(E)]`` are estimated from the same 2000
    parameter draws, with the exclusive constraints counted as evidence on both sides, and their
    ratio must sit within 4.5 delta-method standard errors of the point answer (plus 1e-12 for
    queries that no uncertain parameter reaches, where the error is pure roundoff). The same draws
    must give ``mean_over_draws`` exactly, as the average of the per-draw answers.

    A query that no uncertain parameter reaches passes trivially, so the test also requires that
    enough seeds have a Monte Carlo error above 1e-6. Measured: 23 of 30, with standardised gaps
    between the predictive estimate and the point answer from -1.7 to 2.2. At this number of draws the
    random networks do not separate ``mean_over_draws`` from the point answer beyond Monte Carlo
    error; the four-node network above does.
    """
    draws = 2000
    moving = 0
    for seed in range(30):
        network, target, evidence = _random_case(seed)
        answer = conditional(network, target, evidence, draws=draws, rng=seed)
        samples = sample_parameters(network, draws, rng=seed)
        if not samples:
            # Every parameter is a point value, so the answer cannot move and no band is reported.
            assert answer.band is None
            continue
        numerator, denominator = _predictive(network, target, evidence, samples)
        predictive, error = ratio_and_error(numerator, denominator)
        assert abs(predictive - answer.point) <= 4.5 * error + 1e-12, seed
        np.testing.assert_allclose(answer.mean_over_draws, np.mean(numerator / denominator), rtol=1e-12, atol=0.0)
        moving += error > 1e-6
    assert moving >= 20


class TestBand:
    """When a band is reported, and what it contains."""

    def test_omitted_when_every_parameter_is_a_point(self):
        """Test that a network of point values gets no band, however many draws are asked for."""
        graph = Graph()
        graph.add_node(Node("x", base=0.3))
        graph.add_node(Node("y", base=0.2))
        graph.add_relation(Relation("s", "supports", "x", "y", strength=0.5))
        network = compile_graph(graph)
        answer = conditional(network, "x", {"y": True}, draws=500)
        assert not has_spread(network)
        assert (answer.band, answer.mean_over_draws, answer.draws) == (None, None, 0)
        assert answer.to_dict() == {"point": answer.point}

    def test_omitted_when_no_draws_are_asked_for(self, four_nodes):
        """Test that draws=0 skips the Monte Carlo step."""
        answer = marginal(four_nodes, "D", draws=0)
        assert (answer.band, answer.mean_over_draws, answer.draws) == (None, None, 0)

    def test_default_draws(self, four_nodes):
        """Test that queries draw DEFAULT_DRAWS times unless told otherwise."""
        answer = joint(four_nodes, ["A", "D"], rng=0)
        assert answer.draws == DEFAULT_DRAWS
        assert 0.0 <= answer.band.q05 <= answer.band.q50 <= answer.band.q95 <= 1.0

    def test_reproducible_with_a_seed(self, four_nodes):
        """Test that the same seed, or a generator seeded alike, gives the same band."""
        first = marginal(four_nodes, "D", draws=200, rng=3)
        assert marginal(four_nodes, "D", draws=200, rng=3) == first
        assert marginal(four_nodes, "D", draws=200, rng=np.random.default_rng(3)) == first
        assert marginal(four_nodes, "D", draws=200, rng=4) != first

    def test_intervention_applies_at_every_draw(self, four_nodes):
        """Test that under do(A = 1) the answer P(C) is the base of C at each draw, whatever r is."""
        answer = intervene(four_nodes, "C", {"A": True}, draws=300, rng=5)
        base_c = sample_parameters(four_nodes, 300, rng=5)[KEYS["c"]]
        np.testing.assert_allclose(answer.mean_over_draws, base_c.mean(), rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(answer.band.q50, np.median(base_c), rtol=1e-12, atol=0.0)

    def test_point_override_moves_the_band(self):
        """Test that overriding a Point parameter moves the band, while overriding a Beta one does not.

        The overridden Point value is kept in every draw, so the band is the one of a graph whose
        credence is that value; a Beta parameter is still drawn from its credence.
        """

        def network_with_base_a(base_a):
            graph = Graph()
            graph.add_node(Node("A", base=base_a))
            for name in "BCD":
                graph.add_node(Node(name, base=CREDENCES[name.lower()]))
            graph.add_relation(Relation("r", "requires", "A", "C", strength=CREDENCES["r"]))
            graph.add_relation(Relation("s", "supports", "B", "D", strength=CREDENCES["s"]))
            graph.add_relation(Relation("f", "refutes", "C", "D", strength=CREDENCES["f"]))
            return compile_graph(graph)

        network = network_with_base_a(0.25)
        before = conditional(network, "A", {"D": True}, draws=300, rng=6)
        point_overridden = conditional(network.with_parameters({KEYS["a"]: 0.9}), "A", {"D": True}, draws=300, rng=6)
        beta_overridden = conditional(network.with_parameters({KEYS["r"]: 0.1}), "A", {"D": True}, draws=300, rng=6)
        assert point_overridden == conditional(network_with_base_a(0.9), "A", {"D": True}, draws=300, rng=6)
        assert point_overridden.band.q05 > before.band.q95
        assert beta_overridden.point != before.point
        assert (beta_overridden.band, beta_overridden.mean_over_draws) == (before.band, before.mean_over_draws)

    def test_to_dict(self):
        """Test the JSON form of an answer with a band."""
        answer = Answer(0.4, Band(0.1, 0.35, 0.8), 0.42, 100)
        assert answer.to_dict() == {
            "point": 0.4,
            "band": {"q05": 0.1, "q50": 0.35, "q95": 0.8},
            "mean_over_draws": 0.42,
            "draws": 100,
        }

    @pytest.mark.parametrize(
        "fields",
        [(Band(0.1, 0.2, 0.3), None, 10), (None, 0.2, 10), (None, None, 10), (Band(0.1, 0.2, 0.3), 0.2, 0)],
    )
    def test_answer_fields_come_together(self, fields):
        """Test that a band, its mean and a positive draw count cannot be given one without the others."""
        with pytest.raises(ValidationError, match="together"):
            Answer(0.5, *fields)

    @pytest.mark.parametrize("draws", [-1, 1.5, True, "10"])
    def test_draws_must_be_a_non_negative_integer(self, four_nodes, draws):
        """Test that a bad draw count is rejected."""
        with pytest.raises(ValidationError, match="draws must be a non-negative integer"):
            marginal(four_nodes, "D", draws=draws)


class TestSampling:
    """Drawing the parameters."""

    def test_only_beta_parameters_are_drawn(self):
        """Test that point parameters keep their value and are not drawn."""
        graph = Graph()
        graph.add_node(Node("x", base=Beta(2, 3)))
        graph.add_node(Node("y", base=0.2))
        graph.add_relation(Relation("s", "supports", "x", "y", strength=0.5))
        samples = sample_parameters(compile_graph(graph), 50, rng=0)
        assert list(samples) == [ParameterKey("base", "x")]
        assert samples[ParameterKey("base", "x")].shape == (50,)

    def test_draws_stay_inside_the_open_interval(self):
        """Test that a Beta that rounds to exactly 0 or 1 in floating point is kept strictly inside (0, 1).

        About half the draws of Beta(0.001, 2) and a third of those of Beta(0.01, 0.01) round to an
        endpoint before they are clipped.
        """
        graph = Graph()
        graph.add_node(Node("x", base=Beta(0.001, 2)))
        graph.add_node(Node("y", base=Beta(0.01, 0.01)))
        samples = sample_parameters(compile_graph(graph), 5000, rng=0)
        for values in samples.values():
            assert np.all(values > 0.0)
            assert np.all(values < 1.0)

    def test_unequal_sample_lengths(self, four_nodes):
        """Test that every parameter must have the same number of draws."""
        samples = sample_parameters(four_nodes, 10, rng=0)
        samples[KEYS["a"]] = samples[KEYS["a"]][:5]
        with pytest.raises(ValidationError, match="same number of draws"):
            answers_over_draws(four_nodes, Query({"D": True}), ENGINE, samples)

    def test_zero_probability_at_a_draw_names_the_draw(self):
        """Test that evidence which underflows to probability zero at one draw says which draw."""
        graph = Graph()
        for name in "xyz":
            graph.add_node(Node(name, base=Beta(0.001, 2)))
        network = compile_graph(graph)
        with pytest.raises(ZeroProbabilityError, match=r"parameter draw \d+"):
            conditional(network, "x", {"y": True, "z": True}, draws=200, rng=0)
