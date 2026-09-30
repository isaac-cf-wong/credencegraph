"""Property tests over random networks."""

from __future__ import annotations

import itertools

import numpy as np
from _graphs import proposition_ids, random_assignment, random_graph
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from credencegraph.core import ValidationError
from credencegraph.inference import VariableElimination, ZeroProbabilityError, conditional, intervene
from credencegraph.semantics import PROPOSITION, compile_graph

SEEDS = st.integers(0, 2**32 - 1)
ENGINE = VariableElimination()
SETTINGS = settings(max_examples=150, deadline=None)


@SETTINGS
@given(seed=SEEDS)
def test_probabilities_lie_in_the_unit_interval(seed):
    """Test that every conditional and interventional answer lies in [0, 1]."""
    rng = np.random.default_rng(seed)
    graph = random_graph(rng)
    network = compile_graph(graph)
    ids = proposition_ids(graph)
    for _ in range(5):
        target = random_assignment(rng, ids, 3)
        try:
            answers = [
                conditional(network, target, random_assignment(rng, ids, 3)).point,
                intervene(network, target, random_assignment(rng, ids, 2)).point,
            ]
        except (ZeroProbabilityError, ValidationError):
            continue  # impossible evidence, or merged nodes set to different values
        for answer in answers:
            assert 0.0 <= answer <= 1.0


@SETTINGS
@given(seed=SEEDS)
def test_complements_sum_to_one(seed):
    """Test that the answers over every assignment of a target set sum to 1, under any evidence."""
    rng = np.random.default_rng(seed)
    graph = random_graph(rng)
    network = compile_graph(graph)
    ids = proposition_ids(graph)
    evidence = random_assignment(rng, ids, 3)
    targets = list(rng.choice(ids, size=min(3, len(ids)), replace=False))
    try:
        total = sum(
            conditional(network, dict(zip(targets, values, strict=True)), evidence).point
            for values in itertools.product([False, True], repeat=len(targets))
        )
    except ZeroProbabilityError:
        assume(False)
    np.testing.assert_allclose(total, 1.0, rtol=1e-12, atol=0.0)


def _corners(network, keys, lows, highs):
    """Yield (weight, network) over the corners of a box of parameter values.

    Each parameter independently takes its low value with probability 1/3 and its high value with
    probability 2/3, so the mean of parameter i is (lows[i] + 2 highs[i]) / 3.
    """
    for pick in itertools.product([0, 1], repeat=len(keys)):
        weight = 1.0
        values = {}
        for key, low, high, bit in zip(keys, lows, highs, pick, strict=True):
            weight *= 2 / 3 if bit else 1 / 3
            values[key] = high if bit else low
        yield weight, network.with_parameters(values)


def _raw_assignment(rng, network):
    """Draw values for up to four proposition variables, plus every constraint observed true.

    The result is a variable-index assignment for ``Engine.probability``: its probability is the
    unnormalised joint P(A, E, constraints), which is what is multilinear in the parameters.
    """
    propositions = [v.index for v in network.variables if v.kind == PROPOSITION]
    chosen = rng.choice(propositions, size=int(rng.integers(1, min(4, len(propositions)) + 1)), replace=False)
    return {int(i): int(rng.integers(0, 2)) for i in chosen}, dict.fromkeys(network.constraints, 1)


def _box(rng, network, n_keys):
    """Pick up to ``n_keys`` parameters and the corners of a box of values for them."""
    keys = list(network.parameters)
    chosen = [keys[i] for i in rng.choice(len(keys), size=min(n_keys, len(keys)), replace=False)]
    lows, highs = rng.uniform(0, 1, len(chosen)), rng.uniform(0, 1, len(chosen))
    means = {key: (low + 2 * high) / 3 for key, low, high in zip(chosen, lows, highs, strict=True)}
    return chosen, lows, highs, means


@SETTINGS
@given(seed=SEEDS)
def test_joint_probability_is_multilinear(seed):
    """Test that averaging P(A, E) over independent two-point parameters equals P(A, E) at their means.

    That is exactly multilinearity: each parameter enters the joint probability with degree at most
    one, so its expectation passes through. Up to three parameters are varied at once.
    """
    rng = np.random.default_rng(seed)
    network = compile_graph(random_graph(rng, extremes=False))
    chosen, lows, highs, means = _box(rng, network, 3)
    values, constraints = _raw_assignment(rng, network)
    assignment = {**values, **constraints}

    averaged = sum(w * ENGINE.probability(net, assignment) for w, net in _corners(network, chosen, lows, highs))
    at_means = ENGINE.probability(network.with_parameters(means), assignment)
    # Table entries such as 1 - P(X = 1) lose absolute precision near 1, hence the absolute slack.
    np.testing.assert_allclose(averaged, at_means, rtol=1e-9, atol=1e-12)


@SETTINGS
@given(seed=SEEDS)
def test_point_answer_is_the_predictive_probability(seed):
    """Test that E[P(A, E)] / E[P(E)] over parameter draws equals the point answer P(A | E) at the means.

    The exclusive constraints are part of the evidence on both sides.
    """
    rng = np.random.default_rng(seed)
    graph = random_graph(rng, extremes=False)
    network = compile_graph(graph)
    chosen, lows, highs, means = _box(rng, network, 2)
    values, constraints = _raw_assignment(rng, network)
    target_index, target_value = next(iter(values.items()))
    evidence = {i: v for i, v in values.items() if i != target_index}

    numerator = denominator = 0.0
    for weight, net in _corners(network, chosen, lows, highs):
        numerator += weight * ENGINE.probability(net, {**evidence, **constraints, target_index: target_value})
        denominator += weight * ENGINE.probability(net, {**evidence, **constraints})
    assume(denominator > 1e-6)
    name = network.variables
    point = conditional(
        network.with_parameters(means),
        {name[target_index].name: bool(target_value)},
        {name[i].name: bool(v) for i, v in evidence.items()},
    ).point
    np.testing.assert_allclose(numerator / denominator, point, rtol=1e-8, atol=1e-12)


@SETTINGS
@given(seed=SEEDS, value=st.booleans())
def test_intervening_on_a_root_equals_conditioning(seed, value):
    """Test that do(R = r) and conditioning on R = r agree for a root R, with and without evidence."""
    rng = np.random.default_rng(seed)
    graph = random_graph(rng)
    network = compile_graph(graph)
    roots = [v for v in network.variables if v.kind == PROPOSITION and not v.parents]
    root = roots[int(rng.integers(0, len(roots)))]
    root_id = root.members[int(rng.integers(0, len(root.members)))]
    others = [node_id for node_id in proposition_ids(graph) if node_id not in root.members]
    ids = others or [root_id]
    target = {ids[int(rng.integers(0, len(ids)))]: bool(rng.integers(0, 2))}
    evidence = random_assignment(rng, others, 2)
    try:
        conditioned = conditional(network, target, {**evidence, root_id: value}, engine=ENGINE).point
    except ZeroProbabilityError:
        assume(False)
    intervened = intervene(network, target, {root_id: value}, given=evidence, engine=ENGINE).point
    np.testing.assert_allclose(intervened, conditioned, rtol=1e-12, atol=0.0)
