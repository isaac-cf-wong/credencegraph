"""Property tests for the common-parent pattern that keeps supports from one source from piling up.

The ``supports`` relations into a node combine as a noisy OR of independent reasons. Supports that rest
on one source get a common parent, a proposition each of them requires with strength 1; these tests pin
what that pattern guarantees, against closed forms written independently of the engine, and that the
annotations which record where a support came from never move a credence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _graphs import proposition_ids, random_assignment, random_graph
from hypothesis import given, reject, settings
from hypothesis import strategies as st

from credencegraph.core import Graph, Node, Relation, ValidationError
from credencegraph.inference import ZeroProbabilityError, conditional, marginal
from credencegraph.semantics import compile_graph

SETTINGS = settings(max_examples=150, deadline=None)
# Every probability here is a sum of terms of order 1, and 1 - prod(1 - s_i) cancels when the strengths
# are tiny, so the engine and a closed form agree to roundoff in absolute terms, not relative ones.
ROUNDOFF = 1e-15
PROBABILITIES = st.floats(0.0, 1.0)
EXTRA = st.one_of(
    st.none(), st.tuples(st.sampled_from(["requires", "supports", "refutes"]), PROBABILITIES, PROBABILITIES)
)


def common_parent(base, sound, reasons, extra=None):
    """Build a claim supported by reasons that each require one common parent.

    Args:
        base: The claim's base b.
        sound: The common parent's credence c.
        reasons: ``(base, strength)`` of each reason: its credence given the common parent holds, and
            the strength of its support for the claim.
        extra: ``(type, base, strength)`` of one more parent of the claim, independent of the rest,
            or ``None``.

    Returns:
        The graph.
    """
    graph = Graph()
    graph.add_node(Node("claim", base=base))
    graph.add_node(Node("sound", base=sound))
    for i, (reason_base, strength) in enumerate(reasons):
        graph.add_node(Node(f"reason{i}", base=reason_base))
        graph.add_relation(Relation(f"needs{i}", "requires", "sound", f"reason{i}", strength=1.0))
        graph.add_relation(Relation(f"why{i}", "supports", f"reason{i}", "claim", strength=strength))
    if extra is not None:
        rtype, other_base, strength = extra
        graph.add_node(Node("other", base=other_base))
        graph.add_relation(Relation("other", rtype, "other", "claim", strength=strength))
    return graph


def p_claim(graph):
    """Return the point answer P(claim)."""
    return marginal(compile_graph(graph), "claim", draws=0).point


@SETTINGS
@given(base=PROBABILITIES, sound=PROBABILITIES, n=st.integers(2, 8), extra=EXTRA)
def test_restatements_gain_nothing(base, sound, n, extra):
    """Test that N restatements of one argument give the claim exactly what one gives.

    A restatement holds whenever the common parent does and supports the claim with strength 1, so the
    doubt sits in the common parent alone. Without another parent, one restatement gives b + (1 - b) c.
    """
    one = p_claim(common_parent(base, sound, [(1.0, 1.0)], extra))
    many = p_claim(common_parent(base, sound, [(1.0, 1.0)] * n, extra))
    assert many == pytest.approx(one, rel=1e-12, abs=ROUNDOFF)
    if extra is None:
        assert one == pytest.approx(base + (1 - base) * sound, rel=1e-12, abs=ROUNDOFF)


@SETTINGS
@given(
    base=PROBABILITIES,
    sound=PROBABILITIES,
    reasons=st.lists(st.tuples(PROBABILITIES, PROBABILITIES), min_size=1, max_size=8),
)
def test_supports_saturate_at_the_common_parent(base, sound, reasons):
    """Test that reasons sharing a common parent lift the claim no higher than b + (1 - b) c.

    If the common parent fails every reason fails and the claim keeps b; if it holds, reason i holds
    with its own base b_i and fires with strength s_i, so

        P(claim) = (1 - c) b + c (1 - (1 - b) prod_i (1 - b_i s_i)).
    """
    silent = math.prod(1 - b * s for b, s in reasons)  # P(no reason fires | the common parent holds)
    expected = (1 - sound) * base + sound * (1 - (1 - base) * silent)
    actual = p_claim(common_parent(base, sound, reasons))
    assert actual == pytest.approx(expected, rel=1e-12, abs=ROUNDOFF)
    assert actual <= base + (1 - base) * sound + 1e-12


def test_the_documented_pile():
    """Test the numbers the reading-the-diagnostics page quotes for ten reasons of strength 0.2."""
    independent = Graph()
    independent.add_node(Node("claim", base=0.0))
    for i in range(10):
        independent.add_node(Node(f"reason{i}", base=1.0))
        independent.add_relation(Relation(f"why{i}", "supports", f"reason{i}", "claim", strength=0.2))
    assert p_claim(independent) == pytest.approx(1 - 0.8**10, rel=1e-12, abs=0.0)  # 0.892626
    assert p_claim(common_parent(0.0, 0.5, [(1.0, 0.2)] * 10)) == pytest.approx(0.5 * (1 - 0.8**10), rel=1e-12, abs=0.0)
    assert p_claim(common_parent(0.0, 0.5, [(1.0, 0.2)])) == pytest.approx(0.1, rel=1e-12, abs=0.0)
    for n in (1, 10):
        assert p_claim(common_parent(0.0, 0.2, [(1.0, 1.0)] * n)) == pytest.approx(0.2, rel=1e-12, abs=0.0)


ANNOTATION_TYPES = ("derived_from", "authored_by", "cites", "Supports", "requires ")


@SETTINGS
@given(seed=st.integers(0, 2**32 - 1))
def test_annotations_never_change_a_credence(seed):
    """Test that adding annotation relations, between any nodes and in any direction, moves no answer."""
    graphs = []
    for _ in range(2):
        graph = random_graph(np.random.default_rng(seed))
        graph.add_node(Node("dataset", kind="dataset"))
        graphs.append(graph)
    plain, annotated = graphs
    rng = np.random.default_rng(seed + 1)
    everyone = list(annotated.nodes)
    for k in range(int(rng.integers(1, 8))):
        source, target = (everyone[int(i)] for i in rng.integers(0, len(everyone), size=2))
        rtype = ANNOTATION_TYPES[int(rng.integers(0, len(ANNOTATION_TYPES)))]
        annotated.add_relation(Relation(f"note{k}", rtype, source, target))
    before, after = compile_graph(plain), compile_graph(annotated)
    ids = proposition_ids(plain)
    try:
        points = [marginal(before, node_id, draws=0).point for node_id in ids]
    except ZeroProbabilityError:  # exclusive constraints that no assignment satisfies
        reject()
    else:
        for node_id, point in zip(ids, points, strict=True):
            assert marginal(after, node_id, draws=0).point == point
    evidence = random_assignment(rng, ids, 3)
    target = {ids[int(rng.integers(0, len(ids)))]: True}
    try:
        expected = conditional(before, target, evidence, draws=0).point
    except (ZeroProbabilityError, ValidationError):  # impossible evidence, or a target it contradicts
        reject()
    else:
        assert conditional(after, target, evidence, draws=0).point == expected
