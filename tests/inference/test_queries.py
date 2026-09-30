"""Tests for the query functions and the edge cases every engine shares."""

from __future__ import annotations

import numpy as np
import pytest

from credencegraph.core import Graph, Node, Relation, ValidationError
from credencegraph.inference import (
    Answer,
    Enumeration,
    Query,
    VariableElimination,
    ZeroProbabilityError,
    conditional,
    intervene,
    joint,
    marginal,
)
from credencegraph.semantics import compile_graph

ENGINES = [pytest.param(VariableElimination(), id="elimination"), pytest.param(Enumeration(), id="enumeration")]


@pytest.fixture
def network():
    """A supports C, A is exclusive with B, A' is equivalent to A, and a person is carried."""
    graph = Graph()
    graph.add_node(Node("A", base=0.5))
    graph.add_node(Node("A2"))
    graph.add_node(Node("B", base=0.4))
    graph.add_node(Node("C", base=0.2))
    graph.add_node(Node("person", kind="person"))
    graph.add_relation(Relation("e", "equivalent", "A", "A2"))
    graph.add_relation(Relation("x", "exclusive", "A", "B"))
    graph.add_relation(Relation("s", "supports", "A", "C", strength=0.5))
    graph.add_relation(Relation("by", "authored_by", "C", "person"))
    return compile_graph(graph)


def close(actual, expected):
    """Assert agreement to 1e-12 relative, with no absolute slack."""
    np.testing.assert_allclose(actual.point, expected, rtol=1e-12, atol=0.0)


# Worlds allowed by exclusive(A, B), with weights P(A) P(B): (0,0) 0.3, (1,0) 0.3, (0,1) 0.2.
P_A = 0.3 / 0.8


@pytest.mark.parametrize("engine", ENGINES)
class TestQueryForms:
    """The four query functions."""

    def test_marginal(self, network, engine):
        """Test P(A) under the exclusion constraint, and P(not A)."""
        close(marginal(network, "A", engine=engine), P_A)
        close(marginal(network, "A", False, engine=engine), 1 - P_A)
        close(marginal(network, "C", engine=engine), P_A * 0.6 + (1 - P_A) * 0.2)

    def test_joint_forms(self, network, engine):
        """Test that a joint accepts a mapping or an iterable of ids that are all true."""
        close(joint(network, ["A", "C"], engine=engine), P_A * 0.6)
        close(joint(network, {"A": True, "C": False}, engine=engine), P_A * 0.4)
        close(joint(network, [], engine=engine), 1.0)

    def test_conditional_target_forms(self, network, engine):
        """Test that a conditional target may be a node id or an assignment."""
        close(conditional(network, "B", {"A": False}, engine=engine), 0.2 / 0.5)
        close(conditional(network, {"B": False, "C": True}, {"A": False}, engine=engine), 0.3 / 0.5 * 0.2)

    def test_intervene_with_evidence(self, network, engine):
        """Test do(C = 1) leaves A alone, and that evidence after an intervention is conditioned on."""
        close(intervene(network, "A", {"C": True}, engine=engine), P_A)
        close(intervene(network, "C", {"C": False}, engine=engine), 0.0)
        close(intervene(network, "C", {"A": True}, given={"B": False}, engine=engine), 0.6)

    def test_intervention_keeps_constraints(self, network, engine):
        """Test that do(A = 1) still observes exclusive(A, B), which forces B false."""
        close(intervene(network, "B", {"A": True}, engine=engine), 0.0)

    def test_merged_nodes(self, network, engine):
        """Test that A and A' answer as one proposition."""
        close(marginal(network, "A2", engine=engine), P_A)
        close(joint(network, {"A": True, "A2": False}, engine=engine), 0.0)
        close(conditional(network, "A", {"A2": True}, engine=engine), 1.0)

    def test_target_inside_evidence(self, network, engine):
        """Test that a target implied or contradicted by the evidence gives exactly 1 or 0."""
        assert conditional(network, "C", {"C": True}, engine=engine).point == 1.0
        assert conditional(network, "C", {"C": False}, engine=engine).point == 0.0

    def test_zero_probability_evidence(self, network, engine):
        """Test that impossible evidence raises rather than returning a number."""
        with pytest.raises(ZeroProbabilityError, match="probability zero"):
            conditional(network, "C", {"A": True, "B": True}, engine=engine)
        with pytest.raises(ZeroProbabilityError):
            conditional(network, "C", {"A": True, "A2": False}, engine=engine)

    def test_non_inference_node(self, network, engine):
        """Test that a carried node cannot be queried."""
        with pytest.raises(ValidationError, match="'person' is not an inference variable"):
            marginal(network, "person", engine=engine)


class TestArguments:
    """Validation of query arguments."""

    def test_default_engine(self, network):
        """Test that queries default to variable elimination."""
        assert marginal(network, "A") == Answer(VariableElimination().query(network, Query({"A": True})))

    def test_answer_to_dict(self):
        """Test the JSON form of an answer."""
        assert Answer(0.25).to_dict() == {"point": 0.25}

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"target": {"A": 1}}, "target value for 'A' must be True or False"),
            ({"target": {"A": True}, "evidence": ["A"]}, "evidence must be a mapping"),
            ({"target": {"A": True}, "interventions": {"B": None}}, "interventions value for 'B'"),
        ],
    )
    def test_query_validation(self, kwargs, match):
        """Test that assignments must map node ids to bools."""
        with pytest.raises(ValidationError, match=match):
            Query(**kwargs)

    def test_query_is_read_only(self):
        """Test that a query copies its assignments."""
        target = {"A": True}
        query = Query(target)
        target["B"] = False
        assert dict(query.target) == {"A": True}
        with pytest.raises(TypeError):
            query.target["C"] = True

    @pytest.mark.parametrize("target", [3, [1, 2], None])
    def test_bad_targets(self, network, target):
        """Test that a target must be an id, ids or an assignment."""
        with pytest.raises(ValidationError, match="must be a node id"):
            joint(network, target)
