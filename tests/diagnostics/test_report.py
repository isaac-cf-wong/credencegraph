"""The combined report and the JSON form of a finding."""

from __future__ import annotations

import json

import pytest

from credencegraph.core import Beta, Graph, Node, Relation, SourceAnchor, ValidationError
from credencegraph.diagnostics import Finding, diagnose

ANCHOR = SourceAnchor("doi:10.0000/example")


def argument(stated=0.9):
    """A premise P, cited, that T requires with strength 0.8; T is stated at ``stated``."""
    graph = Graph()
    graph.add_node(Node("P", base=Beta(6, 4), sources=[ANCHOR]))
    graph.add_node(Node("L", base=0.5))
    graph.add_node(Node("T", base=0.3, stated=stated))
    graph.add_relation(Relation("PT", "requires", "P", "T", strength=0.8))
    graph.add_relation(Relation("LT", "supports", "L", "T", strength=0.5))
    return graph


def test_without_target():
    """Test that without a target only the structural and claim diagnostics run."""
    findings = diagnose(argument())
    assert [f.id for f in findings] == ["unanchored:L", "overclaim:T"]


def test_with_target_groups_every_diagnostic():
    """Test the order of the groups, and that they match the single diagnostics' counts."""
    findings = diagnose(argument(), "T")
    groups = [f.diagnostic for f in findings]
    assert groups == [
        "unanchored",
        "overclaim",
        *["sensitivity"] * 5,
        *["crux"] * 5,
        "single-point-of-failure",
        *["value-of-information"] * 2,
    ]
    assert len({f.id for f in findings}) == len(findings)


def test_missing_parameter_stops_the_report():
    """Test that a graph that cannot compile gets only the structural findings, even with a target."""
    graph = argument()
    graph.add_node(Node("Q"))
    graph.add_relation(Relation("QT", "supports", "Q", "T", strength=0.4))
    findings = diagnose(graph, "T")
    assert [f.id for f in findings] == ["missing-parameter:base:Q", "unanchored:L", "unanchored:Q"]


def test_thresholds_are_passed_on():
    """Test that the claim and failure thresholds reach their diagnostics."""
    ids = [f.id for f in diagnose(argument(stated=0.4), "T", claim_threshold=0.05, failure_threshold=0.0)]
    assert "overclaim:T" in ids
    assert not any(i.startswith("single-point-of-failure") for i in ids)


def test_unknown_target():
    """Test that a target that is not a node is rejected."""
    with pytest.raises(ValidationError, match="no node 'nope'"):
        diagnose(argument(), "nope")


def test_report_is_strict_json():
    """Test that a full report serialises without NaN or infinities, and round-trips."""
    records = [finding.to_dict() for finding in diagnose(argument(), "T")]
    text = json.dumps(records, allow_nan=False)
    assert json.loads(text) == records
    for record in records:
        assert set(record) == {"id", "diagnostic", "nodes", "relations", "value", "details", "message"}
        assert "\n" not in record["message"]


def test_finding_to_dict():
    """Test the JSON form of a finding, and that it holds read-only copies."""
    details = {"sd": 0.1}
    finding = Finding("crux:base:x", "crux", "one line", nodes=["x"], value=0.05, details=details)
    details["sd"] = 9.0
    assert finding.to_dict() == {
        "id": "crux:base:x",
        "diagnostic": "crux",
        "nodes": ["x"],
        "relations": [],
        "value": 0.05,
        "details": {"sd": 0.1},
        "message": "one line",
    }
    with pytest.raises(TypeError):
        finding.details["sd"] = 1.0  # type: ignore[index]
