"""The combined report and the JSON form of a finding."""

from __future__ import annotations

import json

import pytest

from credencegraph.core import Beta, Graph, Node, Relation, SourceAnchor, ValidationError
from credencegraph.diagnostics import Finding, Report, diagnose, diagnose_many
from credencegraph.semantics.network import Network

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
    """Test the order of the groups, and that they match the single diagnostics' counts.

    Failing P leaves 0.294 of P(T), so a failure threshold of 0.5 reports it.
    """
    findings = diagnose(argument(), "T", failure_threshold=0.5)
    groups = [f.diagnostic for f in findings]
    assert groups == [
        "unanchored",
        "overclaim",
        *["sensitivity"] * 5,
        *["crux"] * 5,
        "single-point-of-failure",
        *["failure-impact"] * 2,
        *["value-of-information"] * 2,
    ]
    assert len({f.id for f in findings}) == len(findings)


def test_failure_diagnostics_share_one_intervention_pass(monkeypatch):
    """Test that the single points of failure and the failure impact come from one pass over the variables.

    Each target fails every other variable exactly once, not once for each of the two diagnostics.
    """
    failed = []
    intervene = Network.intervene

    def counted(network, assignment):
        failed.append(dict(assignment))
        return intervene(network, assignment)

    monkeypatch.setattr(Network, "intervene", counted)
    diagnose(argument(), "T")
    assert failed == [{"P": False}, {"L": False}]
    failed.clear()
    graph = argument()
    graph.add_node(Node("U", base=0.4))
    graph.add_relation(Relation("TU", "supports", "T", "U", strength=0.5))
    diagnose_many(graph, ["T", "U"])
    assert failed == [{"P": False}, {"L": False}, {"U": False}, {"P": False}, {"L": False}, {"T": False}]


def test_missing_parameter_stops_the_report():
    """Test that a graph that cannot compile gets only the structural findings, even with a target."""
    graph = argument()
    graph.add_node(Node("Q"))
    graph.add_relation(Relation("QT", "supports", "Q", "T", strength=0.4))
    findings = diagnose(graph, "T")
    assert [f.id for f in findings] == ["missing-parameter:base:Q", "unanchored:L", "unanchored:Q"]


def test_thresholds_are_passed_on():
    """Test that the claim and failure thresholds reach their diagnostics."""
    ids = [f.id for f in diagnose(argument(stated=0.4), "T", claim_threshold=0.05, failure_threshold=0.5)]
    assert "overclaim:T" in ids
    assert "single-point-of-failure:P" in ids
    defaults = [f.id for f in diagnose(argument(stated=0.4), "T")]
    assert "overclaim:T" not in defaults
    assert "single-point-of-failure:P" not in defaults


def test_unknown_target():
    """Test that a target that is not a node is rejected."""
    with pytest.raises(ValidationError, match="no node 'nope'"):
        diagnose(argument(), "nope")


def test_report_is_strict_json():
    """Test that a full report serialises without NaN or infinities, and round-trips.

    The certain premise K is never false, so P(T | K = false) is undefined: the report has to leave
    it out rather than divide by zero or emit NaN.
    """
    graph = argument()
    graph.add_node(Node("K", base=1.0, sources=[ANCHOR]))
    graph.add_relation(Relation("KT", "supports", "K", "T", strength=0.3))
    findings = diagnose(graph, "T")
    certain = next(f for f in findings if f.id == "value-of-information:K")
    assert "p_target_given_true" in certain.details
    assert "p_target_given_false" not in certain.details
    records = [finding.to_dict() for finding in findings]
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


# Characters that ``str.splitlines`` treats as line breaks, placed inside node and relation ids.
BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"


def test_messages_stay_on_one_line_whatever_the_ids():
    """Test that ids containing line breaks cannot split the message of any diagnostic.

    Every id below carries every character ``str.splitlines`` breaks on. The first graph compiles
    and yields every inferential diagnostic; the second is missing a base and a strength.
    """
    premise, target, low = f"premise{BREAKS}p", f"target{BREAKS}t", f"low{BREAKS}l"
    graph = Graph()
    graph.add_node(Node(premise, base=Beta(6, 4)))
    graph.add_node(Node(target, base=0.3, stated=0.9))
    graph.add_node(Node(low, base=0.8, stated=0.1, sources=[ANCHOR]))
    graph.add_relation(Relation(f"needs{BREAKS}r", "requires", premise, target, strength=Beta(8, 2)))
    findings = diagnose(graph, target, failure_threshold=0.5)

    broken = Graph()
    broken.add_node(Node(premise, base=0.5))
    broken.add_node(Node(target))
    relation = broken.add_relation(Relation(f"needs{BREAKS}r", "supports", premise, target, strength=0.5))
    object.__setattr__(relation, "strength", None)
    findings += diagnose(broken, target)

    assert {f.diagnostic for f in findings} == {
        "missing-parameter",
        "unanchored",
        "overclaim",
        "underclaim",
        "sensitivity",
        "crux",
        "single-point-of-failure",
        "failure-impact",
        "value-of-information",
    }
    for finding in findings:
        assert len(finding.message.splitlines()) == 1, finding.message


def test_diagnose_many_runs_the_graph_wide_checks_once():
    """Test that several targets share one set of graph-wide findings and each get ``diagnose``'s weak points."""
    graph = argument()
    report = diagnose_many(graph, ["T", "P", "T"], failure_threshold=0.5)
    graph_wide = diagnose(graph)
    assert list(report.findings) == graph_wide
    assert list(report.weak_points) == ["T", "P"]
    for target in ("T", "P"):
        assert list(report.weak_points[target]) == diagnose(graph, target, failure_threshold=0.5)[len(graph_wide) :]
    ids = [f.id for f in report.all_findings()]
    assert ids.count("overclaim:T") == 1
    assert ids.count("sensitivity:base:P") == 2


def test_diagnose_many_names_the_target_of_each_record():
    """Test that the JSON records carry ``None`` for a graph-wide finding and the target for a weak point."""
    report = diagnose_many(argument(), ["T", "P"])
    records = report.to_dicts()
    assert [record["target"] for record in records] == [
        *[None] * len(report.findings),
        *["T"] * len(report.weak_points["T"]),
        *["P"] * len(report.weak_points["P"]),
    ]
    assert [{k: v for k, v in record.items() if k != "target"} for record in records] == [
        finding.to_dict() for finding in report.all_findings()
    ]
    assert json.loads(json.dumps(records, allow_nan=False)) == records


def test_diagnose_many_without_targets_or_parameters():
    """Test no targets, and that a graph missing a parameter leaves every target's list empty."""
    assert diagnose_many(argument(), []) == Report(diagnose(argument()), {})
    graph = argument()
    graph.add_node(Node("Q"))
    graph.add_relation(Relation("QT", "supports", "Q", "T", strength=0.4))
    report = diagnose_many(graph, ["T", "P"])
    assert [f.id for f in report.findings] == ["missing-parameter:base:Q", "unanchored:L", "unanchored:Q"]
    assert dict(report.weak_points) == {"T": (), "P": ()}


def test_diagnose_many_refuses_a_string():
    """Test that a single string is refused rather than read as one target per character."""
    with pytest.raises(ValidationError, match="not the string 'T'"):
        diagnose_many(argument(), "T")
