"""The ``diagnose``, ``check`` and ``version`` commands."""

from __future__ import annotations

import json

import pytest
from _cli import example_graph

import credencegraph
from credencegraph.core import Graph, Node, Relation, ValidationError, dump
from credencegraph.diagnostics import diagnose
from credencegraph.inference import VariableElimination


class TestDiagnose:
    """``diagnose`` prints the library's findings as JSON records."""

    def test_with_target(self, cli, graph_file):
        """Test that the findings are exactly the library's, in its order."""
        output = cli.json("diagnose", graph_file, "--target", "claim")
        expected = [finding.to_dict() for finding in diagnose(example_graph(), "claim")]
        assert output == {"command": "diagnose", "path": str(graph_file), "target": "claim", "findings": expected}
        ids = [finding["id"] for finding in output["findings"]]
        assert "unanchored:calibrated" in ids
        assert "overclaim:claim" in ids
        assert "crux:strength:r2" in ids

    def test_without_target(self, cli, graph_file):
        """Test that without a target only the graph-wide checks run."""
        output = cli.json("diagnose", graph_file)
        assert output["target"] is None
        assert [finding["id"] for finding in output["findings"]] == ["unanchored:calibrated", "overclaim:claim"]

    def test_thresholds_are_passed_on(self, cli, graph_file):
        """Test that a claim threshold above the 1.88 log-odds gap silences the overclaim, and a failure threshold is used."""
        output = cli.json(
            "diagnose", graph_file, "--target", "claim", "--claim-threshold", "2", "--failure-threshold", "0.2"
        )
        expected = diagnose(example_graph(), "claim", claim_threshold=2.0, failure_threshold=0.2)
        assert output["findings"] == [finding.to_dict() for finding in expected]
        assert "overclaim:claim" not in [finding["id"] for finding in output["findings"]]

    def test_unknown_target(self, cli, graph_file):
        """Test that a target missing from the graph is named."""
        error = cli.error("diagnose", graph_file, "--target", "clam")
        assert error["code"] == "unknown-node"
        assert "'claim'" in error["hint"]

    def test_carried_target(self, cli, graph_file):
        """Test that a target that is not an inference variable is refused."""
        assert cli.error("diagnose", graph_file, "--target", "alice")["code"] == "invalid-argument"

    @pytest.mark.parametrize(
        ("option", "value"),
        [("--claim-threshold", "-0.1"), ("--claim-threshold", "inf"), ("--failure-threshold", "1.5")],
    )
    def test_threshold_out_of_range(self, cli, graph_file, option, value):
        """Test that a negative or infinite log-odds threshold, or a probability outside [0, 1], is refused."""
        error = cli.error("diagnose", graph_file, "--target", "claim", option, value)
        assert error["code"] == "invalid-argument"

    def test_max_factor_size_below_one(self, cli, graph_file):
        """Test that a factor-size limit below 1 is refused."""
        error = cli.error("diagnose", graph_file, "--target", "claim", "--max-factor-size", "0")
        assert error["code"] == "invalid-argument"
        assert "--max-factor-size must be 1 or more" in error["message"]

    def test_max_factor_size_is_raised_from_the_command_line(self, cli, tmp_path):
        """Test that a diagnosis over the limit fails naming --max-factor-size, and succeeds once it is raised.

        ``child`` rests on four parents, so its weak points need a factor of 16 table entries.
        """
        graph = graph_with(
            *(Node(f"p{i}", base=0.5) for i in range(4)),
            Node("child", base=0.1),
            relations=tuple(Relation(f"r{i}", "supports", f"p{i}", "child", strength=0.5) for i in range(4)),
        )
        path = tmp_path / "g.json"
        dump(graph, path)
        error = cli.error("diagnose", path, "--target", "child", "--max-factor-size", "8")
        assert error["code"] == "problem-too-large"
        assert error["details"] == {"required": 16, "limit": 8}
        assert "raise it with --max-factor-size, to at least 16" in error["hint"]
        output = cli.json("diagnose", path, "--target", "child", "--max-factor-size", "16")
        expected = diagnose(graph, "child", engine=VariableElimination(16))
        assert output["findings"] == [finding.to_dict() for finding in expected]

    def test_missing_file(self, cli, tmp_path):
        """Test that a missing graph file is reported."""
        assert cli.error("diagnose", tmp_path / "g.json")["code"] == "file-not-found"

    def test_text_output(self, cli, graph_file):
        """Test the human-readable form: one line per finding, id first."""
        result = cli.run("diagnose", graph_file)
        assert result.exit_code == 0
        assert [line.split(":", 2)[:2] for line in result.stdout.splitlines()] == [
            ["unanchored", "calibrated"],
            ["overclaim", "claim"],
        ]


def graph_with(*nodes: Node, relations: tuple[Relation, ...] = ()) -> Graph:
    """Build a graph from nodes and relations."""
    graph = Graph()
    for node in nodes:
        graph.add_node(node)
    for relation in relations:
        graph.add_relation(relation)
    return graph


class TestCheck:
    """``check`` passes a usable graph and fails, with a compile error, one that cannot be compiled.

    The failing verdicts are in ``test_cli_contract.py``.
    """

    def test_passes(self, cli, graph_file):
        """Test a usable graph: counts, and the unanchored variable as a warning only."""
        output = cli.json("check", graph_file)
        assert (output["nodes"], output["relations"]) == (4, 3)
        assert [finding["id"] for finding in output["warnings"]] == ["unanchored:calibrated"]

    def test_empty_graph_passes(self, cli, tmp_path):
        """Test that a fresh graph passes."""
        path = tmp_path / "g.json"
        cli.json("init", path)
        output = cli.json("check", path)
        assert output == {"command": "check", "path": str(path), "nodes": 0, "relations": 0, "warnings": []}

    def test_invalid_file(self, cli, tmp_path):
        """Test that a file that does not parse as a graph is an error, not a verdict."""
        path = tmp_path / "g.json"
        path.write_text(json.dumps({"format": "credencegraph", "version": 2, "nodes": [], "relations": []}))
        error = cli.error("check", path)
        assert error["code"] == "invalid-graph"
        assert "version" in error["message"]

    def test_cycle_in_file(self, cli, tmp_path):
        """Test that a cycle written into the file by hand is reported with the cycle."""
        document = {
            "format": "credencegraph",
            "version": 1,
            "nodes": [{"id": "a", "base": 0.5}, {"id": "b", "base": 0.5}],
            "relations": [
                {"id": "ab", "type": "supports", "source": "a", "target": "b", "strength": 0.5},
                {"id": "ba", "type": "supports", "source": "b", "target": "a", "strength": 0.5},  # typos:disable-line
            ],
        }
        path = tmp_path / "g.json"
        path.write_text(json.dumps(document))
        error = cli.error("check", path)
        assert error["code"] == "invalid-graph"
        assert error["details"] == {"cycle": ["b", "a", "b"], "relations": ["ba", "ab"]}  # typos:disable-line

    def test_missing_file(self, cli, tmp_path):
        """Test that a missing graph file is reported."""
        assert cli.error("check", tmp_path / "g.json")["code"] == "file-not-found"

    def test_text_output(self, cli, graph_file, tmp_path):
        """Test the human-readable verdict of a pass, and that a failure goes to stderr with status 1."""
        result = cli.run("check", graph_file)
        assert result.exit_code == 0
        assert result.stdout.splitlines()[0] == f"{graph_file}: ok (4 nodes, 3 relations)"
        assert result.stdout.splitlines()[1].startswith("warning: unanchored:calibrated:")
        path = tmp_path / "bad.json"
        dump(
            graph_with(Node("a", base=0.5), Node("b"), relations=(Relation("ab", "supports", "a", "b", strength=0.5),)),
            path,
        )
        result = cli.run("check", path)
        assert result.exit_code == 1
        assert result.stdout == ""
        assert result.stderr.splitlines()[0] == "error: inference variables without a base: 'b'"


def test_version(cli):
    """Test the version in JSON and as text."""
    assert cli.json("version") == {"command": "version", "version": credencegraph.__version__}
    assert cli.run("version").stdout == f"{credencegraph.__version__}\n"


def test_unreadable_path(cli, tmp_path):
    """Test that a path that cannot be read as a file is an I/O error, not a missing file."""
    assert cli.error("check", tmp_path)["code"] == "io-error"


def test_unanticipated_validation_error(cli, graph_file, mocker):
    """Test the safety net for a validation error no command checks for first.

    The library is stubbed here because every validation error a command can meet is checked, and
    restated, before the library is called; this pins only what the fallback reports.
    """
    mocker.patch("credencegraph.cli.diagnose.diagnose", side_effect=ValidationError("bad value"))
    error = cli.error("diagnose", graph_file, "--target", "claim")
    assert (error["code"], error["message"], error["details"]) == ("invalid-argument", "bad value", {})
    assert error["hint"] == "check the arguments against 'credencegraph diagnose --help'"
