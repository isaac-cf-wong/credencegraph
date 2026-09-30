"""The ``diagnose``, ``check`` and ``version`` commands."""

from __future__ import annotations

import json

import pytest
from _cli import example_graph

import credencegraph
from credencegraph.core import Graph, Node, Relation, ValidationError, dump
from credencegraph.diagnostics import diagnose
from credencegraph.inference import InferenceError, ProblemTooLargeError


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
        """Test that a claim threshold above the gap silences the overclaim, and a failure threshold is used."""
        output = cli.json(
            "diagnose", graph_file, "--target", "claim", "--claim-threshold", "0.5", "--failure-threshold", "0.2"
        )
        expected = diagnose(example_graph(), "claim", claim_threshold=0.5, failure_threshold=0.2)
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

    @pytest.mark.parametrize("option", ["--claim-threshold", "--failure-threshold"])
    def test_threshold_out_of_range(self, cli, graph_file, option):
        """Test that a threshold outside [0, 1] is refused."""
        error = cli.error("diagnose", graph_file, "--target", "claim", option, "1.5")
        assert error["code"] == "invalid-argument"

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
    """``check`` passes a usable graph and fails, with status 1, one that cannot be compiled."""

    def test_passes(self, cli, graph_file):
        """Test a usable graph: ok, counts, and the unanchored variable as a warning only."""
        output = cli.json("check", graph_file)
        assert output["ok"] is True
        assert (output["nodes"], output["relations"]) == (4, 3)
        assert output["errors"] == []
        assert output["compile_error"] is None
        assert [finding["id"] for finding in output["warnings"]] == ["unanchored:calibrated"]

    def test_empty_graph_passes(self, cli, tmp_path):
        """Test that a fresh graph passes."""
        path = tmp_path / "g.json"
        cli.json("init", path)
        output = cli.json("check", path)
        assert output["ok"] is True
        assert output["warnings"] == []

    def test_missing_base_fails(self, cli, tmp_path):
        """Test that an inference variable without a base fails the check and is named."""
        path = tmp_path / "g.json"
        dump(
            graph_with(
                Node("a", base=0.5, sources=()),
                Node("b"),
                relations=(Relation("ab", "supports", "a", "b", strength=0.5),),
            ),
            path,
        )
        output = cli.json("check", path, exit_code=1)
        assert output["ok"] is False
        assert [finding["id"] for finding in output["errors"]] == ["missing-parameter:base:b"]
        assert output["compile_error"] is None

    def test_conflicting_bases_fail(self, cli, tmp_path):
        """Test that a graph whose structure is complete but does not compile fails with the compile error."""
        path = tmp_path / "g.json"
        dump(
            graph_with(Node("a", base=0.3), Node("b", base=0.4), relations=(Relation("e", "equivalent", "a", "b"),)),
            path,
        )
        output = cli.json("check", path, exit_code=1)
        assert output["ok"] is False
        assert output["errors"] == []
        assert output["compile_error"]["code"] == "compile-error"

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
                {"id": "ba", "type": "supports", "source": "b", "target": "a", "strength": 0.5},
            ],
        }
        path = tmp_path / "g.json"
        path.write_text(json.dumps(document))
        error = cli.error("check", path)
        assert error["code"] == "invalid-graph"
        assert error["details"] == {"cycle": ["b", "a", "b"], "relations": ["ba", "ab"]}

    def test_missing_file(self, cli, tmp_path):
        """Test that a missing graph file is reported."""
        assert cli.error("check", tmp_path / "g.json")["code"] == "file-not-found"

    def test_text_output_and_exit_status(self, cli, tmp_path):
        """Test the human-readable verdict and that a failed check exits with status 1."""
        path = tmp_path / "g.json"
        dump(
            graph_with(Node("a", base=0.5), Node("b"), relations=(Relation("ab", "supports", "a", "b", strength=0.5),)),
            path,
        )
        result = cli.run("check", path)
        assert result.exit_code == 1
        lines = result.stdout.splitlines()
        assert lines[0] == f"{path}: failed (2 nodes, 1 relations)"
        assert lines[1].startswith("error: missing-parameter:base:b:")


def test_version(cli):
    """Test the version in JSON and as text."""
    assert cli.json("version") == {"command": "version", "version": credencegraph.__version__}
    assert cli.run("version").stdout == f"{credencegraph.__version__}\n"


def test_unreadable_path(cli, tmp_path):
    """Test that a path that cannot be read as a file is an I/O error, not a missing file."""
    assert cli.error("check", tmp_path)["code"] == "io-error"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ProblemTooLargeError("too big", required=2**30, limit=2**22),
            ("problem-too-large", {"required": 2**30, "limit": 2**22}),
        ),
        (InferenceError("engine failed"), ("inference-error", {})),
        (ValidationError("bad value"), ("invalid-argument", {})),
    ],
)
def test_library_errors_raised_by_a_command(cli, graph_file, mocker, error, expected):
    """Test that an inference or validation error raised inside a command is reported with its code."""
    mocker.patch("credencegraph.cli.diagnose.diagnose", side_effect=error)
    reported = cli.error("diagnose", graph_file, "--target", "claim")
    assert (reported["code"], reported["details"]) == expected
    assert reported["message"] == str(error)
