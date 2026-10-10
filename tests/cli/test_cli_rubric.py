"""The rubric commands: ``check --rubric --level``, ``coverage``, ``split`` and ``annotate``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from credencegraph.core import Graph, Node, Relation, SourceAnchor, dump, load
from credencegraph.rubric import text_digest

RUBRIC = Path(__file__).parents[2] / "docs" / "examples" / "methods-paper.toml"

SENTENCE = "We show that the bias vanishes for long segments, which implies that the estimator is consistent."
CLAIM = "The false-alarm rate stays below 5% for long segments."


def chunk(node_id: str, text: str) -> Node:
    """An unassigned document chunk, anchored verbatim."""
    anchor = SourceAnchor("paper", f"file=main.tex;lines={node_id}", text, text_digest(text))
    return Node(node_id, "unassigned", text, (anchor,), attributes={"origin": "document", "unit": "sentence"})


@pytest.fixture
def paper(tmp_path: Path) -> Path:
    """A freshly ingested graph of two sentences, written to a file."""
    graph = Graph()
    graph.add_node(chunk("s-1", SENTENCE))
    graph.add_node(chunk("s-2", CLAIM))
    path = tmp_path / "paper.json"
    dump(graph, path)
    return path


def run(cli, *args):
    """Run a command with ``--json`` and return its exit code and output."""
    result = cli.run(*args, "--json")
    return result.exit_code, json.loads(result.stdout)


def test_check_levels(cli, paper):
    """Test that a fresh graph passes ``anchored`` and fails ``typed`` and ``assessed``, listing every violation."""
    code, output = run(cli, "check", paper, "--rubric", RUBRIC, "--level", "anchored")
    assert code == 0
    assert output == {
        "command": "check",
        "path": str(paper),
        "rubric": {"name": "methods-paper", "version": "2026.1"},
        "level": "anchored",
        "nodes": 2,
        "relations": 0,
        "violations": [],
    }
    for level in ("typed", "assessed"):
        code, output = run(cli, "check", paper, "--rubric", RUBRIC, "--level", level)
        assert code == 1
        assert output["error"]["code"] == "rubric-violation"
        ids = [violation["id"] for violation in output["error"]["details"]["violations"]]
        assert ids == ["untyped:s-1", "untyped:s-2"]
        assert output["error"]["details"]["level"] == level
        assert "2 violation(s)" in output["error"]["message"]


def test_check_level_defaults_to_typed(cli, paper):
    """Test that --rubric without --level checks at ``typed``."""
    _, output = run(cli, "check", paper, "--rubric", RUBRIC)
    assert output["error"]["details"]["level"] == "typed"


def test_check_text_lists_every_violation(cli, paper):
    """Test that without --json the message names every violation on one line."""
    result = cli.run("check", paper, "--rubric", RUBRIC)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "untyped: node 's-1'" in result.stderr
    assert "untyped: node 's-2'" in result.stderr


def test_check_without_rubric_is_unchanged(cli, paper):
    """Test that plain ``check`` still checks only that the graph parses and compiles."""
    code, output = run(cli, "check", paper)
    assert code == 0
    assert set(output) == {"command", "path", "nodes", "relations", "warnings"}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--level", "typed"), "--level needs --rubric"),
        (("--rubric", RUBRIC, "--level", "strict"), "--level must be one of anchored, typed, assessed"),
    ],
)
def test_check_argument_errors(cli, paper, args, message):
    """Test that --level without --rubric, or an unknown level, is an invalid argument."""
    error = cli.error("check", paper, *args)
    assert error["code"] == "invalid-argument"
    assert message in error["message"]


def test_rubric_file_errors(cli, paper, tmp_path):
    """Test that a missing rubric file is file-not-found and a malformed one invalid-rubric."""
    assert cli.error("check", paper, "--rubric", tmp_path / "absent.toml")["code"] == "file-not-found"
    bad = tmp_path / "bad.toml"
    bad.write_text("[rubric]\nformat = 1\nname = 'x'\nversion = '1'\n[types.compound]\n", encoding="utf-8")
    error = cli.error("coverage", paper, "--rubric", bad)
    assert error["code"] == "invalid-rubric"
    assert "built in" in error["message"]


def test_full_workflow(cli, paper):
    """Test split, annotate, check and coverage together, as in the documentation."""
    code, output = run(
        cli,
        "split",
        paper,
        "s-1",
        "--at",
        "We show that the bias vanishes for long segments",
        "--at",
        "the estimator is consistent",
    )
    assert (code, output["children"]) == (0, ["s-1a", "s-1b"])
    assert (
        run(cli, "annotate", paper, "s-1a", "--rubric", RUBRIC, "--type", "result", "--set", "quantity=the bias")[0]
        == 0
    )
    code, output = run(
        cli,
        "annotate",
        paper,
        "s-1b",
        "--rubric",
        RUBRIC,
        "--type",
        "claim",
        "--set",
        "shape=universal",
        "--set",
        "statement=the estimator is consistent",
        "--rests-on",
        "s-1a",
    )
    assert code == 0
    assert output["node"]["kind"] == "claim"
    assert output["node"]["base"] == {"alpha": 5.0, "beta": 5.0}
    assert output["relations"] == ["s-1a-requires-s-1b"]
    code, output = run(
        cli,
        "annotate",
        paper,
        "s-2",
        "--rubric",
        RUBRIC,
        "--type",
        "claim",
        "--set",
        "shape=bound",
        "--set",
        "statement=The false-alarm rate stays below 5%",
        "--set",
        "bound=0.05",
    )
    assert code == 0
    _, output = run(cli, "check", paper, "--rubric", RUBRIC, "--level", "typed")
    assert [violation["id"] for violation in output["error"]["details"]["violations"]] == ["rests-on-missing:s-2"]
    code, output = run(cli, "coverage", paper, "--rubric", RUBRIC)
    assert code == 0
    assert output["by_type"] == {
        "compound": {"count": 1, "share": 0.25},
        "result": {"count": 1, "share": 0.25},
        "claim": {"count": 2, "share": 0.5},
    }
    assert output["unexamined"] == ["s-1a", "s-1b", "s-2"]
    assert run(cli, "annotate", paper, "s-2", "--rubric", RUBRIC, "--not-assessed", "outside this review")[0] == 0
    _, output = run(cli, "coverage", paper, "--rubric", RUBRIC)
    assert output["not_assessed"] == {"count": 1, "reasons": {"s-2": "outside this review"}}
    assert output["untyped"] == []


def test_assessed_compile_error_carries_the_compiler_details(cli, paper):
    """Test that a graph that does not compile at ``assessed`` also reports the compile-error details."""
    _, output = run(cli, "check", paper, "--rubric", RUBRIC, "--level", "assessed")
    assert "compile_error" not in output["error"]["details"]
    graph = load(paper)
    graph.add_node(Node("x-1", "check", "Re-measured.", attributes={"origin": "evidence"}))
    graph.add_relation(Relation("ev", "supports", "x-1", "s-1", strength=0.5))
    dump(graph, paper)
    _, output = run(cli, "check", paper, "--rubric", RUBRIC, "--level", "assessed")
    details = output["error"]["details"]
    assert [violation["id"] for violation in details["violations"]][-1] == "compile-error:graph"
    assert [finding["id"] for finding in details["compile_error"]["errors"]] == [
        "missing-parameter:base:s-1",
        "missing-parameter:base:x-1",
    ]


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        (("s-1",), "invalid-argument", "split takes --at or --fields"),
        (("s-1", "--at", "x", "--fields"), "invalid-argument", "split takes --at or --fields"),
        (("s-1", "--at", "not in the text"), "invalid-argument", "occurs 0 times"),
        (("s-9", "--at", "We"), "unknown-node", "NODE 's-9' is not a node"),
    ],
)
def test_split_failures_leave_the_file(cli, paper, args, code, message):
    """Test that a refused split names the problem and leaves the file exactly as it was."""
    before = paper.read_bytes()
    error = cli.error("split", paper, *args)
    assert error["code"] == code
    assert message in error["message"]
    assert paper.read_bytes() == before


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        ((), "invalid-argument", "annotate needs something to do"),
        (("--type", ""), "invalid-argument", "--type must not be empty"),
        (("--verdict", "holds", "--not-assessed", "x"), "invalid-argument", "exclude each other"),
        (("--set", "novalue"), "invalid-argument", "is not FIELD=VALUE"),
        (("--set", "=x"), "invalid-argument", "is not FIELD=VALUE"),
        (("--type", "claim", "--set", "bound=1", "--set", "bound=2"), "invalid-argument", "twice"),
        (("--type", "claim", "--set", "bound=0.04"), "invalid-argument", "no numeral in the text equals 0.04"),
        (("--type", "finding"), "invalid-argument", "not declared by rubric"),
        (("--type", "claim", "--rests-on", "s-9"), "unknown-node", "--rests-on 's-9'"),
        (("--type", "claim", "--rests-on", "s-1"), "invalid-argument", "has type 'unassigned'"),
    ],
)
def test_annotate_failures_leave_the_file(cli, paper, args, code, message):
    """Test that a refused annotation names the problem and leaves the file exactly as it was."""
    before = paper.read_bytes()
    error = cli.error("annotate", paper, "s-2", "--rubric", RUBRIC, *args)
    assert error["code"] == code
    assert message in error["message"]
    assert paper.read_bytes() == before


def test_coverage_text(cli, paper):
    """Test the text summary of coverage."""
    result = cli.run("coverage", paper, "--rubric", RUBRIC)
    assert result.exit_code == 0
    assert "type unassigned: 2 (100.0% of document nodes)" in result.stdout
    assert "untyped: s-1, s-2" in result.stdout
