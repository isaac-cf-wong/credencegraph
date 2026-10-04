"""The failure contract shared by every command, and the hints each kind of failure carries.

Every failure here is provoked by a real invocation against a real graph file; nothing is stubbed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from credencegraph.cli import common
from credencegraph.core import Graph, Node, Relation, dump
from credencegraph.inference import InferenceError, ProblemTooLargeError

DOCS = Path(__file__).parents[2] / "docs" / "cli.md"


def write(path: Path, *nodes: Node, relations: tuple[Relation, ...] = ()) -> Path:
    """Write a graph of the given nodes and relations to ``path``."""
    graph = Graph()
    for node in nodes:
        graph.add_node(node)
    for relation in relations:
        graph.add_relation(relation)
    dump(graph, path)
    return path


def certain_exclusive(tmp_path: Path) -> Path:
    """Two nodes that are certainly true, joined by an ``exclusive`` relation: no world satisfies it."""
    return write(
        tmp_path / "g.json", Node("A", base=1.0), Node("B", base=1.0), relations=(Relation("x", "exclusive", "A", "B"),)
    )


def even_exclusive(tmp_path: Path) -> Path:
    """Two even-odds nodes joined by an ``exclusive`` relation."""
    return write(
        tmp_path / "g.json", Node("a", base=0.5), Node("b", base=0.5), relations=(Relation("x", "exclusive", "a", "b"),)
    )


def missing_base(tmp_path: Path) -> Path:
    """``b`` is an inference variable, the target of a relation, without a base."""
    return write(
        tmp_path / "g.json",
        Node("a", base=0.5, sources=()),
        Node("b"),
        relations=(Relation("ab", "supports", "a", "b", strength=0.5),),
    )


def conflicting_bases(tmp_path: Path) -> Path:
    """Equivalent nodes whose bases disagree."""
    return write(
        tmp_path / "g.json",
        Node("a", base=0.3),
        Node("b", base=0.4),
        relations=(Relation("e", "equivalent", "a", "b"),),
    )


def merged_cycle(tmp_path: Path) -> Path:
    """No cycle as written, but merging the equivalent ``a`` and ``b`` closes ``a -> c -> a``."""
    return write(
        tmp_path / "g.json",
        Node("a", base=0.5),
        Node("b"),
        Node("c", base=0.5),
        relations=(
            Relation("ac", "supports", "a", "c", strength=0.5),
            Relation("cb", "supports", "c", "b", strength=0.5),
            Relation("e", "equivalent", "a", "b"),
        ),
    )


def too_large(tmp_path: Path) -> Path:
    """A node with 23 parents: its factor has 2**24 entries, over the default limit of 2**22."""
    parents = [Node(f"p{i}", base=0.5) for i in range(23)]
    relations = tuple(Relation(f"r{i}", "supports", f"p{i}", "child", strength=0.5) for i in range(23))
    return write(tmp_path / "g.json", *parents, Node("child", base=0.1), relations=relations)


def example(tmp_path: Path) -> Path:
    """A small valid graph: ``claim`` rests on ``premise``; ``alice`` is carried, not inferred."""
    return write(
        tmp_path / "g.json",
        Node("premise", base=0.5),
        Node("claim", base=0.1),
        Node("alice", kind="person"),
        relations=(Relation("r", "supports", "premise", "claim", strength=0.8),),
    )


# One real failure per command, and per documented error code.
FAILURES = {
    "file-exists": ("init", example, ()),
    "file-not-found": ("check", lambda tmp: tmp / "absent.json", ()),
    "io-error": ("check", lambda tmp: tmp, ()),
    "invalid-graph": ("add-node", lambda tmp: _text(tmp, "not json"), ("--id", "x")),
    "invalid-argument": ("add-node", example, ("--id", "x", "--base", "1.5")),
    "duplicate-id": ("add-node", example, ("--id", "claim")),
    "unknown-node": ("diagnose", example, ("--target", "clam")),
    "cycle": ("relate", example, ("claim", "premise", "--type", "supports", "--strength", "0.5")),
    "compile-error": ("check", missing_base, ()),
    "zero-probability": ("query", certain_exclusive, ("marginal", "A", "--draws", "0")),
    "problem-too-large": ("query", too_large, ("marginal", "child", "--draws", "0")),
}


def _text(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "g.json"
    path.write_text(content, encoding="utf-8")
    return path


def provoke(cli, tmp_path, code, *, as_json=True):
    """Run the command that fails with ``code``."""
    command, make, extra = FAILURES[code]
    path = make(tmp_path)
    args = [command, path, *extra] + (["--json"] if as_json else [])
    return cli.run(*args)


def test_every_documented_code_is_provoked():
    """Test that the codes the package defines, the codes the docs list and the codes provoked here agree."""
    sentence = DOCS.read_text(encoding="utf-8").split("The codes are", 1)[1].split(".", 1)[0]
    documented = set(re.findall(r"`([a-z]+(?:-[a-z]+)*)`", sentence))
    assert set(common.ERROR_CODES) == set(FAILURES) == documented


@pytest.mark.parametrize("code", sorted(FAILURES))
def test_failure_envelope_in_json(cli, tmp_path, code):
    """Test that each failure prints one error object on stdout, nothing on stderr, and exits with 1."""
    result = provoke(cli, tmp_path, code)
    assert result.exit_code == 1, (result.stdout, result.exception)
    assert result.stderr == ""
    output = json.loads(result.stdout)
    assert output == {"command": FAILURES[code][0], "error": output["error"]}
    assert set(output["error"]) == {"code", "message", "hint", "details"}
    assert output["error"]["code"] == code
    assert isinstance(output["error"]["message"], str)
    assert output["error"]["message"]


@pytest.mark.parametrize("code", sorted(FAILURES))
def test_failure_in_text(cli, tmp_path, code):
    """Test that without ``--json`` each failure prints its message and hint on stderr and nothing on stdout.

    Both runs use one directory, so the text is compared exactly as printed. A quoted path escapes
    each backslash: on Windows those are the separators, and elsewhere the directory's name holds one.
    """
    directory = tmp_path / "back\\slash"
    directory.mkdir(parents=True)
    error = json.loads(provoke(cli, directory, code).stdout)["error"]
    result = provoke(cli, directory, code, as_json=False)
    assert result.exit_code == 1
    assert result.stdout == ""
    expected = [f"error: {error['message']}"] + ([f"hint: {error['hint']}"] if error["hint"] else [])
    assert result.stderr.splitlines() == expected


def test_problem_too_large_hint_names_a_next_step(cli, tmp_path):
    """Test that the hint for a graph beyond exact inference says how to get an answer, not only that it failed."""
    error = json.loads(provoke(cli, tmp_path, "problem-too-large").stdout)["error"]
    assert error["details"]["required"] > error["details"]["limit"]
    assert error["hint"].startswith("reduce how many relations meet at one node")
    assert "--max-factor-size" in error["hint"]
    assert "raise max_factor_size on VariableElimination" in common.translate(ProblemTooLargeError("m", 2, 1)).hint


def test_every_command_is_covered():
    """Test that the failure table exercises each of the six commands."""
    assert {entry[0] for entry in FAILURES.values()} == {"init", "add-node", "relate", "query", "diagnose", "check"}


def test_inference_errors_have_their_own_codes():
    """Test that every kind of inference error the library defines maps to a code of its own.

    An oracle over the library's classes: a new subclass without a mapping would fall through to
    ``invalid-argument`` and fail here.
    """

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    kinds = list(subclasses(InferenceError))
    assert kinds
    for kind in kinds:
        instance = kind("message", 2, 1) if kind is ProblemTooLargeError else kind("message")
        assert common.translate(instance).code != common.INVALID_ARGUMENT, kind


class TestCheckFailure:
    """A failing ``check`` is an error envelope with the whole verdict under ``details``."""

    def test_missing_base(self, cli, tmp_path):
        """Test the code, the message naming the node, the hint, and the verdict in the details."""
        path = missing_base(tmp_path)
        error = cli.error("check", path)
        assert error["code"] == "compile-error"
        assert "'b'" in error["message"]
        assert "base" in error["hint"]
        assert "credencegraph check" not in error["hint"]
        details = error["details"]
        assert (details["path"], details["nodes"], details["relations"]) == (str(path), 2, 1)
        assert [finding["id"] for finding in details["errors"]] == ["missing-parameter:base:b"]
        assert details["conflicting_bases"] == []
        assert [finding["id"] for finding in details["warnings"]] == ["unanchored:a"]

    def test_conflicting_bases(self, cli, tmp_path):
        """Test that conflicting bases are named with the credences as the user writes them."""
        error = cli.error("check", conflicting_bases(tmp_path))
        assert error["code"] == "compile-error"
        assert error["message"] == "equivalent nodes 'a' and 'b' have conflicting bases 0.3 and 0.4"
        assert "same base" in error["hint"]
        assert "credencegraph check" not in error["hint"]
        assert error["details"]["errors"] == []
        assert error["details"]["conflicting_bases"] == [{"nodes": ["a", "b"], "bases": [0.3, 0.4]}]

    def test_merged_cycle(self, cli, tmp_path):
        """Test that a cycle closed by merging equivalent nodes is a compile error that names the relation to cut."""
        error = cli.error("check", merged_cycle(tmp_path))
        assert error["code"] == "compile-error"
        assert "cycle" in error["message"]
        assert "remove the equivalent relation or one relation of the cycle" in error["hint"]
        assert "credencegraph check" not in error["hint"]

    def test_success_is_not_an_error(self, cli, tmp_path):
        """Test that a passing check is an ordinary result that carries its warnings."""
        path = example(tmp_path)
        output = cli.json("check", path)
        assert output == {
            "command": "check",
            "path": str(path),
            "nodes": 3,
            "relations": 1,
            "warnings": output["warnings"],
        }
        assert [finding["id"] for finding in output["warnings"]] == ["unanchored:premise"]


class TestZeroProbabilityHint:
    """The hint for evidence of probability zero names only what the caller can change."""

    GRAPH_REMEDY = "exactly 0 or 1"

    def test_marginal_without_evidence(self, cli, tmp_path):
        """Test that a failure of the graph itself points at the graph, not at a flag that was not passed."""
        error = cli.error("query", certain_exclusive(tmp_path), "marginal", "A", "--draws", "0")
        assert error["code"] == "zero-probability"
        assert "--given" not in error["hint"]
        assert "--set" not in error["hint"]
        assert self.GRAPH_REMEDY in error["hint"]

    def test_diagnose(self, cli, tmp_path):
        """Test that ``diagnose``, which takes no evidence, points at the graph."""
        error = cli.error("diagnose", certain_exclusive(tmp_path), "--target", "A")
        assert error["code"] == "zero-probability"
        assert "--given" not in error["hint"]
        assert self.GRAPH_REMEDY in error["hint"]

    def test_diagnose_without_target(self, cli, tmp_path):
        """Test the same for the graph-wide checks, which compare stated and computed credences."""
        path = certain_exclusive(tmp_path)
        write(
            path,
            Node("A", base=1.0, stated=0.5),
            Node("B", base=1.0),
            relations=(Relation("x", "exclusive", "A", "B"),),
        )
        error = cli.error("diagnose", path)
        assert error["code"] == "zero-probability"
        assert "--given" not in error["hint"]

    def test_graph_failure_with_evidence_points_at_the_graph(self, cli, tmp_path):
        """Test that evidence given on a graph that fails without it still points at the graph."""
        error = cli.error("query", certain_exclusive(tmp_path), "conditional", "A", "--given", "B=true", "--draws", "0")
        assert "--given" not in error["hint"]
        assert self.GRAPH_REMEDY in error["hint"]

    def test_impossible_evidence(self, cli, tmp_path):
        """Test that evidence the graph rules out points at ``--given``, and only at it."""
        path = even_exclusive(tmp_path)
        error = cli.error("query", path, "conditional", "a", "--given", "a=true", "--given", "b=true", "--draws", "0")
        assert error["code"] == "zero-probability"
        assert "--given" in error["hint"]
        assert "--set" not in error["hint"]

    def test_impossible_intervention(self, cli, tmp_path):
        """Test that interventions the graph rules out point at ``--set``, and only at it."""
        path = even_exclusive(tmp_path)
        error = cli.error("query", path, "intervene", "a", "--set", "a=true", "--set", "b=true", "--draws", "0")
        assert error["code"] == "zero-probability"
        assert "--set" in error["hint"]
        assert "--given" not in error["hint"]


class TestCompileErrorHint:
    """A graph that cannot be compiled is explained the same way by every command that compiles it."""

    @pytest.mark.parametrize("command", [("query", "marginal", "a"), ("diagnose", "--target", "a")])
    def test_conflicting_bases(self, cli, tmp_path, command):
        """Test that the hint fits conflicting bases, and does not send the caller to ``check``."""
        name, *args = command
        error = cli.error(name, conflicting_bases(tmp_path), *args)
        assert error["code"] == "compile-error"
        assert error["message"] == "equivalent nodes 'a' and 'b' have conflicting bases 0.3 and 0.4"
        assert "same base" in error["hint"]
        assert "Point(" not in error["message"]

    def test_missing_base(self, cli, tmp_path):
        """Test that a query on a graph with a missing base names the node and asks for a base."""
        error = cli.error("query", missing_base(tmp_path), "marginal", "b")
        assert error["code"] == "compile-error"
        assert "'b'" in error["message"]
        assert "base" in error["hint"]
        assert [finding["id"] for finding in error["details"]["errors"]] == ["missing-parameter:base:b"]

    def test_merged_cycle(self, cli, tmp_path):
        """Test that a query on a graph whose merged nodes form a cycle names the equivalent relation as the cause."""
        error = cli.error("query", merged_cycle(tmp_path), "marginal", "c")
        assert error["code"] == "compile-error"
        assert "remove the equivalent relation or one relation of the cycle" in error["hint"]


class TestArgumentHints:
    """A rejected argument is restated by the flag or argument the caller typed, with a remedy."""

    @pytest.mark.parametrize(
        ("args", "flag"),
        [
            (("add-node", "--id", ""), "--id"),
            (("add-node", "--id", "x", "--statement", ""), "--statement"),
            (("add-node", "--id", "x", "--kind", ""), "--kind"),
            (("relate", "claim", "alice", "--type", ""), "--type"),
            (("relate", "claim", "alice", "--type", "cites", "--id", ""), "--id"),
            (("diagnose", "--target", "claim", "--claim-threshold", "-0.1"), "--claim-threshold"),
            (("diagnose", "--target", "claim", "--failure-threshold", "-0.1"), "--failure-threshold"),
            (("diagnose", "--claim-threshold", "nan"), "--claim-threshold"),
            (("query", "marginal", "claim", "--draws", "-1"), "--draws"),
        ],
    )
    def test_names_the_flag(self, cli, tmp_path, args, flag):
        """Test that the message and the hint both name the flag, and the file is left alone."""
        path = example(tmp_path)
        before = path.read_bytes()
        command, *rest = args
        error = cli.error(command, path, *rest)
        assert error["code"] == "invalid-argument"
        assert error["message"].startswith(flag)
        assert flag in error["hint"]
        assert path.read_bytes() == before

    @pytest.mark.parametrize("rtype", ["equivalent", "exclusive"])
    def test_self_relation(self, cli, tmp_path, rtype):
        """Test that joining a node to itself names SOURCE and TARGET."""
        error = cli.error("relate", example(tmp_path), "claim", "claim", "--type", rtype)
        assert error["code"] == "invalid-argument"
        assert "SOURCE" in error["hint"]
        assert "TARGET" in error["hint"]

    @pytest.mark.parametrize(
        ("args", "role"),
        [
            (("query", "marginal", "alice"), "target"),
            (("query", "marginal", "claim", "--given", "alice=true"), "--given"),
            (("query", "intervene", "claim", "--set", "alice=true"), "--set"),
            (("diagnose", "--target", "alice"), "--target"),
        ],
    )
    def test_carried_node(self, cli, tmp_path, args, role):
        """Test that a node that takes no part in inference is named with the role it was given."""
        command, *rest = args
        error = cli.error(command, example(tmp_path), *rest)
        assert error["code"] == "invalid-argument"
        assert error["message"].startswith(f"{role} 'alice'")
        assert "base" in error["hint"]
        assert error["details"] == {"node": "alice"}

    def test_equivalent_nodes_set_apart(self, cli, tmp_path):
        """Test that setting two equivalent nodes to different values is refused, naming ``--set``."""
        path = write(
            tmp_path / "g.json", Node("a", base=0.5), Node("b"), relations=(Relation("e", "equivalent", "a", "b"),)
        )
        error = cli.error("query", path, "intervene", "a", "--set", "a=true", "--set", "b=false")
        assert error["code"] == "invalid-argument"
        assert error["message"].startswith("--set")
        assert "--set" in error["hint"]
