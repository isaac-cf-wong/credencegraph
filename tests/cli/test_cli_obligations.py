"""The ``verify`` and ``sorries`` commands, run against graph files with real scripts beside them."""

from __future__ import annotations

from pathlib import Path

import pytest

from credencegraph.core import Discharge, Graph, Node, Obligation, Relation, dump
from credencegraph.core.obligation import file_digest

PASSES = "assert 1 + 1 == 2\n"


def write(directory: Path, *relations: Relation) -> Path:
    """Write ``premise`` and ``claim``, with ``relations`` between them, to ``g.json`` in ``directory``."""
    graph = Graph()
    graph.add_node(Node("premise", base=0.5))
    graph.add_node(Node("claim", base=0.1))
    graph.add_node(Node("alice", kind="person"))
    for relation in relations:
        graph.add_relation(relation)
    path = directory / "g.json"
    dump(graph, path)
    return path


def checked(directory: Path, text: str = PASSES, status: int = 0) -> Relation:
    """A support discharged by ``check.py``, holding ``text``, whose recorded run exited with ``status``."""
    (directory / "check.py").write_text(text, encoding="utf-8")
    discharge = Discharge("symbolic", script="check.py", digest=file_digest(text.encode()), status=status)
    return Relation("r", "supports", "premise", "claim", strength=0.8, obligation=Obligation("holds", discharge))


class TestVerify:
    """``verify`` succeeds only when no discharge is flagged."""

    def test_scripts_are_found_beside_the_graph(self, cli, tmp_path, monkeypatch):
        """Test that a script is found relative to the graph file, not the working directory."""
        directory = tmp_path / "project"
        directory.mkdir()
        path = write(directory, checked(directory))
        monkeypatch.chdir(tmp_path)
        output = cli.json("verify", path)
        assert output == {
            "command": "verify",
            "path": str(path),
            "checks": [
                {"relation": "r", "kind": "symbolic", "outcome": "passed", "status": 0, "reason": None, "findings": []}
            ],
        }

    def test_text(self, cli, tmp_path):
        """Test the text summary, one line per discharge, with the reason for an unchecked one."""
        formal = Obligation("eq. 3", Discharge("formal", theorem="T"))
        path = write(
            tmp_path, checked(tmp_path), Relation("f", "requires", "premise", "claim", strength=1.0, obligation=formal)
        )
        result = cli.run("verify", path)
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert lines[0] == "r: symbolic passed"
        assert lines[1].startswith("f: formal unchecked (no proof checker")

    def test_no_discharges(self, cli, tmp_path):
        """Test that a graph without discharges verifies, and says so."""
        result = cli.run("verify", write(tmp_path))
        assert (result.exit_code, result.stdout) == (0, "no discharges\n")

    def test_flagged_discharge_fails(self, cli, tmp_path):
        """Test that a flagged discharge fails with every check under the details."""
        path = write(tmp_path, checked(tmp_path, "print('ok')\n"))
        error = cli.error("verify", path)
        assert error["code"] == "verification-failed"
        assert error["message"].startswith("1 of 1 discharge(s) failed verification: relation 'r': the script")
        assert [check["outcome"] for check in error["details"]["checks"]] == ["flagged"]
        assert error["details"]["checks"][0]["findings"][0]["diagnostic"] == "cannot-fail"

    @pytest.mark.parametrize("timeout", ["0", "-1", "inf", "nan"])
    def test_timeout_must_be_positive_and_finite(self, cli, tmp_path, timeout):
        """Test that a timeout of no time, or of unbounded time, is refused before anything runs."""
        error = cli.error("verify", write(tmp_path, checked(tmp_path)), "--timeout", timeout)
        assert error["code"] == "invalid-argument"


class TestSorries:
    """``sorries`` lists what a target rests on that is not discharged, most influential first."""

    def test_json(self, cli, tmp_path):
        """Test the report for a graph with one discharged step and one judgement-only step."""
        graph = Graph()
        graph.add_node(Node("premise", base=0.5))
        graph.add_node(Node("other", base=0.5))
        graph.add_node(Node("claim", base=0.1))
        graph.add_relation(checked(tmp_path))
        graph.add_relation(Relation("j", "supports", "other", "claim", strength=0.5))
        path = tmp_path / "g.json"
        dump(graph, path)
        output = cli.json("sorries", path, "--target", "claim")
        assert output["steps"] == ["r", "j"]
        assert output["discharged"] == ["r"]
        (sorry,) = output["sorries"]
        assert {key: sorry[key] for key in ("relation", "reason", "obligation")} == {
            "relation": "j",
            "reason": "no-obligation",
            "obligation": None,
        }
        # dP/ds_j = (1 - 0.1)(1 - 0.8 * 0.5) * 0.5
        assert sorry["sensitivity"] == pytest.approx(0.27, rel=1e-12, abs=0.0)

    def test_text(self, cli, tmp_path):
        """Test the text summary."""
        path = write(tmp_path, Relation("r", "supports", "premise", "claim", strength=0.8))
        result = cli.run("sorries", path, "--target", "claim")
        assert result.exit_code == 0
        assert result.stdout.splitlines() == [
            "claim: 1 sorry(ies) of 1 step(s)",
            "r: no-obligation, dP/dstrength = 0.45",
        ]

    def test_unknown_target(self, cli, tmp_path):
        """Test that a misspelt target is an unknown-node error."""
        error = cli.error("sorries", write(tmp_path), "--target", "clam")
        assert error["code"] == "unknown-node"

    def test_graph_that_does_not_compile(self, cli, tmp_path):
        """Test that a graph that cannot be compiled is a compile-error, naming the node without a base."""
        graph = Graph()
        graph.add_node(Node("premise", base=0.5))
        graph.add_node(Node("claim"))
        graph.add_relation(Relation("r", "supports", "premise", "claim", strength=0.8))
        path = tmp_path / "g.json"
        dump(graph, path)
        error = cli.error("sorries", path, "--target", "claim")
        assert error["code"] == "compile-error"
