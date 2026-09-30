"""The commands that write a graph file: ``init``, ``add-node`` and ``relate``."""

from __future__ import annotations

import json

import pytest
from _cli import example_graph

from credencegraph.core import Beta, Graph, Point, SourceAnchor, load


class TestInit:
    """``init`` creates an empty graph file and refuses to overwrite one."""

    def test_creates_an_empty_graph(self, cli, tmp_path):
        """Test that the file holds an empty graph and the output says so."""
        path = tmp_path / "g.json"
        output = cli.json("init", path)
        assert output == {"command": "init", "path": str(path), "nodes": 0, "relations": 0}
        assert load(path) == Graph()

    def test_refuses_an_existing_file(self, cli, graph_file):
        """Test that an existing file is left untouched without ``--force``."""
        before = graph_file.read_bytes()
        error = cli.error("init", graph_file)
        assert error["code"] == "file-exists"
        assert error["details"] == {"path": str(graph_file)}
        assert "--force" in error["hint"]
        assert graph_file.read_bytes() == before

    def test_force_replaces_the_file(self, cli, graph_file):
        """Test that ``--force`` replaces an existing graph with an empty one."""
        cli.json("init", graph_file, "--force")
        assert load(graph_file) == Graph()

    def test_missing_directory(self, cli, tmp_path):
        """Test that a path in a directory that does not exist is an I/O error."""
        path = tmp_path / "absent" / "g.json"
        assert cli.error("init", path)["code"] == "io-error"
        assert not path.exists()


class TestAddNode:
    """``add-node`` validates every field before it writes."""

    def test_adds_a_node(self, cli, tmp_path):
        """Test every field, read back from the file as well as from the output."""
        path = tmp_path / "g.json"
        cli.json("init", path)
        output = cli.json(
            "add-node",
            path,
            "--id",
            "h1",
            "--statement",
            "The half-life is 5.2 d.",
            "--kind",
            "claim",
            "--base",
            "beta:8,2",
            "--stated",
            "0.95",
            "--source",
            "doi:10.0000/x#p4",
            "--source",
            "https://example.org/a#b#sec2",
            "--source",
            "notes.txt",
        )
        anchors = (
            SourceAnchor("doi:10.0000/x", "p4"),
            SourceAnchor("https://example.org/a#b", "sec2"),
            SourceAnchor("notes.txt"),
        )
        node = load(path).nodes["h1"]
        assert node.kind == "claim"
        assert node.statement == "The half-life is 5.2 d."
        assert node.base == Beta(8, 2)
        assert node.stated == Point(0.95)
        assert node.sources == anchors
        assert output == {
            "command": "add-node",
            "path": str(path),
            "node": {
                "id": "h1",
                "kind": "claim",
                "statement": "The half-life is 5.2 d.",
                "sources": [{"document": a.document, "locator": a.locator} for a in anchors],
                "base": {"alpha": 8.0, "beta": 2.0},
                "stated": 0.95,
            },
        }

    def test_minimal_node(self, cli, graph_file):
        """Test that only the id is needed, and the other nodes are kept."""
        cli.json("add-node", graph_file, "--id", "bob")
        graph = load(graph_file)
        assert graph.nodes["bob"].base is None
        assert graph.nodes["bob"].kind == "proposition"
        assert list(graph.nodes) == [*example_graph().nodes, "bob"]

    def test_duplicate_id(self, cli, graph_file):
        """Test that an id already in the graph is refused by name."""
        before = graph_file.read_bytes()
        error = cli.error("add-node", graph_file, "--id", "claim")
        assert error["code"] == "duplicate-id"
        assert error["details"] == {"node": "claim"}
        assert graph_file.read_bytes() == before

    @pytest.mark.parametrize(
        "value",
        ["abc", "1.5", "-0.1", "nan", "inf", "beta:8", "beta:8,2,1", "beta:0,2", "beta:x,2", "gamma:1,2", ""],
    )
    @pytest.mark.parametrize("option", ["--base", "--stated"])
    def test_invalid_credence(self, cli, graph_file, option, value):
        """Test that a credence that is neither a probability nor a valid Beta is refused, with the forms."""
        before = graph_file.read_bytes()
        error = cli.error("add-node", graph_file, "--id", "new", option, value)
        assert error["code"] == "invalid-argument"
        assert error["message"].startswith(f"{option} {value!r}")
        assert "beta:ALPHA,BETA" in error["hint"]
        assert graph_file.read_bytes() == before

    @pytest.mark.parametrize("value", ["#p4", "doi:x#", "#", ""])
    def test_invalid_source(self, cli, graph_file, value):
        """Test that a source without a document, or with an empty locator after ``#``, is refused."""
        before = graph_file.read_bytes()
        error = cli.error("add-node", graph_file, "--id", "new", "--source", value)
        assert error["code"] == "invalid-argument"
        assert "DOCUMENT#LOCATOR" in error["hint"]
        assert graph_file.read_bytes() == before

    @pytest.mark.parametrize("args", [("--id", ""), ("--id", "new", "--statement", ""), ("--id", "new", "--kind", "")])
    def test_invalid_field(self, cli, graph_file, args):
        """Test that the data model's own checks reach the caller as invalid arguments."""
        before = graph_file.read_bytes()
        assert cli.error("add-node", graph_file, *args)["code"] == "invalid-argument"
        assert graph_file.read_bytes() == before

    def test_missing_file(self, cli, tmp_path):
        """Test that a missing graph file points at ``init``."""
        path = tmp_path / "g.json"
        error = cli.error("add-node", path, "--id", "h1")
        assert error["code"] == "file-not-found"
        assert f"credencegraph init {path}" in error["hint"]
        assert not path.exists()

    @pytest.mark.parametrize(
        "content",
        [
            "not json",
            json.dumps({"format": "other", "version": 1, "nodes": [], "relations": []}),
            json.dumps({"format": "credencegraph", "version": 1, "nodes": [{"id": "a", "base": 2}], "relations": []}),
        ],
    )
    def test_invalid_file(self, cli, tmp_path, content):
        """Test that a file that is not a valid graph is reported and left alone."""
        path = tmp_path / "g.json"
        path.write_text(content, encoding="utf-8")
        assert cli.error("add-node", path, "--id", "h1")["code"] == "invalid-graph"
        assert path.read_text(encoding="utf-8") == content


class TestRelate:
    """``relate`` adds a relation between two existing nodes, keeping the graph acyclic."""

    def test_adds_an_inferential_relation(self, cli, graph_file):
        """Test the default id, the strength and the output."""
        output = cli.json("relate", graph_file, "calibrated", "claim", "--type", "refutes", "--strength", "beta:1,3")
        relation = load(graph_file).relations["calibrated-refutes-claim"]
        assert (relation.type, relation.source, relation.target) == ("refutes", "calibrated", "claim")
        assert relation.strength == Beta(1, 3)
        assert output == {
            "command": "relate",
            "path": str(graph_file),
            "relation": {
                "id": "calibrated-refutes-claim",
                "type": "refutes",
                "source": "calibrated",
                "target": "claim",
                "strength": {"alpha": 1.0, "beta": 3.0},
                "inferential": True,
            },
            "warnings": [],
        }

    @pytest.mark.parametrize("rtype", ["equivalent", "exclusive"])
    def test_relation_without_strength(self, cli, graph_file, rtype):
        """Test that ``equivalent`` and ``exclusive`` relations take no strength."""
        output = cli.json("relate", graph_file, "signal", "calibrated", "--type", rtype)
        assert output["relation"]["strength"] is None
        assert output["relation"]["inferential"] is True
        assert load(graph_file).relations[f"signal-{rtype}-calibrated"].strength is None

    def test_annotation(self, cli, graph_file):
        """Test that any other type is an annotation, stored without a strength or a warning."""
        output = cli.json("relate", graph_file, "signal", "alice", "--type", "cites", "--id", "c1")
        assert output["relation"]["id"] == "c1"
        assert output["relation"]["inferential"] is False
        assert output["warnings"] == []
        assert load(graph_file).relations["c1"].type == "cites"

    def test_near_miss_type_warns(self, cli, graph_file):
        """Test that a type one letter off an inferential one is stored but flagged."""
        output = cli.json("relate", graph_file, "calibrated", "claim", "--type", "support")
        assert output["relation"]["inferential"] is False
        assert len(output["warnings"]) == 1
        assert "'supports'" in output["warnings"][0]

    @pytest.mark.parametrize(("source", "target", "role"), [("nope", "claim", "source"), ("claim", "nope", "target")])
    def test_unknown_node(self, cli, graph_file, source, target, role):
        """Test that an endpoint that is not a node is refused and named."""
        before = graph_file.read_bytes()
        error = cli.error("relate", graph_file, source, target, "--type", "supports", "--strength", "0.5")
        assert error["code"] == "unknown-node"
        assert error["details"] == {"node": "nope"}
        assert error["message"].startswith(role)
        assert graph_file.read_bytes() == before

    def test_unknown_node_suggests_close_ids(self, cli, graph_file):
        """Test that a misspelt node id is answered with the ids it resembles."""
        error = cli.error("relate", graph_file, "signl", "claim", "--type", "supports", "--strength", "0.5")
        assert "'signal'" in error["hint"]

    def test_duplicate_default_id(self, cli, graph_file):
        """Test that a second relation of the same type between the same nodes needs its own id."""
        args = ("relate", graph_file, "calibrated", "claim", "--type", "supports", "--strength", "0.5")
        cli.json(*args)
        before = graph_file.read_bytes()
        error = cli.error(*args)
        assert error["code"] == "duplicate-id"
        assert error["details"] == {"relation": "calibrated-supports-claim"}
        assert "--id" in error["hint"]
        assert graph_file.read_bytes() == before
        cli.json(*args, "--id", "second")
        assert "second" in load(graph_file).relations

    def test_duplicate_explicit_id(self, cli, graph_file):
        """Test that an explicit id already taken is refused."""
        before = graph_file.read_bytes()
        error = cli.error("relate", graph_file, "calibrated", "claim", "--type", "cites", "--id", "r1")
        assert error["code"] == "duplicate-id"
        assert error["details"] == {"relation": "r1"}
        assert graph_file.read_bytes() == before

    @pytest.mark.parametrize("rtype", ["requires", "supports", "refutes"])
    def test_missing_strength(self, cli, graph_file, rtype):
        """Test that a relation that needs a strength is refused without one, with the forms it takes."""
        before = graph_file.read_bytes()
        error = cli.error("relate", graph_file, "calibrated", "claim", "--type", rtype)
        assert error["code"] == "invalid-argument"
        assert "--strength" in error["hint"]
        assert graph_file.read_bytes() == before

    @pytest.mark.parametrize(
        ("rtype", "hint"),
        [("equivalent", "drop --strength"), ("cites", "drop --strength"), ("suports", "--type supports")],
    )
    def test_strength_where_none_is_allowed(self, cli, graph_file, rtype, hint):
        """Test that a strength on a type without one is refused, suggesting the type probably meant."""
        before = graph_file.read_bytes()
        error = cli.error("relate", graph_file, "calibrated", "claim", "--type", rtype, "--strength", "0.5")
        assert error["code"] == "invalid-argument"
        assert hint in error["hint"]
        assert graph_file.read_bytes() == before

    def test_invalid_strength(self, cli, graph_file):
        """Test that the strength is parsed like any other credence."""
        error = cli.error("relate", graph_file, "calibrated", "claim", "--type", "supports", "--strength", "2")
        assert error["code"] == "invalid-argument"
        assert error["message"].startswith("--strength '2'")

    def test_cycle(self, cli, graph_file):
        """Test that a relation closing a cycle is refused and the cycle is named."""
        before = graph_file.read_bytes()
        error = cli.error("relate", graph_file, "claim", "calibrated", "--type", "supports", "--strength", "0.5")
        assert error["code"] == "cycle"
        assert error["details"] == {
            "cycle": ["claim", "calibrated", "signal", "claim"],
            "relations": ["claim-supports-calibrated", "r1", "r2"],
        }
        assert graph_file.read_bytes() == before

    def test_self_equivalence(self, cli, graph_file):
        """Test that the data model's own refusals reach the caller."""
        error = cli.error("relate", graph_file, "claim", "claim", "--type", "equivalent")
        assert error["code"] == "invalid-argument"
        assert "both 'claim'" in error["message"]


def test_text_mode_errors_go_to_stderr(cli, graph_file):
    """Test that without ``--json`` a failure prints nothing on stdout and exits with status 1."""
    result = cli.run("add-node", graph_file, "--id", "claim")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == ["error: a node with id 'claim' already exists", "hint: choose another --id"]


def test_text_mode_success(cli, graph_file):
    """Test that without ``--json`` a success prints a summary, not JSON."""
    result = cli.run("add-node", graph_file, "--id", "bob")
    assert result.exit_code == 0
    assert result.stdout == f"added node 'bob' to {graph_file}\n"


def test_failed_write_keeps_the_old_file(cli, graph_file, mocker):
    """Test that a write that fails at the last step leaves the old file intact and no temporary file."""
    before = graph_file.read_bytes()
    mocker.patch("credencegraph.cli.common.Path.replace", side_effect=OSError("disk full"))
    error = cli.error("add-node", graph_file, "--id", "bob")
    assert error["code"] == "io-error"
    assert "disk full" in error["message"]
    assert graph_file.read_bytes() == before
    assert sorted(p.name for p in graph_file.parent.iterdir()) == [graph_file.name]
