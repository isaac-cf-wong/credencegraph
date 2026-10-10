"""The ``diagnose``, ``check`` and ``version`` commands."""

from __future__ import annotations

import json

import pytest
from _cli import example_graph

import credencegraph
from credencegraph.core import Beta, Graph, Node, Relation, ValidationError, dump
from credencegraph.diagnostics import diagnose, diagnose_many
from credencegraph.inference import VariableElimination


class TestDiagnose:
    """``diagnose`` prints the library's findings as JSON records."""

    def test_with_target(self, cli, graph_file):
        """Test that the findings are exactly the library's, in its order."""
        output = cli.json("diagnose", graph_file, "--target", "claim")
        expected = diagnose_many(example_graph(), ["claim"]).to_dicts()
        assert output == {"command": "diagnose", "path": str(graph_file), "targets": ["claim"], "findings": expected}
        ids = [finding["id"] for finding in output["findings"]]
        assert "unanchored:calibrated" in ids
        assert "overclaim:claim" in ids
        assert "crux:strength:r2" in ids

    def test_every_parameter_a_point(self, cli, tmp_path):
        """Test that a graph of plain numbers gets one crux record saying why, not a ranking of zeros."""
        graph = Graph()
        graph.add_node(Node("y", base=0.8))
        graph.add_node(Node("t", base=0.9))
        graph.add_relation(Relation("a", "requires", "y", "t", strength=0.9))
        path = tmp_path / "points.json"
        dump(graph, path)
        output = cli.json("diagnose", path, "--target", "t")
        cruxes = [finding for finding in output["findings"] if finding["diagnostic"] == "crux"]
        assert [(finding["id"], finding["value"]) for finding in cruxes] == [("crux:t", None)]
        assert cruxes[0]["message"].startswith("crux is undefined")

    def test_without_target(self, cli, graph_file):
        """Test that without a target only the graph-wide checks run."""
        output = cli.json("diagnose", graph_file)
        assert output["targets"] == []
        assert [finding["id"] for finding in output["findings"]] == ["unanchored:calibrated", "overclaim:claim"]

    def test_thresholds_are_passed_on(self, cli, graph_file):
        """Test that a claim threshold above the 1.88 log-odds gap silences the overclaim, and a failure threshold is used."""
        output = cli.json(
            "diagnose", graph_file, "--target", "claim", "--claim-threshold", "2", "--failure-threshold", "0.2"
        )
        expected = diagnose_many(example_graph(), ["claim"], claim_threshold=2.0, failure_threshold=0.2)
        assert output["findings"] == expected.to_dicts()
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
        expected = diagnose_many(graph, ["child"], engine=VariableElimination(16))
        assert output["findings"] == expected.to_dicts()

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
    mocker.patch("credencegraph.cli.diagnose.diagnose_many", side_effect=ValidationError("bad value"))
    error = cli.error("diagnose", graph_file, "--target", "claim")
    assert (error["code"], error["message"], error["details"]) == ("invalid-argument", "bad value", {})
    assert error["hint"] == "check the arguments against 'credencegraph diagnose --help'"


def observed_graph() -> Graph:
    """A hypothesis H, stated at 0.9, with two observations of it, and a node merged with the first."""
    return graph_with(
        Node("H", base=0.3, stated=0.9),
        Node("o1", base=0.2),
        Node("o2", base=0.2),
        Node("o1c"),
        relations=(
            Relation("H1", "supports", "H", "o1", strength=0.75),
            Relation("H2", "supports", "H", "o2", strength=0.75),
            Relation("same", "equivalent", "o1", "o1c"),
        ),
    )


class TestDiagnoseGiven:
    """``diagnose --given`` runs every inference diagnostic under the evidence."""

    @pytest.fixture
    def path(self, tmp_path):
        """Write the observation graph."""
        path = tmp_path / "obs.json"
        dump(observed_graph(), path)
        return path

    def test_findings_are_the_library_ones(self, cli, path):
        """Test that the findings are the library's under the same evidence, and the evidence is echoed."""
        output = cli.json("diagnose", path, "--target", "H", "--given", "o1=true", "--given", "o2=false")
        evidence = {"o1": True, "o2": False}
        expected = diagnose_many(observed_graph(), ["H"], evidence=evidence).to_dicts()
        assert output == {
            "command": "diagnose",
            "path": str(path),
            "targets": ["H"],
            "given": evidence,
            "findings": expected,
        }
        assert expected != diagnose_many(observed_graph(), ["H"]).to_dicts()

    def test_without_given_is_unchanged(self, cli, path):
        """Test that without --given the response has no ``given`` key and the unconditioned findings."""
        output = cli.json("diagnose", path, "--target", "H")
        assert list(output) == ["command", "path", "targets", "findings"]
        assert output["findings"] == diagnose_many(observed_graph(), ["H"]).to_dicts()

    def test_given_without_target(self, cli, path):
        """Test that the graph-wide checks take the evidence too: the posterior settles the overclaim."""
        assert [f["id"] for f in cli.json("diagnose", path)["findings"]] == ["unanchored:H", "overclaim:H"]
        output = cli.json("diagnose", path, "--given", "o1=true", "--given", "o2=true")
        assert [f["id"] for f in output["findings"]] == ["unanchored:H"]

    def test_malformed(self, cli, path):
        """Test that --given needs an explicit value."""
        error = cli.error("diagnose", path, "--target", "H", "--given", "o1")
        assert error["code"] == "invalid-argument"
        assert "NODE=true|false" in error["message"]

    def test_unknown_node(self, cli, path):
        """Test that a --given node missing from the graph is named."""
        error = cli.error("diagnose", path, "--target", "H", "--given", "o3=true")
        assert error["code"] == "unknown-node"
        assert error["details"] == {"node": "o3"}

    def test_equivalent_nodes_given_different_values(self, cli, path):
        """Test that merged nodes given different values are refused before any inference."""
        error = cli.error("diagnose", path, "--target", "H", "--given", "o1=true", "--given", "o1c=false")
        assert error["code"] == "invalid-argument"
        assert "equivalent nodes 'o1' and 'o1c' different values" in error["message"]

    def test_observed_target(self, cli, path):
        """Test that observing the target is refused."""
        error = cli.error("diagnose", path, "--target", "H", "--given", "H=true")
        assert error["code"] == "invalid-argument"
        assert "target 'H' is fixed" in error["message"]

    def test_evidence_is_written_as_typed(self, cli, path):
        """Test that a finding's message and a command-line error write evidence exactly as it was typed."""
        typed = "o1=false"
        findings = {f["id"]: f for f in cli.json("diagnose", path, "--given", typed)["findings"]}
        assert f" given {typed}: " in findings["overclaim:H"]["message"]
        error = cli.error("query", path, "intervene", "H", "--given", typed, "--set", "o1=true")
        assert error["message"] == f"--given {typed} contradicts --set o1=true"

    def test_impossible_evidence_points_at_given(self, cli, tmp_path):
        """Test that evidence the graph rules out is a zero-probability error naming --given."""
        path = tmp_path / "g.json"
        dump(graph_with(Node("A", base=0.0), Node("T", base=0.5)), path)
        error = cli.error("diagnose", path, "--target", "T", "--given", "A=true")
        assert error["code"] == "zero-probability"
        assert "--given" in error["message"]
        assert "--given" in error["hint"]
        assert error["details"] == {"given": {"A": True}}

    def test_impossible_graph_with_evidence_points_at_the_graph(self, cli, tmp_path):
        """Test that a graph that fails without the evidence still points at the graph."""
        path = tmp_path / "g.json"
        dump(
            graph_with(Node("A", base=1.0), Node("B", base=1.0), relations=(Relation("x", "exclusive", "A", "B"),)),
            path,
        )
        error = cli.error("diagnose", path, "--target", "A", "--given", "B=true")
        assert error["code"] == "zero-probability"
        assert "--given" not in error["hint"]
        assert "exactly 0 or 1" in error["hint"]

    def test_missing_parameter_is_reported_with_evidence(self, cli, tmp_path):
        """Test that a graph with a missing base gets its structural findings whatever the evidence."""
        path = tmp_path / "g.json"
        dump(
            graph_with(Node("a", base=0.5), Node("b"), relations=(Relation("ab", "supports", "a", "b", strength=0.5),)),
            path,
        )
        output = cli.json("diagnose", path, "--target", "b", "--given", "a=true")
        assert [f["id"] for f in output["findings"]] == ["missing-parameter:base:b", "unanchored:a"]


def test_failure_impact_ranks_a_premise_below_the_default_line(cli, tmp_path):
    """Test that a premise behind one requires of 0.9 is ranked, though it is not a single point of failure.

    y (base 0.8) is required by t with strength 0.9, so failing y keeps
    0.1 / (0.1 + 0.9 * 0.8) = 0.122 of P(t): above the default threshold 0.1, never below 1 - 0.9.
    """
    path = tmp_path / "g.json"
    dump(
        graph_with(
            Node("y", base=0.8),
            Node("t", base=0.9),
            relations=(Relation("yt", "requires", "y", "t", strength=0.9),),
        ),
        path,
    )
    findings = cli.json("diagnose", path, "--target", "t")["findings"]
    assert not [f for f in findings if f["diagnostic"] == "single-point-of-failure"]
    (impact,) = [f for f in findings if f["diagnostic"] == "failure-impact"]
    assert impact["id"] == "failure-impact:y"
    assert impact["value"] == pytest.approx(0.1 / (0.1 + 0.9 * 0.8), rel=1e-12, abs=0.0)
    assert impact["details"]["single_point_of_failure"] is False
    assert impact["details"]["threshold"] == 0.1
    raised = cli.json("diagnose", path, "--target", "t", "--failure-threshold", "0.2")["findings"]
    (impact,) = [f for f in raised if f["diagnostic"] == "failure-impact"]
    assert impact["details"]["single_point_of_failure"] is True
    assert [f["id"] for f in raised if f["diagnostic"] == "single-point-of-failure"] == ["single-point-of-failure:y"]


def claims_graph() -> Graph:
    """Two stated claims A and B resting on one premise P, a stated observation O merged with O2, and a stated note N.

    N has no base and no inferential relation, so it takes no part in inference.
    """
    return graph_with(
        Node("P", base=Beta(6, 4)),
        Node("A", base=0.3, stated=0.9),
        Node("B", base=0.2, stated=0.5),
        Node("O", base=0.4),
        Node("O2", stated=0.4),
        Node("N", kind="note", stated=0.7),
        relations=(
            Relation("PA", "requires", "P", "A", strength=0.8),
            Relation("PB", "supports", "P", "B", strength=0.6),
            Relation("same", "equivalent", "O", "O2"),
        ),
    )


class TestDiagnoseTargets:
    """``diagnose`` with several targets: the graph-wide checks once, then each target's weak points."""

    @pytest.fixture
    def path(self, tmp_path):
        """Write the claims graph."""
        path = tmp_path / "claims.json"
        dump(claims_graph(), path)
        return path

    def test_repeated_target(self, cli, path):
        """Test that each finding names its target, and the graph-wide findings appear once."""
        output = cli.json("diagnose", path, "--target", "A", "--target", "B", "--target", "A")
        assert output["targets"] == ["A", "B"]
        assert output["findings"] == diagnose_many(claims_graph(), ["A", "B"]).to_dicts()
        graph_wide = [f.to_dict() for f in diagnose(claims_graph())]
        assert [{**f, "target": None} for f in graph_wide] == [f for f in output["findings"] if f["target"] is None]
        for target in ("A", "B"):
            single = [f.to_dict() for f in diagnose(claims_graph(), target)][len(graph_wide) :]
            assert single
            assert [{**f, "target": target} for f in single] == [f for f in output["findings"] if f["target"] == target]
        ids = [f["id"] for f in output["findings"]]
        assert ids.count("overclaim:A") == 1
        assert ids.count("crux:base:P") == 2
        assert ids.count("failure-impact:P") == 2

    def test_targets_stated(self, cli, path):
        """Test that --targets stated picks the stated inference variables, after any --target nodes."""
        assert cli.json("diagnose", path, "--targets", "stated")["targets"] == ["A", "B", "O2"]
        output = cli.json("diagnose", path, "--target", "P", "--target", "B", "--targets", "stated")
        assert output["targets"] == ["P", "B", "A", "O2"]
        assert output["findings"] == diagnose_many(claims_graph(), ["P", "B", "A", "O2"]).to_dicts()

    def test_targets_stated_skips_the_observed(self, cli, path):
        """Test that a stated node observed through an equivalent node is not a target, as it is not a claim."""
        output = cli.json("diagnose", path, "--targets", "stated", "--given", "O=true")
        assert output["targets"] == ["A", "B"]
        evidence = {"O": True}
        assert output["findings"] == diagnose_many(claims_graph(), ["A", "B"], evidence=evidence).to_dicts()

    def test_targets_other_than_stated(self, cli, path):
        """Test that --targets accepts only ``stated``."""
        error = cli.error("diagnose", path, "--targets", "all")
        assert error["code"] == "invalid-argument"
        assert "--targets must be 'stated'" in error["message"]

    def test_text_output(self, cli, path):
        """Test that the text form has one line per finding, for every target."""
        result = cli.run("diagnose", path, "--target", "A", "--target", "B")
        assert result.exit_code == 0
        report = diagnose_many(claims_graph(), ["A", "B"])
        assert result.stdout.splitlines() == [f"{f.id}: {f.message}" for f in report.all_findings()]


def test_support_pile_of_the_reading_page(cli, tmp_path):
    """Test the commands of the reading page's support pile: the finding, then the common parent that clears it."""
    path = tmp_path / "pile.json"
    cli.json("init", path)
    cli.json("add-node", path, "--id", "claim", "--base", "0")
    for i in range(1, 11):
        cli.json("add-node", path, "--id", f"reason-{i}", "--base", "1", "--source", "paper-x")
        cli.json("relate", path, f"reason-{i}", "claim", "--type", "supports", "--strength", "0.2")
    assert cli.json("query", path, "marginal", "claim")["point"] == pytest.approx(1 - 0.8**10, rel=1e-12, abs=0.0)
    (finding,) = cli.json("diagnose", path)["findings"]
    assert finding["id"] == "correlated-support:claim:document:paper-x"
    assert finding["target"] is None
    assert finding["nodes"] == ["claim", *(f"reason-{i}" for i in range(1, 11))]
    assert finding["details"] == {"supports": 10.0, "min_supports": 2.0}
    cli.json("add-node", path, "--id", "sound", "--base", "0.5", "--source", "paper-x")
    for i in range(1, 11):
        cli.json("relate", path, "sound", f"reason-{i}", "--type", "requires", "--strength", "1")
    point = cli.json("query", path, "marginal", "claim")["point"]
    assert point == pytest.approx(0.5 * (1 - 0.8**10), rel=1e-12, abs=0.0)
    assert cli.json("diagnose", path)["findings"] == []
