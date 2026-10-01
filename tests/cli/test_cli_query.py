"""The ``query`` command."""

from __future__ import annotations

import pytest
from _cli import example_graph

from credencegraph.core import Graph, Node, Relation, dump
from credencegraph.inference import conditional, intervene, joint, marginal
from credencegraph.semantics import compile_graph

# Closed forms for the example graph, written independently of the engine (see ``example_graph``).
P_SIGNAL = 0.9 * 0.05
P_CLAIM = 1 - (1 - 0.1) * (1 - 0.8 * P_SIGNAL)


def assert_close(actual, expected):
    """Compare two probabilities to double-precision roundoff."""
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)


class TestSuccess:
    """Each kind of query, checked against closed forms and against the library."""

    def test_marginal(self, cli, graph_file):
        """Test the point answer, the echoed query, and a band made from the given seed."""
        output = cli.json("query", graph_file, "marginal", "claim", "--seed", "7", "--draws", "200")
        assert_close(output["point"], P_CLAIM)
        library = marginal(compile_graph(example_graph()), "claim", draws=200, rng=7).to_dict()
        assert output == {
            "command": "query",
            "path": str(graph_file),
            "kind": "marginal",
            "target": {"claim": True},
            "given": {},
            "set": {},
            "seed": 7,
            **library,
        }
        assert output["band"]["q05"] <= output["band"]["q50"] <= output["band"]["q95"]

    def test_marginal_of_false(self, cli, graph_file):
        """Test that a target can ask for the value false."""
        output = cli.json("query", graph_file, "marginal", "claim=false", "--draws", "0")
        assert output["target"] == {"claim": False}
        assert_close(output["point"], 1 - P_CLAIM)

    def test_no_draws_no_band(self, cli, graph_file):
        """Test that ``--draws 0`` leaves out the band and its companions."""
        output = cli.json("query", graph_file, "marginal", "claim", "--draws", "0")
        assert {"band", "mean_over_draws", "draws"}.isdisjoint(output)

    def test_joint(self, cli, graph_file):
        """Test a joint query over two targets: signal requires calibrated, so P(both) = P(signal)."""
        output = cli.json("query", graph_file, "joint", "calibrated", "signal", "--draws", "0")
        assert output["target"] == {"calibrated": True, "signal": True}
        assert_close(output["point"], P_SIGNAL)
        expected = joint(compile_graph(example_graph()), ["calibrated", "signal"], draws=0).point
        assert output["point"] == expected

    def test_conditional(self, cli, graph_file):
        """Test conditioning on the conclusion updates the premise, by Bayes' rule."""
        output = cli.json("query", graph_file, "conditional", "signal", "--given", "claim=true", "--draws", "0")
        p_claim_given_signal = 1 - (1 - 0.1) * (1 - 0.8)
        assert_close(output["point"], p_claim_given_signal * P_SIGNAL / P_CLAIM)
        assert output["given"] == {"claim": True}
        expected = conditional(compile_graph(example_graph()), "signal", {"claim": True}, draws=0).point
        assert output["point"] == expected

    def test_marginal_with_evidence_is_conditional(self, cli, graph_file):
        """Test that ``marginal`` with ``--given`` gives the conditional answer."""
        args = ("signal", "--given", "claim=false", "--draws", "0")
        by_marginal = cli.json("query", graph_file, "marginal", *args)
        by_conditional = cli.json("query", graph_file, "conditional", *args)
        assert by_marginal["point"] == by_conditional["point"]
        assert by_marginal["kind"] == "marginal"

    def test_intervene(self, cli, graph_file):
        """Test that setting the premise false cuts it off: the claim falls back to its base."""
        output = cli.json("query", graph_file, "intervene", "claim", "--set", "calibrated=false", "--draws", "0")
        assert_close(output["point"], 0.1)
        assert output["set"] == {"calibrated": False}
        expected = intervene(compile_graph(example_graph()), "claim", {"calibrated": False}, draws=0).point
        assert output["point"] == expected

    def test_intervene_differs_from_conditioning(self, cli, graph_file):
        """Test that intervening on the conclusion leaves the premise alone, unlike conditioning."""
        done = cli.json("query", graph_file, "intervene", "signal", "--set", "claim=true", "--draws", "0")
        seen = cli.json("query", graph_file, "conditional", "signal", "--given", "claim=true", "--draws", "0")
        assert_close(done["point"], P_SIGNAL)
        assert seen["point"] > done["point"]

    @pytest.mark.parametrize(
        ("raw", "value"), [("TRUE", True), ("yes", True), ("1", True), ("No", False), ("0", False)]
    )
    def test_truth_values(self, cli, graph_file, raw, value):
        """Test the spellings accepted for true and false."""
        output = cli.json("query", graph_file, "marginal", "claim", "--given", f"signal={raw}", "--draws", "0")
        assert output["given"] == {"signal": value}

    def test_text_output(self, cli, graph_file):
        """Test the human-readable form."""
        result = cli.run(
            "query",
            graph_file,
            "intervene",
            "claim",
            "--set",
            "calibrated=false",
            "--given",
            "signal=false",
            "--draws",
            "0",
        )
        assert result.exit_code == 0
        assert result.stdout == "P(claim=true | do(calibrated=false), signal=false) = 0.1\n"


class TestErrors:
    """Queries that do not fit their kind, or the graph, are refused with a code."""

    @pytest.mark.parametrize(
        ("args", "fragment"),
        [
            (("margin", "claim"), "unknown query kind"),
            (("marginal", "claim", "signal"), "one target node"),
            (("conditional", "claim"), "needs evidence"),
            (("intervene", "claim"), "needs an intervention"),
            (("marginal", "claim", "--set", "signal=false"), "takes no --set"),
            (("joint", "claim", "--set", "signal=false"), "takes no --set"),
            (("conditional", "claim", "--given", "signal=true", "--set", "calibrated=false"), "takes no --set"),
            (("marginal", "claim=maybe"), "is not NODE or NODE=true|false"),
            (("marginal", "=true"), "is not NODE or NODE=true|false"),
            (("marginal", "claim", "--given", "signal"), "is not NODE=true|false"),
            (("marginal", "claim", "--given", "signal="), "is not NODE=true|false"),
            (("joint", "claim", "claim=false"), "both true and false"),
            (("marginal", "claim", "--given", "signal=true", "--given", "signal=false"), "both true and false"),
            (("marginal", "claim", "--draws", "-1"), "--draws must be 0 or more"),
        ],
    )
    def test_invalid_query(self, cli, graph_file, args, fragment):
        """Test each shape rule, and the argument parsing, one violation at a time."""
        error = cli.error("query", graph_file, *args)
        assert error["code"] == "invalid-argument"
        assert fragment in error["message"]

    def test_unknown_kind_lists_the_kinds(self, cli, graph_file):
        """Test that the error for an unknown kind carries the valid ones."""
        error = cli.error("query", graph_file, "margin", "claim")
        assert error["details"] == {"kinds": ["marginal", "joint", "conditional", "intervene"]}

    @pytest.mark.parametrize(
        "args",
        [
            ("marginal", "clam"),
            ("marginal", "claim", "--given", "clam=true"),
            ("intervene", "claim", "--set", "clam=true"),
        ],
    )
    def test_unknown_node(self, cli, graph_file, args):
        """Test that a node missing from the graph is named, with the ids it resembles."""
        error = cli.error("query", graph_file, *args)
        assert error["code"] == "unknown-node"
        assert error["details"] == {"node": "clam"}
        assert "'claim'" in error["hint"]

    def test_carried_node(self, cli, graph_file):
        """Test that a node that is not an inference variable cannot be queried."""
        error = cli.error("query", graph_file, "marginal", "alice")
        assert error["code"] == "invalid-argument"
        assert "takes no part in inference" in error["message"]

    def test_missing_base(self, cli, tmp_path):
        """Test that a graph that cannot compile names the node without a base."""
        graph = Graph()
        graph.add_node(Node("a", base=0.5))
        graph.add_node(Node("b"))
        graph.add_relation(Relation("ab", "supports", "a", "b", strength=0.5))
        path = tmp_path / "g.json"
        dump(graph, path)
        error = cli.error("query", path, "marginal", "b")
        assert error["code"] == "compile-error"
        assert "set a base on each node listed" in error["hint"]

    def test_impossible_evidence(self, cli, tmp_path):
        """Test that evidence of probability zero is reported as such."""
        graph = Graph()
        graph.add_node(Node("a", base=0.5))
        graph.add_node(Node("b", base=0.5))
        graph.add_relation(Relation("x", "exclusive", "a", "b"))
        path = tmp_path / "g.json"
        dump(graph, path)
        error = cli.error("query", path, "conditional", "a", "--given", "a=true", "--given", "b=true")
        assert error["code"] == "zero-probability"

    def test_missing_file(self, cli, tmp_path):
        """Test that a missing graph file is reported."""
        assert cli.error("query", tmp_path / "g.json", "marginal", "a")["code"] == "file-not-found"

    def test_shape_is_checked_before_the_file(self, cli, tmp_path):
        """Test that a malformed query is reported as such even when the file is missing too."""
        assert cli.error("query", tmp_path / "g.json", "margin", "a")["code"] == "invalid-argument"
