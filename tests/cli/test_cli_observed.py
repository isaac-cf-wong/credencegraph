"""``query --observed`` and ``diagnose --observed``: conditioning on the observations a graph records.

A claim ``c`` of prior h is checked by a check ``k`` of sensitivity s = P(pass | c) and false-pass rate
f = P(pass | not c), recorded either as an ``observes`` check edge or as the equivalent ``supports``
relation into a check node of base f. Every expected value is written from Bayes' rule in odds
form, independently of the engine.
"""

from __future__ import annotations

import pytest

from credencegraph.core import Graph, Node, Relation, dump
from credencegraph.diagnostics.report import diagnose_many

PRIOR = 0.3
SENSITIVITY = 0.9
FALSE_PASS = 0.2


def posterior(prior: float, *results: bool) -> float:
    """P(c | results) by Bayes' rule in odds form, one likelihood ratio per check result."""
    odds = prior / (1 - prior)
    for passed in results:
        odds *= SENSITIVITY / FALSE_PASS if passed else (1 - SENSITIVITY) / (1 - FALSE_PASS)
    return odds / (1 + odds)


def checked(form: str, *observations: bool | None) -> Graph:
    """The claim ``c``, with one check ``k0``, ``k1``, ... per observation; ``None`` leaves it unobserved."""
    graph = Graph()
    graph.add_node(Node("c", base=PRIOR))
    for index, observed in enumerate(observations):
        attributes = {} if observed is None else {"observed": observed}
        if form == "observes":
            graph.add_node(Node(f"k{index}", attributes=attributes))
            relation = Relation(
                f"e{index}", "observes", "c", f"k{index}", sensitivity=SENSITIVITY, false_pass=FALSE_PASS
            )
        else:
            graph.add_node(Node(f"k{index}", base=FALSE_PASS, attributes=attributes))
            strength = 1 - (1 - SENSITIVITY) / (1 - FALSE_PASS)
            relation = Relation(f"e{index}", "supports", "c", f"k{index}", strength=strength)
        graph.add_relation(relation)
    return graph


def written(tmp_path, graph: Graph, name: str = "g.json"):
    """Write a graph to a file and return its path."""
    path = tmp_path / name
    dump(graph, path)
    return path


def assert_close(actual, expected):
    """Compare two probabilities to double-precision roundoff."""
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)


FORMS = ("observes", "supports")


class TestQueryObserved:
    """``query --observed`` conditions on every node's ``attributes.observed``."""

    @pytest.mark.parametrize("form", FORMS)
    @pytest.mark.parametrize("passed", [True, False], ids=["pass", "fail"])
    def test_bayes_closed_form(self, cli, tmp_path, form, passed):
        """Test that the claim's posterior given a pass or a fail is Bayes' rule for s and f."""
        path = written(tmp_path, checked(form, passed))
        output = cli.json("query", path, "marginal", "c", "--observed", "--draws", "0")
        assert_close(output["point"], posterior(PRIOR, passed))
        assert output["given"] == {}
        assert output["observed"] == {"k0": passed}

    @pytest.mark.parametrize("form", FORMS)
    def test_closed_form_is_the_explicit_posterior(self, cli, tmp_path, form):
        """Test the pass and fail posteriors against s h / (s h + f (1 - h)) and its complement form."""
        h, s, f = PRIOR, SENSITIVITY, FALSE_PASS
        for passed, expected in (
            (True, s * h / (s * h + f * (1 - h))),
            (False, (1 - s) * h / ((1 - s) * h + (1 - f) * (1 - h))),
        ):
            path = written(tmp_path, checked(form, passed), f"{passed}.json")
            assert_close(cli.json("query", path, "marginal", "c", "--observed", "--draws", "0")["point"], expected)

    @pytest.mark.parametrize("form", FORMS)
    def test_same_as_given(self, cli, tmp_path, form):
        """Test that --observed gives the answer of passing every observation as --given."""
        path = written(tmp_path, checked(form, True, False))
        observed = cli.json("query", path, "marginal", "c", "--observed", "--draws", "0")
        given = cli.json("query", path, "marginal", "c", "--given", "k0=true", "--given", "k1=false", "--draws", "0")
        assert observed["point"] == given["point"]
        assert_close(observed["point"], posterior(PRIOR, True, False))

    @pytest.mark.parametrize("form", FORMS)
    def test_absent_observation_is_no_check(self, cli, tmp_path, form):
        """Test that a check with no attributes.observed changes nothing: the answer is the one without it."""
        unchecked = cli.json("query", written(tmp_path, checked(form), "none.json"), "marginal", "c", "--draws", "0")
        path = written(tmp_path, checked(form, None))
        output = cli.json("query", path, "marginal", "c", "--observed", "--draws", "0")
        assert_close(unchecked["point"], PRIOR)
        assert output["point"] == unchecked["point"]
        assert output["observed"] == {}

    @pytest.mark.parametrize("form", FORMS)
    def test_merges_with_given(self, cli, tmp_path, form):
        """Test that --observed adds to --given: one check observed in the graph, one given on the command line."""
        path = written(tmp_path, checked(form, True, None))
        output = cli.json("query", path, "marginal", "c", "--observed", "--given", "k1=false", "--draws", "0")
        assert_close(output["point"], posterior(PRIOR, True, False))
        assert output["given"] == {"k1": False}
        assert output["observed"] == {"k0": True}

    def test_given_agreeing_with_observed(self, cli, tmp_path):
        """Test that --given may repeat an observed value."""
        path = written(tmp_path, checked("observes", True))
        output = cli.json("query", path, "marginal", "c", "--observed", "--given", "k0=true", "--draws", "0")
        assert_close(output["point"], posterior(PRIOR, True))

    @pytest.mark.parametrize("form", FORMS)
    def test_conflict_with_given(self, cli, tmp_path, form):
        """Test that --given giving an observed node the other value is an invalid-argument naming both."""
        path = written(tmp_path, checked(form, True))
        error = cli.error("query", path, "marginal", "c", "--observed", "--given", "k0=false")
        assert error["code"] == "invalid-argument"
        assert error["message"] == (
            "--given k0=false contradicts --observed k0=true, the value in its attributes.observed"
        )
        assert "drop" in error["hint"]
        assert error["details"] == {"node": "k0"}

    def test_conflict_through_an_equivalent_node(self, cli, tmp_path):
        """Test that --given on a node equivalent to an observed one must agree with the observation."""
        graph = checked("supports", True)
        graph.add_node(Node("k0c"))
        graph.add_relation(Relation("same", "equivalent", "k0", "k0c"))
        path = written(tmp_path, graph)
        error = cli.error("query", path, "marginal", "c", "--observed", "--given", "k0c=false")
        assert error["code"] == "invalid-argument"
        assert error["message"] == "--given k0c=false contradicts --observed k0=true, and 'k0c' and 'k0' are equivalent"

    def test_conflict_with_set(self, cli, tmp_path):
        """Test that an observed value contradicting --set is named as --observed."""
        graph = checked("supports", True)
        path = written(tmp_path, graph)
        error = cli.error("query", path, "intervene", "c", "--observed", "--set", "k0=false")
        assert error["code"] == "invalid-argument"
        assert error["message"] == "--observed k0=true contradicts --set k0=false"

    def test_without_the_flag_observations_are_ignored(self, cli, tmp_path):
        """Test that without --observed an observed node is not conditioned on, and no observed key is printed."""
        path = written(tmp_path, checked("observes", True))
        output = cli.json("query", path, "marginal", "c", "--draws", "0")
        assert_close(output["point"], PRIOR)
        assert "observed" not in output

    def test_conditional_with_observed(self, cli, tmp_path):
        """Test that a conditional query may take its evidence from --observed alone, and needs some."""
        path = written(tmp_path, checked("observes", False))
        output = cli.json("query", path, "conditional", "c", "--observed", "--draws", "0")
        assert_close(output["point"], posterior(PRIOR, False))
        error = cli.error(
            "query", written(tmp_path, checked("observes", None), "u.json"), "conditional", "c", "--observed"
        )
        assert error["code"] == "invalid-argument"
        assert "needs evidence" in error["message"]

    @pytest.mark.parametrize("value", ["yes", None, 1])
    def test_observed_not_a_boolean(self, cli, tmp_path, value):
        """Test that an attributes.observed that is not true or false is refused, naming the node."""
        graph = checked("observes")
        graph.add_node(Node("k0", attributes={"observed": value}))
        graph.add_relation(Relation("e0", "observes", "c", "k0", sensitivity=SENSITIVITY, false_pass=FALSE_PASS))
        error = cli.error("query", written(tmp_path, graph), "marginal", "c", "--observed")
        assert error["code"] == "invalid-argument"
        assert "attributes.observed of 'k0'" in error["message"]
        assert error["details"] == {"node": "k0"}

    def test_impossible_observation_is_named(self, cli, tmp_path):
        """Test that an observation the graph rules out is a zero-probability error naming --observed."""
        graph = Graph()
        graph.add_node(Node("c", base=0.0))
        graph.add_node(Node("k0", attributes={"observed": True}))
        graph.add_relation(Relation("e0", "observes", "c", "k0", sensitivity=1.0, false_pass=0.0))
        error = cli.error("query", written(tmp_path, graph), "marginal", "c", "--observed")
        assert error["code"] == "zero-probability"
        assert "--observed" in error["message"]
        assert "--observed k0=true" in error["details"]["drop_any_one_of"]


class TestDiagnoseObserved:
    """``diagnose --observed`` diagnoses under every node's ``attributes.observed``."""

    def test_findings_are_the_library_ones(self, cli, tmp_path):
        """Test that the findings are the library's under the observed evidence, and the evidence is echoed."""
        graph = checked("observes", True, None)
        path = written(tmp_path, graph)
        output = cli.json("diagnose", path, "--target", "c", "--observed")
        expected = diagnose_many(graph, ["c"], evidence={"k0": True}).to_dicts()
        assert output == {
            "command": "diagnose",
            "path": str(path),
            "targets": ["c"],
            "observed": {"k0": True},
            "findings": expected,
        }
        assert expected != diagnose_many(graph, ["c"]).to_dicts()

    def test_merges_with_given(self, cli, tmp_path):
        """Test that --observed and --given together diagnose under both."""
        graph = checked("observes", True, None)
        path = written(tmp_path, graph)
        output = cli.json("diagnose", path, "--target", "c", "--observed", "--given", "k1=false")
        assert output["given"] == {"k1": False}
        assert output["observed"] == {"k0": True}
        assert output["findings"] == diagnose_many(graph, ["c"], evidence={"k1": False, "k0": True}).to_dicts()

    def test_absent_observation_is_no_check(self, cli, tmp_path):
        """Test that with nothing observed the findings are the unconditioned ones."""
        graph = checked("observes", None)
        output = cli.json("diagnose", written(tmp_path, graph), "--target", "c", "--observed")
        assert output["observed"] == {}
        assert output["findings"] == diagnose_many(graph, ["c"]).to_dicts()

    def test_conflict_with_given(self, cli, tmp_path):
        """Test that --given contradicting an observation is an invalid-argument."""
        path = written(tmp_path, checked("observes", False))
        error = cli.error("diagnose", path, "--target", "c", "--observed", "--given", "k0=true")
        assert error["code"] == "invalid-argument"
        assert error["message"] == (
            "--given k0=true contradicts --observed k0=false, the value in its attributes.observed"
        )

    def test_impossible_observation_names_observed(self, cli, tmp_path):
        """Test that an observation the graph rules out is a zero-probability error naming --observed."""
        graph = Graph()
        graph.add_node(Node("c", base=0.0))
        graph.add_node(Node("k0", attributes={"observed": True}))
        graph.add_relation(Relation("e0", "observes", "c", "k0", sensitivity=1.0, false_pass=0.0))
        error = cli.error("diagnose", written(tmp_path, graph), "--target", "c", "--observed")
        assert error["code"] == "zero-probability"
        assert "the --observed values" in error["message"]
        assert error["details"] == {"given": {}, "observed": {"k0": True}}
