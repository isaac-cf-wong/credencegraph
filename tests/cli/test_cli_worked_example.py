"""The small graphs of the worked-example page, run through the command line.

Every number the page quotes is checked here against a closed form written independently of the engine.
"""

from __future__ import annotations

import math

import pytest


def assert_close(actual, expected):
    """Compare two probabilities to double-precision roundoff."""
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)


def build(cli, path, nodes, relations):
    """Write a graph file through the command line.

    Args:
        cli: The command-line driver.
        path: The file.
        nodes: ``(id, base)`` pairs.
        relations: ``(source, target, type, strength)`` tuples.
    """
    cli.json("init", path)
    for node, base in nodes:
        cli.json("add-node", path, "--id", node, "--base", base)
    for source, target, kind, strength in relations:
        cli.json("relate", path, source, target, "--type", kind, "--strength", strength)
    cli.json("check", path)


def point(cli, path, *query):
    """Return the point answer of a query.

    Args:
        cli: The command-line driver.
        path: The file.
        *query: The query arguments after the file.

    Returns:
        The point answer.
    """
    return cli.json("query", path, *query)["point"]


def logit(p):
    """Return the log-odds of a probability."""
    return math.log(p / (1 - p))


def entropy(p):
    """Return the entropy in bits of a binary variable that is true with probability ``p``."""
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


PRIOR, LIKE_H, LIKE_NOT_H = 0.3, 0.8, 0.2


class TestLikelihoodDirection:
    """Observations encoded as consequences of the hypothesis update it by Bayes' theorem."""

    @pytest.fixture
    def path(self, cli, tmp_path):
        """Write the graph with a hypothesis and two observations of it."""
        strength = 1 - (1 - LIKE_H) / (1 - LIKE_NOT_H)
        assert_close(strength, 0.75)
        path = tmp_path / "obs.json"
        build(
            cli,
            path,
            [("H", str(PRIOR)), ("o1", str(LIKE_NOT_H)), ("o2", str(LIKE_NOT_H))],
            [("H", "o1", "supports", str(strength)), ("H", "o2", "supports", str(strength))],
        )
        return path

    def test_likelihoods(self, cli, path):
        """Test that the encoding gives the intended P(obs | H) and P(obs | not H)."""
        assert_close(point(cli, path, "conditional", "o1", "--given", "H=true"), LIKE_H)
        assert_close(point(cli, path, "conditional", "o1", "--given", "H=false"), LIKE_NOT_H)

    def test_posteriors(self, cli, path):
        """Test the posteriors after one observation, two, and one for and one against."""
        one = PRIOR * LIKE_H / (PRIOR * LIKE_H + (1 - PRIOR) * LIKE_NOT_H)
        two = PRIOR * LIKE_H**2 / (PRIOR * LIKE_H**2 + (1 - PRIOR) * LIKE_NOT_H**2)
        assert_close(point(cli, path, "conditional", "H", "--given", "o1=true"), one)
        assert_close(point(cli, path, "conditional", "H", "--given", "o1=true", "--given", "o2=true"), two)
        assert_close(point(cli, path, "conditional", "H", "--given", "o1=true", "--given", "o2=false"), PRIOR)
        assert round(one, 4) == 0.6316
        assert round(two, 4) == 0.8727

    def test_diagnostics_see_the_prior(self, cli, path):
        """Test that the diagnostics describe H before observation, apart from the value of information."""
        findings = {finding["id"]: finding for finding in cli.json("diagnose", path, "--target", "H")["findings"]}
        for parameter in ("base:o1", "strength:H-supports-o1", "base:o2", "strength:H-supports-o2"):
            assert findings[f"sensitivity:{parameter}"]["value"] == 0
        p_obs = PRIOR * LIKE_H + (1 - PRIOR) * LIKE_NOT_H
        information = entropy(p_obs) - (PRIOR * entropy(LIKE_H) + (1 - PRIOR) * entropy(LIKE_NOT_H))
        assert findings["value-of-information:o1"]["value"] == pytest.approx(information, rel=1e-9, abs=0.0)
        assert round(information, 3) == 0.236


class TestVeto:
    """A refuter acts as a veto on a support of the same strength."""

    @pytest.fixture
    def path(self, cli, tmp_path):
        """Write the graph with one support and one refuter of the same proposition."""
        path = tmp_path / "veto.json"
        build(
            cli,
            path,
            [("h", "0.1"), ("e", "1"), ("d", "1")],
            [("e", "h", "supports", "0.9"), ("d", "h", "refutes", "0.9")],
        )
        return path

    def test_combination(self, cli, path):
        """Test each reason alone and both together against the noisy-OR and inhibition closed forms."""
        support_alone = 1 - (1 - 0.1) * (1 - 0.9)
        refuter_alone = 0.1 * (1 - 0.9)
        assert_close(point(cli, path, "intervene", "h", "--set", "d=false"), support_alone)
        assert_close(point(cli, path, "intervene", "h", "--set", "e=false"), refuter_alone)
        assert_close(point(cli, path, "marginal", "h"), support_alone * (1 - 0.9))
        assert round(support_alone * (1 - 0.9), 3) == 0.091

    def test_log_odds_reading(self):
        """Test the number the page quotes for adding the two reasons as weights of evidence."""
        combined = logit(0.1) + (logit(0.91) - logit(0.1)) + (logit(0.01) - logit(0.1))
        assert round(1 / (1 + math.exp(-combined)), 3) == 0.479
