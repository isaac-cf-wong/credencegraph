"""A query whose ``--given`` and ``--set`` values have probability zero is explained by its real cause.

Every hint is checked against the graph by running the remedy it names: each value the error says to
drop is dropped, and the query must then succeed, while dropping any value it does not name must not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from credencegraph.core import Graph, Node, Relation, dump


def write(path: Path, *nodes: Node, relations: tuple[Relation, ...] = ()) -> Path:
    """Write a graph of the given nodes and relations to ``path``."""
    graph = Graph()
    for node in nodes:
        graph.add_node(node)
    for relation in relations:
        graph.add_relation(relation)
    dump(graph, path)
    return path


def impossible_base(path: Path) -> Path:
    """A has a base of 0 and B of 0.5, with no relation at all."""
    return write(path, Node("A", base=0.0), Node("B", base=0.5))


def two_impossible_bases(path: Path) -> Path:
    """A and C both have a base of 0; B has 0.5. No relation."""
    return write(path, Node("A", base=0.0), Node("B", base=0.5), Node("C", base=0.0))


def impossible_on_both_sides(path: Path) -> Path:
    """A has a base of 0, and b and c are exclusive: ``--given A=true`` and ``--set b,c=true`` each fail alone."""
    return write(
        path,
        Node("A", base=0.0),
        Node("b", base=0.5),
        Node("c", base=0.5),
        Node("T", base=0.5),
        relations=(Relation("x", "exclusive", "b", "c"),),
    )


def certain_requirement(path: Path) -> Path:
    """B requires A with strength 1, so B cannot hold without A; there is no exclusive relation."""
    return write(
        path, Node("A", base=0.5), Node("B", base=0.5), relations=(Relation("AB", "requires", "A", "B", strength=1.0),)
    )


def certain_partner(path: Path) -> Path:
    """A and B are exclusive, and B is certain, so A cannot hold."""
    return write(path, Node("A", base=0.5), Node("B", base=1.0), relations=(Relation("x", "exclusive", "A", "B"),))


def even_exclusive(path: Path) -> Path:
    """Two even-odds nodes joined by the exclusive relation ``x``."""
    return write(path, Node("a", base=0.5), Node("b", base=0.5), relations=(Relation("x", "exclusive", "a", "b"),))


def equivalent_pair(path: Path) -> Path:
    """``a`` and ``b`` are one proposition; there is no exclusive relation."""
    return write(path, Node("a", base=0.5), Node("b"), relations=(Relation("e", "equivalent", "a", "b"),))


def query_args(kind: str, target: str, items: list[tuple[str, str]]) -> list[str]:
    """Build the arguments of a query from its kind, target and ``(flag, NODE=value)`` items."""
    args = [kind, target, "--draws", "0"]
    for flag, value in items:
        args += [flag, value]
    return args


# (graph, kind, target, items, what the hint must and must not say, the parameters at 0 or 1)
CASES = {
    "base-of-zero-with-set": (
        impossible_base,
        "intervene",
        "B",
        [("--given", "A=true"), ("--set", "B=true")],
        {"must": ["--given A=true"], "must_not": ["--set", "exclusive"]},
        [{"id": "base:A", "value": 0.0}],
    ),
    "base-of-zero": (
        impossible_base,
        "conditional",
        "B",
        [("--given", "A=true")],
        {"must": ["--given A=true"], "must_not": ["exclusive"]},
        [{"id": "base:A", "value": 0.0}],
    ),
    "two-impossible-givens-with-set": (
        two_impossible_bases,
        "intervene",
        "B",
        [("--given", "A=true"), ("--given", "C=true"), ("--set", "B=true")],
        {"must": ["drop all of --given A=true, --given C=true"], "must_not": ["--set", "exclusive"]},
        [{"id": "base:A", "value": 0.0}, {"id": "base:C", "value": 0.0}],
    ),
    "impossible-given-and-impossible-set": (
        impossible_on_both_sides,
        "intervene",
        "T",
        [("--given", "A=true"), ("--set", "b=true"), ("--set", "c=true")],
        {"must": ["'x'", "drop all of --given A=true, --set b=true, --set c=true"], "must_not": []},
        [{"id": "base:A", "value": 0.0}],
    ),
    "strength-of-one": (
        certain_requirement,
        "conditional",
        "A",
        [("--given", "A=false"), ("--given", "B=true")],
        {"must": ["--given A=false", "--given B=true"], "must_not": ["exclusive", "--set"]},
        [{"id": "strength:AB", "value": 1.0}],
    ),
    "exclusive-with-certain-partner": (
        certain_partner,
        "conditional",
        "B",
        [("--given", "A=true")],
        {"must": ["--given A=true", "exclusive"], "must_not": ["--set"]},
        [{"id": "base:B", "value": 1.0}],
    ),
    "exclusive-broken-by-given": (
        even_exclusive,
        "conditional",
        "a",
        [("--given", "a=true"), ("--given", "b=true")],
        {"must": ["'x'", "--given a=true", "--given b=true"], "must_not": ["--set"]},
        [],
    ),
    "exclusive-broken-by-given-and-set": (
        even_exclusive,
        "intervene",
        "a",
        [("--given", "a=true"), ("--set", "b=true")],
        {"must": ["'x'", "--given a=true", "--set b=true"], "must_not": []},
        [],
    ),
}


def kind_for(items: list[tuple[str, str]]) -> str:
    """The kind of query that takes these items: ``intervene`` with a ``--set``, else ``conditional`` or ``marginal``."""
    flags = {flag for flag, _ in items}
    return "intervene" if "--set" in flags else "conditional" if "--given" in flags else "marginal"


def without(items: list[tuple[str, str]], dropped: list[str]) -> list[tuple[str, str]]:
    """Remove the items written ``FLAG NODE=value`` in ``dropped``."""
    return [item for item in items if f"{item[0]} {item[1]}" not in dropped]


@pytest.mark.parametrize("name", sorted(CASES))
def test_hint_names_the_cause(cli, tmp_path, name):
    """Test that the hint names what the error says, and nothing the graph does not bear out."""
    make, kind, target, items, words, extreme = CASES[name]
    error = cli.error("query", make(tmp_path / "g.json"), *query_args(kind, target, items))
    assert error["code"] == "zero-probability"
    for word in words["must"]:
        assert word in error["hint"], word
    for word in words["must_not"]:
        assert word not in error["hint"], word
    if "exclusive" in words["must_not"]:
        assert "exclusive" not in error["message"]
    assert error["details"]["extreme_parameters"] == extreme


@pytest.mark.parametrize("name", sorted(CASES))
def test_remedy_is_true(cli, tmp_path, name):
    """Test the remedy against the graph: dropping a named value succeeds, dropping an unnamed one does not."""
    make, kind, target, items, _, _ = CASES[name]
    path = make(tmp_path / "g.json")
    details = cli.error("query", path, *query_args(kind, target, items))["details"]
    any_one, all_of = details["drop_any_one_of"], details["drop_all_of"]
    assert bool(any_one) != bool(all_of)
    labels = [f"{flag} {value}" for flag, value in items]
    assert set(any_one) | set(all_of) <= set(labels)
    for label in labels:
        remaining = without(items, [label])
        result = cli.run("query", path, *query_args(kind_for(remaining), target, remaining), "--json")
        assert (result.exit_code == 0) is (label in any_one), (label, result.stdout)
    if all_of:
        remaining = without(items, all_of)
        assert cli.run("query", path, *query_args(kind_for(remaining), target, remaining), "--json").exit_code == 0


@pytest.mark.parametrize(
    ("make", "args", "code"),
    [
        (impossible_base, ["intervene", "B", "--given", "A=true", "--set", "B=true"], "zero-probability"),
        (equivalent_pair, ["conditional", "a", "--given", "a=true", "--given", "b=false"], "invalid-argument"),
        (certain_requirement, ["conditional", "A", "--given", "A=false", "--given", "B=true"], "zero-probability"),
    ],
)
def test_reported_queries_on_graphs_without_exclusive_relations(cli, tmp_path, make, args, code):
    """Test the three reported queries: none of these graphs has an exclusive relation, so none is blamed."""
    error = cli.error("query", make(tmp_path / "g.json"), *args, "--draws", "0")
    assert error["code"] == code
    assert "exclusive" not in error["hint"]
    assert "exclusive" not in error["message"]


class TestContradictions:
    """Values that contradict each other outright are refused before any inference."""

    def test_equivalent_nodes_given_apart(self, cli, tmp_path):
        """Test the ``--given`` counterpart of setting two equivalent nodes to different values."""
        error = cli.error(
            "query", equivalent_pair(tmp_path / "g.json"), "conditional", "a", "--given", "a=true", "--given", "b=false"
        )
        assert error["code"] == "invalid-argument"
        assert error["message"] == "--given gives the equivalent nodes 'a' and 'b' different values"
        assert "--given" in error["hint"]

    def test_equivalent_nodes_set_apart(self, cli, tmp_path):
        """Test that the ``--set`` check keeps its wording."""
        error = cli.error(
            "query", equivalent_pair(tmp_path / "g.json"), "intervene", "a", "--set", "a=true", "--set", "b=false"
        )
        assert error["code"] == "invalid-argument"
        assert error["message"] == "--set gives the equivalent nodes 'a' and 'b' different values"

    @pytest.mark.parametrize(
        ("given", "message"),
        [
            ("a=false", "--given a=false contradicts --set a=true"),
            ("b=false", "--given b=false contradicts --set a=true, and 'a' and 'b' are equivalent"),
        ],
    )
    def test_given_contradicts_set(self, cli, tmp_path, given, message):
        """Test that evidence against a value fixed by intervention is refused and names both flags."""
        path = equivalent_pair(tmp_path / "g.json")
        error = cli.error("query", path, "intervene", "a", "--set", "a=true", "--given", given)
        assert error["code"] == "invalid-argument"
        assert error["message"] == message
        assert "--set" in error["hint"]
        assert "--given" in error["hint"]

    def test_given_agreeing_with_set(self, cli, tmp_path):
        """Test that evidence that agrees with an intervention is accepted: the node holds that value."""
        output = cli.json(
            "query",
            equivalent_pair(tmp_path / "g.json"),
            "intervene",
            "a",
            "--set",
            "a=true",
            "--given",
            "b=true",
            "--draws",
            "0",
        )
        assert output["point"] == 1.0
