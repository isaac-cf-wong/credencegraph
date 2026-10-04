"""A query whose ``--given`` and ``--set`` values have probability zero is explained by its real cause.

Every cause and remedy the error names is checked by running the same command, of the same kind, with
that one thing changed: a named flag's values must fail on their own, a value to drop must make the
command succeed when dropped, a base or strength to move must make it succeed when moved off 0 and 1,
and a named exclusive relation must make it succeed when removed. What the error does not name must not.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from credencegraph.cli import query as query_module
from credencegraph.core import Graph, Node, Point, Relation, dump, load


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


def unrelated_extremes(path: Path) -> Path:
    """A has a base of 0; D, which no query here uses, has a base of 0, and ``s`` a strength of 0."""
    return write(
        path,
        Node("A", base=0.0),
        Node("B", base=0.5),
        Node("D", base=0.0),
        relations=(Relation("s", "supports", "A", "B", strength=0.0),),
    )


def two_impossible_among_four(path: Path) -> Path:
    """A and C have a base of 0, and F, G and B of 0.5. No relation."""
    return write(
        path, Node("A", base=0.0), Node("C", base=0.0), Node("F", base=0.5), Node("G", base=0.5), Node("B", base=0.5)
    )


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


def certain_partner_of_set(path: Path) -> Path:
    """B and c are exclusive, and c is certain, so setting b true fails; A is unrelated."""
    return write(
        path,
        Node("A", base=0.5),
        Node("b", base=0.5),
        Node("c", base=1.0),
        relations=(Relation("x", "exclusive", "b", "c"),),
    )


def impossible_exclusive_partner(path: Path) -> Path:
    """A has a base of 0 and is exclusive with B: passing both true fails even without the relation."""
    return write(path, Node("A", base=0.0), Node("B", base=0.5), relations=(Relation("x", "exclusive", "A", "B"),))


def even_exclusive(path: Path) -> Path:
    """Two even-odds nodes joined by the exclusive relation ``x``."""
    return write(path, Node("a", base=0.5), Node("b", base=0.5), relations=(Relation("x", "exclusive", "a", "b"),))


def equivalent_pair(path: Path) -> Path:
    """``a`` and ``b`` are one proposition; there is no exclusive relation."""
    return write(path, Node("a", base=0.5), Node("b"), relations=(Relation("e", "equivalent", "a", "b"),))


def two_broken_exclusives(path: Path) -> Path:
    """``a`` and ``b`` are exclusive, and so are ``c`` and ``d``; every node has even odds."""
    return write(
        path,
        *(Node(n, base=0.5) for n in "abcdT"),
        relations=(Relation("x", "exclusive", "a", "b"), Relation("y", "exclusive", "c", "d")),
    )


def equivalent_impossible_bases(path: Path) -> Path:
    """``a`` and ``b`` are one proposition with a base of 0 each, so neither base can move alone."""
    return write(
        path,
        Node("a", base=0.0),
        Node("b", base=0.0),
        Node("T", base=0.5),
        relations=(Relation("e", "equivalent", "a", "b"),),
    )


def many_unrelated_extremes(path: Path) -> Path:
    """Twenty nodes ``u0``..``u19`` no query here uses have a base of 0, and so have ``g0``, ``g1`` and ``g2``, last."""
    nodes = [Node(f"u{i}", base=0.0) for i in range(20)] + [Node(f"g{i}", base=0.0) for i in range(3)]
    return write(path, *nodes, Node("T", base=0.5))


def combined_remedy_only(path: Path) -> Path:
    """``a`` is exclusive with ``b`` and with ``Y``, whose base is 1: no single drop, move or removal clears ``--set a``."""
    return write(
        path,
        Node("a", base=0.5),
        Node("b", base=0.5),
        Node("Y", base=1.0),
        Node("T", base=0.5),
        relations=(Relation("x", "exclusive", "a", "b"), Relation("z", "exclusive", "a", "Y")),
    )


def exclusive_bound_by_set(path: Path) -> Path:
    """``A`` makes ``C`` certain and ``D`` even, and ``C`` and ``D`` are exclusive; every base is 0 but ``T``'s.

    Without ``--set``, ``--given D=true`` fails on ``D``'s base alone; with ``--set A=true`` it fails only through ``x``.
    """
    return write(
        path,
        *(Node(n, base=0.0) for n in "ACD"),
        Node("T", base=0.5),
        relations=(
            Relation("r1", "supports", "A", "C", strength=1.0),
            Relation("r2", "supports", "A", "D", strength=0.5),
            Relation("x", "exclusive", "C", "D"),
        ),
    )


def query_args(kind: str, target: str, items: list[tuple[str, str]]) -> list[str]:
    """Build the arguments of a query from its kind, target and ``(flag, NODE=value)`` items."""
    args = [kind, target, "--draws", "0"]
    for flag, value in items:
        args += [flag, value]
    return args


GIVEN = "the --given values have probability zero, so the query has no answer"
SET = "the --set values have probability zero, so the query has no answer"
TOGETHER = "the --given and --set values have probability zero together, so the query has no answer"
EACH = "the --given values and the --set values each have probability zero, so the query has no answer"

# (graph, kind, target, items, message, what the hint must and must not say, the exclusive relation named)
CASES = {
    "base-of-zero-with-set": (
        impossible_base,
        "intervene",
        "B",
        [("--given", "A=true"), ("--set", "B=true")],
        GIVEN,
        {"must": ["drop --given A=true", "move base:A"], "must_not": ["--set", "exclusive"]},
        None,
    ),
    "base-of-zero": (
        impossible_base,
        "conditional",
        "B",
        [("--given", "A=true")],
        GIVEN,
        {"must": ["move base:A"], "must_not": ["exclusive", "drop"]},
        None,
    ),
    "unrelated-base-of-zero": (
        two_impossible_bases,
        "conditional",
        "B",
        [("--given", "A=true")],
        GIVEN,
        {"must": ["move base:A"], "must_not": ["base:C", "drop"]},
        None,
    ),
    "unrelated-base-and-strength-of-zero": (
        unrelated_extremes,
        "conditional",
        "B",
        [("--given", "A=true")],
        GIVEN,
        {"must": ["move base:A"], "must_not": ["base:D", "strength:s"]},
        None,
    ),
    "two-impossible-givens-with-set": (
        two_impossible_bases,
        "intervene",
        "B",
        [("--given", "A=true"), ("--given", "C=true"), ("--set", "B=true")],
        GIVEN,
        {
            "must": ["drop all of --given A=true, --given C=true", "move all of base:A, base:C"],
            "must_not": ["--set", "exclusive"],
        },
        None,
    ),
    "smallest-drop": (
        two_impossible_among_four,
        "conditional",
        "B",
        [("--given", "A=true"), ("--given", "C=true"), ("--given", "F=true"), ("--given", "G=true")],
        GIVEN,
        {"must": ["drop all of --given A=true, --given C=true"], "must_not": ["F=true", "G=true"]},
        None,
    ),
    "impossible-given-and-impossible-set": (
        impossible_on_both_sides,
        "intervene",
        "T",
        [("--given", "A=true"), ("--set", "b=true"), ("--set", "c=true")],
        EACH,
        {"must": ["drop all of --given A=true, --set b=true", "exclusive relations"], "must_not": ["'x'"]},
        None,
    ),
    "strength-of-one": (
        certain_requirement,
        "conditional",
        "A",
        [("--given", "A=false"), ("--given", "B=true")],
        GIVEN,
        {
            "must": ["drop any one of --given A=false, --given B=true", "strength:AB"],
            "must_not": ["exclusive", "--set"],
        },
        None,
    ),
    "exclusive-with-certain-partner": (
        certain_partner,
        "conditional",
        "B",
        [("--given", "A=true")],
        GIVEN,
        {"must": ["exclusive relations", "move base:B"], "must_not": ["--set", "drop", "'x'"]},
        None,
    ),
    "set-with-certain-partner": (
        certain_partner_of_set,
        "intervene",
        "A",
        [("--given", "A=true"), ("--set", "b=true")],
        SET,
        {"must": ["exclusive relations", "move base:c"], "must_not": ["--given", "drop"]},
        None,
    ),
    "exclusive-not-the-cause": (
        impossible_exclusive_partner,
        "conditional",
        "B",
        [("--given", "A=true"), ("--given", "B=true")],
        GIVEN,
        {"must": ["drop --given A=true"], "must_not": ["exclusive", "'x'", "B=true", "base:A"]},
        None,
    ),
    "exclusive-broken-by-given": (
        even_exclusive,
        "conditional",
        "a",
        [("--given", "a=true"), ("--given", "b=true")],
        GIVEN,
        {
            "must": ["'x'", "drop any one of --given a=true, --given b=true", "remove the exclusive relation 'x'"],
            "must_not": ["--set"],
        },
        "x",
    ),
    "exclusive-broken-by-given-and-set": (
        even_exclusive,
        "intervene",
        "a",
        [("--given", "a=true"), ("--set", "b=true")],
        TOGETHER,
        {"must": ["'x'", "drop --given a=true", "remove the exclusive relation 'x'"], "must_not": ["drop any"]},
        "x",
    ),
    "one-of-several-smallest-drops": (
        two_broken_exclusives,
        "conditional",
        "T",
        [("--given", "a=true"), ("--given", "b=true"), ("--given", "c=true"), ("--given", "d=true")],
        GIVEN,
        {"must": ["drop all of --given a=true, --given c=true", "exclusive relations"], "must_not": ["'x'", "'y'"]},
        None,
    ),
    "given-under-set-meets-exclusive": (
        exclusive_bound_by_set,
        "intervene",
        "T",
        [("--set", "A=true"), ("--given", "D=true")],
        GIVEN,
        {"must": ["exclusive relations", "drop --given D=true", "move strength:r1"], "must_not": ["drop any"]},
        None,
    ),
    "equivalent-bases-move-together": (
        equivalent_impossible_bases,
        "conditional",
        "T",
        [("--given", "a=true")],
        GIVEN,
        {"must": ["move all of base:a, base:b"], "must_not": ["drop", "exclusive"]},
        None,
    ),
}


def kind_for(items: list[tuple[str, str]]) -> str:
    """The kind of query that takes these items: ``intervene`` with a ``--set``, else ``conditional`` or ``marginal``."""
    flags = {flag for flag, _ in items}
    return "intervene" if "--set" in flags else "conditional" if "--given" in flags else "marginal"


def without(items: list[tuple[str, str]], dropped: list[str]) -> list[tuple[str, str]]:
    """Remove the items written ``FLAG NODE=value`` in ``dropped``."""
    return [item for item in items if f"{item[0]} {item[1]}" not in dropped]


def changed(path: Path, softened: set[str] = frozenset(), removed: set[str] = frozenset(), value: float = 0.4) -> Path:
    """Write a copy of the graph at ``path`` with some parameters moved to ``value`` and some relations removed."""
    graph, copy = load(path), Graph()
    for node in graph.nodes.values():
        copy.add_node(replace(node, base=Point(value)) if f"base:{node.id}" in softened else node)
    for relation in graph.relations.values():
        if relation.id not in removed:
            soft = f"strength:{relation.id}" in softened
            copy.add_relation(replace(relation, strength=Point(value)) if soft else relation)
    out = path.with_name(f"changed-{len(list(path.parent.iterdir()))}.json")
    dump(copy, out)
    return out


def extreme_ids(path: Path) -> list[str]:
    """Every base and strength of the graph at ``path`` that is exactly 0 or 1."""
    graph = load(path)
    ids = [f"base:{n.id}" for n in graph.nodes.values() if isinstance(n.base, Point) and n.base.p in {0.0, 1.0}]
    return ids + [
        f"strength:{r.id}"
        for r in graph.relations.values()
        if isinstance(r.strength, Point) and r.strength.p in {0.0, 1.0}
    ]


def succeeds(cli, path: Path, kind: str, target: str, items: list[tuple[str, str]]) -> bool:
    """Run a query and tell whether it succeeds."""
    return cli.run("query", path, *query_args(kind, target, items), "--json").exit_code == 0


@pytest.mark.parametrize("name", sorted(CASES))
def test_message_and_hint(cli, tmp_path, name):
    """Test that the error names what the case expects, and none of what it must not."""
    make, kind, target, items, message, words, exclusive = CASES[name]
    error = cli.error("query", make(tmp_path / "g.json"), *query_args(kind, target, items))
    assert error["code"] == "zero-probability"
    assert error["message"] == message
    for word in words["must"]:
        assert word in error["hint"], word
    for word in words["must_not"]:
        assert word not in error["hint"], word
    assert error["details"].get("exclusive_relation") == exclusive
    assert error["details"]["search_truncated"] == []
    assert all("stopped_at" not in tried for tried in error["details"]["search_tries"].values())
    assert "stopped before trying" not in error["hint"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_named_flags_are_the_cause(cli, tmp_path, name):
    """Test that a flag the message names fails on its own values, and a flag it does not name succeeds.

    When the message says the values fail together, each flag's values alone succeed.
    """
    make, kind, target, items, _, _, _ = CASES[name]
    path = make(tmp_path / "g.json")
    message = cli.error("query", path, *query_args(kind, target, items))["message"]
    for flag in sorted({flag for flag, _ in items}):
        alone = [item for item in items if item[0] == flag]
        named = "together" not in message and (message == EACH or f"the {flag} values" in message)
        assert succeeds(cli, path, kind_for(alone), target, alone) is not named, flag


@pytest.mark.parametrize("name", sorted(CASES))
def test_drop_remedy_on_the_same_command(cli, tmp_path, name):
    """Test the drops against the same command: a named drop succeeds, and no smaller or other drop does."""
    make, kind, target, items, _, _, _ = CASES[name]
    path = make(tmp_path / "g.json")
    details = cli.error("query", path, *query_args(kind, target, items))["details"]
    any_one, all_of = details["drop_any_one_of"], details["drop_all_of"]
    assert not (any_one and all_of)
    labels = [f"{flag} {value}" for flag, value in items]
    assert set(any_one) | set(all_of) <= set(labels)
    for label in labels:
        assert succeeds(cli, path, kind, target, without(items, [label])) is (label in any_one), label
    if all_of:
        assert succeeds(cli, path, kind, target, without(items, all_of))
        for label in all_of:
            assert not succeeds(cli, path, kind, target, without(items, [x for x in all_of if x != label])), label


@pytest.mark.parametrize("name", sorted(CASES))
def test_move_remedy_on_the_same_command(cli, tmp_path, name):
    """Test the parameters against the same command: moving a named one off 0 and 1 succeeds, an unnamed one not."""
    make, kind, target, items, _, _, _ = CASES[name]
    path = make(tmp_path / "g.json")
    details = cli.error("query", path, *query_args(kind, target, items))["details"]
    any_one, all_of = details["move_any_one_of"], details["move_all_of"]
    assert not (any_one and all_of)
    assert [p["id"] for p in details["extreme_parameters"]] == any_one + all_of
    for parameter in extreme_ids(path):
        moved = succeeds(cli, changed(path, softened={parameter}), kind, target, items)
        assert moved is (parameter in any_one), parameter
    if all_of:
        assert succeeds(cli, changed(path, softened=set(all_of)), kind, target, items)
        for parameter in all_of:
            assert not succeeds(cli, changed(path, softened=set(all_of) - {parameter}), kind, target, items)


@pytest.mark.parametrize("name", sorted(CASES))
def test_move_at_any_other_value(cli, tmp_path, name):
    """Test that 0.5 stands for every value: a parameter not named does not clear the zero at any other value."""
    make, kind, target, items, _, _, _ = CASES[name]
    path = make(tmp_path / "g.json")
    details = cli.error("query", path, *query_args(kind, target, items))["details"]
    graph = load(path)
    for parameter in extreme_ids(path):
        kind_of, key = parameter.split(":")
        current = (graph.nodes[key].base if kind_of == "base" else graph.relations[key].strength).p
        for value in (1e-6, 0.999999, 1.0 - current):
            moved = succeeds(cli, changed(path, softened={parameter}, value=value), kind, target, items)
            assert moved is (parameter in details["move_any_one_of"]), (parameter, value)


@pytest.mark.parametrize("name", sorted(CASES))
def test_exclusive_remedy_on_the_same_command(cli, tmp_path, name):
    """Test that removing the named exclusive relation makes the same command succeed."""
    make, kind, target, items, _, _, exclusive = CASES[name]
    path = make(tmp_path / "g.json")
    cli.error("query", path, *query_args(kind, target, items))
    if exclusive is not None:
        assert succeeds(cli, changed(path, removed={exclusive}), kind, target, items)


def test_exclusive_relation_that_is_not_the_cause(cli, tmp_path):
    """Test the relation that the passed values break, but whose removal leaves the query at zero."""
    path = impossible_exclusive_partner(tmp_path / "g.json")
    items = [("--given", "A=true"), ("--given", "B=true")]
    assert not succeeds(cli, changed(path, removed={"x"}), "conditional", "B", items)
    assert "exclusive_relation" not in cli.error("query", path, *query_args("conditional", "B", items))["details"]


def test_exclusive_bound_by_set_is_the_cause(cli, tmp_path):
    """Test that the context of ``--given`` values that fail under ``--set`` is judged under that ``--set``.

    The ``--given`` value fails without ``--set`` on a base alone, but the command fails only through
    the exclusive relation: without it, the same command succeeds.
    """
    path = exclusive_bound_by_set(tmp_path / "g.json")
    items = [("--set", "A=true"), ("--given", "D=true")]
    hint = cli.error("query", path, *query_args("intervene", "T", items))["hint"]
    assert "the --given values have probability zero under the graph's credences and exclusive relations" in hint
    result = cli.run("query", changed(path, removed={"x"}), *query_args("intervene", "T", items), "--json")
    assert result.exit_code == 0
    assert '"point": 0.5' in result.output


def count_moves(monkeypatch) -> list[int]:
    """Count the graphs the query command rebuilds with parameters moved, one entry per move tried."""
    moves, rebuilt = [], query_module._rebuilt

    def counting(graph, removed=frozenset(), softened=frozenset()):
        if softened:
            moves.append(len(softened))
        return rebuilt(graph, removed=removed, softened=softened)

    monkeypatch.setattr(query_module, "_rebuilt", counting)
    return moves


def test_no_remedy_found(cli, tmp_path, monkeypatch):
    """Test a search whose budget leaves no set tried: the error offers nothing and says no set was tried."""
    monkeypatch.setattr(query_module, "SEARCH_BUDGET", 0)
    moves = count_moves(monkeypatch)
    path = two_impossible_bases(tmp_path / "g.json")
    items = [("--given", "A=true"), ("--given", "C=true")]
    error = cli.error("query", path, *query_args("conditional", "B", items))
    assert "that was tried makes this command succeed" in error["hint"]
    assert error["hint"].endswith(
        "the drop search tried all 2 single drops, but no set of several values, and stopped before trying every "
        "set of 2; the move search tried all 2 single moves, but no set of several parameters, and stopped before "
        "trying every set of 2, so a set not tried may make it succeed (details.search_truncated)"
    )
    details = error["details"]
    assert details["drop_any_one_of"] == details["drop_all_of"] == []
    assert details["move_any_one_of"] == details["move_all_of"] == details["extreme_parameters"] == []
    assert details["search_truncated"] == ["drop", "move"]
    stopped = {"single": 2, "several": 0, "stopped_at": 2}
    assert details["search_tries"] == {"drop": stopped, "move": stopped}
    assert moves == [1, 1]
    assert succeeds(cli, changed(path, softened={"base:A", "base:C"}), "conditional", "B", items)


def test_truncated_search_is_told_apart_from_no_remedy(cli, tmp_path, monkeypatch):
    """Test a move remedy past the budget: the error says what the search tried, and the remedy does succeed.

    The 23 parameters at 0 give 23 single moves and 253 pairs, so 256 sets end 3 sets into the triples,
    long before the last triple, the one that clears the zero.
    """
    moves = count_moves(monkeypatch)
    path = many_unrelated_extremes(tmp_path / "g.json")
    items = [("--given", "g0=true"), ("--given", "g1=true"), ("--given", "g2=true")]
    error = cli.error("query", path, *query_args("conditional", "T", items))
    details = error["details"]
    assert details["search_truncated"] == ["move"]
    assert details["search_tries"] == {
        "drop": {"single": 3, "several": 4},
        "move": {"single": 23, "several": 256, "stopped_at": 3},
    }
    assert details["move_all_of"] == []
    assert sorted(set(moves)) == [1, 2, 3]
    assert [moves.count(size) for size in (1, 2, 3)] == [23, 253, 3]
    assert (
        "the move search tried all 23 single moves, then 256 sets of several parameters, and stopped before trying "
        "every set of 3, so a set not tried may make it succeed (details.search_truncated)"
    ) in error["hint"]
    assert succeeds(cli, changed(path, softened={"base:g0", "base:g1", "base:g2"}), "conditional", "T", items)


def test_single_changes_do_not_spend_the_budget(cli, tmp_path, monkeypatch):
    """Test more candidates than the budget: every single change is tried, and the budget still buys a set."""
    monkeypatch.setattr(query_module, "SEARCH_BUDGET", 1)
    path = two_impossible_among_four(tmp_path / "g.json")
    items = [("--given", "A=true"), ("--given", "C=true"), ("--given", "F=true")]
    details = cli.error("query", path, *query_args("conditional", "B", items))["details"]
    assert details["drop_all_of"] == ["--given A=true", "--given C=true"]
    assert details["move_all_of"] == ["base:A", "base:C"]
    assert details["search_tries"] == {"drop": {"single": 3, "several": 1}, "move": {"single": 2, "several": 1}}
    assert details["search_truncated"] == []


def test_no_remedy_says_combined_changes_were_not_tried(cli, tmp_path):
    """Test a query that only a drop together with a move clears: the hint says combinations were not tried."""
    path = combined_remedy_only(tmp_path / "g.json")
    items = [("--set", "a=true"), ("--given", "b=true")]
    error = cli.error("query", path, *query_args("intervene", "T", items))
    details = error["details"]
    assert details["search_truncated"] == []
    assert details["drop_any_one_of"] == details["drop_all_of"] == details["move_all_of"] == []
    assert "exclusive_relation" not in details
    assert (
        "a drop combined with a move or a removal, and the removal of several exclusive relations, were not tried"
        in (error["hint"])
    )
    assert not succeeds(cli, changed(path, softened={"base:Y"}), "intervene", "T", items)
    assert succeeds(cli, changed(path, softened={"base:Y"}), "intervene", "T", items[:1])


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
