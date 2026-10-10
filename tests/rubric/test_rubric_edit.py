"""Splitting a chunk into a compound and annotating a node, as library calls."""

from __future__ import annotations

import pytest
from _ingested import RESULT_TEXT, RUBRIC, build, chunk, violations, with_attributes

from credencegraph.core import Beta, CycleError, Node, Point, Relation, SourceAnchor, ValidationError
from credencegraph.rubric import annotate_node, loads_rubric, parse_value, split_node, text_digest

SENTENCE = "We show that the bias vanishes for long segments, which implies that the estimator is consistent."
LOCATOR = "file=sec/results.tex;lines=12-13"


def ingested():
    """One chunk, the documentation's example sentence, and a reference after it."""
    sentence = chunk("s-012", SENTENCE, sources=_anchor(SENTENCE, LOCATOR))
    reference = chunk("r-1", "A. Author, Journal 1 (2020).", "reference")
    return build(sentence, reference)


def _anchor(text, locator):
    """A verbatim anchor of ``text`` at ``locator``."""
    return (SourceAnchor("paper", locator, text, text_digest(text)),)


def test_split_into_spans_matches_the_documented_example():
    """Test that splitting gives the children, locators and order of the documentation's example."""
    graph, children = split_node(
        ingested(), "s-012", ["We show that the bias vanishes for long segments", "the estimator is consistent"]
    )
    assert children == ["s-012a", "s-012b"]
    assert list(graph.nodes) == ["s-012", "s-012a", "s-012b", "r-1"]
    assert graph.nodes["s-012"].kind == "compound"
    assert graph.nodes["s-012"].statement == SENTENCE
    first, second = graph.nodes["s-012a"], graph.nodes["s-012b"]
    assert first.sources[0].locator == f"{LOCATOR};chars=0-48"
    assert second.sources[0].locator == f"{LOCATOR};chars=69-96"
    assert second.sources[0].digest == text_digest("the estimator is consistent")
    assert (first.kind, dict(first.attributes)) == ("unassigned", {"origin": "document", "unit": "sentence"})
    assert {(r.source, r.type, r.target) for r in graph.relations.values()} == {
        ("s-012a", "part-of", "s-012"),
        ("s-012b", "part-of", "s-012"),
    }
    assert violations(graph, "anchored") == []
    assert ingested().nodes["s-012"].kind == "unassigned"


def test_split_again_adds_children():
    """Test that splitting a compound adds children after the existing ones, with the next letters."""
    graph, _ = split_node(ingested(), "s-012", ["We show", "the estimator"])
    graph, children = split_node(graph, "s-012", ["long segments"])
    assert children == ["s-012c"]
    assert list(graph.nodes) == ["s-012", "s-012a", "s-012b", "s-012c", "r-1"]


def test_split_skips_taken_ids():
    """Test that a child id already used by another node is skipped."""
    graph = build(*ingested(), chunk("s-012a", "Unrelated."))
    _, children = split_node(graph, "s-012", ["We show", "the estimator"])
    assert children == ["s-012b", "s-012c"]


def test_split_fields():
    """Test that a field child has no text, a copy of its parent's anchor, and is invalid until it has a form."""
    graph, _ = split_node(ingested(), "s-012", ["We show"])
    graph, (child,) = split_node(graph, "s-012", fields=True)
    field = graph.nodes[child]
    assert field.statement is None
    assert field.sources == graph.nodes["s-012"].sources
    assert violations(graph, "anchored") == [f"field-child-invalid:{child}"]
    graph = annotate_node(graph, RUBRIC, child, node_type="result", settings={"quantity": "the bias"})
    assert violations(graph, "anchored") == []


@pytest.mark.parametrize(
    ("clauses", "message"),
    [
        (["the estimator is unbiased"], "occurs 0 times"),
        (["th"], "occurs 4 times"),
        ([""], "occurs 0 times"),
    ],
)
def test_split_clause_must_occur_exactly_once(clauses, message):
    """Test that a clause must occur exactly once in the text, overlapping occurrences counted."""
    with pytest.raises(ValidationError, match=message):
        split_node(ingested(), "s-012", ["We show", *clauses])


def test_split_counts_overlapping_occurrences():
    """Test that a clause occurring twice with an overlap is refused."""
    graph = build(chunk("s-1", "aaa then b", sources=_anchor("aaa then b", "l")))
    with pytest.raises(ValidationError, match="occurs 2 times"):
        split_node(graph, "s-1", ["aa"])


@pytest.mark.parametrize(
    ("node", "message"),
    [
        (Node("a-1", "assumption", "Implicit.", base=0.5, attributes={"origin": "analyst"}), "not a document node"),
        (chunk("s-1", RESULT_TEXT, base=0.5), "carries base"),
        (with_attributes(chunk("s-1", RESULT_TEXT), form={}), "carries form"),
        (with_attributes(chunk("s-1", RESULT_TEXT), assessment={"verdict": "holds"}), "carries assessment"),
        (chunk("s-1", RESULT_TEXT, sources=()), "exactly one source with a locator"),
    ],
)
def test_split_refuses_nodes_a_compound_cannot_be(node, message):
    """Test that a node carrying content, or not a document chunk, cannot be split."""
    with pytest.raises(ValidationError, match=message):
        split_node(build(node), node.id, ["We"])


def test_split_refuses_a_node_in_a_relation():
    """Test that a node that rests on something, or is a child, cannot be split."""
    premise = Node("a-1", "assumption", "Implicit.", base=0.5, attributes={"origin": "analyst"})
    leaning = Relation("lean", "supports", "a-1", "s-012", strength=0.5)
    with pytest.raises(ValidationError, match="relations a compound may not have"):
        split_node(build(*ingested(), premise, leaning), "s-012", ["We show"])
    graph, _ = split_node(ingested(), "s-012", ["We show", "the estimator"])
    with pytest.raises(ValidationError, match="is a child of a compound"):
        split_node(graph, "s-012a", ["We"])


def test_split_needs_clauses_or_fields():
    """Test that split takes clauses or a field child, not both and not neither."""
    for clauses, fields in (([], False), (["We"], True)):
        with pytest.raises(ValidationError, match="either clauses or fields"):
            split_node(ingested(), "s-012", clauses, fields=fields)


# --- annotate ------------------------------------------------------------------------------------


def test_annotate_types_fills_links_and_assesses():
    """Test the documented sequence: type with base, form, rests-on with the rule's strength, verdict."""
    graph, _ = split_node(
        ingested(), "s-012", ["We show that the bias vanishes for long segments", "the estimator is consistent"]
    )
    graph = annotate_node(
        graph, RUBRIC, "s-012a", node_type="result", settings={"quantity": "the bias", "trend": "vanishes"}
    )
    graph = annotate_node(
        graph,
        RUBRIC,
        "s-012b",
        node_type="claim",
        settings={"shape": "universal", "statement": "the estimator is consistent"},
        rests_on=["s-012a"],
    )
    result, claim = graph.nodes["s-012a"], graph.nodes["s-012b"]
    assert (result.kind, result.base, dict(result.attributes["form"])) == (
        "result",
        Beta(8, 2),
        {"quantity": "the bias", "trend": "vanishes"},
    )
    assert (claim.kind, claim.base) == ("claim", Beta(5, 5))
    relation = graph.relations["s-012a-requires-s-012b"]
    assert (relation.type, relation.source, relation.target, relation.strength) == (
        "requires",
        "s-012a",
        "s-012b",
        Beta(9, 1),
    )
    assert violations(graph, "typed") == []
    graph = annotate_node(graph, RUBRIC, "s-012b", reason="outside the scope of this review")
    assert dict(graph.nodes["s-012b"].attributes["assessment"]) == {
        "verdict": "not_assessed",
        "reason": "outside the scope of this review",
    }
    graph = annotate_node(graph, RUBRIC, "s-012a", verdict="undetermined")
    assert dict(graph.nodes["s-012a"].attributes["assessment"]) == {"verdict": "undetermined"}


def test_annotate_keeps_an_existing_base():
    """Test that typing a node that has a base keeps it."""
    graph = build(chunk("s-1", RESULT_TEXT, base=0.3))
    assert annotate_node(graph, RUBRIC, "s-1", node_type="result").nodes["s-1"].base == Point(0.3)


def test_annotate_unassigned_removes_type_form_assessment_and_base():
    """Test that --type unassigned undoes typing, so the node can be split."""
    node = with_attributes(
        chunk("s-1", RESULT_TEXT, "result", base=0.5), form={"quantity": "the bias"}, assessment={"verdict": "holds"}
    )
    cleared = annotate_node(build(node), RUBRIC, "s-1", node_type="unassigned").nodes["s-1"]
    assert (cleared.kind, cleared.base, dict(cleared.attributes)) == (
        "unassigned",
        None,
        {"origin": "document", "unit": "sentence"},
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"node_type": "finding"}, "not declared by rubric"),
        ({"node_type": "check"}, "allows origins \\['evidence'\\]"),
        ({"settings": {"quantity": "the variance"}}, "not a substring of the text"),
        ({"settings": {"trend": "oscillates"}}, "not a valid enum"),
        ({"settings": {"colour": "red"}}, "not a field of type 'result'"),
        ({"settings": {"value": {"value": 0.2, "unit": "Hz"}}}, "no numeral in the text equals 0.2"),
        ({"rests_on": ["s-0"]}, "has no edge rule"),
        ({"verdict": "plausible"}, "verdict must be"),
        ({"reason": " "}, "must not be empty"),
        ({"verdict": "holds", "reason": "x"}, "not both"),
    ],
)
def test_annotate_validates_each_write(changes, message):
    """Test that each kind of invalid write is refused, naming the problem."""
    graph = build(chunk("s-0", "Earlier."), chunk("s-1", RESULT_TEXT, "result"))
    with pytest.raises(ValidationError, match=message):
        annotate_node(graph, RUBRIC, "s-1", **changes)


def test_annotate_form_needs_a_declared_type_and_a_document_node():
    """Test that a form is filled only on a typed document node."""
    with pytest.raises(ValidationError, match="give it a declared type"):
        annotate_node(build(chunk("s-1", RESULT_TEXT)), RUBRIC, "s-1", settings={"quantity": "the bias"})
    premise = Node("a-1", "assumption", "Implicit.", attributes={"origin": "analyst"})
    with pytest.raises(ValidationError, match="only document nodes carry a form"):
        annotate_node(build(premise), RUBRIC, "a-1", settings={"x": 1})
    with pytest.raises(ValidationError, match="only document nodes may be unassigned"):
        annotate_node(build(premise), RUBRIC, "a-1", node_type="unassigned")


def test_annotate_rests_on_checks_the_rule():
    """Test that the supporting node must have a type the rule names, and the relation must be new."""
    reference = chunk("s-0", "A. Author, Journal 1 (2020).", "reference")
    claim = chunk("s-2", "Our filter works.", "claim")
    with pytest.raises(ValidationError, match="rests on \\["):
        annotate_node(build(reference, claim), RUBRIC, "s-2", rests_on=["s-0"])
    method = chunk("s-0", "We tested it.", "method")
    graph = annotate_node(build(method, claim), RUBRIC, "s-2", rests_on=["s-0", "s-0"])
    assert list(graph.relations) == ["s-0-requires-s-2"]
    with pytest.raises(ValidationError, match="already exists"):
        annotate_node(graph, RUBRIC, "s-2", rests_on=["s-0"])


def test_annotate_rests_on_refuses_a_cycle():
    """Test that a rests-on relation that would close a cycle is refused, and nothing is changed."""
    rubric = loads_rubric(
        '[rubric]\nformat = 1\nname = "steps"\nversion = "1"\n'
        '[types.step]\nrests_on = { types = ["step"], relation = "requires", strength = 0.9 }\n'
    )
    graph = build(chunk("s-1", "First.", "step"), chunk("s-2", "Second.", "step"))
    graph = annotate_node(graph, rubric, "s-2", rests_on=["s-1"])
    with pytest.raises(CycleError):
        annotate_node(graph, rubric, "s-1", rests_on=["s-2"])


def test_annotate_refuses_compounds_and_invalid_origins():
    """Test that a compound is annotated through its children, and a node without an origin not at all."""
    graph, _ = split_node(ingested(), "s-012", ["We show", "the estimator"])
    with pytest.raises(ValidationError, match="is a compound"):
        annotate_node(graph, RUBRIC, "s-012", node_type="result")
    with pytest.raises(ValidationError, match=r"no valid attributes\.origin"):
        annotate_node(build(Node("x")), RUBRIC, "x", node_type="result")


def test_annotate_leaves_the_input_unchanged():
    """Test that annotate returns a new graph."""
    graph = build(chunk("s-1", RESULT_TEXT))
    annotate_node(graph, RUBRIC, "s-1", node_type="result")
    assert graph.nodes["s-1"].kind == "unassigned"


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("0.05", 0.05),
        ("universal", "universal"),
        ('{"value": 1, "unit": "s"}', {"value": 1, "unit": "s"}),
        ('"0.05"', "0.05"),
    ],
)
def test_parse_value(text, value):
    """Test that a command-line value is JSON when it parses and a string otherwise."""
    assert parse_value(text) == value
