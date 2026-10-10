"""The ``anchored``, ``typed`` and ``assessed`` checks, one rule at a time, on synthetic graphs.

Each test starts from a graph that passes its level and breaks exactly one rule, so it fails if that
rule's check is removed.
"""

from __future__ import annotations

import dataclasses

import pytest
from _ingested import (
    CLAIM_TEXT,
    RESULT_TEXT,
    RUBRIC,
    assessed_graph,
    build,
    chunk,
    typed_graph,
    violations,
    with_attributes,
)

from credencegraph.core import Beta, Node, Relation, SourceAnchor
from credencegraph.rubric import VIOLATION_CODES, check_graph, text_digest


def test_baselines_pass():
    """Test that the graphs the other tests break pass their levels to begin with."""
    assert violations(build(*typed_graph()), "typed") == []
    assert violations(build(*assessed_graph()), "assessed") == []


def test_unknown_level_is_refused():
    """Test that only the three levels are accepted."""
    with pytest.raises(ValueError, match="level must be one of"):
        check_graph(build(), RUBRIC, "strict")


# --- the dummy cases named for this feature -------------------------------------------------------


def test_claim_without_rests_on_fails_at_typed():
    """Test that a claim with no relation to what it rests on fails ``typed``."""
    result, claim, _ = typed_graph()
    assert violations(build(result, claim), "typed") == ["rests-on-missing:s-2"]


def test_rests_on_counts_only_the_rule_relation_from_the_rule_types():
    """Test that a supports relation, or a requires relation from a type outside the rule, does not count."""
    result, claim, _ = typed_graph()
    wrong_relation = Relation("r", "supports", "s-1", "s-2", strength=0.5)
    assert violations(build(result, claim, relations=(wrong_relation,))) == ["rests-on-missing:s-2"]
    reference = chunk("s-0", "A. Author, Journal 1 (2020).", "reference")
    wrong_type = Relation("r", "requires", "s-0", "s-2", strength=0.5)
    assert violations(build(reference, claim, relations=(wrong_type,))) == ["rests-on-missing:s-2"]


def test_number_absent_from_text_fails():
    """Test that a form number the text does not contain fails: "below 5%" holds 5 and 0.05, not 0.04."""
    result, claim, rests = typed_graph()
    form = {**claim.attributes["form"], "bound": 0.04}
    bad = with_attributes(claim, form=form)
    assert violations(build(result, bad, rests)) == ["form-not-in-text:s-2"]


def test_child_with_free_text_fails():
    """Test that a child whose text is not a span of its parent fails, as a span or as a field child."""
    parent, first, second, *relations = _split()
    rewritten = dataclasses.replace(first, statement="the bias is small")
    rewritten = dataclasses.replace(
        rewritten,
        sources=(
            dataclasses.replace(first.sources[0], quote="the bias is small", digest=text_digest("the bias is small")),
        ),
    )
    assert violations(build(parent, rewritten, second, relations=tuple(relations)), "anchored") == [
        "span-not-in-parent:s-5a"
    ]
    field_child = Node(
        "s-5a",
        "result",
        "the bias is small",
        parent.sources,
        attributes={"origin": "document", "form": {"quantity": "the bias"}},
    )
    assert violations(build(parent, field_child, second, relations=tuple(relations)), "anchored") == [
        "field-child-invalid:s-5a"
    ]


@pytest.mark.parametrize("level", ["typed", "assessed"])
def test_untyped_node_fails_at_typed_and_assessed(level):
    """Test that an unassigned node fails ``typed`` and ``assessed`` but not ``anchored``."""
    graph = build(chunk("s-1", RESULT_TEXT))
    assert violations(graph, "anchored") == []
    assert "untyped:s-1" in violations(graph, level)


def test_not_assessed_without_reason_fails_at_assessed():
    """Test that a not_assessed verdict needs a non-empty reason."""
    result, claim, *rest = assessed_graph()
    for assessment in ({"verdict": "not_assessed"}, {"verdict": "not_assessed", "reason": "  "}):
        bad = with_attributes(claim, assessment=assessment)
        assert violations(build(result, bad, *rest), "assessed") == ["missing-reason:s-2"]
        assert violations(build(result, bad, *rest), "typed") == []


# --- anchored ------------------------------------------------------------------------------------


def _split() -> list:
    """A compound split into a result and a claim, as in the documentation's example."""
    text = "We show that the bias vanishes for long segments, which implies that the estimator is consistent."
    parent = chunk("s-5", text, "compound")
    locator = parent.sources[0].locator
    children = []
    for suffix, span in (
        ("a", "We show that the bias vanishes for long segments"),
        ("b", "the estimator is consistent"),
    ):
        start = text.index(span)
        source = SourceAnchor("paper", f"{locator};chars={start}-{start + len(span)}", span, text_digest(span))
        children.append(Node(f"s-5{suffix}", "unassigned", span, (source,), attributes={"origin": "document"}))
    relations = [Relation(f"p{child.id}", "part-of", child.id, "s-5") for child in children]
    return [parent, *children, *relations]


def test_split_graph_passes_anchored():
    """Test that a well-formed compound with two span children passes ``anchored``."""
    assert violations(build(*_split()[:3], relations=tuple(_split()[3:])), "anchored") == []


def test_origin_invalid():
    """Test that a missing or unknown origin is reported."""
    for attributes in ({}, {"origin": "reader"}):
        graph = build(chunk("s-1", RESULT_TEXT, attributes=attributes))
        assert violations(graph, "anchored") == ["origin-invalid:s-1"]


def test_unknown_type():
    """Test that a type the rubric does not declare is reported."""
    assert violations(build(chunk("s-1", RESULT_TEXT, "finding")), "anchored") == ["unknown-type:s-1"]


def test_origin_not_allowed():
    """Test that a type used with an origin outside its origins, or a built-in type off a document node, is reported."""
    premise = Node("a-1", "result", "Noise is Gaussian.", attributes={"origin": "analyst"})
    assert violations(build(premise), "anchored") == ["origin-not-allowed:a-1"]
    unassigned = Node("a-1", "unassigned", "Noise is Gaussian.", attributes={"origin": "analyst"})
    assert violations(build(unassigned), "anchored") == ["origin-not-allowed:a-1"]
    allowed = Node("a-1", "assumption", "Noise is Gaussian.", base=0.8, attributes={"origin": "analyst"})
    assert violations(build(allowed), "anchored") == []


def test_anchor_invalid():
    """Test that a document node needs exactly one source with a locator, a quote and a digest."""
    assert violations(build(chunk("s-1", RESULT_TEXT, sources=())), "anchored") == ["anchor-invalid:s-1"]
    two = chunk("s-1", RESULT_TEXT)
    two = dataclasses.replace(two, sources=two.sources * 2)
    assert violations(build(two), "anchored") == ["anchor-invalid:s-1"]
    for field in ("locator", "digest"):
        node = chunk("s-1", RESULT_TEXT)
        node = dataclasses.replace(node, sources=(dataclasses.replace(node.sources[0], **{field: None}),))
        assert violations(build(node), "anchored") == ["anchor-invalid:s-1"]


def test_anchor_mismatch():
    """Test that a statement differing from its quote is reported, even by whitespace alone."""
    node = chunk("s-1", RESULT_TEXT)
    for statement in ("We show that the bias vanishes.", RESULT_TEXT + " ", None):
        assert violations(build(dataclasses.replace(node, statement=statement)), "anchored") == ["anchor-mismatch:s-1"]


def test_quote_without_statement_is_a_mismatch_and_quote_missing_is_invalid():
    """Test that a quote missing from the source is an invalid anchor, not a silent pass."""
    node = chunk("s-1", RESULT_TEXT)
    node = dataclasses.replace(node, sources=(dataclasses.replace(node.sources[0], quote=None),))
    assert violations(build(node), "anchored") == ["anchor-invalid:s-1"]


def test_digest_mismatch():
    """Test that a digest that is not the digest of the quote is reported."""
    node = chunk("s-1", RESULT_TEXT)
    stale = dataclasses.replace(node.sources[0], digest=text_digest("an earlier version of the sentence"))
    assert violations(build(dataclasses.replace(node, sources=(stale,))), "anchored") == ["digest-mismatch:s-1"]


def test_analyst_has_source():
    """Test that an implicit premise with a source is reported."""
    premise = Node(
        "a-1", "assumption", "Noise is Gaussian.", sources=[SourceAnchor("paper")], attributes={"origin": "analyst"}
    )
    assert violations(build(premise), "anchored") == ["analyst-has-source:a-1"]


def test_part_of_invalid():
    """Test that a child needs exactly one part-of, to a compound, and is not a compound itself."""
    parent, first, second, *relations = _split()
    not_compound = dataclasses.replace(parent, kind="unassigned")
    assert "part-of-invalid:s-5a" in violations(
        build(not_compound, first, second, relations=tuple(relations)), "anchored"
    )
    other = chunk("s-6", "Another sentence, then another clause.", "compound")
    extra = Relation("extra", "part-of", "s-5a", "s-6")
    found = violations(build(parent, first, second, other, relations=(*relations, extra)), "anchored")
    assert "part-of-invalid:s-5a" in found
    nested = dataclasses.replace(first, kind="compound")
    assert "part-of-invalid:s-5a" in violations(build(parent, nested, second, relations=tuple(relations)), "anchored")


def test_span_not_in_parent():
    """Test that a span child must sit at its chars range, under its parent's locator."""
    parent, first, second, *relations = _split()
    anchor = first.sources[0]
    shifted = dataclasses.replace(anchor, locator=anchor.locator.replace("chars=0-", "chars=1-"))
    moved = dataclasses.replace(first, sources=(shifted,))
    assert violations(build(parent, moved, second, relations=tuple(relations)), "anchored") == [
        "span-not-in-parent:s-5a"
    ]
    elsewhere = dataclasses.replace(anchor, locator="file=other.tex;lines=1-2;chars=0-48")
    relocated = dataclasses.replace(first, sources=(elsewhere,))
    assert violations(build(parent, relocated, second, relations=tuple(relations)), "anchored") == [
        "span-not-in-parent:s-5a"
    ]
    beyond = dataclasses.replace(anchor, locator=anchor.locator.replace("chars=0-48", "chars=90-200"))
    assert violations(
        build(parent, dataclasses.replace(first, sources=(beyond,)), second, relations=tuple(relations)), "anchored"
    ) == ["span-not-in-parent:s-5a"]


def test_field_child():
    """Test that a field child passes with no text, its parent's anchor and a form, and fails without either."""
    parent, first, _, *relations = _split()
    good = Node(
        "s-5b", "claim", None, parent.sources, attributes={"origin": "document", "form": {"shape": "universal"}}
    )
    relations = (relations[0], Relation("ps-5b", "part-of", "s-5b", "s-5"))
    assert violations(build(parent, first, good, relations=relations), "anchored") == []
    formless = Node("s-5b", "claim", None, parent.sources, attributes={"origin": "document"})
    assert violations(build(parent, first, formless, relations=relations), "anchored") == ["field-child-invalid:s-5b"]
    other_anchor = (SourceAnchor("paper", "file=main.tex;lines=9-9", "x", text_digest("x")),)
    foreign = dataclasses.replace(good, sources=other_anchor)
    assert violations(build(parent, first, foreign, relations=relations), "anchored") == ["field-child-invalid:s-5b"]


def test_field_child_form_is_checked_against_parent_text():
    """Test that a field child's form must be found in its parent's text."""
    parent, first, _, *relations = _split()
    relations = (relations[0], Relation("ps-5b", "part-of", "s-5b", "s-5"))
    form = {"shape": "universal", "statement": "the estimator is consistent"}
    good = Node("s-5b", "claim", None, parent.sources, attributes={"origin": "document", "form": form})
    rests = Relation("r", "requires", "s-5a", "s-5b", strength=0.9)
    typed_first = dataclasses.replace(
        first, kind="result", attributes={"origin": "document", "form": {"quantity": "the bias"}}
    )
    assert violations(build(parent, typed_first, good, relations=(*relations, rests))) == []
    bad = with_attributes(good, form={**form, "statement": "the estimator is unbiased"})
    assert violations(build(parent, typed_first, bad, relations=(*relations, rests))) == ["form-not-in-text:s-5b"]


def test_compound_children():
    """Test that a compound needs at least two children."""
    parent, first, _, *relations = _split()
    assert violations(build(parent, first, relations=(relations[0],)), "anchored") == ["compound-children:s-5"]


def test_compound_has_content():
    """Test that a compound carries no form, assessment or base and takes part in no other relation."""
    parent, first, second, *relations = _split()
    for changed in (
        with_attributes(parent, form={"shape": "universal"}),
        with_attributes(parent, assessment={"verdict": "holds"}),
        dataclasses.replace(parent, base=0.5),
    ):
        assert violations(build(changed, first, second, relations=tuple(relations)), "anchored") == [
            "compound-has-content:s-5"
        ]
    premise = Node("a-1", "assumption", "Noise is Gaussian.", base=0.8, attributes={"origin": "analyst"})
    leaning = Relation("lean", "supports", "a-1", "s-5", strength=0.5)
    found = violations(build(parent, first, second, premise, relations=(*relations, leaning)), "anchored")
    assert found == ["compound-has-content:s-5"]
    reference = chunk("s-9", "A. Author, Journal 1 (2020).", "reference")
    cites = Relation("cites", "cites", "s-5", "s-9")
    assert violations(build(parent, first, second, reference, relations=(*relations, cites)), "anchored") == []


# --- typed ---------------------------------------------------------------------------------------


def test_form_missing_field():
    """Test that an absent required field is reported, and an absent optional one is not."""
    result, claim, rests = typed_graph()
    form = dict(claim.attributes["form"])
    del form["shape"]
    assert violations(build(result, with_attributes(claim, form=form), rests)) == ["form-missing-field:s-2"]
    del form["bound"]
    form["shape"] = "bound"
    assert violations(build(result, with_attributes(claim, form=form), rests)) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("colour", "red"),
        ("shape", "sometimes"),
        ("bound", "0.05"),
        ("bound", True),
        ("statement", ""),
        ("scope", {"temperature": [0, 1]}),
        ("scope", {"sample_rate": [100, 10]}),
        ("scope", {"sample_rate": [10]}),
    ],
)
def test_form_invalid_field(field, value):
    """Test that an undeclared field, or a value of the wrong field type, is reported."""
    result, claim, rests = typed_graph()
    bad = with_attributes(claim, form={**claim.attributes["form"], field: value})
    assert violations(build(result, bad, rests)) == ["form-invalid-field:s-2"]


def test_scope_is_exempt_from_faithfulness():
    """Test that a scope the text does not state passes: it is the analyst's reading, not the text's."""
    result, claim, rests = typed_graph()
    scoped = with_attributes(claim, form={**claim.attributes["form"], "scope": {"sample_rate": [1, None]}})
    assert violations(build(result, scoped, rests)) == []


def test_quantity_and_eqref_faithfulness():
    """Test that a quantity needs its value and unit in the text, and an equation reference its label."""
    text = "The bias falls to 0.2 Hz, as Eq. \\eqref{eq:bias} predicts."
    node = chunk("s-1", text, "result", base=0.8, attributes={"origin": "document", "form": {"quantity": "The bias"}})

    def check(form):
        return violations(build(with_attributes(node, form={"quantity": "The bias", **form})))

    assert check({"value": {"value": 0.2, "unit": "Hz"}}) == []
    assert check({"value": {"value": 0.3, "unit": "Hz"}}) == ["form-not-in-text:s-1"]
    assert check({"value": {"value": 0.2, "unit": "kHz"}}) == ["form-not-in-text:s-1"]
    derivation = chunk(
        "s-1", text, "derivation", base=0.9, attributes={"origin": "document", "form": {"equation": "eq:bias"}}
    )
    assert violations(build(derivation)) == []
    other = with_attributes(derivation, form={"equation": "eq:variance"})
    assert violations(build(other)) == ["form-not-in-text:s-1"]


def test_verbatim_field_must_quote_the_text():
    """Test that a verbatim field that rephrases the text fails, the case the rule exists for."""
    result, claim, rests = typed_graph()
    rephrased = with_attributes(claim, form={**claim.attributes["form"], "statement": "The detection threshold is 5%"})
    assert violations(build(result, rephrased, rests)) == ["form-not-in-text:s-2"]


def test_form_on_non_document_node_is_invalid():
    """Test that only document nodes carry a form."""
    premise = Node("a-1", "assumption", "Noise is Gaussian.", base=0.8, attributes={"origin": "analyst", "form": {}})
    assert violations(build(premise)) == ["form-invalid-field:a-1"]


# --- assessed ------------------------------------------------------------------------------------


def test_unassessed():
    """Test that a node of an assess = true type with no assessment, or an unknown verdict, is reported."""
    result, claim, *rest = assessed_graph()
    bare = dataclasses.replace(
        result, attributes={key: value for key, value in result.attributes.items() if key != "assessment"}
    )
    assert violations(build(bare, claim, *rest), "assessed") == ["unassessed:s-1"]
    unknown = with_attributes(result, assessment={"verdict": "plausible"})
    assert violations(build(unknown, claim, *rest), "assessed") == ["unassessed:s-1"]


def test_no_evidence():
    """Test that a holds, fails or undetermined verdict needs a relation from an evidence node."""
    result, claim, check, rests, _ = assessed_graph()
    for verdict in ("holds", "fails", "undetermined"):
        judged = with_attributes(result, assessment={"verdict": verdict})
        assert violations(build(judged, claim, check, relations=(rests,)), "assessed") == ["no-evidence:s-1"]
    annotation = Relation("ev", "assesses", "x-1", "s-1")
    assert violations(build(result, claim, check, relations=(rests, annotation)), "assessed") == []
    premise = Node("a-1", "assumption", "The data are clean.", base=0.9, attributes={"origin": "analyst"})
    not_evidence = Relation("ev", "supports", "a-1", "s-1", strength=0.9)
    assert violations(build(result, claim, check, premise, relations=(rests, not_evidence)), "assessed") == [
        "no-evidence:s-1"
    ]


def test_compile_error_only_at_assessed():
    """Test that a graph that does not compile fails ``assessed`` only, naming the node without a base."""
    result, claim, *rest = assessed_graph()
    baseless = dataclasses.replace(result, base=None)
    graph = build(baseless, claim, *rest)
    assert violations(graph, "typed") == []
    findings = check_graph(graph, RUBRIC, "assessed")
    assert [finding.id for finding in findings] == ["compile-error:graph"]
    assert findings[0].nodes == ("s-1",)


def test_levels_include_the_ones_below():
    """Test that every violation is reported, ordered by level and then by node."""
    result, claim, _ = typed_graph()
    stale = dataclasses.replace(claim.sources[0], digest="sha256:0")
    graph = build(chunk("s-0", CLAIM_TEXT), result, dataclasses.replace(claim, sources=(stale,)))
    assert violations(graph, "anchored") == ["digest-mismatch:s-2"]
    assert violations(graph, "typed") == ["digest-mismatch:s-2", "untyped:s-0", "rests-on-missing:s-2"]
    assessed = violations(graph, "assessed")
    assert assessed[:3] == ["digest-mismatch:s-2", "untyped:s-0", "rests-on-missing:s-2"]
    assert assessed[3:] == ["unassessed:s-1", "unassessed:s-2"]


def test_every_code_is_documented():
    """Test that the codes the package defines are exactly the ones the documentation lists, level by level."""
    from pathlib import Path

    text = (Path(__file__).parents[2] / "docs" / "verbatim-ingestion.md").read_text(encoding="utf-8")
    for level, codes in VIOLATION_CODES.items():
        section = text.split(f"**`{level}`**", 1)[1].split("**`", 1)[0].split("## ", 1)[0]
        documented = [line.split("`")[1] for line in section.splitlines() if line.startswith("- `")]
        assert documented == list(codes), level


def test_finding_record():
    """Test that a violation is a finding with the code, an id naming the node, and the node involved."""
    result, claim, _ = typed_graph()
    (finding,) = check_graph(build(result, claim), RUBRIC, "typed")
    assert finding.to_dict() == {
        "id": "rests-on-missing:s-2",
        "diagnostic": "rests-on-missing",
        "nodes": ["s-2"],
        "relations": [],
        "value": None,
        "details": {},
        "message": finding.message,
    }
    assert "'claim'" in finding.message
    assert "found 0" in finding.message


def test_baseline_bases_are_beta():
    """Test the fixture: the typed graph's nodes carry the bases the rubric would give them."""
    result, claim, _ = typed_graph()
    assert result.base == Beta(8, 2)
    assert claim.base == RUBRIC.types["claim"].base
    assert RESULT_TEXT in result.statement
