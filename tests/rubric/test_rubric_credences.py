"""The ``rubric_credences`` rules: every base and strength is the rubric's, and only an observed check raises.

Each rule test starts from a graph that passes and breaks exactly one rule. The property tests draw
random graphs of rubric-typed nodes, some with one violation injected, and check that the rules
report every injected violation and that a graph they pass keeps every credence under its base.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _ingested import RUBRIC, build, chunk, typed_graph, violations, with_attributes
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from credencegraph.core import Beta, Graph, Node, Point, Relation
from credencegraph.inference import conditional
from credencegraph.rubric import RUBRIC_CREDENCE_CODES, check_graph
from credencegraph.semantics import compile_graph

RECOMPUTE = RUBRIC.checks["recompute-numbers"]


def extracted_graph() -> list:
    """The parts of a graph that passes ``typed`` and the rubric-credence rules.

    A claim rests on a result and on an implicit premise, and a passed check was run on the result.

    Returns:
        ``[result, claim, premise, check, rests, premise relation, check edge]``.
    """
    result, claim, rests = typed_graph()
    premise = Node("a-1", "assumption", "The noise is stationary.", base=Beta(2, 2), attributes={"origin": "analyst"})
    check = Node(
        "c-1",
        "recompute-numbers",
        "recompute-numbers on s-1: passed",
        attributes={"origin": "evidence", "check": {"result": "passed"}, "observed": True},
    )
    assumed = Relation("ra", "requires", "a-1", "s-2", strength=Beta(9, 1))
    edge = check_edge("s-1", "c-1", RECOMPUTE)
    return [result, claim, premise, check, rests, assumed, edge]


def check_edge(source: str, target: str, check, relation_id: str = "ce") -> Relation:
    """The check edge of a check type: an observes relation carrying the check's sensitivity and false-pass rate."""
    return Relation(relation_id, "observes", source, target, sensitivity=check.sensitivity, false_pass=check.false_pass)


def credence_violations(graph: Graph, level: str = "typed") -> list[str]:
    """The ids of every violation with the rubric-credence rules added, in report order."""
    return [finding.id for finding in check_graph(graph, RUBRIC, level, rubric_credences=True)]


def replaced(parts: list, index: int, item: Node | Relation) -> list:
    """A copy of ``parts`` with one item replaced."""
    return [item if i == index else part for i, part in enumerate(parts)]


def test_baseline_passes():
    """Test that the graph the rule tests break passes every level with the rules added."""
    for level in ("anchored", "typed"):
        assert credence_violations(build(*extracted_graph()), level) == []


def test_check_edge_gives_the_check_its_sensitivity_and_false_pass():
    """Test the check edge: a pass is E[sensitivity] likely on a true target, E[false_pass] on a false one."""
    graph = build(*extracted_graph())
    network = compile_graph(graph)
    for value, expected in ((True, 19 / 20), (False, 1 / 10)):
        point = conditional(network, {"c-1": True}, {"s-1": value}, draws=0).point
        assert point == pytest.approx(expected, rel=1e-12)


# --- credence-not-rubric ---------------------------------------------------------------------------


def test_document_base_not_the_types():
    """Test that a document node whose base is not its type's is reported."""
    parts = extracted_graph()
    parts[0] = Node(**{**_fields(parts[0]), "base": Beta(9, 1)})
    assert credence_violations(build(*parts)) == ["credence-not-rubric:s-1"]


def test_analyst_base_not_the_premises():
    """Test that an analyst node takes premises.base, not its type's base (beta:8,2 for an assumption)."""
    parts = extracted_graph()
    parts[2] = Node(**{**_fields(parts[2]), "base": Beta(8, 2)})
    assert credence_violations(build(*parts)) == ["credence-not-rubric:a-1"]


def test_node_without_base_is_not_reported():
    """Test that a missing base is not a non-rubric base; it is a compile error at ``assessed`` instead."""
    parts = extracted_graph()
    parts[0] = Node(**{**_fields(parts[0]), "base": None})
    assert credence_violations(build(*parts)) == []


def test_requires_strength_not_the_edge_rules():
    """Test that a requires into a claim with a strength other than the edge rule's is reported."""
    parts = extracted_graph()
    parts[4] = Relation("r", "requires", "s-1", "s-2", strength=0.9)
    assert credence_violations(build(*parts)) == ["credence-not-rubric:r"]


def test_premise_strength_not_the_premises():
    """Test that a requires from an analyst node takes premises.strength."""
    parts = extracted_graph()
    parts[5] = Relation("ra", "requires", "a-1", "s-2", strength=Beta(8, 2))
    assert credence_violations(build(*parts)) == ["credence-not-rubric:ra"]


def test_requires_into_a_type_without_a_requires_rule():
    """Test that a requires into a type whose rubric gives no requires strength is reported: no value can match."""
    parts = extracted_graph()
    method = chunk("s-0", "We ran a test.", "method", base=Beta(9, 1))
    relation = Relation("rm", "requires", "s-0", "s-1", strength=Beta(9, 1))
    assert credence_violations(build(*parts, method, relation), "anchored") == ["credence-not-rubric:rm"]


@pytest.mark.parametrize("base", [Beta(1, 9), Beta(19, 1)], ids=["its-false-pass", "another"])
def test_check_node_with_a_base(base):
    """Test that a check node with any base is reported, even its false_pass: the check edge sets its probability."""
    parts = extracted_graph()
    parts[3] = Node(**{**_fields(parts[3]), "base": base})
    assert credence_violations(build(*parts)) == ["credence-not-rubric:c-1"]


@pytest.mark.parametrize(
    "credences",
    [{"sensitivity": Beta(9, 1)}, {"false_pass": Beta(1, 19)}, {"sensitivity": 0.95, "false_pass": 0.1}],
    ids=["sensitivity", "false-pass", "points-at-the-means"],
)
def test_check_edge_credences_not_the_checks(credences):
    """Test that a check edge with a sensitivity or false_pass other than its check type's is reported."""
    parts = extracted_graph()
    fields = {"sensitivity": RECOMPUTE.sensitivity, "false_pass": RECOMPUTE.false_pass, **credences}
    parts[6] = Relation("ce", "observes", "s-1", "c-1", **fields)
    assert credence_violations(build(*parts)) == ["credence-not-rubric:ce"]


def test_supports_check_edge_is_not_the_rubrics():
    """Test that the supports form of a check edge, which no-evidence accepts, is not allowed by the rules.

    The supports form puts the false-pass rate on the check node's base, so that is reported too.
    """
    parts = extracted_graph()
    parts[3] = Node(**{**_fields(parts[3]), "base": RECOMPUTE.false_pass})
    strength = 1 - (1 - RECOMPUTE.sensitivity.mean) / (1 - RECOMPUTE.false_pass.mean)
    parts[6] = Relation("ce", "supports", "s-1", "c-1", strength=strength)
    graph = build(*parts)
    assert credence_violations(graph) == ["credence-not-rubric:c-1", "relation-not-allowed:ce"]
    for value, expected in ((True, 19 / 20), (False, 1 / 10)):
        point = conditional(compile_graph(graph), {"c-1": True}, {"s-1": value}, draws=0).point
        assert point == pytest.approx(expected, rel=1e-12)


# --- relation-not-allowed ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relation",
    [
        Relation("x", "supports", "s-1", "s-2", strength=0.5),
        Relation("x", "refutes", "a-1", "s-2", strength=0.5),
        Relation("x", "equivalent", "s-1", "a-1"),
        Relation("x", "exclusive", "s-2", "a-1"),
        Relation("x", "requires", "c-1", "s-2", strength=Beta(9, 1)),
        Relation("x", "requires", "s-1", "c-1", strength=Beta(9, 1)),
    ],
    ids=["supports", "refutes", "equivalent", "exclusive", "out-of-a-check", "requires-into-a-check"],
)
def test_relation_not_allowed(relation):
    """Test that an inferential relation touching a document, analyst or check node, other than the two kinds, fails."""
    assert credence_violations(build(*extracted_graph(), relation)) == ["relation-not-allowed:x"]


def test_evidence_supporting_a_document_node():
    """Test that evidence pointing at the node it bears on, as hand-built graphs do, is not allowed."""
    evidence = Node("x-1", "check", "Re-measured.", base=0.9, attributes={"origin": "evidence"})
    relation = Relation("x", "supports", "x-1", "s-1", strength=0.9)
    assert credence_violations(build(*extracted_graph(), evidence, relation)) == ["relation-not-allowed:x"]


def test_check_type_without_record_is_not_a_check_node():
    """Test that a check edge needs a check record on its target: without one it is an observes like any other."""
    parts = extracted_graph()
    attributes = {"origin": "evidence"}
    parts[3] = Node(**{**_fields(parts[3]), "attributes": attributes})
    assert credence_violations(build(*parts)) == ["relation-not-allowed:ce"]


def test_relations_away_from_document_and_analyst_nodes_are_not_checked():
    """Test that relations among evidence nodes that touch no document, analyst or check node are left alone."""
    first = Node("x-1", "check", "One.", base=0.9, attributes={"origin": "evidence"})
    second = Node("x-2", "check", "Two.", base=0.9, attributes={"origin": "evidence"})
    relations = (Relation("x", "supports", "x-1", "x-2", strength=0.9), Relation("y", "exclusive", "x-1", "x-2"))
    assert credence_violations(build(*extracted_graph(), first, second, relations=relations)) == []


def test_annotations_are_not_checked():
    """Test that an annotation relation is not an inferential relation, so it is never reported."""
    relation = Relation("x", "cites", "s-1", "s-2")
    assert credence_violations(build(*extracted_graph(), relation)) == []


# --- check-unobserved-invalid -----------------------------------------------------------------------


def test_observed_on_a_document_node():
    """Test that an observation on a node that is not a check node is reported."""
    parts = extracted_graph()
    parts[0] = with_attributes(parts[0], observed=True)
    assert credence_violations(build(*parts)) == ["check-unobserved-invalid:s-1"]


@pytest.mark.parametrize(
    ("result", "observed"),
    [("passed", False), ("passed", None), ("failed", True), ("failed", None), ("error", True), ("error", False)],
)
def test_observed_disagrees_with_result(result, observed):
    """Test that observed must be true for passed, false for failed and absent for error; None means absent."""
    parts = extracted_graph()
    attributes = {"origin": "evidence", "check": {"result": result}}
    attributes.update({} if observed is None else {"observed": observed})
    parts[3] = Node(**{**_fields(parts[3]), "attributes": attributes})
    assert credence_violations(build(*parts)) == ["check-unobserved-invalid:c-1"]


@pytest.mark.parametrize(("result", "observed"), [("passed", True), ("failed", False), ("error", None)])
def test_observed_agrees_with_result(result, observed):
    """Test the three agreeing cases pass."""
    parts = extracted_graph()
    attributes = {"origin": "evidence", "check": {"result": result}}
    attributes.update({} if observed is None else {"observed": observed})
    parts[3] = Node(**{**_fields(parts[3]), "attributes": attributes})
    assert credence_violations(build(*parts)) == []


@pytest.mark.parametrize("observed", [1, None, "true"])
def test_observed_must_be_a_boolean(observed):
    """Test that an observation equal to true but not the boolean, or null, does not agree with passed."""
    parts = extracted_graph()
    parts[3] = Node(**{**_fields(parts[3]), "attributes": {**parts[3].attributes, "observed": observed}})
    assert credence_violations(build(*parts)) == ["check-unobserved-invalid:c-1"]


def test_unknown_result():
    """Test that a check record whose result is not passed, failed or error is reported."""
    parts = extracted_graph()
    parts[3] = Node(**{**_fields(parts[3]), "attributes": {"origin": "evidence", "check": {"result": "skipped"}}})
    assert credence_violations(build(*parts)) == ["check-unobserved-invalid:c-1"]


# --- without the flag --------------------------------------------------------------------------------


def test_without_the_flag_nothing_changes():
    """Test that a graph breaking every rule is checked exactly as before when the rules are not asked for."""
    result, claim, premise, check, rests, assumed, _ = extracted_graph()
    broken = [
        Node(**{**_fields(result), "base": Beta(9, 1)}),
        with_attributes(claim, observed=True),
        premise,
        check,
        rests,
        assumed,
        Relation("x", "supports", "s-1", "s-2", strength=0.5),
    ]
    graph = build(*broken)
    for level in ("anchored", "typed", "assessed"):
        before = violations(graph, level)
        assert check_graph(graph, RUBRIC, level, rubric_credences=False) == check_graph(graph, RUBRIC, level)
        assert not any(code.split(":")[0] in RUBRIC_CREDENCE_CODES for code in before)
        assert credence_violations(graph, level) == [
            *before,
            "credence-not-rubric:s-1",
            "check-unobserved-invalid:s-2",
            "relation-not-allowed:x",
        ]


def test_documented_check_edge_is_observes():
    """Test that the documentation names observes as the check edge the rules allow, and supports as accepted."""
    text = (Path(__file__).parents[2] / "docs" / "verbatim-ingestion.md").read_text(encoding="utf-8")
    flat = " ".join(text.split())
    section = flat.split("**`--rubric-credences`**", 1)[1].split("## ", 1)[0]
    assert "A **check edge** is an `observes` relation from a document or analyst node to a check node" in section
    assert "a `supports` into one, though `no-evidence` accepts it" in section
    assert "no check edge (`observes`) or `supports` relation out of it to a check node" in flat


def test_every_code_is_documented():
    """Test that the rules' codes are exactly the ones the documentation lists under the flag."""
    text = (Path(__file__).parents[2] / "docs" / "verbatim-ingestion.md").read_text(encoding="utf-8")
    section = text.split("**`--rubric-credences`**", 1)[1].split("## ", 1)[0]
    documented = [line.split("`")[1] for line in section.splitlines() if line.startswith("- `")]
    assert documented == list(RUBRIC_CREDENCE_CODES)


def _fields(node: Node) -> dict:
    """The fields of a node, to rebuild it with some replaced."""
    return {name: getattr(node, name) for name in Node.__dataclass_fields__}


# --- the ceiling, on random graphs -------------------------------------------------------------------

SETTINGS = settings(max_examples=300, deadline=None)
SEEDS = st.integers(0, 2**32 - 1)

# The declared types a document or analyst node may take, each with a base.
TYPES = ("method", "result", "derivation", "literature_claim", "claim", "assumption")

# The violations a random graph may have injected, and the code each is reported with.
INJECTIONS = {
    "base": "credence-not-rubric",
    "strength": "credence-not-rubric",
    "check-base": "credence-not-rubric",
    "check-edge-credences": "credence-not-rubric",
    "supports-check-edge": "relation-not-allowed",
    "supports": "relation-not-allowed",
    "refutes": "relation-not-allowed",
    "equivalent": "relation-not-allowed",
    "exclusive": "relation-not-allowed",
    "evidence-supports": "relation-not-allowed",
    "out-of-check": "relation-not-allowed",
    "observed-off-check": "check-unobserved-invalid",
    "observed-disagrees": "check-unobserved-invalid",
}

# The attributes.observed each check result implies, None for absent.
OBSERVED = {"passed": True, "failed": False, "error": None}


def expected_base(kind: str, origin: str):
    """The rubric base of a document or analyst node of the example rubric, which has [premises]."""
    return RUBRIC.premises.base if origin == "analyst" else RUBRIC.types[kind].base


def add(graph: Graph, rtype: str, source: str, target: str, strength=None) -> None:
    """Add a relation, numbered in the order relations are added."""
    graph.add_relation(Relation(f"r{len(graph.relations)}", rtype, source, target, strength=strength))


def add_check_edge(graph: Graph, source: str, target: str, check) -> None:
    """Add a check edge, numbered in the order relations are added."""
    graph.add_relation(check_edge(source, target, check, f"r{len(graph.relations)}"))


def random_extracted_graph(rng: np.random.Generator, injection: str | None) -> Graph:
    """Build a random graph shaped like an extracted one, with at most one violation of the rules injected.

    Document and analyst nodes come in a random order; a requires runs only from an earlier node to a
    later one, into a claim (with the edge rule's strength) or from an analyst node (with the premise
    strength). Check nodes hang off random nodes by check edges, with observations that agree with their
    results. An injected violation raises credences where it can: high bases and strong supports.

    Args:
        rng: The random generator.
        injection: A key of ``INJECTIONS``, or ``None`` for a graph that passes.

    Returns:
        The graph.
    """
    count = int(rng.integers(2, 8))
    nodes = []
    for i in range(count):
        kind = TYPES[int(rng.integers(0, len(TYPES)))]
        origin = "analyst" if kind == "assumption" and rng.random() < 0.5 else "document"
        nodes.append(Node(f"n{i}", kind, base=expected_base(kind, origin), attributes={"origin": origin}))
    i = int(rng.integers(0, count))
    if injection == "base":
        nodes[i] = Node(**{**_fields(nodes[i]), "base": Point(float(rng.uniform(0.9, 0.99)))})
    if injection == "observed-off-check":
        nodes[i] = with_attributes(nodes[i], observed=bool(rng.integers(0, 2)))
    graph = build(*nodes)
    density = rng.uniform(0.2, 0.8)
    for j in range(count):
        for i in range(j):
            source, target = nodes[i], nodes[j]
            analyst = source.attributes["origin"] == "analyst"
            if (analyst or target.kind == "claim") and rng.random() < density:
                strength = RUBRIC.premises.strength if analyst else RUBRIC.types["claim"].rests_on.strength
                add(graph, "requires", source.id, target.id, strength)
    names = list(RUBRIC.checks)
    for k in range(int(rng.integers(0, 4))):
        check = RUBRIC.checks[names[int(rng.integers(0, len(names)))]]
        result = ("passed", "failed", "error")[int(rng.integers(0, 3))]
        attributes = {"origin": "evidence", "check": {"result": result}}
        if OBSERVED[result] is not None:
            attributes["observed"] = OBSERVED[result]
        graph.add_node(Node(f"c{k}", check.name, attributes=attributes))
        add_check_edge(graph, nodes[int(rng.integers(0, count))].id, f"c{k}", check)
    if injection is not None:
        inject(graph, [node.id for node in nodes], rng, injection)
    return graph


def inject(graph: Graph, ids: list[str], rng: np.random.Generator, injection: str) -> None:
    """Add the relations, and nodes, of an injected violation from an earlier node ``i`` to a later one ``j``."""
    i, j = (ids[k] for k in sorted(int(x) for x in rng.choice(len(ids), size=2, replace=False)))
    high = Point(float(rng.uniform(0.8, 0.99)))
    if injection == "strength":
        add(graph, "requires", i, j, Point(float(rng.uniform(0.01, 0.5))))
    elif injection in ("supports", "refutes"):
        add(graph, injection, i, j, high)
    elif injection in ("equivalent", "exclusive"):
        add(graph, injection, i, j)
    elif injection == "evidence-supports":
        graph.add_node(Node("e", "check", base=high, attributes={"origin": "evidence"}))
        add(graph, "supports", "e", j, high)
    elif injection in (
        "check-base",
        "check-edge-credences",
        "supports-check-edge",
        "observed-disagrees",
        "out-of-check",
    ):
        check = RUBRIC.checks["recompute-numbers"]
        observed = injection != "observed-disagrees"
        attributes = {"origin": "evidence", "check": {"result": "passed"}, "observed": observed}
        graph.add_node(Node("c", check.name, base=high if injection == "check-base" else None, attributes=attributes))
        if injection == "check-edge-credences":
            relation_id = f"r{len(graph.relations)}"
            graph.add_relation(Relation(relation_id, "observes", i, "c", sensitivity=high, false_pass=check.false_pass))
        elif injection == "supports-check-edge":
            add(graph, "supports", i, "c", high)
        else:
            add_check_edge(graph, i, "c", check)
        if injection == "out-of-check":
            add(graph, "supports", "c", j, high)


def rule_codes(graph: Graph) -> set[str]:
    """The codes of the rubric-credence rules the graph breaks."""
    findings = check_graph(graph, RUBRIC, "anchored", rubric_credences=True)
    return {finding.diagnostic for finding in findings if finding.diagnostic in RUBRIC_CREDENCE_CODES}


@SETTINGS
@given(seed=SEEDS, injection=st.sampled_from([None, *INJECTIONS]))
def test_rules_report_exactly_the_injected_violation(seed, injection):
    """Test that a graph built from rubric values passes, and that each injected violation is reported."""
    graph = random_extracted_graph(np.random.default_rng(seed), injection)
    assert rule_codes(graph) == (set() if injection is None else {INJECTIONS[injection]})


@SETTINGS
@given(seed=SEEDS, injection=st.sampled_from([None, None, None, *INJECTIONS]))
def test_unobserved_credences_stay_under_their_rubric_base(seed, injection):
    """Test that on a graph that passes the rules, before anything is observed, no credence exceeds its base.

    Graphs with a violation injected are drawn too and kept only if the rules pass them, so a rule that
    lets a credence-raising violation through fails this test.
    """
    graph = random_extracted_graph(np.random.default_rng(seed), injection)
    assume(not rule_codes(graph))
    network = compile_graph(graph)
    for node in graph:
        if node.attributes["origin"] in ("document", "analyst"):
            base = expected_base(node.kind, node.attributes["origin"]).mean
            point = conditional(network, {node.id: True}, {}, draws=0).point
            assert point <= base + 1e-12, (node.id, point, base)
