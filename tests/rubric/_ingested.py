"""Synthetic graphs built from a document's own text, and the example rubric they are checked against."""

from __future__ import annotations

from pathlib import Path

from credencegraph.core import Beta, Graph, Node, Relation, SourceAnchor
from credencegraph.rubric import load_rubric, text_digest
from credencegraph.rubric.validate import check_graph

EXAMPLE_RUBRIC_PATH = Path(__file__).parents[2] / "docs" / "examples" / "methods-paper.toml"
RUBRIC = load_rubric(EXAMPLE_RUBRIC_PATH)

RESULT_TEXT = "We show that the bias vanishes for long segments."
CLAIM_TEXT = "The false-alarm rate stays below 5% for segments longer than 64 s."


def chunk(node_id: str, text: str | None, kind: str = "unassigned", **changes) -> Node:
    """A document chunk anchored verbatim, with ``changes`` overriding any field of the node."""
    fields = {
        "kind": kind,
        "statement": text,
        "sources": [SourceAnchor("paper", f"file=main.tex;lines={node_id}", text, text and text_digest(text))],
        "attributes": {"origin": "document", "unit": "sentence"},
    }
    fields.update(changes)
    return Node(node_id, **fields)


def build(*items: Node | Relation, relations: tuple[Relation, ...] = ()) -> Graph:
    """A graph of the given nodes and relations, nodes added first, each kind in the order given."""
    graph = Graph()
    for node in items:
        if isinstance(node, Node):
            graph.add_node(node)
    for relation in [*items, *relations]:
        if isinstance(relation, Relation):
            graph.add_relation(relation)
    return graph


def typed_graph() -> list:
    """The parts of a graph that passes ``typed``: a claim resting on a result.

    Returns:
        ``[result, claim, relation]``, to be changed one at a time and passed to ``build``.
    """
    result = chunk(
        "s-1",
        RESULT_TEXT,
        "result",
        base=Beta(8, 2),
        attributes={"origin": "document", "form": {"quantity": "the bias", "trend": "vanishes"}},
    )
    claim = chunk(
        "s-2",
        CLAIM_TEXT,
        "claim",
        base=Beta(5, 5),
        attributes={
            "origin": "document",
            "form": {"shape": "bound", "statement": "The false-alarm rate stays below 5%", "bound": 0.05},
        },
    )
    rests = Relation("r", "requires", "s-1", "s-2", strength=Beta(9, 1))
    return [result, claim, rests]


def assessed_graph() -> list:
    """The parts of a graph that passes ``assessed``: the typed graph, assessed, with a check as evidence.

    Returns:
        ``[result, claim, check, relation, evidence relation]``.
    """
    result, claim, rests = typed_graph()
    result = with_attributes(result, assessment={"verdict": "holds"})
    claim = with_attributes(claim, assessment={"verdict": "not_assessed", "reason": "outside the scope of this review"})
    check = Node("x-1", "check", "The bias was re-measured on new data.", base=0.9, attributes={"origin": "evidence"})
    evidence = Relation("ev", "supports", "x-1", "s-1", strength=0.9)
    return [result, claim, check, rests, evidence]


def with_attributes(node: Node, **attributes) -> Node:
    """A copy of a node with attributes added."""
    return Node(
        node.id, node.kind, node.statement, node.sources, node.base, node.stated, {**node.attributes, **attributes}
    )


def violations(graph: Graph, level: str = "typed") -> list[str]:
    """The ids of every violation of the example rubric, in report order."""
    return [finding.id for finding in check_graph(graph, RUBRIC, level)]
