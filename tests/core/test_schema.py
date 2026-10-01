"""Tests that the published JSON Schema agrees with the loader on structure."""

from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator

from credencegraph.core import (
    Beta,
    CredenceGraphError,
    Graph,
    Node,
    Relation,
    SourceAnchor,
    graph_from_dict,
    graph_to_dict,
    json_schema,
)


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    """Provide a validator for the published schema."""
    return Draft202012Validator(json_schema())


def full_doc() -> dict:
    """Return a document using every node field, both credence kinds and every relation type."""
    g = Graph()
    g.add_node(
        Node(
            "h",
            statement="s",
            sources=[SourceAnchor("d", "p1", "q", "sha256:0")],
            base=Beta(3, 7),
            stated=0.9,
            attributes={"k": [1, None]},
        )
    )
    g.add_node(Node("e", base=0.5))
    g.add_node(Node("p", kind="person"))
    g.add_relation(Relation("r1", "supports", "e", "h", strength=Beta(1, 1)))
    g.add_relation(Relation("r2", "refutes", "e", "h", strength=0.0))
    g.add_relation(Relation("r3", "requires", "e", "h", strength=1))
    g.add_relation(Relation("r4", "equivalent", "e", "h"))
    g.add_relation(Relation("r5", "exclusive", "e", "h"))
    g.add_relation(Relation("r6", "authored_by", "h", "p", attributes={"x": 1}))
    return graph_to_dict(g)


def test_schema_is_a_valid_draft_2020_12_schema():
    """Test that the published schema is itself well formed."""
    Draft202012Validator.check_schema(json_schema())


def test_schema_declares_the_format_version():
    """Test that the schema pins the format tag and version it describes."""
    props = json_schema()["properties"]
    assert props["format"] == {"const": "credencegraph"}
    assert props["version"] == {"const": 1}


def test_dumped_documents_validate(validator):
    """Test that whatever the writer produces satisfies the schema."""
    validator.validate(full_doc())
    validator.validate(graph_to_dict(Graph()))


def test_minimal_documents_validate(validator):
    """Test that omitted optional fields are allowed, as in the loader."""
    doc = {
        "format": "credencegraph",
        "version": 1,
        "nodes": [{"id": "a"}],
        "relations": [{"id": "r", "type": "cites", "source": "a", "target": "a"}],
    }
    validator.validate(doc)
    graph_from_dict(doc)


def _set(path: tuple, value):
    def apply(doc):
        target = doc
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return apply


def _drop(path: tuple):
    def apply(doc):
        target = doc
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return apply


STRUCTURAL_BREAKS = {
    "wrong format": _set(("format",), "other"),
    "wrong version": _set(("version",), 2),
    "missing nodes": _drop(("nodes",)),
    "extra top-level field": _set(("extra",), 1),
    "node without id": _drop(("nodes", 0, "id")),
    "empty node id": _set(("nodes", 0, "id"), ""),
    "empty kind": _set(("nodes", 0, "kind"), ""),
    "unknown node field": _set(("nodes", 0, "color"), "red"),
    "empty statement": _set(("nodes", 0, "statement"), ""),
    "point above one": _set(("nodes", 0, "base"), 1.5),
    "point below zero": _set(("nodes", 0, "stated"), -0.1),
    "beta with zero alpha": _set(("nodes", 0, "base"), {"alpha": 0, "beta": 1}),
    "beta with negative beta": _set(("nodes", 0, "base"), {"alpha": 1, "beta": -1}),
    "beta missing beta": _set(("nodes", 0, "base"), {"alpha": 1}),
    "beta with extra key": _set(("nodes", 0, "base"), {"alpha": 1, "beta": 1, "mean": 0.5}),
    "credence is a string": _set(("nodes", 0, "base"), "0.5"),
    "anchor without document": _drop(("nodes", 0, "sources", 0, "document")),
    "anchor unknown field": _set(("nodes", 0, "sources", 0, "page"), 3),
    "sources not an array": _set(("nodes", 0, "sources"), "d"),
    "attributes not an object": _set(("nodes", 0, "attributes"), [1]),
    "relation without type": _drop(("relations", 0, "type")),
    "empty relation type": _set(("relations", 0, "type"), ""),
    "relation unknown field": _set(("relations", 0, "weight"), 1),
    "supports without strength": _drop(("relations", 0, "strength")),
    "supports with null strength": _set(("relations", 0, "strength"), None),
    "requires with bad strength": _set(("relations", 2, "strength"), 2),
    "strength on equivalent": _set(("relations", 3, "strength"), 0.5),
    "strength on exclusive": _set(("relations", 4, "strength"), {"alpha": 1, "beta": 1}),
    "strength on annotation": _set(("relations", 5, "strength"), 0.5),
}


@pytest.mark.parametrize("name", STRUCTURAL_BREAKS)
def test_structural_errors_fail_both_schema_and_loader(validator, name):
    """Test that a structural defect is rejected by the schema and by the loader alike."""
    doc = full_doc()
    STRUCTURAL_BREAKS[name](doc)
    assert not validator.is_valid(doc), "schema accepted a defective document"
    with pytest.raises(CredenceGraphError):
        graph_from_dict(doc)


def test_full_document_used_above_is_itself_valid(validator):
    """Test the baseline of the mutation table: the unbroken document passes both."""
    doc = full_doc()
    assert validator.is_valid(doc)
    graph_from_dict(copy.deepcopy(doc))


def test_null_optional_strength_is_valid_for_annotations(validator):
    """Test that an explicit null strength is accepted where none is allowed."""
    doc = full_doc()
    doc["relations"][5]["strength"] = None
    doc["relations"][3]["strength"] = None
    assert validator.is_valid(doc)
    graph_from_dict(doc)


def test_document_wide_rules_are_left_to_the_loader(validator):
    """Test the documented boundary: cycles, dangling ids, duplicates and self-relations pass the schema, fail the loader."""
    cyc = full_doc()
    cyc["relations"].append({"id": "back", "type": "supports", "source": "h", "target": "e", "strength": 0.5})
    cyc["relations"].append({"id": "loop", "type": "supports", "source": "e", "target": "h", "strength": 0.5})
    dangling = full_doc()
    dangling["relations"][0]["target"] = "ghost"
    duplicate = full_doc()
    duplicate["nodes"].append({"id": "h"})
    for doc in (cyc, dangling, duplicate):
        assert validator.is_valid(doc)
        with pytest.raises(CredenceGraphError):
            graph_from_dict(doc)
    for index in (3, 4):
        self_relation = full_doc()
        rel = self_relation["relations"][index]
        rel["target"] = rel["source"]
        assert validator.is_valid(self_relation)
        with pytest.raises(CredenceGraphError, match=rf"of type '{rel['type']}' joins node 'e' to itself"):
            graph_from_dict(self_relation)
