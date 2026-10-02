"""Tests for SourceAnchor, Node and Relation validation."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from credencegraph.core import Beta, Node, Point, Relation, SourceAnchor, ValidationError
from credencegraph.core.attributes import copy_json


class TestSourceAnchor:
    """Validation of SourceAnchor."""

    def test_defaults(self):
        """Test that only the document is required."""
        a = SourceAnchor("doi:10.1/x")
        assert (a.locator, a.quote, a.digest) == (None, None, None)

    @pytest.mark.parametrize("document", ["", None, 3])
    def test_document_must_be_non_empty_string(self, document):
        """Test that the document identifier is a non-empty string."""
        with pytest.raises(ValidationError, match=r"SourceAnchor\.document"):
            SourceAnchor(document)

    @pytest.mark.parametrize("field", ["locator", "quote", "digest"])
    @pytest.mark.parametrize("value", ["", 3])
    def test_optional_fields_reject_empty_and_non_strings(self, field, value):
        """Test that optional text fields are None or a non-empty string."""
        with pytest.raises(ValidationError, match=field):
            SourceAnchor("d", **{field: value})


class TestNode:
    """Validation and normalisation of Node."""

    def test_defaults(self):
        """Test the defaults of a bare node."""
        n = Node("a")
        assert n.kind == "proposition"
        assert n.statement is None
        assert n.sources == ()
        assert n.base is None
        assert n.stated is None
        assert n.attributes == {}

    @pytest.mark.parametrize("node_id", ["", None, 1])
    def test_id_must_be_non_empty_string(self, node_id):
        """Test that the id is a non-empty string."""
        with pytest.raises(ValidationError, match=r"Node\.id"):
            Node(node_id)

    @pytest.mark.parametrize("kind", ["", None])
    def test_kind_must_be_non_empty_string(self, kind):
        """Test that the kind is a non-empty string."""
        with pytest.raises(ValidationError, match=r"Node\.kind"):
            Node("a", kind=kind)

    @pytest.mark.parametrize("statement", ["", 3])
    def test_statement_must_be_none_or_non_empty_string(self, statement):
        """Test that the statement is None or a non-empty string."""
        with pytest.raises(ValidationError, match="statement"):
            Node("a", statement=statement)

    def test_float_shortcut_for_base_and_stated(self):
        """Test that a bare float is read as a Point for base and stated."""
        n = Node("a", base=0.3, stated=Beta(3, 7))
        assert n.base == Point(0.3)
        assert n.stated == Beta(3, 7)

    @pytest.mark.parametrize("field", ["base", "stated"])
    @pytest.mark.parametrize("value", [1.5, "0.5", True])
    def test_bad_credence_is_rejected(self, field, value):
        """Test that an invalid base or stated credence is rejected."""
        with pytest.raises(ValidationError):
            Node("a", **{field: value})

    def test_sources_accepts_a_list_and_becomes_a_tuple(self):
        """Test that sources may be given as a list and are stored immutably."""
        anchor = SourceAnchor("d")
        assert Node("a", sources=[anchor]).sources == (anchor,)

    @pytest.mark.parametrize("sources", ["doc", b"doc", 3])
    def test_sources_must_be_a_sequence(self, sources):
        """Test that a string, bytes or a number is not read as a sequence of anchors."""
        with pytest.raises(ValidationError, match="must be a sequence of SourceAnchor"):
            Node("a", sources=sources)

    @pytest.mark.parametrize("sources", [[SourceAnchor("d"), "doc"], [{"document": "d"}]])
    def test_sources_must_hold_anchors(self, sources):
        """Test that every element of sources is a SourceAnchor."""
        with pytest.raises(ValidationError, match="must hold SourceAnchor objects"):
            Node("a", sources=sources)

    def test_attributes_are_deep_copied(self):
        """Test that later edits to the caller's dict do not reach the node."""
        attrs = {"k": [1, {"z": 2}]}
        n = Node("a", attributes=attrs)
        attrs["k"][1]["z"] = 99
        assert n.attributes == {"k": (1, {"z": 2})}

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda attrs: attrs.__setitem__("x", math.nan),
            lambda attrs: attrs.__delitem__("k"),
            lambda attrs: attrs["k"][1].__setitem__("z", math.nan),
            lambda attrs: attrs["k"].append(math.nan),
        ],
        ids=["set-top", "delete-top", "set-nested", "append-nested"],
    )
    @pytest.mark.parametrize(
        "make",
        [
            lambda attrs: Node("a", attributes=attrs),
            lambda attrs: Relation("r", "cites", "a", "b", attributes=attrs),
        ],
        ids=["node", "relation"],
    )
    def test_stored_attributes_refuse_in_place_mutation(self, make, mutate):
        """Test that the stored attributes cannot be changed in place, at any depth."""
        obj = make({"k": [1, {"z": 2}]})
        with pytest.raises((TypeError, AttributeError)):
            mutate(obj.attributes)
        assert copy_json(obj.attributes, "attributes") == {"k": [1, {"z": 2}]}

    def test_frozen_attributes_survive_replace(self):
        """Test that a node rebuilt from its own frozen attributes keeps them."""
        n = Node("a", attributes={"k": [1, {"z": 2}]})
        assert replace(n, base=0.5).attributes == n.attributes

    @pytest.mark.parametrize(
        "attrs",
        [[1], {1: "x"}, {"k": object()}, {"k": math.nan}, {"k": {"n": math.inf}}, {"k": {1, 2}}],
    )
    def test_attributes_must_be_json(self, attrs):
        """Test that attributes must be a JSON object with finite values."""
        with pytest.raises(ValidationError, match="attributes"):
            Node("a", attributes=attrs)


class TestRelation:
    """Validation of Relation, above all the strength rules."""

    @pytest.mark.parametrize("field", ["id", "type", "source", "target"])
    @pytest.mark.parametrize("value", ["", None])
    def test_identifying_fields_must_be_non_empty_strings(self, field, value):
        """Test that id, type, source and target are non-empty strings."""
        kwargs = {"id": "r", "type": "cites", "source": "a", "target": "b", field: value}
        with pytest.raises(ValidationError):
            Relation(**kwargs)

    @pytest.mark.parametrize("rtype", ["requires", "supports", "refutes"])
    def test_strength_required_for_strength_types(self, rtype):
        """Test that requires, supports and refutes need a strength."""
        with pytest.raises(ValidationError, match="requires a strength"):
            Relation("r", rtype, "a", "b")

    @pytest.mark.parametrize("rtype", ["requires", "supports", "refutes"])
    def test_strength_float_shortcut(self, rtype):
        """Test that a float strength becomes a Point and a Beta is kept."""
        assert Relation("r", rtype, "a", "b", strength=0.7).strength == Point(0.7)
        assert Relation("r", rtype, "a", "b", strength=Beta(8, 2)).strength == Beta(8, 2)

    @pytest.mark.parametrize("rtype", ["requires", "supports", "refutes"])
    @pytest.mark.parametrize("strength", [1.5, "0.5", True])
    def test_bad_strength_is_rejected(self, rtype, strength):
        """Test that an invalid strength is rejected."""
        with pytest.raises(ValidationError):
            Relation("r", rtype, "a", "b", strength=strength)

    @pytest.mark.parametrize("rtype", ["equivalent", "exclusive"])
    def test_strength_forbidden_for_equivalent_and_exclusive(self, rtype):
        """Test that equivalent and exclusive carry no strength."""
        with pytest.raises(ValidationError, match="must not have a strength"):
            Relation("r", rtype, "a", "b", strength=0.5)
        assert Relation("r", rtype, "a", "b").strength is None

    @pytest.mark.parametrize("rtype", ["cites", "authored_by", "Supports", "supports "])
    def test_strength_forbidden_for_annotations(self, rtype):
        """Test that annotation types (anything else, case-sensitively) carry no strength."""
        with pytest.raises(ValidationError, match="annotation"):
            Relation("r", rtype, "a", "b", strength=0.5)
        rel = Relation("r", rtype, "a", "b")
        assert rel.strength is None
        assert not rel.is_inferential

    @pytest.mark.parametrize("rtype", ["equivalent", "exclusive"])
    def test_equivalent_and_exclusive_reject_self_relation(self, rtype):
        """Test that equivalent and exclusive may not join a node to itself."""
        with pytest.raises(ValidationError, match="to itself"):
            Relation("r", rtype, "a", "a")

    def test_annotation_self_relation_is_allowed(self):
        """Test that an annotation may join a node to itself."""
        assert Relation("r", "cites", "a", "a").source == "a"

    @pytest.mark.parametrize("rtype", ["requires", "supports", "refutes", "equivalent", "exclusive"])
    def test_inferential_flag(self, rtype):
        """Test that the five fixed types are inferential."""
        strength = 0.5 if rtype in {"requires", "supports", "refutes"} else None
        assert Relation("r", rtype, "a", "b", strength=strength).is_inferential

    def test_attributes_must_be_json(self):
        """Test that relation attributes are validated like node attributes."""
        with pytest.raises(ValidationError, match="attributes"):
            Relation("r", "cites", "a", "b", attributes={"k": object()})
