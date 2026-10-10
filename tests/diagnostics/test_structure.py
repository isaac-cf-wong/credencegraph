"""Structural diagnostics: missing parameters, unanchored variables and correlated supports."""

from __future__ import annotations

import pytest

from credencegraph.core import Beta, Graph, Node, Relation, SourceAnchor, ValidationError
from credencegraph.diagnostics import (
    CORRELATED_SUPPORT,
    MISSING_PARAMETER,
    UNANCHORED,
    correlated_support,
    missing_parameters,
    unanchored,
)

ANCHOR = SourceAnchor("doi:10.0000/example", locator="p. 4")


class TestMissingParameters:
    """An inference variable needs a base; a strength-carrying relation needs a strength."""

    def test_complete_graph_has_none(self):
        """Test that a graph with every base and strength gives no finding."""
        graph = Graph()
        graph.add_node(Node("a", base=0.3))
        graph.add_node(Node("b", base=0.1))
        graph.add_relation(Relation("ab", "supports", "a", "b", strength=0.6))
        assert missing_parameters(graph) == []

    def test_endpoint_without_base(self):
        """Test that the target of a relation without a base is reported, and the carried node is not."""
        graph = Graph()
        graph.add_node(Node("a", base=0.3))
        graph.add_node(Node("b"))
        graph.add_node(Node("person", kind="person"))
        graph.add_relation(Relation("ab", "supports", "a", "b", strength=0.6))
        graph.add_relation(Relation("auth", "authored_by", "b", "person"))
        (finding,) = missing_parameters(graph)
        assert finding.id == "missing-parameter:base:b"
        assert finding.diagnostic == MISSING_PARAMETER
        assert finding.nodes == ("b",)
        assert finding.relations == ()
        assert finding.value is None
        assert "'b'" in finding.message
        assert "without a base" in finding.message

    def test_merged_nodes_share_one_base(self):
        """Test that equivalent nodes need one base between them, and are reported together without one."""
        graph = Graph()
        for node_id, base in (("a", 0.4), ("a2", None), ("c", None), ("c2", None)):
            graph.add_node(Node(node_id, base=base))
        graph.add_relation(Relation("eq_a", "equivalent", "a", "a2"))
        graph.add_relation(Relation("eq_c", "equivalent", "c2", "c"))
        (finding,) = missing_parameters(graph)
        assert finding.id == "missing-parameter:base:c"
        assert finding.nodes == ("c", "c2")
        assert "merged with 'c2'" in finding.message

    def test_relation_without_strength(self):
        """Test that a relation stripped of its strength after validation is reported."""
        graph = Graph()
        graph.add_node(Node("a", base=0.3))
        graph.add_node(Node("b", base=0.1))
        relation = graph.add_relation(Relation("ab", "requires", "a", "b", strength=0.6))
        object.__setattr__(relation, "strength", None)
        (finding,) = missing_parameters(graph)
        assert finding.id == "missing-parameter:strength:ab"
        assert finding.nodes == ("a", "b")
        assert finding.relations == ("ab",)
        assert "requires relation 'ab'" in finding.message


class TestUnanchored:
    """A variable with no source and no inferential parents is an assumption nobody wrote down."""

    def test_root_without_source(self):
        """Test that of two roots only the one without a source is reported, and a child is not."""
        graph = Graph()
        graph.add_node(Node("bare", base=0.5))
        graph.add_node(Node("cited", base=0.5, sources=[ANCHOR]))
        graph.add_node(Node("child", base=0.1))
        graph.add_relation(Relation("r", "supports", "bare", "child", strength=0.7))
        (finding,) = unanchored(graph)
        assert finding.id == "unanchored:bare"
        assert finding.diagnostic == UNANCHORED
        assert finding.nodes == ("bare",)
        assert "no source and no inferential parents" in finding.message

    def test_annotation_parent_does_not_anchor(self):
        """Test that an annotation relation into a node does not count as an inferential parent."""
        graph = Graph()
        graph.add_node(Node("x", base=0.5))
        graph.add_node(Node("doc", kind="document"))
        graph.add_relation(Relation("cite", "cites", "doc", "x"))
        assert [f.id for f in unanchored(graph)] == ["unanchored:x"]

    def test_exclusive_does_not_anchor(self):
        """Test that taking part in an exclusive constraint does not anchor a node."""
        graph = Graph()
        graph.add_node(Node("x", base=0.5))
        graph.add_node(Node("y", base=0.5, sources=[ANCHOR]))
        graph.add_relation(Relation("xy", "exclusive", "x", "y"))
        assert [f.id for f in unanchored(graph)] == ["unanchored:x"]

    def test_merged_twin_anchors_the_variable(self):
        """Test that a source, or a parent, on one equivalent node anchors the whole merged variable."""
        graph = Graph()
        for node_id in ("a", "a2", "b", "b2", "p"):
            graph.add_node(Node(node_id, base=0.5, sources=[ANCHOR] if node_id in ("a2", "p") else ()))
        graph.add_relation(Relation("eq_a", "equivalent", "a", "a2"))
        graph.add_relation(Relation("eq_b", "equivalent", "b", "b2"))
        graph.add_relation(Relation("pb", "requires", "p", "b2", strength=0.9))
        assert unanchored(graph) == []

    def test_merged_bare_variable_lists_every_member(self):
        """Test that an unanchored merged variable is reported once, with all its nodes."""
        graph = Graph()
        graph.add_node(Node("a", base=0.5))
        graph.add_node(Node("a2"))
        graph.add_relation(Relation("eq", "equivalent", "a", "a2"))
        (finding,) = unanchored(graph)
        assert finding.nodes == ("a", "a2")


def pile(n, anchor=ANCHOR):
    """A claim of base 0 with ``n`` supports of strength 0.2, ``r0`` to ``r<n-1>``, each anchored in ``anchor``."""
    graph = Graph()
    graph.add_node(Node("claim", base=0.0))
    for i in range(n):
        graph.add_node(Node(f"r{i}", base=1.0, sources=[anchor]))
        graph.add_relation(Relation(f"s{i}", "supports", f"r{i}", "claim", strength=0.2))
    return graph


class TestCorrelatedSupport:
    """Supports of one node that share a source and no common parent are reported."""

    def test_supports_from_one_document(self):
        """Test the record for three supports anchored in the same document."""
        (finding,) = correlated_support(pile(3))
        assert finding.id == "correlated-support:claim:document:doi:10.0000/example"
        assert finding.diagnostic == CORRELATED_SUPPORT
        assert finding.nodes == ("claim", "r0", "r1", "r2")
        assert finding.relations == ("s0", "s1", "s2")
        assert finding.value is None
        assert dict(finding.details) == {"supports": 3.0, "min_supports": 2.0}
        assert finding.message.startswith("node 'claim' has 3 supports, 'r0', 'r1', 'r2', that share document ")
        assert "no common parent" in finding.message
        assert len(finding.message.splitlines()) == 1

    def test_common_parent_silences_it(self):
        """Test that a proposition every support requires is the common parent the finding asks for."""
        graph = pile(3)
        graph.add_node(Node("sound", base=0.5, sources=[ANCHOR]))
        for i in range(3):
            graph.add_relation(Relation(f"c{i}", "requires", "sound", f"r{i}", strength=1.0))
        assert correlated_support(graph) == []

    def test_parent_of_only_some_is_not_common(self):
        """Test that a parent shared by two of three supports does not silence the group."""
        graph = pile(3)
        graph.add_node(Node("sound", base=0.5, sources=[ANCHOR]))
        for i in range(2):
            graph.add_relation(Relation(f"c{i}", "requires", "sound", f"r{i}", strength=1.0))
        (finding,) = correlated_support(graph)
        assert finding.nodes == ("claim", "r0", "r1", "r2")

    def test_any_inferential_type_makes_a_common_parent(self):
        """Test that a common parent through supports or refutes counts as one through requires does."""
        for rtype in ("supports", "refutes"):
            graph = pile(2)
            graph.add_node(Node("p", base=0.5))
            for i in range(2):
                graph.add_relation(Relation(f"c{i}", rtype, "p", f"r{i}", strength=0.5))
            assert correlated_support(graph) == [], rtype

    @pytest.mark.parametrize("rtype", ["requires", "supports", "refutes"])
    def test_zero_strength_parent_is_not_common(self, rtype):
        """Test that a shared parent whose relations have strength 0 does not silence the group.

        A term of strength 0 multiplies the child's table by 1 - 0 = 1 whatever the parent's value, so
        the parent has no effect on the supports and does not correlate them.
        """
        graph = pile(2)
        graph.add_node(Node("p", base=0.5, sources=[ANCHOR]))
        for i in range(2):
            graph.add_relation(Relation(f"c{i}", rtype, "p", f"r{i}", strength=0.0))
        (finding,) = correlated_support(graph)
        assert finding.nodes == ("claim", "r0", "r1")

    def test_one_effective_relation_makes_a_common_parent(self):
        """Test that a parent with a zero-strength and an effective relation into each support still counts."""
        graph = pile(2)
        graph.add_node(Node("p", base=0.5, sources=[ANCHOR]))
        for i in range(2):
            graph.add_relation(Relation(f"z{i}", "requires", "p", f"r{i}", strength=0.0))
            graph.add_relation(Relation(f"c{i}", "requires", "p", f"r{i}", strength=Beta(1, 99)))
        assert correlated_support(graph) == []

    def test_min_supports(self):
        """Test that a group smaller than ``min_supports`` is not reported and one that reaches it is."""
        assert len(correlated_support(pile(2))) == 1
        assert correlated_support(pile(2), min_supports=3) == []
        assert correlated_support(pile(3), min_supports=3)[0].details["min_supports"] == 3.0
        assert correlated_support(pile(1)) == []

    @pytest.mark.parametrize("value", [1, 0, -2, True, 2.0, "2", None])
    def test_min_supports_is_validated(self, value):
        """Test that ``min_supports`` must be an integer of at least 2."""
        with pytest.raises(ValidationError, match="min_supports"):
            correlated_support(pile(2), min_supports=value)

    def test_different_documents_are_not_grouped(self):
        """Test that supports anchored in different documents, or in none, do not share a source."""
        graph = Graph()
        graph.add_node(Node("claim", base=0.0))
        for i, sources in enumerate(([SourceAnchor("a")], [SourceAnchor("b")], [])):
            graph.add_node(Node(f"r{i}", base=1.0, sources=sources))
            graph.add_relation(Relation(f"s{i}", "supports", f"r{i}", "claim", strength=0.2))
        assert correlated_support(graph) == []

    def test_locator_does_not_separate_a_document(self):
        """Test that two places in one document are one source."""
        graph = pile(1, SourceAnchor("paper", locator="p. 1"))
        graph.add_node(Node("r1", base=1.0, sources=[SourceAnchor("paper", locator="p. 9")]))
        graph.add_relation(Relation("s1", "supports", "r1", "claim", strength=0.2))
        (finding,) = correlated_support(graph)
        assert finding.id == "correlated-support:claim:document:paper"

    @pytest.mark.parametrize("annotation", ["derived_from", "authored_by"])
    def test_provenance_annotations(self, annotation):
        """Test that supports pointing at one node through a provenance annotation share a source."""
        graph = Graph()
        graph.add_node(Node("claim", base=0.0, sources=[ANCHOR]))
        graph.add_node(Node("origin", kind="dataset"))
        for i in range(2):
            graph.add_node(Node(f"r{i}", base=1.0))
            graph.add_relation(Relation(f"s{i}", "supports", f"r{i}", "claim", strength=0.2))
            graph.add_relation(Relation(f"a{i}", annotation, f"r{i}", "origin"))
        (finding,) = correlated_support(graph)
        assert finding.id == f"correlated-support:claim:{annotation}:origin"
        assert f"share {annotation} 'origin'" in finding.message

    def test_other_annotations_are_not_provenance(self):
        """Test that an annotation outside ``PROVENANCE_TYPES``, or one pointing into a support, shares nothing."""
        graph = Graph()
        graph.add_node(Node("claim", base=0.0))
        graph.add_node(Node("origin", kind="dataset"))
        for i in range(2):
            graph.add_node(Node(f"r{i}", base=1.0))
            graph.add_relation(Relation(f"s{i}", "supports", f"r{i}", "claim", strength=0.2))
            graph.add_relation(Relation(f"c{i}", "cites", f"r{i}", "origin"))
            graph.add_relation(Relation(f"d{i}", "derived_from", "origin", f"r{i}"))
        assert correlated_support(graph) == []

    def test_requires_and_refutes_parents_are_not_supports(self):
        """Test that only supports relations make up a group."""
        graph = Graph()
        graph.add_node(Node("claim", base=0.5))
        for i, rtype in enumerate(("requires", "refutes", "supports")):
            graph.add_node(Node(f"r{i}", base=0.9, sources=[ANCHOR]))
            graph.add_relation(Relation(f"s{i}", rtype, f"r{i}", "claim", strength=0.5))
        assert correlated_support(graph) == []

    def test_group_sharing_several_sources_is_reported_once(self):
        """Test that supports sharing a document and an author give one finding naming both."""
        graph = pile(2)
        graph.add_node(Node("alice", kind="person"))
        for i in range(2):
            graph.add_relation(Relation(f"by{i}", "authored_by", f"r{i}", "alice"))
        (finding,) = correlated_support(graph)
        assert finding.id == "correlated-support:claim:document:doi:10.0000/example"
        assert "share document 'doi:10.0000/example' and authored_by 'alice' but" in finding.message

    def test_one_finding_per_group(self):
        """Test that two distinct groups into one node are reported separately, in first-seen order."""
        graph = pile(2, SourceAnchor("a"))
        for i in (2, 3):
            graph.add_node(Node(f"r{i}", base=1.0, sources=[SourceAnchor("b")]))
            graph.add_relation(Relation(f"s{i}", "supports", f"r{i}", "claim", strength=0.2))
        assert [f.id for f in correlated_support(graph)] == [
            "correlated-support:claim:document:a",
            "correlated-support:claim:document:b",
        ]

    def test_merged_nodes_are_one_variable(self):
        """Test that equivalent nodes pool their sources and are named after the earliest of them."""
        graph = pile(1, SourceAnchor("paper"))
        graph.add_node(Node("x", base=1.0))
        graph.add_node(Node("x2", sources=[SourceAnchor("paper")]))
        graph.add_relation(Relation("eq", "equivalent", "x", "x2"))
        graph.add_relation(Relation("s1", "supports", "x2", "claim", strength=0.2))
        (finding,) = correlated_support(graph)
        assert finding.nodes == ("claim", "r0", "x")
        assert finding.relations == ("s0", "s1")
