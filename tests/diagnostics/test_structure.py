"""Structural diagnostics: missing parameters and unanchored variables."""

from __future__ import annotations

from credencegraph.core import Graph, Node, Relation, SourceAnchor
from credencegraph.diagnostics import MISSING_PARAMETER, UNANCHORED, missing_parameters, unanchored

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
