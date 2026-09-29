"""Tests for Graph writes: duplicates, dangling endpoints and cycle rejection."""

from __future__ import annotations

import pytest

from credencegraph.core import CycleError, Graph, Node, Relation, ValidationError


def build(*node_ids: str) -> Graph:
    """Build a graph of bare nodes."""
    g = Graph()
    for node_id in node_ids:
        g.add_node(Node(node_id, base=0.5))
    return g


def rel(rel_id: str, rtype: str, source: str, target: str) -> Relation:
    """Build a relation, with a strength when the type needs one."""
    strength = 0.5 if rtype in {"requires", "supports", "refutes"} else None
    return Relation(rel_id, rtype, source, target, strength=strength)


class TestWrites:
    """Duplicate ids, dangling endpoints and atomicity."""

    def test_duplicate_node_id(self):
        """Test that a second node with the same id is rejected."""
        g = build("a")
        with pytest.raises(ValidationError, match="duplicate node id 'a'"):
            g.add_node(Node("a"))
        assert len(g) == 1

    def test_duplicate_relation_id(self):
        """Test that a second relation with the same id is rejected."""
        g = build("a", "b", "c")
        g.add_relation(rel("r", "supports", "a", "b"))
        with pytest.raises(ValidationError, match="duplicate relation id 'r'"):
            g.add_relation(rel("r", "cites", "b", "c"))
        assert list(g.relations) == ["r"]

    @pytest.mark.parametrize(
        ("source", "target", "missing", "role"), [("x", "b", "x", "source"), ("a", "y", "y", "target")]
    )
    def test_dangling_endpoint(self, source, target, missing, role):
        """Test that a relation must join existing nodes, for annotations too."""
        g = build("a", "b")
        for rtype in ("supports", "cites"):
            with pytest.raises(ValidationError, match=f"{role} '{missing}'"):
                g.add_relation(rel("r", rtype, source, target))
        assert not g.relations

    def test_type_checks(self):
        """Test that only Node and Relation objects can be added."""
        g = Graph()
        with pytest.raises(ValidationError, match="expects a Node"):
            g.add_node({"id": "a"})
        with pytest.raises(ValidationError, match="expects a Relation"):
            g.add_relation({"id": "r"})

    def test_views_are_read_only_and_ordered(self):
        """Test the read accessors."""
        g = build("b", "a")
        g.add_relation(rel("r", "cites", "a", "b"))
        assert list(g.nodes) == ["b", "a"]
        assert [n.id for n in g] == ["b", "a"]
        assert "a" in g
        assert "z" not in g
        with pytest.raises(TypeError):
            g.nodes["c"] = Node("c")  # type: ignore[index]
        with pytest.raises(TypeError):
            g.relations["q"] = rel("q", "cites", "a", "b")  # type: ignore[index]
        assert repr(g) == "Graph(nodes=2, relations=1)"

    def test_equality_ignores_order(self):
        """Test that graphs compare by content."""
        assert build("a", "b") == build("b", "a")
        with_relation = build("a", "b")
        with_relation.add_relation(rel("r", "cites", "a", "b"))
        assert with_relation != build("b", "a")
        assert build("a") != build("b")
        assert build("a") != "graph"


class TestCycles:
    """Write-time cycle rejection in the requires/supports/refutes subgraph."""

    @pytest.mark.parametrize("rtype", ["requires", "supports", "refutes"])
    def test_self_loop(self, rtype):
        """Test that a self-loop is a cycle of length one."""
        g = build("a")
        with pytest.raises(CycleError) as info:
            g.add_relation(rel("r1", rtype, "a", "a"))
        assert info.value.cycle == ("a", "a")
        assert info.value.relations == ("r1",)
        assert f"a --{rtype}[r1]--> a" in str(info.value)

    @pytest.mark.parametrize("rtype", ["requires", "supports", "refutes"])
    def test_two_cycle_names_both_nodes_and_relations(self, rtype):
        """Test that the message names the nodes and relation ids around the cycle."""
        g = build("a", "b")
        g.add_relation(rel("r1", rtype, "a", "b"))
        with pytest.raises(CycleError) as info:
            g.add_relation(rel("r2", rtype, "b", "a"))
        assert info.value.cycle == ("b", "a", "b")
        assert info.value.relations == ("r2", "r1")
        assert str(info.value) == f"relation 'r2' would create a cycle: b --{rtype}[r2]--> a --{rtype}[r1]--> b"

    def test_long_cycle_is_named_in_full(self):
        """Test a five-node cycle, with the closing relation added last."""
        ids = ["n0", "n1", "n2", "n3", "n4"]
        g = build(*ids)
        for i in range(4):
            g.add_relation(rel(f"r{i}", "supports", ids[i], ids[i + 1]))
        with pytest.raises(CycleError) as info:
            g.add_relation(rel("close", "supports", "n4", "n0"))
        assert info.value.cycle == ("n4", "n0", "n1", "n2", "n3", "n4")
        assert info.value.relations == ("close", "r0", "r1", "r2", "r3")
        for name in [*ids, "r0", "r1", "r2", "r3", "close"]:
            assert name in str(info.value)

    def test_cycle_mixes_the_three_types(self):
        """Test that requires, supports and refutes together form one subgraph."""
        g = build("a", "b", "c")
        g.add_relation(rel("r1", "requires", "a", "b"))
        g.add_relation(rel("r2", "supports", "b", "c"))
        with pytest.raises(CycleError) as info:
            g.add_relation(rel("r3", "refutes", "c", "a"))
        assert info.value.cycle == ("c", "a", "b", "c")
        assert "--refutes[r3]--> a --requires[r1]--> b --supports[r2]--> c" in str(info.value)

    def test_found_path_is_a_real_path_not_a_wandering_one(self):
        """Test that the reported cycle skips dead-end branches explored on the way."""
        g = build("a", "b", "dead", "c")
        g.add_relation(rel("dead_end", "supports", "b", "dead"))
        g.add_relation(rel("bc", "supports", "b", "c"))
        g.add_relation(rel("ab", "supports", "a", "b"))
        with pytest.raises(CycleError) as info:
            g.add_relation(rel("ca", "supports", "c", "a"))
        assert info.value.cycle == ("c", "a", "b", "c")
        assert info.value.relations == ("ca", "ab", "bc")

    def test_diamond_is_not_a_cycle(self):
        """Test that converging paths are accepted."""
        g = build("a", "b", "c", "d")
        for i, (s, t) in enumerate([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]):
            g.add_relation(rel(f"r{i}", "supports", s, t))
        assert len(g.relations) == 4

    def test_search_through_a_downstream_diamond_finds_no_cycle(self):
        """Test that a write into a diamond, which the search reaches twice, is accepted."""
        g = build("z", "a", "b", "c", "d", "e")
        for i, (s, t) in enumerate([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "e")]):
            g.add_relation(rel(f"r{i}", "supports", s, t))
        g.add_relation(rel("in", "supports", "z", "a"))
        assert "in" in g.relations

    def test_parallel_relations_are_not_a_cycle(self):
        """Test that two relations in the same direction are accepted."""
        g = build("a", "b")
        g.add_relation(rel("r1", "supports", "a", "b"))
        g.add_relation(rel("r2", "refutes", "a", "b"))
        assert len(g.relations) == 2

    def test_rejected_write_leaves_graph_unchanged_and_usable(self):
        """Test that a rejected relation is not stored and later writes still work."""
        g = build("a", "b", "c")
        g.add_relation(rel("r1", "supports", "a", "b"))
        with pytest.raises(CycleError):
            g.add_relation(rel("r2", "supports", "b", "a"))
        assert list(g.relations) == ["r1"]
        g.add_relation(rel("r2", "supports", "b", "c"))
        with pytest.raises(CycleError) as info:
            g.add_relation(rel("r3", "supports", "c", "a"))
        assert info.value.relations == ("r3", "r1", "r2")

    def test_cycle_error_is_a_validation_error(self):
        """Test that callers catching ValidationError also catch cycles."""
        g = build("a")
        with pytest.raises(ValidationError):
            g.add_relation(rel("r", "supports", "a", "a"))

    @pytest.mark.parametrize("rtype", ["cites", "authored_by", "derived_from"])
    def test_annotations_never_form_a_cycle(self, rtype):
        """Test that annotation relations may loop, and do not hide or create cycles."""
        g = build("a", "b")
        g.add_relation(rel("r1", rtype, "a", "b"))
        g.add_relation(rel("r2", rtype, "b", "a"))
        g.add_relation(rel("r3", rtype, "a", "a"))
        g.add_relation(rel("s1", "supports", "a", "b"))
        assert len(g.relations) == 4

    def test_annotation_path_does_not_complete_an_inferential_cycle(self):
        """Test that a support edge plus an annotation edge back is not a cycle."""
        g = build("a", "b")
        g.add_relation(rel("cite", "cites", "b", "a"))
        g.add_relation(rel("s", "supports", "a", "b"))
        assert len(g.relations) == 2

    @pytest.mark.parametrize("rtype", ["equivalent", "exclusive"])
    def test_equivalent_and_exclusive_are_outside_the_acyclic_subgraph(self, rtype):
        """Test that symmetric relations do not take part in the cycle check."""
        g = build("a", "b")
        g.add_relation(rel("r1", rtype, "a", "b"))
        g.add_relation(rel("r2", rtype, "b", "a"))
        g.add_relation(rel("s1", "supports", "a", "b"))
        assert len(g.relations) == 3

    def test_deep_chain_does_not_recurse(self):
        """Test that cycle detection over a 5000-node chain does not hit the recursion limit."""
        n = 5000
        g = build(*(f"n{i}" for i in range(n)))
        for i in range(n - 1):
            g.add_relation(rel(f"r{i}", "supports", f"n{i}", f"n{i + 1}"))
        with pytest.raises(CycleError) as info:
            g.add_relation(rel("close", "supports", f"n{n - 1}", "n0"))
        assert len(info.value.cycle) == n + 1
