"""Tests for versioned JSON I/O: round trips and document validation."""

from __future__ import annotations

import copy
import json
import math

import pytest

from credencegraph.core import (
    Beta,
    CycleError,
    Graph,
    Node,
    Point,
    Relation,
    SourceAnchor,
    ValidationError,
    dump,
    dumps,
    graph_from_dict,
    graph_to_dict,
    load,
    loads,
)


def sample_graph() -> Graph:
    """Build a graph exercising every field, every relation type and both credence kinds."""
    g = Graph()
    g.add_node(
        Node(
            "h",
            statement="The half-life of X is 5.2 d.",
            sources=[SourceAnchor("doi:10.1/x", locator="p. 4", quote="t1/2 = 5.2 d", digest="sha256:ab")],
            base=Beta(3, 7),
            stated=0.95,
            attributes={"note": "core claim", "tags": ["a", 1, 2.5, None, True], "nested": {"k": {"z": []}}},
        )
    )
    g.add_node(Node("e", kind="observation", base=0.9))
    g.add_node(Node("f", base=Point(0.0)))
    g.add_node(Node("p", kind="person"))
    g.add_node(Node("h2", base=1.0))
    g.add_relation(Relation("r1", "supports", "e", "h", strength=Beta(8, 2), attributes={"w": 1}))
    g.add_relation(Relation("r2", "refutes", "f", "h", strength=0.05))
    g.add_relation(Relation("r3", "requires", "h2", "h", strength=1))
    g.add_relation(Relation("r4", "equivalent", "h", "h2"))
    g.add_relation(Relation("r5", "exclusive", "e", "f"))
    g.add_relation(Relation("r6", "authored_by", "h", "p", attributes={"role": "first"}))
    g.add_relation(Relation("r7", "cites", "h", "h"))
    return g


class TestRoundTrip:
    """A graph survives serialisation unchanged."""

    def test_dict_round_trip(self):
        """Test to_dict then from_dict reproduces the graph."""
        g = sample_graph()
        assert graph_from_dict(graph_to_dict(g)) == g

    def test_json_string_round_trip(self):
        """Test dumps then loads reproduces the graph, for both indent settings."""
        g = sample_graph()
        assert loads(dumps(g)) == g
        assert loads(dumps(g, indent=None)) == g
        assert "\n" not in dumps(g, indent=None)

    def test_stored_attributes_cannot_smuggle_a_non_finite_value(self):
        """Test that a non-finite value cannot be written into stored attributes and reach the output."""
        g = sample_graph()
        with pytest.raises(TypeError):
            g.nodes["h"].attributes["tags"] += (math.nan,)
        with pytest.raises(TypeError):
            g.relations["r1"].attributes["w"] = math.inf
        assert loads(dumps(g)) == g

    def test_file_round_trip(self, tmp_path):
        """Test dump then load through a file."""
        g = sample_graph()
        path = tmp_path / "g.json"
        dump(g, path)
        assert load(path) == g
        assert path.read_text(encoding="utf-8").endswith("}\n")

    def test_serialisation_is_stable(self):
        """Test that serialising a loaded graph gives the same text."""
        text = dumps(sample_graph())
        assert dumps(loads(text)) == text

    def test_order_is_preserved(self):
        """Test that node and relation order survives, since it fixes which cycle is named."""
        g = loads(dumps(sample_graph()))
        assert list(g.nodes) == ["h", "e", "f", "p", "h2"]
        assert list(g.relations) == ["r1", "r2", "r3", "r4", "r5", "r6", "r7"]

    def test_annotation_relations_are_stored_and_round_tripped(self):
        """Test that annotations keep type, endpoints and attributes."""
        rel = loads(dumps(sample_graph())).relations["r6"]
        assert (rel.type, rel.source, rel.target, rel.strength) == ("authored_by", "h", "p", None)
        assert rel.attributes == {"role": "first"}
        assert not rel.is_inferential

    def test_credence_encoding(self):
        """Test that a Point is a bare number and a Beta an alpha/beta object."""
        doc = graph_to_dict(sample_graph())
        by_id = {n["id"]: n for n in doc["nodes"]}
        assert by_id["h"]["base"] == {"alpha": 3.0, "beta": 7.0}
        assert by_id["h"]["stated"] == 0.95
        assert by_id["e"]["base"] == 0.9
        assert by_id["p"]["base"] is None

    def test_floats_round_trip_exactly(self):
        """Test that no precision is lost through JSON."""
        g = Graph()
        g.add_node(Node("a", base=Beta(0.1 + 0.2, 1 / 3), stated=1e-300))
        assert loads(dumps(g)) == g

    def test_document_header(self):
        """Test the format and version header."""
        doc = graph_to_dict(Graph())
        assert doc == {"format": "credencegraph", "version": 1, "nodes": [], "relations": []}

    def test_minimal_node_and_relation_objects_use_defaults(self):
        """Test that omitted optional fields take their defaults."""
        doc = {
            "format": "credencegraph",
            "version": 1,
            "nodes": [{"id": "a"}, {"id": "b"}],
            "relations": [{"id": "r", "type": "cites", "source": "a", "target": "b"}],
        }
        g = graph_from_dict(doc)
        assert g.nodes["a"] == Node("a")
        assert g.relations["r"] == Relation("r", "cites", "a", "b")

    def test_loaded_attributes_are_detached_from_the_document(self):
        """Test that mutating the parsed document afterwards does not change the graph."""
        doc = graph_to_dict(sample_graph())
        g = graph_from_dict(doc)
        doc["nodes"][0]["attributes"]["tags"].append("late")
        assert "late" not in g.nodes["h"].attributes["tags"]


def valid_doc() -> dict:
    """Return a small valid document for mutation in error tests."""
    return {
        "format": "credencegraph",
        "version": 1,
        "nodes": [{"id": "a", "base": 0.5}, {"id": "b", "base": 0.5}],
        "relations": [{"id": "r", "type": "supports", "source": "a", "target": "b", "strength": 0.5}],
    }


def mutated(mutate) -> dict:
    """Apply ``mutate`` to a fresh valid document and return it."""
    doc = valid_doc()
    mutate(doc)
    return doc


class TestDocumentErrors:
    """Each malformed or rule-breaking document is rejected with a located message."""

    @pytest.mark.parametrize("data", [[], "graph", 3, None])
    def test_document_must_be_an_object(self, data):
        """Test that the top level must be a JSON object."""
        with pytest.raises(ValidationError, match="document must be an object"):
            graph_from_dict(data)

    @pytest.mark.parametrize("key", ["format", "version", "nodes", "relations"])
    def test_missing_top_level_field(self, key):
        """Test that each header and body field is required."""
        doc = valid_doc()
        del doc[key]
        with pytest.raises(ValidationError, match=f"missing required field.*{key}"):
            graph_from_dict(doc)

    def test_unknown_top_level_field(self):
        """Test that unknown top-level fields are rejected rather than dropped."""
        with pytest.raises(ValidationError, match=r"unknown field.*extra"):
            graph_from_dict({**valid_doc(), "extra": 1})

    @pytest.mark.parametrize("fmt", ["other", None, 1])
    def test_wrong_format(self, fmt):
        """Test that the format tag must be exactly 'credencegraph'."""
        with pytest.raises(ValidationError, match="format must be 'credencegraph'"):
            graph_from_dict({**valid_doc(), "format": fmt})

    @pytest.mark.parametrize("version", [0, 2, "1", 1.5, True, None])
    def test_unsupported_version(self, version):
        """Test that only version 1 is read, and the message names what was found."""
        with pytest.raises(ValidationError, match=r"unsupported credencegraph version"):
            graph_from_dict({**valid_doc(), "version": version})

    def test_version_error_names_the_version(self):
        """Test that the message carries the offending version."""
        with pytest.raises(ValidationError, match="version 7; this release reads version 1"):
            graph_from_dict({**valid_doc(), "version": 7})

    @pytest.mark.parametrize("key", ["nodes", "relations"])
    def test_collections_must_be_arrays(self, key):
        """Test that nodes and relations are arrays."""
        with pytest.raises(ValidationError, match=f"{key} must be an array"):
            graph_from_dict({**valid_doc(), key: {}})

    def test_node_must_be_an_object(self):
        """Test that each node entry is an object, with its index in the message."""
        with pytest.raises(ValidationError, match=r"nodes\[1\] must be an object"):
            graph_from_dict(mutated(lambda d: d["nodes"].__setitem__(1, "b")))

    def test_node_needs_an_id(self):
        """Test that a node without id is rejected."""
        with pytest.raises(ValidationError, match=r"nodes\[0\] is missing required field.*id"):
            graph_from_dict(mutated(lambda d: d["nodes"].__setitem__(0, {"base": 0.5})))

    def test_unknown_node_field(self):
        """Test that unknown node fields are rejected."""
        with pytest.raises(ValidationError, match=r"nodes\[0\] has unknown field.*color"):
            graph_from_dict(mutated(lambda d: d["nodes"][0].update(color="red")))

    def test_duplicate_node_id(self):
        """Test that duplicate node ids in a document are rejected with the node's index."""
        with pytest.raises(ValidationError, match=r"nodes\[1\]: duplicate node id 'a'"):
            graph_from_dict(mutated(lambda d: d["nodes"][1].update(id="a")))

    def test_duplicate_relation_id(self):
        """Test that duplicate relation ids are rejected."""
        doc = mutated(lambda d: d["relations"].append(dict(d["relations"][0])))
        with pytest.raises(ValidationError, match=r"relations\[1\]: duplicate relation id 'r'"):
            graph_from_dict(doc)

    @pytest.mark.parametrize("end", ["source", "target"])
    def test_dangling_relation_endpoint(self, end):
        """Test that a relation naming a missing node is rejected."""
        doc = mutated(lambda d: d["relations"][0].update({end: "ghost"}))
        with pytest.raises(ValidationError, match=rf"relations\[0\].*{end} 'ghost'"):
            graph_from_dict(doc)

    @pytest.mark.parametrize(
        "bad", [1.5, -0.1, "0.5", True, {"alpha": 1}, {"alpha": 0, "beta": 1}, {"alpha": 1, "beta": 1, "x": 1}]
    )
    def test_bad_node_credence(self, bad):
        """Test that malformed credences on nodes are rejected, located at the field."""
        with pytest.raises(ValidationError, match=r"nodes\[0\]"):
            graph_from_dict(mutated(lambda d: d["nodes"][0].update(base=bad)))
        with pytest.raises(ValidationError, match=r"nodes\[0\]"):
            graph_from_dict(mutated(lambda d: d["nodes"][0].update(stated=bad)))

    def test_bad_beta_shape_in_relation(self):
        """Test that a malformed Beta strength is located at the relation."""
        doc = mutated(lambda d: d["relations"][0].update(strength={"alpha": 1, "beta": -2}))
        with pytest.raises(ValidationError, match=r"relations\[0\].*Beta\.beta"):
            graph_from_dict(doc)

    def test_missing_strength_on_inferential_relation(self):
        """Test that supports without a strength is rejected."""
        doc = mutated(lambda d: d["relations"][0].pop("strength"))
        with pytest.raises(ValidationError, match=r"relations\[0\].*requires a strength"):
            graph_from_dict(doc)

    def test_strength_on_annotation(self):
        """Test that an annotation with a strength is rejected."""
        doc = mutated(lambda d: d["relations"][0].update(type="cites"))
        with pytest.raises(ValidationError, match=r"relations\[0\].*annotation.*must not have a strength"):
            graph_from_dict(doc)

    @pytest.mark.parametrize("key", ["id", "type", "source", "target"])
    def test_relation_missing_required_field(self, key):
        """Test that each identifying field of a relation is required."""
        doc = mutated(lambda d: d["relations"][0].pop(key))
        with pytest.raises(ValidationError, match=rf"relations\[0\] is missing required field.*{key}"):
            graph_from_dict(doc)

    def test_unknown_relation_field(self):
        """Test that unknown relation fields are rejected."""
        with pytest.raises(ValidationError, match=r"relations\[0\] has unknown field.*weight"):
            graph_from_dict(mutated(lambda d: d["relations"][0].update(weight=1)))

    def test_sources_must_be_an_array_of_anchors(self):
        """Test that sources is an array and each anchor an object with a document."""
        with pytest.raises(ValidationError, match=r"nodes\[0\]\.sources must be an array"):
            graph_from_dict(mutated(lambda d: d["nodes"][0].update(sources="doc")))
        with pytest.raises(ValidationError, match=r"nodes\[0\]\.sources\[0\] must be an object"):
            graph_from_dict(mutated(lambda d: d["nodes"][0].update(sources=["doc"])))
        with pytest.raises(ValidationError, match=r"nodes\[0\]\.sources\[0\] is missing required field.*document"):
            graph_from_dict(mutated(lambda d: d["nodes"][0].update(sources=[{"quote": "q"}])))
        with pytest.raises(ValidationError, match=r"nodes\[0\]\.sources\[0\] has unknown field.*page"):
            graph_from_dict(mutated(lambda d: d["nodes"][0].update(sources=[{"document": "d", "page": 3}])))

    def test_attributes_must_be_an_object(self):
        """Test that non-object attributes are rejected."""
        with pytest.raises(ValidationError, match=r"nodes\[0\].*attributes must be a dict"):
            graph_from_dict(mutated(lambda d: d["nodes"][0].update(attributes=[1])))

    def test_cycle_in_document_is_rejected_and_named(self):
        """Test that a cyclic file is rejected on load, with the cycle in the message."""
        doc = mutated(
            lambda d: d["relations"].append(
                {"id": "back", "type": "refutes", "source": "b", "target": "a", "strength": 0.1}
            )
        )
        with pytest.raises(CycleError, match=r"b --refutes\[back\]--> a --supports\[r\]--> b") as info:
            graph_from_dict(doc)
        assert info.value.cycle == ("b", "a", "b")

    def test_cycle_only_in_annotations_is_accepted(self):
        """Test that an annotation loop in a file is not a cycle."""
        doc = mutated(lambda d: d["relations"].append({"id": "back", "type": "cites", "source": "b", "target": "a"}))
        assert len(graph_from_dict(doc).relations) == 2

    @pytest.mark.parametrize("text", ["", "{", "[1,", "nope"])
    def test_invalid_json(self, text):
        """Test that text that is not JSON raises ValidationError."""
        with pytest.raises(ValidationError, match="not valid JSON"):
            loads(text)

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_constants_are_rejected(self, constant):
        """Test that the non-standard NaN and Infinity tokens are refused."""
        text = json.dumps(valid_doc()).replace('"base": 0.5', f'"base": {constant}', 1)
        with pytest.raises(ValidationError, match="not valid JSON"):
            loads(text)

    def test_writer_never_emits_non_finite_numbers(self):
        """Test that a value smuggled past validation cannot produce invalid JSON."""
        g = Graph()
        g.add_node(Node("a", base=0.5))
        object.__setattr__(g.nodes["a"].base, "p", float("nan"))
        with pytest.raises(ValueError, match="Out of range float"):
            dumps(g)

    def test_load_reports_missing_file(self, tmp_path):
        """Test that a missing file raises the ordinary OSError."""
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "absent.json")

    def test_from_dict_does_not_mutate_its_input(self):
        """Test that parsing leaves the document untouched."""
        doc = graph_to_dict(sample_graph())
        before = copy.deepcopy(doc)
        graph_from_dict(doc)
        assert doc == before
