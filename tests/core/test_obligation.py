"""Tests for obligations, discharges, and their JSON form."""

from __future__ import annotations

import copy
import math

import pytest
from jsonschema import Draft202012Validator

from credencegraph.core import (
    DISCHARGE_KINDS,
    Discharge,
    Graph,
    Node,
    Obligation,
    Relation,
    SourceAnchor,
    ValidationError,
    graph_from_dict,
    graph_to_dict,
    json_schema,
)
from credencegraph.core.obligation import file_digest

DIGEST = file_digest(b"assert True\n")

# One valid discharge of each kind, using every field the kind takes.
DISCHARGES = {
    "formal": Discharge("formal", theorem="Paper.eq3_of_eq2"),
    "symbolic": Discharge("symbolic", script="eq3.py", digest=DIGEST, status=0),
    "numerical": Discharge("numerical", script="bias.py", digest=DIGEST, status=1, value=0.0, tolerance=0.01),
    "citation": Discharge("citation", anchor=SourceAnchor("paper.tex", "sec2", "the noise is Gaussian", None)),
    "judgement": Discharge("judgement"),
}


def graph_with(discharge: Discharge | None, relation_type: str = "supports") -> Graph:
    """Return ``a -> b`` whose one relation carries an obligation with ``discharge``."""
    g = Graph()
    g.add_node(Node("a", base=0.5))
    g.add_node(Node("b", base=0.1))
    strength = 0.8 if relation_type == "supports" else None
    g.add_relation(Relation("r", relation_type, "a", "b", strength=strength, obligation=Obligation("holds", discharge)))
    return g


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    """Provide a validator for the published schema."""
    return Draft202012Validator(json_schema())


def test_every_kind_is_listed():
    """Test that the five kinds of the model are exactly the ones accepted."""
    assert set(DISCHARGE_KINDS) == set(DISCHARGES) == {"formal", "symbolic", "numerical", "citation", "judgement"}


class TestDischarge:
    """Each kind takes exactly its own fields."""

    def test_unknown_kind(self):
        """Test that a kind outside the five is refused."""
        with pytest.raises(ValidationError, match="kind must be one of"):
            Discharge("hunch")

    @pytest.mark.parametrize(
        ("kind", "fields", "missing"),
        [
            ("formal", {}, "theorem"),
            ("symbolic", {}, "script"),
            ("numerical", {"script": "x.py", "value": 0.0}, "tolerance"),
            ("numerical", {"script": "x.py", "tolerance": 0.1}, "value"),
            ("numerical", {"value": 0.0, "tolerance": 0.1}, "script"),
            ("citation", {}, "anchor"),
        ],
    )
    def test_required_fields(self, kind, fields, missing):
        """Test that each kind's required fields are required."""
        with pytest.raises(ValidationError, match=f"a {kind} discharge needs {missing}"):
            Discharge(kind, **fields)

    @pytest.mark.parametrize(
        ("kind", "fields", "extra"),
        [
            ("judgement", {"theorem": "t"}, "theorem"),
            ("formal", {"theorem": "t", "script": "x.py"}, "script"),
            ("symbolic", {"script": "x.py", "tolerance": 0.1}, "tolerance"),
            ("citation", {"anchor": SourceAnchor("d", quote="q"), "digest": DIGEST, "status": 0}, "digest, status"),
        ],
    )
    def test_foreign_fields(self, kind, fields, extra):
        """Test that a field of another kind is refused."""
        with pytest.raises(ValidationError, match=f"a {kind} discharge takes no {extra}"):
            Discharge(kind, **fields)

    def test_script_must_be_python(self):
        """Test that a script discharge names a Python script."""
        with pytest.raises(ValidationError, match=r"ending in \.py"):
            Discharge("symbolic", script="check.sh")

    @pytest.mark.parametrize("fields", [{"digest": DIGEST}, {"status": 0}])
    def test_digest_and_status_come_together(self, fields):
        """Test that a recorded run has both its digest and its status."""
        with pytest.raises(ValidationError, match="both the digest and the status"):
            Discharge("symbolic", script="x.py", **fields)

    @pytest.mark.parametrize("digest", ["sha256:abc", "md5:" + "0" * 64, "sha256:" + "A" * 64, 7])
    def test_digest_format(self, digest):
        """Test that a digest is ``sha256:`` and 64 lowercase hex digits."""
        with pytest.raises(ValidationError, match=r"Discharge\.digest"):
            Discharge("symbolic", script="x.py", digest=digest, status=0)

    @pytest.mark.parametrize("status", [True, 0.0, "0"])
    def test_status_is_an_integer(self, status):
        """Test that an exit status is an integer, not a bool, float or string."""
        with pytest.raises(ValidationError, match=r"Discharge\.status"):
            Discharge("symbolic", script="x.py", digest=DIGEST, status=status)

    @pytest.mark.parametrize("tolerance", [0, -0.1, math.inf, math.nan, True, "0.1"])
    def test_tolerance_is_positive_and_finite(self, tolerance):
        """Test that a tolerance is a finite number above zero."""
        with pytest.raises(ValidationError, match=r"Discharge\.tolerance"):
            Discharge("numerical", script="x.py", value=0.0, tolerance=tolerance)

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, True, "0.1"])
    def test_value_is_finite(self, value):
        """Test that the expected value of a numerical check is a finite number."""
        with pytest.raises(ValidationError, match=r"Discharge\.value must be a finite number"):
            Discharge("numerical", script="x.py", value=value, tolerance=0.1)

    def test_value_belongs_to_numerical_only(self):
        """Test that a symbolic discharge, which prints no result, takes no value."""
        with pytest.raises(ValidationError, match="a symbolic discharge takes no value"):
            Discharge("symbolic", script="x.py", value=0.0)

    def test_citation_needs_a_quote(self):
        """Test that a citation anchors a quote, not only a document."""
        with pytest.raises(ValidationError, match="anchor with a quote"):
            Discharge("citation", anchor=SourceAnchor("paper.tex", "sec2"))

    def test_citation_anchor_type(self):
        """Test that a citation's anchor is a ``SourceAnchor``."""
        with pytest.raises(ValidationError, match="must be a SourceAnchor"):
            Discharge("citation", anchor="paper.tex")


class TestObligation:
    """An obligation states its condition and sits only on inferential relations."""

    def test_statement_required(self):
        """Test that an obligation states what must hold."""
        with pytest.raises(ValidationError, match=r"Obligation\.statement"):
            Obligation("")

    def test_discharge_type(self):
        """Test that a discharge must be a ``Discharge``."""
        with pytest.raises(ValidationError, match="must be a Discharge"):
            Obligation("holds", "judgement")

    def test_relation_obligation_type(self):
        """Test that a relation's obligation must be an ``Obligation``."""
        with pytest.raises(ValidationError, match="must be an Obligation"):
            Relation("r", "supports", "a", "b", strength=0.5, obligation="holds")

    def test_annotation_cannot_carry_one(self):
        """Test that an annotation relation, which inference ignores, carries no obligation."""
        with pytest.raises(ValidationError, match="annotation; it must not have an obligation"):
            Relation("r", "authored_by", "a", "b", obligation=Obligation("holds"))

    @pytest.mark.parametrize("relation_type", ["requires", "supports", "refutes", "equivalent", "exclusive"])
    def test_every_inferential_type_can_carry_one(self, relation_type):
        """Test that any inferential relation may carry an obligation."""
        strength = 0.5 if relation_type in {"requires", "supports", "refutes"} else None
        relation = Relation("r", relation_type, "a", "b", strength=strength, obligation=Obligation("holds"))
        assert relation.obligation == Obligation("holds")


class TestSerialization:
    """Obligations round-trip, and a relation without one is written as before."""

    @pytest.mark.parametrize("kind", sorted(DISCHARGES))
    def test_round_trip(self, kind):
        """Test that each kind of discharge survives a round trip unchanged."""
        g = graph_with(DISCHARGES[kind])
        assert graph_from_dict(graph_to_dict(g)) == g

    def test_open_obligation_round_trips(self):
        """Test that an obligation with no discharge round-trips as ``discharge: null``."""
        g = graph_with(None)
        data = graph_to_dict(g)
        assert data["relations"][0]["obligation"] == {"statement": "holds", "discharge": None}
        assert graph_from_dict(data) == g

    def test_only_given_fields_are_written(self):
        """Test that a discharge is written with its kind and the fields it has, nothing else."""
        data = graph_to_dict(graph_with(Discharge("numerical", script="bias.py", value=0, tolerance=0.01)))
        assert data["relations"][0]["obligation"]["discharge"] == {
            "kind": "numerical",
            "script": "bias.py",
            "value": 0.0,
            "tolerance": 0.01,
        }

    def test_relation_without_obligation_has_no_field(self):
        """Test that a relation without an obligation is written exactly as before obligations existed."""
        g = Graph()
        g.add_node(Node("a", base=0.5))
        g.add_node(Node("b", base=0.1))
        g.add_relation(Relation("r", "supports", "a", "b", strength=0.8))
        assert "obligation" not in graph_to_dict(g)["relations"][0]

    @pytest.mark.parametrize(
        ("patch", "match"),
        [
            ({"obligation": {"statement": "s", "extra": 1}}, "unknown field"),
            ({"obligation": {"discharge": None}}, "missing required field"),
            ({"obligation": {"statement": "s", "discharge": {"kind": "hunch"}}}, "kind must be one of"),
            ({"obligation": {"statement": "s", "discharge": {"kind": "judgement", "x": 1}}}, "unknown field"),
            ({"obligation": {"statement": "s", "discharge": {"theorem": "t"}}}, "missing required field"),
            (
                {"obligation": {"statement": "s", "discharge": {"kind": "citation", "anchor": {"quote": "q"}}}},
                r"discharge\.anchor is missing required field",
            ),
        ],
    )
    def test_malformed_obligations_are_located(self, patch, match):
        """Test that a malformed obligation is refused, with its place in the document."""
        data = graph_to_dict(graph_with(None))
        data["relations"][0].update(patch)
        with pytest.raises(ValidationError, match=match) as info:
            graph_from_dict(data)
        assert str(info.value).startswith("relations[0]")


class TestSchema:
    """The schema accepts what the loader accepts and refuses the malformed shapes it can express."""

    @pytest.mark.parametrize("kind", [*sorted(DISCHARGES), None])
    def test_dumped_obligations_validate(self, validator, kind):
        """Test that every kind of discharge, and an open obligation, is valid under the schema."""
        validator.validate(graph_to_dict(graph_with(DISCHARGES.get(kind))))

    def test_annotation_obligation_fails_both(self, validator):
        """Test that an obligation on an annotation fails the schema and the loader alike."""
        data = graph_to_dict(graph_with(None))
        data["relations"][0].update(type="cites", strength=None)
        assert not validator.is_valid(data)
        with pytest.raises(ValidationError, match="annotation; it must not have an obligation"):
            graph_from_dict(data)

    @pytest.mark.parametrize(
        "discharge",
        [
            {"kind": "formal"},
            {"kind": "symbolic", "script": "x.sh"},
            {"kind": "symbolic", "script": "x.py", "digest": DIGEST},
            {"kind": "symbolic", "script": "x.py", "digest": "sha256:0", "status": 0},
            {"kind": "numerical", "script": "x.py", "value": 0},
            {"kind": "numerical", "script": "x.py", "tolerance": 0.1},
            {"kind": "numerical", "script": "x.py", "value": 0, "tolerance": 0},
            {"kind": "citation", "anchor": {"document": "d"}},
            {"kind": "judgement", "theorem": "t"},
            {"kind": "hunch"},
        ],
    )
    def test_malformed_discharges_fail_both(self, validator, discharge):
        """Test that each malformed discharge fails the schema and the loader alike."""
        data = graph_to_dict(graph_with(None))
        data["relations"][0]["obligation"]["discharge"] = copy.deepcopy(discharge)
        assert not validator.is_valid(data)
        with pytest.raises(ValidationError):
            graph_from_dict(data)
