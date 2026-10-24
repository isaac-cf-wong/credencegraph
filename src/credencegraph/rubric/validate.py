"""Checking a graph against a rubric, level by level, and reporting how far classification has got.

The levels nest: ``anchored`` checks that the graph is structurally sound, ``typed`` that every
node has a type and what the type requires, with scopes and units that agree with what it rests
on, and ``assessed`` that every node that needs an assessment has one and that the graph
compiles. Every violation is reported, as a ``Finding`` whose ``diagnostic`` is the violation's
code and whose id is ``<code>:<node id>``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from credencegraph.core.credence import Credence, format_credence
from credencegraph.core.graph import Graph
from credencegraph.core.node import Node
from credencegraph.core.relation import REQUIRES, SUPPORTS, Relation
from credencegraph.diagnostics.records import Finding
from credencegraph.diagnostics.structure import missing_parameters
from credencegraph.rubric.faithfulness import unfaithful
from credencegraph.rubric.model import (
    ANALYST,
    BOOL,
    BUILTIN_TYPES,
    COMPOUND,
    DOCUMENT,
    ENUM,
    EQREF,
    EVIDENCE,
    NUMBER,
    ORIGINS,
    QUANTITY,
    SCOPE,
    UNASSIGNED,
    VERBATIM,
    Check,
    FormField,
    Rubric,
)
from credencegraph.rubric.scope import UNBOUNDED, Interval, contains, intersection, is_empty, show
from credencegraph.rubric.text import text_digest
from credencegraph.rubric.units import agree
from credencegraph.semantics.compiler import compile_graph
from credencegraph.semantics.errors import CompileError

ANCHORED = "anchored"
TYPED = "typed"
ASSESSED = "assessed"
LEVELS = (ANCHORED, TYPED, ASSESSED)

# Annotation relations with a fixed meaning in a graph built from a document.
PART_OF = "part-of"
CITES = "cites"
ASSESSES = "assesses"

# Assessment verdicts.
HOLDS = "holds"
FAILS = "fails"
UNDETERMINED = "undetermined"
NOT_ASSESSED = "not_assessed"
VERDICTS = (HOLDS, FAILS, UNDETERMINED, NOT_ASSESSED)

# Violation codes, by level. They are part of the JSON output and stay stable across releases.
ORIGIN_INVALID = "origin-invalid"
UNKNOWN_TYPE = "unknown-type"
ORIGIN_NOT_ALLOWED = "origin-not-allowed"
ANCHOR_INVALID = "anchor-invalid"
ANCHOR_MISMATCH = "anchor-mismatch"
DIGEST_MISMATCH = "digest-mismatch"
ANALYST_HAS_SOURCE = "analyst-has-source"
PART_OF_INVALID = "part-of-invalid"
SPAN_NOT_IN_PARENT = "span-not-in-parent"
FIELD_CHILD_INVALID = "field-child-invalid"
COMPOUND_CHILDREN = "compound-children"
COMPOUND_HAS_CONTENT = "compound-has-content"
UNTYPED = "untyped"
FORM_MISSING_FIELD = "form-missing-field"
FORM_INVALID_FIELD = "form-invalid-field"
FORM_NOT_IN_TEXT = "form-not-in-text"
RESTS_ON_MISSING = "rests-on-missing"
OUT_OF_DOMAIN = "out-of-domain"
SCOPE_EXCEEDS_SUPPORT = "scope-exceeds-support"
UNIT_MISMATCH = "unit-mismatch"
UNASSESSED = "unassessed"
NO_EVIDENCE = "no-evidence"
MISSING_REASON = "missing-reason"
COMPILE_ERROR = "compile-error"

VIOLATION_CODES = {
    ANCHORED: (
        ORIGIN_INVALID,
        UNKNOWN_TYPE,
        ORIGIN_NOT_ALLOWED,
        ANCHOR_INVALID,
        ANCHOR_MISMATCH,
        DIGEST_MISMATCH,
        ANALYST_HAS_SOURCE,
        PART_OF_INVALID,
        SPAN_NOT_IN_PARENT,
        FIELD_CHILD_INVALID,
        COMPOUND_CHILDREN,
        COMPOUND_HAS_CONTENT,
    ),
    TYPED: (
        UNTYPED,
        FORM_MISSING_FIELD,
        FORM_INVALID_FIELD,
        FORM_NOT_IN_TEXT,
        RESTS_ON_MISSING,
        OUT_OF_DOMAIN,
        SCOPE_EXCEEDS_SUPPORT,
        UNIT_MISMATCH,
    ),
    ASSESSED: (UNASSESSED, NO_EVIDENCE, MISSING_REASON, COMPILE_ERROR),
}

# The opt-in rules of ``rubric_credences``: every base and strength is the rubric's, and only an observed check
# can raise a credence above its node's base.
CREDENCE_NOT_RUBRIC = "credence-not-rubric"
RELATION_NOT_ALLOWED = "relation-not-allowed"
CHECK_UNOBSERVED_INVALID = "check-unobserved-invalid"
RUBRIC_CREDENCE_CODES = (CREDENCE_NOT_RUBRIC, RELATION_NOT_ALLOWED, CHECK_UNOBSERVED_INVALID)

# The relation of a check edge: from the node a check was run on to the check's evidence node.
CHECK_EDGE = SUPPORTS

# The check results, and the ``attributes.observed`` each implies; an errored check is not observed.
_OBSERVED = {"passed": True, "failed": False, "error": None}

# The last component of a span child's locator: the half-open character range within its parent.
_CHARS = re.compile(r";chars=(\d+)-(\d+)$")


def origin_of(node: Node) -> str | None:
    """Return a node's origin, or ``None`` if ``attributes.origin`` is missing or not a known origin.

    Args:
        node: The node.

    Returns:
        ``document``, ``analyst``, ``evidence`` or ``None``.
    """
    origin = node.attributes.get("origin")
    return origin if origin in ORIGINS else None


def chars_range(locator: str | None) -> tuple[int, int] | None:
    """Read the ``chars=<start>-<end>`` range a span child's locator ends with.

    Args:
        locator: The locator.

    Returns:
        ``(start, end)``, or ``None`` if the locator does not end with a range.
    """
    match = _CHARS.search(locator or "")
    return None if match is None else (int(match[1]), int(match[2]))


@dataclass(frozen=True)
class _Structure:
    """The part-of structure of a graph: each child's parents and each node's children."""

    parents: Mapping[str, list[Relation]]
    children: Mapping[str, list[Relation]]

    @classmethod
    def of(cls, graph: Graph) -> _Structure:
        parents: dict[str, list[Relation]] = {}
        children: dict[str, list[Relation]] = {}
        for relation in graph.relations.values():
            if relation.type == PART_OF:
                parents.setdefault(relation.source, []).append(relation)
                children.setdefault(relation.target, []).append(relation)
        return cls(parents, children)

    def parent(self, graph: Graph, node: Node) -> Node | None:
        """Return the single parent of a child, or ``None`` if it has not exactly one."""
        relations = self.parents.get(node.id, [])
        return graph.nodes[relations[0].target] if len(relations) == 1 else None

    def text(self, graph: Graph, node: Node) -> str:
        """Return the text a node's form is checked against: its parent's for a field child."""
        if self.is_field_child(node):
            parent = self.parent(graph, node)
            return (parent.statement if parent is not None else None) or ""
        return node.statement or ""

    def is_field_child(self, node: Node) -> bool:
        """Tell whether a child is a field child: one whose locator carries no character range."""
        return node.id in self.parents and not (len(node.sources) == 1 and chars_range(node.sources[0].locator))


def _finding(code: str, node: Node, message: str, relations: tuple[str, ...] = ()) -> Finding:
    """Build a violation about one node."""
    return Finding(f"{code}:{node.id}", code, f"node {node.id!r}: {message}", (node.id,), relations)


def _origin_and_type(rubric: Rubric, node: Node) -> list[Finding]:
    """The ``origin-invalid``, ``unknown-type`` and ``origin-not-allowed`` checks of one node."""
    findings = []
    origin = origin_of(node)
    if origin is None:
        findings.append(
            _finding(
                ORIGIN_INVALID,
                node,
                f"attributes.origin is {node.attributes.get('origin')!r}, not one of {list(ORIGINS)}",
            )
        )
    if node.kind not in BUILTIN_TYPES and node.kind not in rubric.types:
        findings.append(_finding(UNKNOWN_TYPE, node, f"type {node.kind!r} is not declared by the rubric"))
    elif origin is not None and node.kind in BUILTIN_TYPES and origin != DOCUMENT:
        findings.append(_finding(ORIGIN_NOT_ALLOWED, node, f"type {node.kind!r} is only for document nodes"))
    elif origin is not None and node.kind in rubric.types and origin not in rubric.types[node.kind].origins:
        allowed = list(rubric.types[node.kind].origins)
        findings.append(
            _finding(ORIGIN_NOT_ALLOWED, node, f"type {node.kind!r} allows origins {allowed}, not {origin!r}")
        )
    return findings


def _anchor(structure: _Structure, node: Node) -> list[Finding]:
    """The ``anchor-invalid``, ``anchor-mismatch`` and ``digest-mismatch`` checks of a document node."""
    if len(node.sources) != 1:
        return [_finding(ANCHOR_INVALID, node, f"a document node needs exactly one source, has {len(node.sources)}")]
    anchor = node.sources[0]
    findings = []
    lacking = [name for name in ("locator", "quote", "digest") if getattr(anchor, name) is None]
    if lacking:
        findings.append(_finding(ANCHOR_INVALID, node, f"its source lacks {', '.join(lacking)}"))
    if anchor.quote is not None and not structure.is_field_child(node) and node.statement != anchor.quote:
        findings.append(_finding(ANCHOR_MISMATCH, node, "its statement differs from its source's quote"))
    if anchor.quote is not None and anchor.digest is not None and anchor.digest != text_digest(anchor.quote):
        findings.append(
            _finding(
                DIGEST_MISMATCH,
                node,
                f"digest {anchor.digest!r} is not the digest of the quote, {text_digest(anchor.quote)!r}",
            )
        )
    return findings


def _child(graph: Graph, structure: _Structure, node: Node) -> list[Finding]:
    """The ``part-of-invalid``, ``span-not-in-parent`` and ``field-child-invalid`` checks of a child."""
    relations = structure.parents[node.id]
    ids = tuple(relation.id for relation in relations)
    parent = structure.parent(graph, node)
    problems = []
    if parent is None:
        problems.append(f"it is part of {len(relations)} nodes, not one")
    elif parent.kind != COMPOUND:
        problems.append(f"its parent {parent.id!r} is not a compound")
    if node.kind == COMPOUND:
        problems.append("a child may not itself be a compound")
    if origin_of(node) != DOCUMENT:
        problems.append("a child must be a document node")
    if problems:
        return [_finding(PART_OF_INVALID, node, "; ".join(problems), ids)]
    assert parent is not None  # noqa: S101 - established above
    if structure.is_field_child(node):
        problems = []
        if node.statement is not None:
            problems.append("a field child carries no text of its own, but it has a statement")
        if not isinstance(node.attributes.get("form"), Mapping):
            problems.append("a field child needs a form")
        if node.sources != parent.sources:
            problems.append(f"its source is not a copy of its parent {parent.id!r}'s")
        return [_finding(FIELD_CHILD_INVALID, node, "; ".join(problems), ids)] if problems else []
    start, end = chars_range(node.sources[0].locator)  # type: ignore[misc]
    parent_anchor = parent.sources[0] if len(parent.sources) == 1 else None
    expected_locator = None if parent_anchor is None else f"{parent_anchor.locator};chars={start}-{end}"
    text = parent.statement or ""
    if node.statement is None or text[start:end] != node.statement or end > len(text):
        found = text[start:end] if start <= end <= len(text) else None
        return [
            _finding(
                SPAN_NOT_IN_PARENT,
                node,
                f"its text is not at chars {start}-{end} of its parent, which hold {found!r}",
                ids,
            )
        ]
    if node.sources[0].locator != expected_locator or node.sources[0].document != parent_anchor.document:  # type: ignore[union-attr]
        return [_finding(SPAN_NOT_IN_PARENT, node, f"its source is not its parent's plus chars={start}-{end}", ids)]
    return []


def _compound(graph: Graph, structure: _Structure, node: Node) -> list[Finding]:
    """The ``compound-children`` and ``compound-has-content`` checks of a compound."""
    findings = []
    children = structure.children.get(node.id, [])
    if len(children) < 2:  # noqa: PLR2004 - a compound splits a chunk in at least two
        findings.append(
            _finding(COMPOUND_CHILDREN, node, f"a compound needs at least two children, has {len(children)}")
        )
    content = [name for name in ("form", "assessment") if name in node.attributes]
    content.extend(["base"] if node.base is not None else [])
    stray = tuple(
        relation.id
        for relation in graph.relations.values()
        if (relation.target == node.id and relation.type != PART_OF)
        or (relation.source == node.id and relation.type != CITES)
    )
    problems = [f"it carries {', '.join(content)}"] if content else []
    if stray:
        problems.append(f"it is an endpoint of relations other than incoming part-of and outgoing cites: {list(stray)}")
    if problems:
        findings.append(_finding(COMPOUND_HAS_CONTENT, node, "; ".join(problems), stray))
    return findings


def _anchored(graph: Graph, rubric: Rubric, structure: _Structure) -> list[Finding]:
    """Every ``anchored`` violation, node by node in graph order."""
    findings = []
    for node in graph:
        findings.extend(_origin_and_type(rubric, node))
        origin = origin_of(node)
        if origin == DOCUMENT:
            findings.extend(_anchor(structure, node))
        if origin == ANALYST and node.sources:
            findings.append(
                _finding(ANALYST_HAS_SOURCE, node, f"an implicit premise has no source, but it has {len(node.sources)}")
            )
        if node.id in structure.parents:
            findings.extend(_child(graph, structure, node))
        if node.kind == COMPOUND:
            findings.extend(_compound(graph, structure, node))
    return findings


def _is_number(value: object) -> bool:
    """Tell whether a value is a JSON number."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_text(value: object) -> bool:
    """Tell whether a value is a non-empty string."""
    return isinstance(value, str) and bool(value)


def _has_type(spec: FormField, value: object, rubric: Rubric) -> bool:
    """Tell whether a form value has its field's type."""
    checks = {
        VERBATIM: lambda: _is_text(value),
        EQREF: lambda: _is_text(value),
        NUMBER: lambda: _is_number(value),
        QUANTITY: lambda: (
            isinstance(value, Mapping)
            and set(value) == {"value", "unit"}
            and _is_number(value["value"])
            and _is_text(value["unit"])
        ),
        ENUM: lambda: value in spec.values,
        BOOL: lambda: isinstance(value, bool),
        SCOPE: lambda: (
            isinstance(value, Mapping)
            and all(name in rubric.parameters and _is_interval(interval) for name, interval in value.items())
        ),
    }
    return checks[spec.type]()


def _is_interval(value: object) -> bool:
    """Tell whether a scope value is ``[low, high]`` of numbers or ``None``, with ``low <= high``."""
    if not isinstance(value, tuple | list) or len(value) != 2:  # noqa: PLR2004 - low and high
        return False
    if not all(end is None or _is_number(end) for end in value):
        return False
    return value[0] is None or value[1] is None or value[0] <= value[1]


def node_text(graph: Graph, node: Node) -> str:
    """Return the text a node's form is checked against: its statement, or its parent's for a field child.

    Args:
        graph: The graph.
        node: The node.

    Returns:
        The text, empty if there is none.
    """
    return _Structure.of(graph).text(graph, node)


def form_problems(rubric: Rubric, node_type: str, form: object, text: str) -> tuple[list[str], list[str], list[str]]:
    """Check a form against its type's fields and against the text it restates.

    Args:
        rubric: The rubric.
        node_type: The node's type, declared by the rubric.
        form: The form, ``attributes.form``.
        text: The node's text, from ``node_text``.

    Returns:
        The missing required fields, the invalid fields (undeclared, or of the wrong field type) and
        the fields not found in the text, each as messages.
    """
    fields = rubric.types[node_type].form
    if not isinstance(form, Mapping):
        return [], [f"the form must be an object, got {form!r}"], []
    missing = [f"{name!r}" for name, spec in fields.items() if spec.required and name not in form]
    invalid: list[str] = []
    unfound: list[str] = []
    for name, value in form.items():
        spec = fields.get(name)
        if spec is None:
            invalid.append(f"{name!r} is not a field of type {node_type!r}")
        elif not _has_type(spec, value, rubric):
            invalid.append(f"{name!r} = {_plain(value)!r} is not a valid {spec.type}")
        elif (reason := unfaithful(spec.type, value, text)) is not None:
            unfound.append(f"{name!r}: {reason}")
    return missing, invalid, unfound


def _plain(value: object) -> object:
    """Turn a frozen JSON value back into plain lists and dicts, for messages."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _fields(rubric: Rubric, node: Node, field_type: str) -> dict[str, object]:
    """Return a document node's form fields of one field type that have that type, by name."""
    node_type = rubric.types.get(node.kind)
    form = node.attributes.get("form")
    if node_type is None or origin_of(node) != DOCUMENT or not isinstance(form, Mapping):
        return {}
    return {
        name: value
        for name, value in form.items()
        if (spec := node_type.form.get(name)) is not None and spec.type == field_type and _has_type(spec, value, rubric)
    }


def node_scope(rubric: Rubric, node: Node) -> dict[str, Interval]:
    """Return the scope a node declares: per parameter, the intersection of its ``scope`` fields.

    A parameter the node names in no valid ``scope`` field is absent: the node makes no claim about
    it, so it is not checked on that parameter and, as something another node rests on, restricts
    nothing there. A node with no such field, including every analyst and evidence node, therefore
    declares no scope.

    Args:
        rubric: The rubric.
        node: The node.

    Returns:
        The node's interval for each parameter it names.
    """
    scope: dict[str, Interval] = {}
    for value in _fields(rubric, node, SCOPE).values():
        for parameter, (low, high) in value.items():  # type: ignore[attr-defined]
            scope[parameter] = intersection(scope.get(parameter, UNBOUNDED), (low, high))
    return scope


def _rests_on(graph: Graph, rubric: Rubric, node: Node) -> list[Relation]:
    """Return the relations a node rests on: every ``requires`` into it, and those its type's edge rule counts."""
    node_type = rubric.types.get(node.kind)
    rule = node_type.rests_on if node_type is not None else None
    return [
        relation
        for relation in graph.relations.values()
        if relation.target == node.id
        and (
            relation.type == REQUIRES
            or (rule is not None and relation.type == rule.relation and graph.nodes[relation.source].kind in rule.types)
        )
    ]


def _scope_and_units(graph: Graph, rubric: Rubric, node: Node) -> list[Finding]:
    """The ``out-of-domain``, ``scope-exceeds-support`` and ``unit-mismatch`` checks of one node."""
    findings = []
    scope = node_scope(rubric, node)
    outside = [
        f"{name} {show(interval)} is outside its domain {show(rubric.parameters[name].domain)}"
        for name, interval in scope.items()
        if not contains(rubric.parameters[name].domain, interval)
    ]
    if outside:
        findings.append(_finding(OUT_OF_DOMAIN, node, "; ".join(outside)))
    relations = _rests_on(graph, rubric, node)
    support: dict[str, Interval] = {}
    for relation in relations:
        for name, interval in node_scope(rubric, graph.nodes[relation.source]).items():
            support[name] = intersection(support.get(name, UNBOUNDED), interval)
    exceeds = [
        f"{name} {show(scope[name])} is not within {show(support[name])}"
        + (", as the scopes it rests on do not overlap" if is_empty(support[name]) else ", what it rests on covers")
        for name in scope
        if name in support and not contains(support[name], scope[name])
    ]
    if exceeds:
        ids = tuple(relation.id for relation in relations if node_scope(rubric, graph.nodes[relation.source]))
        findings.append(_finding(SCOPE_EXCEEDS_SUPPORT, node, "; ".join(exceeds), ids))
    quantities = _fields(rubric, node, QUANTITY)
    mismatched: dict[str, list[str]] = {}
    for relation in relations:
        source = graph.nodes[relation.source]
        for name, theirs in _fields(rubric, source, QUANTITY).items():
            mine = quantities.get(name)
            if mine is not None and not agree(mine["unit"], theirs["unit"]):  # type: ignore[index]
                mismatched.setdefault(relation.id, []).append(
                    f"{name!r} is in {mine['unit']!r} here but in {theirs['unit']!r} on {source.id!r}"  # type: ignore[index]
                )
    if mismatched:
        messages = [message for messages in mismatched.values() for message in messages]
        findings.append(_finding(UNIT_MISMATCH, node, "; ".join(messages), tuple(mismatched)))
    return findings


def _typed(graph: Graph, rubric: Rubric, structure: _Structure) -> list[Finding]:
    """Every ``typed`` violation, node by node in graph order."""
    findings = []
    for node in graph:
        if node.kind == UNASSIGNED:
            findings.append(_finding(UNTYPED, node, "it has no type yet"))
        node_type = rubric.types.get(node.kind)
        if node_type is None:
            continue
        if origin_of(node) == DOCUMENT and (node_type.form or "form" in node.attributes):
            text = structure.text(graph, node)
            missing, invalid, unfound = form_problems(rubric, node.kind, node.attributes.get("form", {}), text)
            if missing:
                findings.append(
                    _finding(FORM_MISSING_FIELD, node, f"required form fields absent: {', '.join(missing)}")
                )
            if invalid:
                findings.append(_finding(FORM_INVALID_FIELD, node, "; ".join(invalid)))
            if unfound:
                findings.append(_finding(FORM_NOT_IN_TEXT, node, "; ".join(unfound)))
        elif "form" in node.attributes:
            findings.append(_finding(FORM_INVALID_FIELD, node, "only document nodes carry a form"))
        rule = node_type.rests_on
        if rule is not None:
            satisfying = tuple(
                relation.id
                for relation in graph.relations.values()
                if relation.target == node.id
                and relation.type == rule.relation
                and graph.nodes[relation.source].kind in rule.types
            )
            if len(satisfying) < rule.min:
                findings.append(
                    _finding(
                        RESTS_ON_MISSING,
                        node,
                        f"type {node.kind!r} rests on at least {rule.min} {rule.relation} relation(s) from "
                        f"{list(rule.types)}, found {len(satisfying)}",
                        satisfying,
                    )
                )
        findings.extend(_scope_and_units(graph, rubric, node))
    return findings


def _assessed(graph: Graph, rubric: Rubric) -> list[Finding]:
    """Every ``assessed`` violation: assessments node by node in graph order, then compilation."""
    findings = []
    for node in graph:
        node_type = rubric.types.get(node.kind)
        if node_type is None or not node_type.assess:
            continue
        assessment = node.attributes.get("assessment")
        verdict = assessment.get("verdict") if isinstance(assessment, Mapping) else None
        if assessment is None:
            findings.append(_finding(UNASSESSED, node, f"type {node.kind!r} needs an assessment, and it has none"))
        elif verdict not in VERDICTS or set(assessment) - {"verdict", "reason"}:  # type: ignore[arg-type]
            findings.append(
                _finding(
                    UNASSESSED, node, f"its assessment {_plain(assessment)!r} is not a verdict of {list(VERDICTS)}"
                )
            )
        elif verdict == NOT_ASSESSED:
            reason = assessment.get("reason")  # type: ignore[union-attr]
            if not isinstance(reason, str) or not reason.strip():
                findings.append(_finding(MISSING_REASON, node, "a not_assessed verdict needs a non-empty reason"))
        elif not any(
            relation.target == node.id and origin_of(graph.nodes[relation.source]) == EVIDENCE
            for relation in graph.relations.values()
        ):
            findings.append(
                _finding(NO_EVIDENCE, node, f"the verdict {verdict!r} has no relation from an evidence node")
            )
    try:
        compile_graph(graph)
    except CompileError as error:
        nodes = tuple(node_id for finding in missing_parameters(graph) for node_id in finding.nodes)
        findings.append(Finding(f"{COMPILE_ERROR}:graph", COMPILE_ERROR, f"the graph does not compile: {error}", nodes))
    return findings


def check_of(rubric: Rubric, node: Node) -> Check | None:
    """Return the check type of a check node: an evidence node of a check type that carries a check record.

    Args:
        rubric: The rubric.
        node: The node.

    Returns:
        The check type, or ``None`` if the node is not a check node.
    """
    if origin_of(node) == EVIDENCE and node.kind in rubric.checks and isinstance(node.attributes.get("check"), Mapping):
        return rubric.checks[node.kind]
    return None


def rubric_base(rubric: Rubric, node: Node) -> Credence | None:
    """Return the base the rubric gives a document or analyst node.

    Args:
        rubric: The rubric.
        node: A document or analyst node.

    Returns:
        ``premises.base`` for an analyst node when the rubric has ``[premises]``, else the base of the
        node's type; ``None`` if the type has none or is not declared.
    """
    if origin_of(node) == ANALYST and rubric.premises is not None:
        return rubric.premises.base
    node_type = rubric.types.get(node.kind)
    return node_type.base if node_type is not None else None


def _requires_strength(rubric: Rubric, source: Node, target: Node) -> Credence | None:
    """The strength the rubric gives a ``requires`` from ``source`` into ``target``, if it gives one."""
    if origin_of(source) == ANALYST and rubric.premises is not None:
        return rubric.premises.strength
    node_type = rubric.types.get(target.kind)
    rule = node_type.rests_on if node_type is not None else None
    return rule.strength if rule is not None and rule.relation == REQUIRES else None


def _show(credence: Credence | None) -> str:
    """Write a credence for a message, or say there is none."""
    return "none" if credence is None else format_credence(credence)


def _observed(node: Node, check: Check | None) -> list[Finding]:
    """The ``check-unobserved-invalid`` check of one node."""
    has = "observed" in node.attributes
    if check is None:
        message = "attributes.observed is only for an evidence node of a check type with a check record"
        return [_finding(CHECK_UNOBSERVED_INVALID, node, message)] if has else []
    result = node.attributes["check"].get("result")  # type: ignore[union-attr]
    if not isinstance(result, str) or result not in _OBSERVED:
        message = f"its check result is {_plain(result)!r}, not one of {list(_OBSERVED)}"
        return [_finding(CHECK_UNOBSERVED_INVALID, node, message)]
    expected = _OBSERVED[result]
    if (not has) if expected is None else (has and node.attributes["observed"] is expected):
        return []
    wanted = "absent" if expected is None else str(expected).lower()
    found = f"{_plain(node.attributes['observed'])!r}" if has else "absent"
    message = f"a {result} check needs attributes.observed {wanted}, and it is {found}"
    return [_finding(CHECK_UNOBSERVED_INVALID, node, message)]


def _relation_finding(code: str, relation: Relation, message: str) -> Finding:
    """Build a violation about one relation."""
    nodes = (relation.source, relation.target)
    return Finding(f"{code}:{relation.id}", code, f"relation {relation.id!r}: {message}", nodes, (relation.id,))


def _rubric_credences(graph: Graph, rubric: Rubric) -> list[Finding]:
    """Every ``rubric_credences`` violation: nodes in graph order, then relations in graph order.

    Document and analyst nodes may be joined only by ``requires``, and only to each other or by a check
    edge to a check node; their bases and strengths, and those of check nodes and check edges, are
    the rubric's. With only ``requires`` parents a node's probability is its base times a product of
    factors at most 1, so without an observation no credence exceeds the mean of its node's base.
    """
    findings = []
    propositions = {node.id for node in graph if origin_of(node) in (DOCUMENT, ANALYST)}
    checks = {node.id: check for node in graph if (check := check_of(rubric, node)) is not None}
    for node in graph:
        check = checks.get(node.id)
        expected = check.false_pass if check is not None else rubric_base(rubric, node)
        if (node.id in propositions or check is not None) and node.base is not None and node.base != expected:
            whose = f"check type {node.kind!r}'s false_pass" if check is not None else "the rubric's"
            message = f"its base {_show(node.base)} is not {whose}, {_show(expected)}"
            findings.append(_finding(CREDENCE_NOT_RUBRIC, node, message))
        findings.extend(_observed(node, check))
    for relation in graph.relations.values():
        if not relation.is_inferential or not {relation.source, relation.target} & (propositions | checks.keys()):
            continue
        source, target = graph.nodes[relation.source], graph.nodes[relation.target]
        if relation.type == REQUIRES and source.id in propositions and target.id in propositions:
            strength = _requires_strength(rubric, source, target)
        elif relation.type == CHECK_EDGE and source.id in propositions and target.id in checks:
            strength = checks[target.id].edge_strength
        else:
            message = (
                f"a {relation.type} from {source.id!r} to {target.id!r}: a document or analyst node rests only on "
                f"requires from document or analyst nodes, and a check reaches one only by a {CHECK_EDGE} from it"
            )
            findings.append(_relation_finding(RELATION_NOT_ALLOWED, relation, message))
            continue
        if relation.strength != strength:
            message = f"its strength {_show(relation.strength)} is not the rubric's, {_show(strength)}"
            findings.append(_relation_finding(CREDENCE_NOT_RUBRIC, relation, message))
    return findings


def check_graph(graph: Graph, rubric: Rubric, level: str = TYPED, *, rubric_credences: bool = False) -> list[Finding]:
    """Check a graph against a rubric at a validation level and every level below it.

    Args:
        graph: The graph.
        rubric: The rubric.
        level: ``anchored``, ``typed`` or ``assessed``.
        rubric_credences: Whether to add the rules of ``RUBRIC_CREDENCE_CODES``: every base and
            strength is the rubric's, and only an observed check can raise a credence.

    Returns:
        Every violation: the ``anchored`` ones in graph order, then the ``typed`` ones, then the
        ``assessed`` ones, then, with ``rubric_credences``, those rules'. Empty when the graph
        passes. The graph is compiled only at ``assessed``.

    Raises:
        ValueError: If the level is not one of ``LEVELS``.
    """
    if level not in LEVELS:
        msg = f"level must be one of {list(LEVELS)}, got {level!r}"
        raise ValueError(msg)
    structure = _Structure.of(graph)
    findings = _anchored(graph, rubric, structure)
    if level in (TYPED, ASSESSED):
        findings.extend(_typed(graph, rubric, structure))
    if level == ASSESSED:
        findings.extend(_assessed(graph, rubric))
    if rubric_credences:
        findings.extend(_rubric_credences(graph, rubric))
    return findings


def coverage(graph: Graph, rubric: Rubric) -> dict[str, object]:
    """Report how far classification has got. Never fails: it is a report, not a gate.

    Args:
        graph: The graph.
        rubric: The rubric.

    Returns:
        ``{"rubric", "nodes", "by_origin", "by_type", "untyped", "unexamined", "not_assessed"}``.
        ``by_type`` counts document nodes only, children and compounds included, in the order types
        first appear, each with its ``share`` of the document nodes. ``untyped`` lists the
        ``unassigned`` nodes and ``unexamined`` the nodes of ``assess = true`` types without an
        assessment, in graph order. A node without a valid origin is counted in ``nodes`` only.
    """
    by_origin = dict.fromkeys(ORIGINS, 0)
    by_type: dict[str, int] = {}
    untyped: list[str] = []
    unexamined: list[str] = []
    reasons: dict[str, object] = {}
    for node in graph:
        origin = origin_of(node)
        if origin is not None:
            by_origin[origin] += 1
        if origin == DOCUMENT:
            by_type[node.kind] = by_type.get(node.kind, 0) + 1
        if node.kind == UNASSIGNED:
            untyped.append(node.id)
        node_type = rubric.types.get(node.kind)
        assessment = node.attributes.get("assessment")
        if node_type is not None and node_type.assess and assessment is None:
            unexamined.append(node.id)
        if isinstance(assessment, Mapping) and assessment.get("verdict") == NOT_ASSESSED:
            reasons[node.id] = assessment.get("reason")
    documents = by_origin[DOCUMENT]
    return {
        "rubric": rubric.identity(),
        "nodes": len(graph),
        "by_origin": by_origin,
        "by_type": {name: {"count": count, "share": count / documents} for name, count in by_type.items()},
        "untyped": untyped,
        "unexamined": unexamined,
        "not_assessed": {"count": len(reasons), "reasons": reasons},
    }
