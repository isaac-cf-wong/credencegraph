"""Rubric-driven classification of graphs built from a document's own text.

A rubric declares the node types, their forms and their edge rules. ``check_graph`` checks a graph
against it at the ``anchored``, ``typed`` or ``assessed`` level, ``coverage`` reports how far the
classification has got, and ``split_node`` and ``annotate_node`` make the edits that classify it.
"""

from __future__ import annotations

from credencegraph.rubric.edit import annotate_node, parse_value, split_node
from credencegraph.rubric.faithfulness import has_eqref, has_numeral, has_unit, numerals
from credencegraph.rubric.model import (
    FormField,
    NodeType,
    Parameter,
    RestsOn,
    Rubric,
    RubricError,
    load_rubric,
    loads_rubric,
    rubric_from_dict,
)
from credencegraph.rubric.text import normalise_text, text_digest
from credencegraph.rubric.validate import LEVELS, VIOLATION_CODES, check_graph, coverage

__all__ = [
    "LEVELS",
    "VIOLATION_CODES",
    "FormField",
    "NodeType",
    "Parameter",
    "RestsOn",
    "Rubric",
    "RubricError",
    "annotate_node",
    "check_graph",
    "coverage",
    "has_eqref",
    "has_numeral",
    "has_unit",
    "load_rubric",
    "loads_rubric",
    "normalise_text",
    "numerals",
    "parse_value",
    "rubric_from_dict",
    "split_node",
    "text_digest",
]
