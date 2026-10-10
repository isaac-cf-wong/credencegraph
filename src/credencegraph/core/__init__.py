"""The graph data model: nodes, relations, credences, and their JSON form."""

from __future__ import annotations

from credencegraph.core.anchor import SourceAnchor
from credencegraph.core.credence import Beta, Credence, Point, as_credence, parse_credence
from credencegraph.core.errors import CredenceGraphError, CycleError, ValidationError
from credencegraph.core.graph import Graph
from credencegraph.core.node import Node
from credencegraph.core.relation import (
    ACYCLIC_TYPES,
    INFERENTIAL_TYPES,
    STRENGTH_TYPES,
    Relation,
    is_inferential,
)
from credencegraph.core.serialization import (
    FORMAT,
    VERSION,
    dump,
    dumps,
    graph_from_dict,
    graph_to_dict,
    json_schema,
    load,
    loads,
)

__all__ = [
    "ACYCLIC_TYPES",
    "FORMAT",
    "INFERENTIAL_TYPES",
    "STRENGTH_TYPES",
    "VERSION",
    "Beta",
    "Credence",
    "CredenceGraphError",
    "CycleError",
    "Graph",
    "Node",
    "Point",
    "Relation",
    "SourceAnchor",
    "ValidationError",
    "as_credence",
    "dump",
    "dumps",
    "graph_from_dict",
    "graph_to_dict",
    "is_inferential",
    "json_schema",
    "load",
    "loads",
    "parse_credence",
]
