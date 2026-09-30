"""The commands that write a graph file: ``init``, ``add-node`` and ``relate``.

Each one reads the file, applies the change through the validated writes of ``Graph``, and replaces
the file only if every check passed, so a rejected write leaves the file exactly as it was.
"""

from __future__ import annotations

import difflib
from typing import Annotated

import typer

from credencegraph.cli.common import (
    CREDENCE_FORMS,
    DUPLICATE_ID,
    FILE_EXISTS,
    INVALID_ARGUMENT,
    CliError,
    GraphPath,
    JsonOption,
    Result,
    parse_credence,
    parse_source,
    read_graph,
    require_nodes,
    require_text,
    respond,
    write_graph,
)
from credencegraph.core.graph import Graph
from credencegraph.core.node import DEFAULT_KIND, Node
from credencegraph.core.relation import EQUIVALENT, EXCLUSIVE, INFERENTIAL_TYPES, STRENGTH_TYPES, Relation
from credencegraph.core.serialization import credence_to_json


def init_command(
    path: GraphPath,
    force: Annotated[bool, typer.Option("--force", help="Replace the file if it already exists.")] = False,
    as_json: JsonOption = False,
) -> None:
    """Create an empty graph file."""

    def action() -> Result:
        if path.exists() and not force:
            raise CliError(
                FILE_EXISTS,
                f"{str(path)!r} already exists",
                "pass --force to replace it, or choose another path",
                {"path": str(path)},
            )
        write_graph(Graph(), path)
        return Result({"path": str(path), "nodes": 0, "relations": 0}, [f"created empty graph {path}"])

    respond("init", as_json, action)


def add_node_command(  # noqa: PLR0913, PLR0917 - Typer maps one parameter to each option
    path: GraphPath,
    node_id: Annotated[str, typer.Option("--id", help="The node's id, unique within the graph.", show_default=False)],
    statement: Annotated[str | None, typer.Option(help="The normalised proposition.", show_default=False)] = None,
    kind: Annotated[str, typer.Option(help="A free label the engine attaches no meaning to.")] = DEFAULT_KIND,
    base: Annotated[
        str | None,
        typer.Option(help=f"The prior credence: {CREDENCE_FORMS}.", show_default=False),
    ] = None,
    stated: Annotated[
        str | None,
        typer.Option(help=f"The credence the source itself asserts: {CREDENCE_FORMS}.", show_default=False),
    ] = None,
    source: Annotated[
        list[str] | None,
        typer.Option(help="Where it came from, DOCUMENT or DOCUMENT#LOCATOR; repeat for several.", show_default=False),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Add a node to a graph file."""

    def action() -> Result:
        for value, option in ((node_id, "--id"), (statement, "--statement"), (kind, "--kind")):
            require_text(value, option)
        graph = read_graph(path)
        if node_id in graph:
            raise CliError(
                DUPLICATE_ID,
                f"a node with id {node_id!r} already exists",
                "choose another --id",
                {"node": node_id},
            )
        node = Node(
            node_id,
            kind=kind,
            statement=statement,
            sources=tuple(parse_source(item) for item in source or ()),
            base=None if base is None else parse_credence(base, "--base"),
            stated=None if stated is None else parse_credence(stated, "--stated"),
        )
        graph.add_node(node)
        write_graph(graph, path)
        payload = {
            "path": str(path),
            "node": {
                "id": node.id,
                "kind": node.kind,
                "statement": node.statement,
                "sources": [{"document": a.document, "locator": a.locator} for a in node.sources],
                "base": None if node.base is None else credence_to_json(node.base),
                "stated": None if node.stated is None else credence_to_json(node.stated),
            },
        }
        return Result(payload, [f"added node {node.id!r} to {path}"])

    respond("add-node", as_json, action)


def _default_relation_id(graph: Graph, source: str, relation_type: str, target: str) -> str:
    """Name a relation after its endpoints and type.

    Args:
        graph: The graph.
        source: The source node id.
        relation_type: The relation type.
        target: The target node id.

    Returns:
        ``<source>-<type>-<target>``.

    Raises:
        CliError: If that id is taken.
    """
    relation_id = f"{source}-{relation_type}-{target}"
    if relation_id in graph.relations:
        raise CliError(
            DUPLICATE_ID,
            f"a relation with id {relation_id!r} already exists",
            "pass --id to add another relation between the same nodes",
            {"relation": relation_id},
        )
    return relation_id


def _near_inferential(relation_type: str) -> str | None:
    """Suggest the inferential type a mistyped annotation type was probably meant to be.

    Args:
        relation_type: The relation type.

    Returns:
        The closest inferential type, or ``None`` if the type is inferential or resembles none.
    """
    if relation_type in INFERENTIAL_TYPES:
        return None
    close = difflib.get_close_matches(relation_type.lower(), sorted(INFERENTIAL_TYPES), n=1, cutoff=0.75)
    return close[0] if close else None


def relate_command(  # noqa: PLR0913, PLR0917 - Typer maps one parameter to each option
    path: GraphPath,
    source: Annotated[str, typer.Argument(help="The source node id.", show_default=False)],
    target: Annotated[str, typer.Argument(help="The target node id.", show_default=False)],
    relation_type: Annotated[
        str,
        typer.Option(
            "--type",
            help="requires, supports, refutes, equivalent or exclusive; any other type is an annotation.",
            show_default=False,
        ),
    ],
    strength: Annotated[
        str | None,
        typer.Option(
            help=f"Required for requires, supports and refutes, and forbidden otherwise: {CREDENCE_FORMS}.",
            show_default=False,
        ),
    ] = None,
    relation_id: Annotated[
        str | None,
        typer.Option("--id", help="The relation's id; SOURCE-TYPE-TARGET by default.", show_default=False),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Add a relation between two nodes of a graph file."""

    def action() -> Result:
        require_text(relation_type, "--type")
        require_text(relation_id, "--id")
        if relation_type in (EQUIVALENT, EXCLUSIVE) and source == target:
            raise CliError(
                INVALID_ARGUMENT,
                f"an {relation_type} relation joins two different nodes, but SOURCE and TARGET are both {source!r}",
                "name two different nodes as SOURCE and TARGET",
            )
        graph = read_graph(path)
        require_nodes(graph, (source,), "source")
        require_nodes(graph, (target,), "target")
        if relation_id is not None and relation_id in graph.relations:
            raise CliError(
                DUPLICATE_ID,
                f"a relation with id {relation_id!r} already exists",
                "choose another --id",
                {"relation": relation_id},
            )
        new_id = relation_id if relation_id is not None else _default_relation_id(graph, source, relation_type, target)
        suggestion = _near_inferential(relation_type)
        if relation_type in STRENGTH_TYPES and strength is None:
            raise CliError(
                INVALID_ARGUMENT,
                f"a {relation_type} relation needs a strength",
                f"pass --strength, {CREDENCE_FORMS}",
            )
        if relation_type not in STRENGTH_TYPES and strength is not None:
            reading = "carries no strength" if relation_type in INFERENTIAL_TYPES else "is an annotation"
            hint = f"did you mean --type {suggestion}?" if suggestion else "drop --strength"
            raise CliError(INVALID_ARGUMENT, f"type {relation_type!r} {reading}; it takes no --strength", hint)
        relation = Relation(
            new_id,
            relation_type,
            source,
            target,
            strength=None if strength is None else parse_credence(strength, "--strength"),
        )
        graph.add_relation(relation)
        write_graph(graph, path)
        warnings = []
        if suggestion:
            warnings.append(
                f"type {relation_type!r} is an annotation, which inference ignores; did you mean {suggestion!r}?"
            )
        payload = {
            "path": str(path),
            "relation": {
                "id": relation.id,
                "type": relation.type,
                "source": relation.source,
                "target": relation.target,
                "strength": None if relation.strength is None else credence_to_json(relation.strength),
                "inferential": relation.is_inferential,
            },
            "warnings": warnings,
        }
        role = "inferential" if relation.is_inferential else "annotation"
        text = [f"added {role} relation {relation.id!r}: {source} --{relation_type}--> {target}"]
        text.extend(f"warning: {warning}" for warning in warnings)
        return Result(payload, text)

    respond("relate", as_json, action)
