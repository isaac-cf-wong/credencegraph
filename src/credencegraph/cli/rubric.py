"""The commands that classify a graph against a rubric: ``coverage``, ``split`` and ``annotate``.

``split`` and ``annotate`` write the graph file, replacing it only when every check passed, so a
rejected command leaves it exactly as it was.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from credencegraph.cli.common import (
    INVALID_ARGUMENT,
    CliError,
    GraphPath,
    JsonOption,
    NodeArgument,
    Result,
    read_graph,
    read_rubric,
    require_nodes,
    require_text,
    respond,
    write_graph,
)
from credencegraph.core.graph import Graph
from credencegraph.core.serialization import graph_to_dict
from credencegraph.rubric.edit import annotate_node, parse_value, split_node
from credencegraph.rubric.validate import coverage

RubricPath = Annotated[Path, typer.Option("--rubric", help="The TOML rubric.", show_default=False)]


def _node_json(graph: Graph, node_id: str) -> dict[str, Any]:
    """Return one node as it is written in the graph file.

    Args:
        graph: The graph.
        node_id: The node's id.

    Returns:
        The node's JSON object.
    """
    return next(node for node in graph_to_dict(graph)["nodes"] if node["id"] == node_id)


def coverage_command(path: GraphPath, rubric: RubricPath, as_json: JsonOption = False) -> None:
    """Report how far the classification of a graph against a rubric has got.

    Counts the nodes by origin and the document nodes by type, and lists the untyped nodes, the
    nodes that need an assessment and have none, and the reasons given for not assessing. It is a
    report, not a gate: it does not fail on a graph that has violations.
    """

    def action() -> Result:
        graph = read_graph(path)
        report = coverage(graph, read_rubric(rubric))
        name = report["rubric"]
        text = [f"{path}: {report['nodes']} nodes against rubric {name['name']} {name['version']}"]  # type: ignore[index]
        text.extend(f"origin {origin}: {count}" for origin, count in report["by_origin"].items())  # type: ignore[attr-defined]
        text.extend(
            f"type {kind}: {entry['count']} ({entry['share']:.1%} of document nodes)"
            for kind, entry in report["by_type"].items()  # type: ignore[attr-defined]
        )
        text.append(f"untyped: {', '.join(report['untyped']) or 'none'}")  # type: ignore[arg-type]
        text.append(f"unexamined: {', '.join(report['unexamined']) or 'none'}")  # type: ignore[arg-type]
        text.extend(f"not assessed: {node}: {reason}" for node, reason in report["not_assessed"]["reasons"].items())  # type: ignore[index]
        return Result({"path": str(path), **report}, text)

    respond("coverage", as_json, action)


def split_command(
    path: GraphPath,
    node: NodeArgument,
    at: Annotated[
        list[str] | None,
        typer.Option(
            "--at",
            help="A verbatim clause of the node's text, occurring exactly once in it; one span child per --at.",
            show_default=False,
        ),
    ] = None,
    fields: Annotated[
        bool, typer.Option("--fields", help="Add one field child, with no text, instead of span children.")
    ] = False,
    as_json: JsonOption = False,
) -> None:
    """Make a document chunk a compound, with one span child per --at clause, or one field child."""

    def action() -> Result:
        if bool(at) == fields:
            raise CliError(
                INVALID_ARGUMENT,
                "split takes --at or --fields, and not both",
                "pass one --at per clause for span children, or --fields for one field child",
            )
        graph = read_graph(path)
        require_nodes(graph, (node,), "NODE")
        updated, children = split_node(graph, node, at or (), fields=fields)
        write_graph(updated, path)
        return Result(
            {"path": str(path), "node": node, "children": children},
            [f"split {node!r} into {', '.join(repr(child) for child in children)}"],
        )

    respond("split", as_json, action)


def _settings(items: list[str]) -> dict[str, Any]:
    """Parse ``--set FIELD=VALUE`` items; a value is read as JSON if it parses, else as a string.

    Args:
        items: The command-line values.

    Returns:
        The values by field.

    Raises:
        CliError: If an item has no ``=`` or no field, or a field is given twice.
    """
    settings: dict[str, Any] = {}
    for item in items:
        name, equals, raw = item.partition("=")
        if not equals or not name:
            raise CliError(
                INVALID_ARGUMENT,
                f"--set {item!r} is not FIELD=VALUE",
                "write it as FIELD=VALUE, such as shape=universal or value=0.05",
            )
        if name in settings:
            raise CliError(INVALID_ARGUMENT, f"--set gives field {name!r} twice", "set each field once")
        settings[name] = parse_value(raw)
    return settings


def annotate_command(  # noqa: PLR0913, PLR0917 - Typer maps one parameter to each option
    path: GraphPath,
    node: NodeArgument,
    rubric: RubricPath,
    node_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            help="A type the rubric declares and allows for the node's origin; 'unassigned' removes the type, form, assessment and base.",
            show_default=False,
        ),
    ] = None,
    setting: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="A form field, FIELD=VALUE; the value is read as JSON if it parses, else as a string. Repeat for several.",
            show_default=False,
        ),
    ] = None,
    rests_on: Annotated[
        list[str] | None,
        typer.Option(
            "--rests-on",
            help="A node this one rests on: adds the type's edge-rule relation from it. Repeat for several.",
            show_default=False,
        ),
    ] = None,
    verdict: Annotated[
        str | None,
        typer.Option(help="The assessment: holds, fails or undetermined.", show_default=False),
    ] = None,
    not_assessed: Annotated[
        str | None,
        typer.Option("--not-assessed", help="Why the node was not assessed.", show_default=False),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Type, fill in, link and assess a node, validating each write against the rubric.

    Required form fields may be left out, since they are filled one at a time; check --level typed
    reports the ones still missing.
    """

    def action() -> Result:
        require_text(node_type, "--type")
        if not (node_type or setting or rests_on or verdict or not_assessed is not None):
            raise CliError(
                INVALID_ARGUMENT,
                "annotate needs something to do",
                "pass --type, --set, --rests-on, --verdict or --not-assessed",
            )
        if verdict is not None and not_assessed is not None:
            raise CliError(INVALID_ARGUMENT, "--verdict and --not-assessed exclude each other", "pass one of them")
        settings = _settings(setting or [])
        graph = read_graph(path)
        require_nodes(graph, (node,), "NODE")
        require_nodes(graph, rests_on or (), "--rests-on")
        updated = annotate_node(
            graph,
            read_rubric(rubric),
            node,
            node_type=node_type,
            settings=settings,
            rests_on=rests_on or (),
            verdict=verdict,
            reason=not_assessed,
        )
        write_graph(updated, path)
        added = [relation for relation in updated.relations if relation not in graph.relations]
        return Result(
            {"path": str(path), "node": _node_json(updated, node), "relations": added},
            [f"annotated {node!r}", *(f"added relation {relation!r}" for relation in added)],
        )

    respond("annotate", as_json, action)
