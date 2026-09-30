"""The ``query`` command: marginal, joint, conditional and interventional probabilities."""

from __future__ import annotations

from typing import Annotated

import typer

from credencegraph.cli.common import (
    INVALID_ARGUMENT,
    CliError,
    GraphPath,
    JsonOption,
    Result,
    parse_assignment,
    read_graph,
    require_nodes,
    respond,
)
from credencegraph.inference.engine import Query
from credencegraph.inference.queries import Answer, conditional, intervene, joint, marginal
from credencegraph.inference.uncertainty import DEFAULT_DRAWS
from credencegraph.semantics.compiler import compile_graph

KINDS = ("marginal", "joint", "conditional", "intervene")


def _check_shape(kind: str, target: dict[str, bool], given: dict[str, bool], setting: dict[str, bool]) -> None:
    """Check that the targets and options fit the kind of query.

    Args:
        kind: The kind of query.
        target: The target assignment.
        given: The evidence.
        setting: The interventions.

    Raises:
        CliError: If the kind is unknown or the query does not fit it.
    """
    if kind not in KINDS:
        raise CliError(
            INVALID_ARGUMENT,
            f"unknown query kind {kind!r}",
            f"the kind is one of {', '.join(KINDS)}",
            {"kinds": list(KINDS)},
        )
    if kind == "marginal" and len(target) != 1:
        raise CliError(
            INVALID_ARGUMENT,
            f"a marginal query takes one target node, got {len(target)}",
            "use 'joint' for the probability that several nodes hold together",
        )
    if kind == "conditional" and not given:
        raise CliError(INVALID_ARGUMENT, "a conditional query needs evidence", "pass --given NODE=true|false")
    if kind == "intervene" and not setting:
        raise CliError(INVALID_ARGUMENT, "an intervene query needs an intervention", "pass --set NODE=true|false")
    if kind != "intervene" and setting:
        raise CliError(INVALID_ARGUMENT, f"a {kind} query takes no --set", "use 'intervene' to set a node")


def query_command(  # noqa: PLR0913, PLR0917 - Typer maps one parameter to each option
    path: GraphPath,
    kind: Annotated[str, typer.Argument(help=f"The kind of query: {', '.join(KINDS)}.", show_default=False)],
    targets: Annotated[
        list[str],
        typer.Argument(help="The target nodes, NODE (true) or NODE=true|false.", show_default=False),
    ],
    given: Annotated[
        list[str] | None,
        typer.Option(help="Evidence to condition on, NODE=true|false; repeat for several.", show_default=False),
    ] = None,
    setting: Annotated[
        list[str] | None,
        typer.Option(
            "--set", help="A node to set by intervention, NODE=true|false; repeat for several.", show_default=False
        ),
    ] = None,
    draws: Annotated[
        int, typer.Option(help="Parameter draws behind the uncertainty band; 0 skips the band.")
    ] = DEFAULT_DRAWS,
    seed: Annotated[int | None, typer.Option(help="Seed for the parameter draws.", show_default=False)] = None,
    as_json: JsonOption = False,
) -> None:
    """Compute a probability from a graph file.

    Every kind accepts --given; a marginal or joint query with evidence is the conditional one.
    """

    def action() -> Result:
        target = parse_assignment(targets, "target", default=True)
        evidence = parse_assignment(given or (), "--given")
        interventions = parse_assignment(setting or (), "--set")
        _check_shape(kind, target, evidence, interventions)
        graph = read_graph(path)
        require_nodes(graph, target, "target")
        require_nodes(graph, evidence, "--given node")
        require_nodes(graph, interventions, "--set node")
        network = compile_graph(graph)
        options = {"draws": draws, "rng": seed}
        answer: Answer
        if kind == "intervene":
            answer = intervene(network, target, interventions, given=evidence, **options)
        elif evidence:
            answer = conditional(network, target, evidence, **options)
        elif kind == "marginal":
            ((node, value),) = target.items()
            answer = marginal(network, node, value, **options)
        else:
            answer = joint(network, target, **options)
        query = Query(target, evidence, interventions)
        payload = {
            "path": str(path),
            "kind": kind,
            "target": dict(query.target),
            "given": dict(query.evidence),
            "set": dict(query.interventions),
            "seed": seed,
            **answer.to_dict(),
        }
        return Result(payload, _describe(query, answer))

    respond("query", as_json, action, path)


def _describe(query: Query, answer: Answer) -> list[str]:
    """Render an answer for a human reader.

    Args:
        query: The query.
        answer: Its answer.

    Returns:
        The lines to print.
    """

    def render(assignment: dict[str, bool]) -> str:
        return ", ".join(f"{node}={str(value).lower()}" for node, value in assignment.items())

    conditions = [render(dict(query.evidence))] if query.evidence else []
    if query.interventions:
        conditions.insert(0, f"do({render(dict(query.interventions))})")
    expression = f"P({render(dict(query.target))}{' | ' + ', '.join(conditions) if conditions else ''})"
    lines = [f"{expression} = {answer.point:.6g}"]
    if answer.band is not None:
        band = answer.band
        lines.append(
            f"band over {answer.draws} draws: q05={band.q05:.6g} q50={band.q50:.6g} q95={band.q95:.6g}; "
            f"mean over draws {answer.mean_over_draws:.6g}"
        )
    return lines
