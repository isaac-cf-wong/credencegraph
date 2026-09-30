"""The ``query`` command: marginal, joint, conditional and interventional probabilities."""

from __future__ import annotations

from typing import Annotated

import typer

from credencegraph.cli.common import (
    IMPOSSIBLE_GRAPH_HINT,
    INVALID_ARGUMENT,
    ZERO_PROBABILITY,
    CliError,
    GraphPath,
    JsonOption,
    Result,
    compile_checked,
    parse_assignment,
    read_graph,
    require_variables,
    respond,
)
from credencegraph.inference.elimination import VariableElimination
from credencegraph.inference.engine import Query
from credencegraph.inference.errors import ZeroProbabilityError
from credencegraph.inference.queries import Answer, conditional, intervene, joint, marginal
from credencegraph.inference.uncertainty import DEFAULT_DRAWS
from credencegraph.semantics.network import Network

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


def _check_merged_settings(network: Network, interventions: dict[str, bool]) -> None:
    """Check that ``--set`` gives nodes merged by ``equivalent`` relations the same value.

    Args:
        network: The compiled network.
        interventions: The interventions.

    Raises:
        CliError: If two merged nodes are set to different values.
    """
    seen: dict[int, tuple[str, bool]] = {}
    for node_id, value in interventions.items():
        index = network.index(node_id)
        other, previous = seen.setdefault(index, (node_id, value))
        if previous != value:
            raise CliError(
                INVALID_ARGUMENT,
                f"--set gives the equivalent nodes {other!r} and {node_id!r} different values",
                "equivalent nodes are one proposition; --set them to the same value, or set only one of them",
            )


def _impossible(network: Network, query: Query, error: ZeroProbabilityError) -> CliError:
    """Explain a query whose evidence has probability zero, naming only what the caller passed.

    Args:
        network: The compiled network.
        query: The query.
        error: The engine's error.

    Returns:
        A ``zero-probability`` error. Its hint names ``--given`` or ``--set`` only when they were
        passed and the graph without them has worlds that satisfy every exclusive relation;
        otherwise it points at the graph.
    """
    flags = [flag for flag, values in (("--given", query.evidence), ("--set", query.interventions)) if values]
    if flags:
        try:
            VariableElimination().query(network, Query(query.target))
        except ZeroProbabilityError:
            flags = []
    if not flags:
        return CliError(ZERO_PROBABILITY, str(error), IMPOSSIBLE_GRAPH_HINT)
    named = " and ".join(flags)
    return CliError(
        ZERO_PROBABILITY,
        str(error),
        f"the {named} values cannot all hold under the graph's exclusive relations; drop or change one of them",
    )


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
        if draws < 0:
            raise CliError(
                INVALID_ARGUMENT, f"--draws must be 0 or more, got {draws}", "pass --draws 0 to skip the band"
            )
        graph = read_graph(path)
        require_variables(graph, target, "target")
        require_variables(graph, evidence, "--given")
        require_variables(graph, interventions, "--set")
        network = compile_checked(graph)
        _check_merged_settings(network, interventions)
        query = Query(target, evidence, interventions)
        options = {"draws": draws, "rng": seed}
        answer: Answer
        try:
            if kind == "intervene":
                answer = intervene(network, target, interventions, given=evidence, **options)
            elif evidence:
                answer = conditional(network, target, evidence, **options)
            elif kind == "marginal":
                ((node, value),) = target.items()
                answer = marginal(network, node, value, **options)
            else:
                answer = joint(network, target, **options)
        except ZeroProbabilityError as error:
            raise _impossible(network, query, error) from None
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

    respond("query", as_json, action)


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
