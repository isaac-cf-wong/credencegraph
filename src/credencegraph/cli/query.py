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
from credencegraph.core.credence import Point
from credencegraph.core.graph import Graph
from credencegraph.core.relation import EXCLUSIVE
from credencegraph.inference.elimination import VariableElimination
from credencegraph.inference.engine import Query
from credencegraph.inference.errors import ZeroProbabilityError
from credencegraph.inference.queries import Answer, conditional, intervene, joint, marginal
from credencegraph.inference.uncertainty import DEFAULT_DRAWS
from credencegraph.semantics.compiler import compile_graph
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


def _check_consistent(network: Network, evidence: dict[str, bool], interventions: dict[str, bool]) -> None:
    """Refuse values that contradict each other outright, before any inference.

    Nodes merged by ``equivalent`` relations are one proposition, so ``--set`` or ``--given`` must give
    them the same value, and ``--given`` must agree with ``--set`` on a proposition it fixes.

    Args:
        network: The compiled network.
        evidence: The ``--given`` values.
        interventions: The ``--set`` values.

    Raises:
        CliError: If two merged nodes get different values from one flag, or ``--given`` contradicts ``--set``.
    """
    for flag, assignment in (("--set", interventions), ("--given", evidence)):
        seen: dict[int, tuple[str, bool]] = {}
        for node_id, value in assignment.items():
            other, previous = seen.setdefault(network.index(node_id), (node_id, value))
            if previous != value:
                raise CliError(
                    INVALID_ARGUMENT,
                    f"{flag} gives the equivalent nodes {other!r} and {node_id!r} different values",
                    f"equivalent nodes are one proposition; give them the same {flag} value, or pass {flag} for only one",
                )
    fixed = {network.index(node_id): (node_id, value) for node_id, value in interventions.items()}
    for node_id, value in evidence.items():
        other, setting = fixed.get(network.index(node_id), (node_id, value))
        if setting != value:
            merged = "" if other == node_id else f", and {other!r} and {node_id!r} are equivalent"
            raise CliError(
                INVALID_ARGUMENT,
                f"--given {_item(node_id, value)} contradicts --set {_item(other, setting)}{merged}",
                "a proposition fixed by --set holds that value; drop the --given, or give it the --set value",
            )


def _item(node_id: str, value: bool) -> str:
    """Write an assignment the way it is typed: ``NODE=true``."""
    return f"{node_id}={str(value).lower()}"


def _fails(
    network: Network, target: dict[str, bool], evidence: dict[str, bool], interventions: dict[str, bool]
) -> bool:
    """Tell whether a query's evidence has probability zero.

    Args:
        network: The compiled network.
        target: The target.
        evidence: The ``--given`` values.
        interventions: The ``--set`` values.

    Returns:
        ``True`` if the query raises ``ZeroProbabilityError``.
    """
    try:
        VariableElimination().query(network, Query(target, evidence, interventions))
    except ZeroProbabilityError:
        return True
    return False


def _extreme_parameters(graph: Graph) -> list[dict[str, object]]:
    """List the bases and strengths that are exactly 0 or 1.

    Args:
        graph: The graph.

    Returns:
        ``{"id": "base:<node>" or "strength:<relation>", "value": ...}`` for each, in graph order.
    """
    extreme: list[dict[str, object]] = []
    for node in graph.nodes.values():
        if isinstance(node.base, Point) and node.base.p in {0.0, 1.0}:
            extreme.append({"id": f"base:{node.id}", "value": node.base.p})
    for relation in graph.relations.values():
        if isinstance(relation.strength, Point) and relation.strength.p in {0.0, 1.0}:
            extreme.append({"id": f"strength:{relation.id}", "value": relation.strength.p})
    return extreme


def _without_exclusive(graph: Graph) -> Graph:
    """Copy a graph without its ``exclusive`` relations."""
    copy = Graph()
    for node in graph.nodes.values():
        copy.add_node(node)
    for relation in graph.relations.values():
        if relation.type != EXCLUSIVE:
            copy.add_relation(relation)
    return copy


def _broken_exclusive(
    graph: Graph, network: Network, items: list[tuple[str, str, bool]]
) -> tuple[str, str, str] | None:
    """Find an ``exclusive`` relation whose two propositions the passed values both make true.

    Args:
        graph: The graph.
        network: The compiled network.
        items: The passed values as ``(flag, node id, value)``.

    Returns:
        The relation id and the two items, written ``FLAG NODE=value``, or ``None``.
    """
    true = {}
    for flag, node_id, value in items:
        if value:
            true.setdefault(network.index(node_id), f"{flag} {_item(node_id, value)}")
    for relation in graph.relations.values():
        if relation.type != EXCLUSIVE:
            continue
        ends = network.index(relation.source), network.index(relation.target)
        if all(end in true for end in ends):
            return relation.id, true[ends[0]], true[ends[1]]
    return None


def _impossible(graph: Graph, network: Network, query: Query, error: ZeroProbabilityError) -> CliError:
    """Explain a query whose evidence has probability zero by its cause, with a remedy that is checked.

    With nothing passed, or when the query fails without its ``--given`` and ``--set`` values, the
    cause is the graph. Otherwise the passed values are the cause, and either they make both
    propositions of an ``exclusive`` relation true, or, since every world has positive probability
    when every base and strength lies strictly between 0 and 1, some base or strength is exactly 0 or
    1; those are listed, and exclusive relations are mentioned only when the query succeeds without them.
    The remedy lists only values whose removal was tried and makes the query succeed.

    Args:
        graph: The graph.
        network: The compiled network.
        query: The query.
        error: The engine's error.

    Returns:
        A ``zero-probability`` error. Its details hold ``extreme_parameters``, and either
        ``drop_any_one_of`` (removing any one of those values makes the query succeed) or
        ``drop_all_of`` (removing all of them does), with ``exclusive_relation`` when one is broken.
    """
    target = dict(query.target)
    items = [("--given", n, v) for n, v in query.evidence.items()] + [
        ("--set", n, v) for n, v in query.interventions.items()
    ]
    if not items or _fails(network, target, {}, {}):
        return CliError(ZERO_PROBABILITY, str(error), IMPOSSIBLE_GRAPH_HINT)
    labels = [f"{flag} {_item(node_id, value)}" for flag, node_id, value in items]
    passed = " and ".join(flag for flag in ("--given", "--set") if any(item[0] == flag for item in items))
    message = f"the {passed} values have probability zero together, so the query has no answer"

    def fails_keeping(keep: list[int]) -> bool:
        kept = [items[i] for i in keep]
        return _fails(
            network,
            target,
            {n: v for flag, n, v in kept if flag == "--given"},
            {n: v for flag, n, v in kept if flag == "--set"},
        )

    everything = range(len(items))
    any_one = [labels[i] for i in everything if not fails_keeping([j for j in everything if j != i])]
    all_of: list[str] = []
    if not any_one:
        for flag in ("--given", "--set"):
            group = [i for i in everything if items[i][0] == flag]
            if group and not fails_keeping([i for i in everything if i not in group]):
                all_of = [labels[i] for i in group]
                break
        else:
            all_of = labels
    remedy = (
        f"drop {any_one[0]}"
        if len(any_one) == 1
        else f"drop any one of {', '.join(any_one)}"
        if any_one
        else f"drop all of {', '.join(all_of)}"
    )
    details: dict[str, object] = {
        "extreme_parameters": _extreme_parameters(graph),
        "drop_any_one_of": any_one,
        "drop_all_of": all_of,
    }
    broken = _broken_exclusive(graph, network, items)
    if broken is not None:
        relation, first, second = broken
        details["exclusive_relation"] = relation
        return CliError(
            ZERO_PROBABILITY,
            message,
            f"{first} and {second} make both propositions of the exclusive relation {relation!r} true; {remedy}",
            details,
        )
    responsible = any_one or all_of
    flags = " and ".join(flag for flag in ("--given", "--set") if any(label.startswith(flag) for label in responsible))
    uses_exclusive = any(r.type == EXCLUSIVE for r in graph.relations.values()) and not _fails(
        compile_graph(_without_exclusive(graph)), target, dict(query.evidence), dict(query.interventions)
    )
    context = "the graph's credences and exclusive relations" if uses_exclusive else "the graph's credences"
    return CliError(
        ZERO_PROBABILITY,
        message,
        f"the {flags} values have probability zero under {context}, which needs a base or strength of exactly 0 or 1 "
        f"(listed in details.extreme_parameters); {remedy}, or move every listed parameter off 0 and 1",
        details,
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
        _check_consistent(network, evidence, interventions)
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
            raise _impossible(graph, network, query, error) from None
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
