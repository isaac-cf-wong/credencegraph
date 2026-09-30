"""The ``query`` command: marginal, joint, conditional and interventional probabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import combinations
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
from credencegraph.semantics.errors import CompileError
from credencegraph.semantics.network import Network

KINDS = ("marginal", "joint", "conditional", "intervene")

# The most changes tried in search of a remedy of more than one value or parameter.
SEARCH_BUDGET = 256


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


def _smallest(count: int, clears: Callable[[tuple[int, ...]], bool]) -> tuple[list[int], list[int], bool]:
    """Find the smallest changes that clear a failure, by trying them.

    Every single change is tried, then sets of two, three and so on, until ``SEARCH_BUDGET`` tries.

    Args:
        count: The number of candidate changes.
        clears: Tells whether making the changes at the given indices clears the failure.

    Returns:
        The indices that clear it one at a time, or, when none does, the first smallest set that clears
        it together, and whether the search stopped at ``SEARCH_BUDGET`` with sets left untried. The
        lists are both empty when nothing tried clears it.
    """
    any_one = [i for i in range(count) if clears((i,))]
    if any_one:
        return any_one, [], False
    tries = count
    for size in range(2, count + 1):
        for group in combinations(range(count), size):
            if tries >= SEARCH_BUDGET:
                return [], [], True
            tries += 1
            if clears(group):
                return [], list(group), False
    return [], [], False


def _rebuilt(graph: Graph, removed: frozenset[str] = frozenset(), softened: frozenset[str] = frozenset()) -> Graph:
    """Copy a graph without some relations, and with some bases or strengths moved to 0.5.

    Args:
        graph: The graph.
        removed: The ids of the relations to leave out.
        softened: The parameters to move to 0.5, as ``base:<node>`` or ``strength:<relation>``.

    Returns:
        The copy.
    """
    copy = Graph()
    for node in graph.nodes.values():
        copy.add_node(replace(node, base=Point(0.5)) if f"base:{node.id}" in softened else node)
    for relation in graph.relations.values():
        if relation.id not in removed:
            soft = f"strength:{relation.id}" in softened
            copy.add_relation(replace(relation, strength=Point(0.5)) if soft else relation)
    return copy


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


def _broken_exclusive(graph: Graph, network: Network, items: list[tuple[str, str, bool]]) -> list[tuple[str, str, str]]:
    """Find the ``exclusive`` relations whose two propositions the passed values both make true.

    Args:
        graph: The graph.
        network: The compiled network.
        items: The passed values as ``(flag, node id, value)``.

    Returns:
        The relation id and the two items, written ``FLAG NODE=value``, for each, in graph order.
    """
    true = {}
    for flag, node_id, value in items:
        if value:
            true.setdefault(network.index(node_id), f"{flag} {_item(node_id, value)}")
    broken = []
    for relation in graph.relations.values():
        if relation.type != EXCLUSIVE:
            continue
        ends = network.index(relation.source), network.index(relation.target)
        if all(end in true for end in ends):
            broken.append((relation.id, true[ends[0]], true[ends[1]]))
    return broken


def _either(words: list[str]) -> str:
    """Write ``words`` as one choice: ``x`` or ``any one of x, y``."""
    return words[0] if len(words) == 1 else f"any one of {', '.join(words)}"


def _split(items: list[tuple[str, str, bool]]) -> tuple[dict[str, bool], dict[str, bool]]:
    """Split passed values, as ``(flag, node id, value)``, into the ``--given`` and the ``--set`` values."""
    return {n: v for flag, n, v in items if flag == "--given"}, {n: v for flag, n, v in items if flag == "--set"}


def _failing_values(
    graph: Graph, network: Network, target: dict[str, bool], items: list[tuple[str, str, bool]]
) -> tuple[str, str]:
    """Name the flags whose values fail, and what they fail under, by re-running the query.

    A flag is named when the query fails with its values alone; when no flag's values fail alone,
    all of them fail together. Exclusive relations are named as part of the context only when the
    named values succeed without them.

    Args:
        graph: The graph.
        network: The compiled network.
        target: The target.
        items: The passed values as ``(flag, node id, value)``.

    Returns:
        The error message, and the cause for the hint.
    """
    flags = [flag for flag in ("--given", "--set") if any(item[0] == flag for item in items)]
    groups = {flag: [item for item in items if item[0] == flag] for flag in flags}
    named = [(flag, group) for flag, group in groups.items() if _fails(network, target, *_split(group))]
    if len(named) > 1:
        message = "the --given values and the --set values each have probability zero, so the query has no answer"
    elif named:
        message = f"the {named[0][0]} values have probability zero, so the query has no answer"
    else:
        named = [(" and ".join(flags), items)]
        message = f"the {named[0][0]} values have probability zero together, so the query has no answer"
    exclusive = frozenset(r.id for r in graph.relations.values() if r.type == EXCLUSIVE)
    loose = compile_graph(_rebuilt(graph, removed=exclusive)) if exclusive else network
    causes = []
    for flag, group in named:
        uses_exclusive = exclusive and not _fails(loose, target, *_split(group))
        context = "the graph's credences and exclusive relations" if uses_exclusive else "the graph's credences"
        causes.append(f"the {flag} values have probability zero under {context}")
    return message, "; ".join(causes)


def _impossible(kind: str, graph: Graph, network: Network, query: Query, error: ZeroProbabilityError) -> CliError:
    """Explain a query whose evidence has probability zero by its cause, with remedies that are checked.

    With nothing passed, or when the query fails without its ``--given`` and ``--set`` values, the
    cause is the graph. Otherwise every cause and every remedy named is established by re-running
    this same command, of the same kind, with something changed: a flag is named when its values fail
    on their own (or all flags, when only together they fail), an ``exclusive`` relation when the
    passed values make both its propositions true and removing it makes the command succeed, values
    to drop when dropping them does, and bases or strengths when moving them off 0 and 1 does. A drop
    that leaves the command ill-formed, such as an ``intervene`` query without ``--set``, is no remedy.
    Drops, moves and the removal of one relation are tried separately, never combined, and a search
    that stops at ``SEARCH_BUDGET`` says so, so no remedy found does not read as none existing.

    Args:
        kind: The kind of query.
        graph: The graph.
        network: The compiled network.
        query: The query.
        error: The engine's error.

    Returns:
        A ``zero-probability`` error. Its details hold ``drop_any_one_of`` (dropping any one of those
        values makes the command succeed) or ``drop_all_of`` (dropping all of them does),
        ``move_any_one_of`` or ``move_all_of`` for the bases and strengths at 0 or 1 whose move does,
        ``extreme_parameters`` for those parameters with their values, and ``exclusive_relation`` when
        removing one relation the passed values break does. Each list is empty when no such remedy
        was found. ``search_truncated`` lists ``drop`` and ``move`` for the searches that stopped at
        ``SEARCH_BUDGET`` with larger sets untried.
    """
    target = dict(query.target)
    items = [("--given", n, v) for n, v in query.evidence.items()] + [
        ("--set", n, v) for n, v in query.interventions.items()
    ]
    if not items or _fails(network, target, {}, {}):
        return CliError(ZERO_PROBABILITY, str(error), IMPOSSIBLE_GRAPH_HINT)
    labels = [f"{flag} {_item(node_id, value)}" for flag, node_id, value in items]

    def drop_clears(dropped: tuple[int, ...]) -> bool:
        kept = _split([item for i, item in enumerate(items) if i not in dropped])
        try:
            _check_shape(kind, target, *kept)
        except CliError:
            return False
        return not _fails(network, target, *kept)

    def succeeds_on(changed: Graph) -> bool:
        try:
            return not _fails(compile_graph(changed), target, *_split(items))
        except CompileError:
            return False

    any_drop, all_drop, drops_cut = _smallest(len(items), drop_clears)
    extreme = _extreme_parameters(graph)
    any_move, all_move, moves_cut = _smallest(
        len(extreme), lambda group: succeeds_on(_rebuilt(graph, softened=frozenset(extreme[i]["id"] for i in group)))
    )
    drops = [labels[i] for i in any_drop or all_drop]
    moves = [str(extreme[i]["id"]) for i in any_move or all_move]
    truncated = [name for name, cut in (("drop", drops_cut), ("move", moves_cut)) if cut]
    details: dict[str, object] = {
        "extreme_parameters": [extreme[i] for i in any_move or all_move],
        "drop_any_one_of": drops if any_drop else [],
        "drop_all_of": drops if all_drop else [],
        "move_any_one_of": moves if any_move else [],
        "move_all_of": moves if all_move else [],
        "search_truncated": truncated,
    }
    remedies = []
    if drops:
        remedies.append(f"drop {_either(drops)}" if any_drop else f"drop all of {', '.join(drops)}")
    message, cause = _failing_values(graph, network, target, items)
    for relation, first, second in _broken_exclusive(graph, network, items):
        if succeeds_on(_rebuilt(graph, removed=frozenset([relation]))):
            details["exclusive_relation"] = relation
            cause = f"{first} and {second} make both propositions of the exclusive relation {relation!r} true"
            remedies.append(f"remove the exclusive relation {relation!r}")
            break
    if moves:
        which = _either(moves) if any_move else f"all of {', '.join(moves)}"
        remedies.append(f"move {which} off 0 and 1 (details.extreme_parameters)")
    remedy = ", or ".join(remedies) or (
        "no drop of passed values, no move of bases or strengths off 0 and 1, and no removal of one exclusive "
        "relation that was tried makes this command succeed; a drop combined with a move or a removal, and the "
        "removal of several exclusive relations, were not tried"
    )
    if truncated:
        searched = {"drop": "drops of several values", "move": "moves of several parameters"}
        what = " and ".join(searched[name] for name in truncated)
        remedy += (
            f"; the search stopped after {SEARCH_BUDGET} tries of {what}, so one not tried may make it succeed "
            "(details.search_truncated)"
        )
    return CliError(ZERO_PROBABILITY, message, f"{cause}; {remedy}", details)


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
            raise _impossible(kind, graph, network, query, error) from None
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
