"""The ``diagnose`` and ``check`` commands: weak points of targets, and whether a graph file is usable."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import typer

from credencegraph.cli.common import (
    INVALID_ARGUMENT,
    PROBLEM_TOO_LARGE,
    RUBRIC_VIOLATION,
    ZERO_PROBABILITY,
    CliError,
    GraphPath,
    JsonOption,
    Result,
    check_consistent,
    compile_checked,
    compile_failure,
    parse_assignment,
    read_graph,
    read_rubric,
    require_variables,
    respond,
    translate,
)
from credencegraph.core.graph import Graph
from credencegraph.diagnostics.claims import DEFAULT_CLAIM_THRESHOLD
from credencegraph.diagnostics.records import Finding
from credencegraph.diagnostics.report import diagnose_many
from credencegraph.diagnostics.structure import missing_parameters, unanchored
from credencegraph.diagnostics.weak_points import DEFAULT_FAILURE_THRESHOLD
from credencegraph.inference.elimination import DEFAULT_MAX_FACTOR_SIZE, VariableElimination
from credencegraph.inference.engine import Query
from credencegraph.inference.errors import ProblemTooLargeError, ZeroProbabilityError
from credencegraph.rubric.validate import COMPILE_ERROR, LEVELS, TYPED, check_graph
from credencegraph.semantics.compiler import compile_graph, inference_sets
from credencegraph.semantics.errors import CompileError
from credencegraph.semantics.network import Network


def _lines(findings: list[Finding]) -> list[str]:
    """Render findings one per line.

    Args:
        findings: The findings.

    Returns:
        ``<id>: <message>`` for each finding.
    """
    return [f"{finding.id}: {finding.message}" for finding in findings]


def _threshold(value: float, option: str) -> float:
    """Check that a threshold is a number in [0, 1].

    Args:
        value: The value given on the command line.
        option: The option it came from, used in the error message.

    Returns:
        The value.

    Raises:
        CliError: If the value is not in [0, 1].
    """
    if not 0.0 <= value <= 1.0:
        raise CliError(
            INVALID_ARGUMENT,
            f"{option} must lie in [0, 1], got {value!r}",
            f"pass {option} a number in [0, 1]",
        )
    return value


def _log_odds_threshold(value: float, option: str) -> float:
    """Check that a threshold is a finite gap in log-odds.

    Args:
        value: The value given on the command line.
        option: The option it came from, used in the error message.

    Returns:
        The value.

    Raises:
        CliError: If the value is negative, infinite or not a number.
    """
    if not 0.0 <= value < math.inf:
        raise CliError(
            INVALID_ARGUMENT,
            f"{option} must be a finite number >= 0, got {value!r}",
            f"pass {option} a gap in natural log-odds, such as 0.69 for a factor of 2 in the odds",
        )
    return value


def _targets(graph: Graph, explicit: list[str], selection: str | None, evidence: dict[str, bool]) -> list[str]:
    """Resolve ``--target`` and ``--targets`` to the nodes to diagnose.

    Args:
        graph: The graph.
        explicit: The ``--target`` values, in the order given.
        selection: The ``--targets`` value, if any; only ``stated`` is accepted.
        evidence: The ``--given`` values.

    Returns:
        The ``--target`` nodes, then, for ``--targets stated``, every node with a ``stated``
        credence in the graph's order, without repeats. A stated node that takes no part in
        inference, or is observed directly or through an equivalent node, is left out, as the claim
        check leaves it out: it has no weak points to report.

    Raises:
        CliError: If ``--targets`` is not ``stated``, or a ``--target`` node is not an inference variable.
    """
    require_variables(graph, explicit, "--target")
    if selection is None:
        return list(dict.fromkeys(explicit))
    if selection != "stated":
        raise CliError(
            INVALID_ARGUMENT,
            f"--targets must be 'stated', got {selection!r}",
            "pass --targets stated for every node with a stated credence, or name nodes with --target",
        )
    observed: set[str] = set()
    variables: set[str] = set()
    for members in inference_sets(graph).values():
        variables.update(members)
        if any(member in evidence for member in members):
            observed.update(members)
    stated = [
        node_id
        for node_id, node in graph.nodes.items()
        if node.stated is not None and node_id in variables and node_id not in observed
    ]
    return list(dict.fromkeys([*explicit, *stated]))


def _check_evidence(engine: VariableElimination, network: Network, evidence: dict[str, bool]) -> None:
    """Refuse evidence that the graph gives probability zero, naming the cause.

    Args:
        engine: The engine the command diagnoses with.
        network: The compiled network.
        evidence: The ``--given`` values.

    Raises:
        CliError: A ``zero-probability`` error pointing at the graph when no world satisfies its
            ``exclusive`` relations even without the evidence, and at ``--given`` otherwise.
    """
    try:
        engine.query(network, Query({}, evidence))
    except ZeroProbabilityError:
        try:
            engine.query(network, Query({}))
        except ZeroProbabilityError as error:
            raise translate(error, "diagnose") from None
        raise CliError(
            ZERO_PROBABILITY,
            "the --given values have probability zero, so there is nothing to diagnose under them",
            "the --given values have probability zero under the graph's credences; drop or change one of them. "
            "'credencegraph query' with the same --given values names the drops and the moves of bases or "
            "strengths off 0 and 1 that make them possible",
            {"given": evidence},
        ) from None


def diagnose_command(  # noqa: PLR0913, PLR0917 - Typer maps one parameter to each option
    path: GraphPath,
    target: Annotated[
        list[str] | None,
        typer.Option(
            help="A node whose weak points are wanted; repeat for several. Without it, or --targets, only the graph-wide checks run.",
            show_default=False,
        ),
    ] = None,
    targets: Annotated[
        str | None,
        typer.Option(
            help="'stated' diagnoses every node with a stated credence, after any --target nodes.",
            show_default=False,
        ),
    ] = None,
    given: Annotated[
        list[str] | None,
        typer.Option(
            help="Evidence to diagnose under, NODE=true|false; repeat for several. Every inference diagnostic is then conditioned on it.",
            show_default=False,
        ),
    ] = None,
    claim_threshold: Annotated[
        float,
        typer.Option(
            help="Report |logit(stated) - logit(computed)| above this, in natural log-odds, as an overclaim or underclaim."
        ),
    ] = DEFAULT_CLAIM_THRESHOLD,
    failure_threshold: Annotated[
        float,
        typer.Option(
            help=(
                "Report a premise whose failure leaves the target below this fraction of its own probability; "
                "every premise is ranked by that fraction, and this marks the line. A premise behind one requires "
                "of strength r crosses it only if r > 1 - threshold."
            )
        ),
    ] = DEFAULT_FAILURE_THRESHOLD,
    max_factor_size: Annotated[
        int, typer.Option(help="The largest intermediate factor exact inference may build, in table entries.")
    ] = DEFAULT_MAX_FACTOR_SIZE,
    as_json: JsonOption = False,
) -> None:
    """Report the weak points of a graph file and, with --target or --targets, of some of its nodes.

    The graph-wide checks run once; each target's weak points follow, and in --json every finding
    names its target, null for a graph-wide one.

    With --given, every inference diagnostic is taken given the evidence: stated credences are
    compared with P(X | evidence), and each target's sensitivity, crux, single points of failure,
    failure impact and value of information are those of P(target | evidence).
    """

    def action() -> Result:
        evidence = parse_assignment(given or (), "--given")
        claims_at = _log_odds_threshold(claim_threshold, "--claim-threshold")
        failures_at = _threshold(failure_threshold, "--failure-threshold")
        if max_factor_size < 1:
            raise CliError(
                INVALID_ARGUMENT,
                f"--max-factor-size must be 1 or more, got {max_factor_size}",
                f"the default is {DEFAULT_MAX_FACTOR_SIZE} table entries",
            )
        graph = read_graph(path)
        chosen = _targets(graph, target or [], targets, evidence)
        require_variables(graph, evidence, "--given")
        engine = VariableElimination(max_factor_size)
        try:
            # A graph with a missing parameter is reported by its structural findings, whatever the evidence.
            if evidence and not missing_parameters(graph):
                network = compile_checked(graph)
                check_consistent(network, evidence, {})
                _check_evidence(engine, network, evidence)
            report = diagnose_many(
                graph,
                chosen,
                evidence=evidence,
                claim_threshold=claims_at,
                failure_threshold=failures_at,
                engine=engine,
            )
        except CompileError as error:
            raise compile_failure(graph, error) from None
        except ProblemTooLargeError as error:
            raise CliError(
                PROBLEM_TOO_LARGE,
                str(error),
                f"the graph is too large for exact inference at this limit; raise it with --max-factor-size, "
                f"to at least {error.required}",
                {"required": error.required, "limit": error.limit},
            ) from None
        # Without --given the response is exactly the one for the graph before anything is observed.
        payload = {"path": str(path), "targets": chosen, **({"given": evidence} if evidence else {})}
        payload["findings"] = report.to_dicts()
        return Result(payload, _lines(report.all_findings()) or ["no findings"])

    respond("diagnose", as_json, action)


RubricOption = Annotated[
    Path | None,
    typer.Option("--rubric", help="A TOML rubric to check the graph against.", show_default=False),
]


def check_command(
    path: GraphPath,
    rubric: RubricOption = None,
    level: Annotated[
        str | None,
        typer.Option(
            help="With --rubric: anchored, typed or assessed; each level includes the ones before it. Default: typed.",
            show_default=False,
        ),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Check that a graph file is valid and can be compiled for inference, or that it meets a rubric.

    The file must parse as a credencegraph graph: known fields, unique ids, relations between existing
    nodes, no cycle among requires, supports and refutes, and no equivalent or exclusive relation
    joining a node to itself. Without --rubric it must also compile: every inference
    variable needs a base, and equivalent nodes need the same one. A graph that does not compile is
    reported as a compile-error, with the counts and every finding under details. An unanchored
    variable is reported as a warning and does not fail the check.

    With --rubric, the graph is checked against the rubric at --level instead, and every violation
    is reported as a rubric-violation. The graph is compiled only at the assessed level.
    """

    def action() -> Result:
        if level is not None and rubric is None:
            raise CliError(INVALID_ARGUMENT, "--level needs --rubric", "pass --rubric with the rubric to check against")
        if rubric is not None:
            return _rubric_check(path, rubric, level or TYPED)
        graph = read_graph(path)
        warnings = unanchored(graph)
        counts = {"path": str(path), "nodes": len(graph.nodes), "relations": len(graph.relations)}
        try:
            compile_graph(graph)
        except CompileError as error:
            failure = compile_failure(graph, error)
            failure.details = {**counts, **failure.details, "warnings": [finding.to_dict() for finding in warnings]}
            raise failure from None
        payload = {**counts, "warnings": [finding.to_dict() for finding in warnings]}
        text = [f"{path}: ok ({len(graph.nodes)} nodes, {len(graph.relations)} relations)"]
        text.extend(f"warning: {line}" for line in _lines(warnings))
        return Result(payload, text)

    respond("check", as_json, action)


def _rubric_check(path: Path, rubric_path: Path, level: str) -> Result:
    """Check a graph file against a rubric at a level.

    Args:
        path: The graph file.
        rubric_path: The rubric file.
        level: The validation level.

    Returns:
        ``{"path", "rubric", "level", "nodes", "relations", "violations"}``, with no violations.

    Raises:
        CliError: A ``rubric-violation`` listing every violation in ``details.violations``, with the
            compiler's ``compile-error`` details under ``details.compile_error`` when the graph does
            not compile; or ``invalid-argument`` for an unknown level.
    """
    if level not in LEVELS:
        raise CliError(
            INVALID_ARGUMENT,
            f"--level must be one of {', '.join(LEVELS)}, got {level!r}",
            f"pass --level {' or '.join(LEVELS)}",
        )
    graph = read_graph(path)
    rubric = read_rubric(rubric_path)
    violations = check_graph(graph, rubric, level)
    payload = {
        "path": str(path),
        "rubric": rubric.identity(),
        "level": level,
        "nodes": len(graph.nodes),
        "relations": len(graph.relations),
        "violations": [finding.to_dict() for finding in violations],
    }
    if not violations:
        return Result(payload, [f"{path}: ok at level {level} against rubric {rubric.name} {rubric.version}"])
    if any(finding.diagnostic == COMPILE_ERROR for finding in violations):
        try:
            compile_graph(graph)
        except CompileError as error:
            payload["compile_error"] = compile_failure(graph, error).details
    listed = "; ".join(f"{finding.diagnostic}: {finding.message}" for finding in violations)
    raise CliError(
        RUBRIC_VIOLATION,
        f"{len(violations)} violation(s) of rubric {rubric.name!r} at level {level}: {listed}",
        "each violation names its node and rule; fix them, or check at a lower --level",
        payload,
    )
