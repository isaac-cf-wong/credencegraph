"""The ``diagnose`` and ``check`` commands: weak points of a target, and whether a graph file is usable."""

from __future__ import annotations

from typing import Annotated

import typer

from credencegraph.cli.common import (
    INVALID_ARGUMENT,
    CliError,
    GraphPath,
    JsonOption,
    Result,
    compile_failure,
    read_graph,
    require_variables,
    respond,
)
from credencegraph.diagnostics.claims import DEFAULT_CLAIM_THRESHOLD
from credencegraph.diagnostics.records import Finding
from credencegraph.diagnostics.report import diagnose
from credencegraph.diagnostics.structure import unanchored
from credencegraph.diagnostics.weak_points import DEFAULT_FAILURE_THRESHOLD
from credencegraph.semantics.compiler import compile_graph
from credencegraph.semantics.errors import CompileError


def _lines(findings: list[Finding]) -> list[str]:
    """Render findings one per line.

    Args:
        findings: The findings.

    Returns:
        ``<id>: <message>`` for each finding.
    """
    return [f"{finding.id}: {finding.message}" for finding in findings]


def _threshold(value: float, option: str) -> float:
    """Check that a threshold is a probability.

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
            f"pass {option} a probability between 0 and 1",
        )
    return value


def diagnose_command(
    path: GraphPath,
    target: Annotated[
        str | None,
        typer.Option(help="The node whose weak points are wanted; without it only the graph-wide checks run."),
    ] = None,
    claim_threshold: Annotated[
        float, typer.Option(help="Report |stated - computed| above this as an overclaim or underclaim.")
    ] = DEFAULT_CLAIM_THRESHOLD,
    failure_threshold: Annotated[
        float, typer.Option(help="Report a premise whose failure leaves the target below this.")
    ] = DEFAULT_FAILURE_THRESHOLD,
    as_json: JsonOption = False,
) -> None:
    """Report the weak points of a graph file and, with --target, of one of its nodes."""

    def action() -> Result:
        claims_at = _threshold(claim_threshold, "--claim-threshold")
        failures_at = _threshold(failure_threshold, "--failure-threshold")
        graph = read_graph(path)
        if target is not None:
            require_variables(graph, (target,), "--target")
        try:
            findings = diagnose(graph, target, claim_threshold=claims_at, failure_threshold=failures_at)
        except CompileError as error:
            raise compile_failure(graph, error) from None
        payload = {"path": str(path), "target": target, "findings": [finding.to_dict() for finding in findings]}
        return Result(payload, _lines(findings) or ["no findings"])

    respond("diagnose", as_json, action)


def check_command(path: GraphPath, as_json: JsonOption = False) -> None:
    """Check that a graph file is valid and can be compiled for inference.

    The file must parse as a credencegraph graph: known fields, unique ids, relations between existing
    nodes, no cycle among requires, supports and refutes, and no equivalent or exclusive relation
    joining a node to itself. It must also compile: every inference
    variable needs a base, and equivalent nodes need the same one. A graph that does not compile is
    reported as a compile-error, with the counts and every finding under details. An unanchored
    variable is reported as a warning and does not fail the check.
    """

    def action() -> Result:
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
