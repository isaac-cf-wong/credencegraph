"""Run every diagnostic on a graph at once."""

from __future__ import annotations

from collections.abc import Mapping

from credencegraph.core.graph import Graph
from credencegraph.diagnostics.claims import DEFAULT_CLAIM_THRESHOLD, claims
from credencegraph.diagnostics.records import Finding
from credencegraph.diagnostics.structure import missing_parameters, unanchored
from credencegraph.diagnostics.weak_points import (
    DEFAULT_FAILURE_THRESHOLD,
    crux_findings,
    derivatives,
    sensitivity_findings,
    single_points_of_failure,
    value_of_information,
)
from credencegraph.inference.elimination import VariableElimination
from credencegraph.inference.engine import Engine
from credencegraph.semantics.compiler import compile_graph


def diagnose(  # noqa: PLR0913 - the options after the target are keyword-only
    graph: Graph,
    target: str | None = None,
    *,
    evidence: Mapping[str, bool] | None = None,
    claim_threshold: float = DEFAULT_CLAIM_THRESHOLD,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    engine: Engine | None = None,
) -> list[Finding]:
    """Run the diagnostics that apply to a graph and, optionally, a target node.

    The structural checks (missing parameters, unanchored variables) always run. If a parameter is
    missing the graph cannot be compiled, and the report stops there. Otherwise the stated credences
    are compared with the computed ones, and, when a target is given, its sensitivity, crux,
    single points of failure and value of information follow.

    With evidence every inference diagnostic is taken given it: the computed credences are
    ``P(X | evidence)``, the sensitivities and cruxes are derivatives of ``P(target | evidence)``,
    the single points of failure compare ``P(target | do(Y = false), evidence)`` with
    ``P(target | evidence)``, and the value of information is ``I(target; Y | evidence)``. Observed
    variables are left out of the comparisons and rankings over nodes, and the target may not be
    observed. Without evidence the report is the one for the graph before anything is observed.

    Args:
        graph: The graph.
        target: The id of the node whose weak points are wanted, if any.
        evidence: Node ids and their observed values; none by default.
        claim_threshold: The threshold passed to ``claims``, a gap in natural log-odds; the default,
            ``2 ln(11 / 9)``, is a convention of this package.
        failure_threshold: The threshold passed to ``single_points_of_failure``, a fraction of the
            target's own probability; the default, 0.1, is a convention of this package.
        engine: The exact engine; variable elimination by default.

    Returns:
        The findings, grouped by diagnostic in the order above; ``[f.to_dict() for f in ...]`` is
        ready for ``json.dumps``.

    Raises:
        ValidationError: If a threshold is out of range, ``target`` or an evidence node is not an
            inference variable, or the evidence observes the target.
        CompileError: If the graph has no missing parameter but still fails to compile, for example
            over conflicting bases on merged nodes.
        ZeroProbabilityError: If the evidence and the ``exclusive`` constraints have probability zero.
    """
    findings = missing_parameters(graph)
    anchors = unanchored(graph)
    if findings:
        return [*findings, *anchors]
    findings.extend(anchors)
    engine = engine or VariableElimination()
    network = compile_graph(graph)
    findings.extend(claims(graph, network, evidence=evidence, threshold=claim_threshold, engine=engine))
    if target is None:
        return findings
    slopes = derivatives(network, target, evidence=evidence, engine=engine)
    findings.extend(sensitivity_findings(network, target, slopes, evidence=evidence))
    findings.extend(crux_findings(network, target, slopes, evidence=evidence))
    findings.extend(
        single_points_of_failure(network, target, evidence=evidence, threshold=failure_threshold, engine=engine)
    )
    findings.extend(value_of_information(network, target, evidence=evidence, engine=engine))
    return findings
