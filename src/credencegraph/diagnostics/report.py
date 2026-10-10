"""Run every diagnostic on a graph at once."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from credencegraph.core.attributes import JSON
from credencegraph.core.errors import ValidationError
from credencegraph.core.graph import Graph
from credencegraph.diagnostics.claims import DEFAULT_CLAIM_THRESHOLD, claims
from credencegraph.diagnostics.records import Finding
from credencegraph.diagnostics.structure import DEFAULT_MIN_SUPPORTS, correlated_support, missing_parameters, unanchored
from credencegraph.diagnostics.weak_points import (
    DEFAULT_FAILURE_THRESHOLD,
    crux_findings,
    derivatives,
    failure_impact_findings,
    interventions,
    sensitivity_findings,
    single_point_of_failure_findings,
    value_of_information,
)
from credencegraph.inference.elimination import VariableElimination
from credencegraph.inference.engine import Engine
from credencegraph.semantics.compiler import compile_graph


@dataclass(frozen=True, slots=True)
class Report:
    """The findings for a graph and for each of several target nodes.

    Attributes:
        findings: The graph-wide findings: missing parameters, unanchored variables, correlated
            supports, overclaims and underclaims. They do not depend on a target and appear once.
        weak_points: Each target's sensitivity, crux, single points of failure, failure impact and
            value of information, by target id in the order the targets were given. A finding id is
            unique within one target's list; the same id, such as ``crux:strength:r1``, can recur for
            another target.
    """

    findings: tuple[Finding, ...] = ()
    weak_points: Mapping[str, tuple[Finding, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Store read-only copies of the findings and of the weak points."""
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(
            self,
            "weak_points",
            MappingProxyType({target: tuple(found) for target, found in self.weak_points.items()}),
        )

    def all_findings(self) -> list[Finding]:
        """Return the graph-wide findings, then each target's.

        Returns:
            Every finding, in report order.
        """
        return [*self.findings, *(finding for found in self.weak_points.values() for finding in found)]

    def to_dicts(self) -> list[dict[str, JSON]]:
        """Return every finding as JSON-ready data, naming the target of each.

        Returns:
            The graph-wide findings, then each target's, as ``Finding.to_dict`` with a ``"target"``
            key: ``None`` for a graph-wide finding, and the target's id for a weak point.
        """
        records = [{**finding.to_dict(), "target": None} for finding in self.findings]
        for target, found in self.weak_points.items():
            records.extend({**finding.to_dict(), "target": target} for finding in found)
        return records


def diagnose(  # noqa: PLR0913 - the options after the target are keyword-only
    graph: Graph,
    target: str | None = None,
    *,
    evidence: Mapping[str, bool] | None = None,
    claim_threshold: float = DEFAULT_CLAIM_THRESHOLD,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    min_supports: int = DEFAULT_MIN_SUPPORTS,
    engine: Engine | None = None,
) -> list[Finding]:
    """Run the diagnostics that apply to a graph and, optionally, a target node.

    The structural checks (missing parameters, unanchored variables, correlated supports) always
    run. If a parameter is missing the graph cannot be compiled, and the report stops there. Otherwise the stated credences
    are compared with the computed ones, and, when a target is given, its sensitivity, crux,
    single points of failure, failure impact and value of information follow. ``diagnose_many``
    does the same for several targets, running the graph-wide checks once.

    With evidence every inference diagnostic is taken given it: the computed credences are
    ``P(X | evidence)``, the sensitivities and cruxes are derivatives of ``P(target | evidence)``,
    the single points of failure and the failure impact compare ``P(target | do(Y = false), evidence)``
    with ``P(target | evidence)``, and the value of information is ``I(target; Y | evidence)``. Observed
    variables are left out of the comparisons and rankings over nodes, and the target may not be
    observed. Without evidence the report is the one for the graph before anything is observed.

    Args:
        graph: The graph.
        target: The id of the node whose weak points are wanted, if any.
        evidence: Node ids and their observed values; none by default.
        claim_threshold: The threshold passed to ``claims``, a gap in natural log-odds; the default,
            ``2 ln(11 / 9)``, is a convention of this package.
        failure_threshold: The threshold passed to ``single_points_of_failure``, and marked in the
            ``failure_impact`` ranking, a fraction of the target's own probability; the default, 0.1,
            is a convention of this package. A premise behind one ``requires`` of strength r crosses
            it only if r > 1 - threshold.
        min_supports: The smallest group passed to ``correlated_support``, at least 2.
        engine: The exact engine; variable elimination by default.

    Returns:
        The findings, grouped by diagnostic in the order above; ``[f.to_dict() for f in ...]`` is
        ready for ``json.dumps``.

    Raises:
        ValidationError: If a threshold or ``min_supports`` is out of range, ``target`` or an evidence
            node is not an inference variable, or the evidence observes the target.
        CompileError: If the graph has no missing parameter but still fails to compile, for example
            over conflicting bases on merged nodes.
        ZeroProbabilityError: If the evidence and the ``exclusive`` constraints have probability zero.
    """
    report = diagnose_many(
        graph,
        () if target is None else (target,),
        evidence=evidence,
        claim_threshold=claim_threshold,
        failure_threshold=failure_threshold,
        min_supports=min_supports,
        engine=engine,
    )
    return report.all_findings()


def diagnose_many(  # noqa: PLR0913 - the options after the targets are keyword-only
    graph: Graph,
    targets: Iterable[str],
    *,
    evidence: Mapping[str, bool] | None = None,
    claim_threshold: float = DEFAULT_CLAIM_THRESHOLD,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    min_supports: int = DEFAULT_MIN_SUPPORTS,
    engine: Engine | None = None,
) -> Report:
    """Run the graph-wide diagnostics once and the weak-point diagnostics for each of several targets.

    The graph-wide findings are those ``diagnose`` reports without a target; each target's weak
    points are those ``diagnose`` adds for it, under the same evidence and thresholds. A target named
    twice is diagnosed once. If a parameter is missing the graph cannot be compiled, and every
    target's list is empty.

    Args:
        graph: The graph.
        targets: The ids of the nodes whose weak points are wanted; may be empty.
        evidence: Node ids and their observed values; none by default.
        claim_threshold: The threshold passed to ``claims``, a gap in natural log-odds.
        failure_threshold: The threshold passed to ``single_points_of_failure``, a fraction of each
            target's own probability.
        min_supports: The smallest group passed to ``correlated_support``, at least 2.
        engine: The exact engine; variable elimination by default.

    Returns:
        The graph-wide findings, and each target's weak points in the order the targets were given;
        ``report.to_dicts()`` is ready for ``json.dumps``.

    Raises:
        ValidationError: If ``targets`` is a single string, a threshold or ``min_supports`` is out of
            range, a target or an evidence node is not an inference variable, or the evidence observes
            a target.
        CompileError: If the graph has no missing parameter but still fails to compile, for example
            over conflicting bases on merged nodes.
        ZeroProbabilityError: If the evidence and the ``exclusive`` constraints have probability zero.
    """
    if isinstance(targets, str):
        msg = f"targets must be a collection of node ids, not the string {targets!r}"
        raise ValidationError(msg)
    targets = list(dict.fromkeys(targets))
    findings = missing_parameters(graph)
    structural = [*unanchored(graph), *correlated_support(graph, min_supports)]
    if findings:
        return Report((*findings, *structural), dict.fromkeys(targets, ()))
    findings.extend(structural)
    engine = engine or VariableElimination()
    network = compile_graph(graph)
    findings.extend(claims(graph, network, evidence=evidence, threshold=claim_threshold, engine=engine))
    weak_points: dict[str, list[Finding]] = {}
    for target in targets:
        slopes = derivatives(network, target, evidence=evidence, engine=engine)
        found = weak_points[target] = []
        found.extend(sensitivity_findings(network, target, slopes, evidence=evidence))
        found.extend(crux_findings(network, target, slopes, evidence=evidence))
        run = interventions(network, target, evidence=evidence, threshold=failure_threshold, engine=engine)
        found.extend(single_point_of_failure_findings(run))
        found.extend(failure_impact_findings(run))
        found.extend(value_of_information(network, target, evidence=evidence, engine=engine))
    return Report(findings, weak_points)
