"""Consistency: the credence a source states against the credence its own argument delivers."""

from __future__ import annotations

import math
from collections.abc import Mapping

from credencegraph.core.errors import ValidationError
from credencegraph.core.graph import Graph
from credencegraph.diagnostics.records import OVERCLAIM, ROUNDING, UNDERCLAIM, Finding, render_evidence
from credencegraph.inference.elimination import VariableElimination
from credencegraph.inference.engine import Engine, Query
from credencegraph.semantics.compiler import compile_graph
from credencegraph.semantics.network import Network

# A convention of this package, not something the model determines. The gap is in log-odds rather than
# probability so that 0.07 stated against a computed 0.001 is reported, though it is only 0.069 apart.
# The default is the infimum of the log-odds gap between two probabilities more than 0.1 apart: no such
# pair reaches it, but pairs just over 0.1 apart around 0.5 come arbitrarily close. Over a window
# [p, p + 0.1], logit(p + 0.1) - logit(p) is convex in p and its derivative vanishes where both ends have
# the same slope 1 / (p (1 - p)), that is at the window [0.45, 0.55] centred on even odds:
# 2 ln(0.55 / 0.45) = 2 ln(11 / 9), about 0.401, odds that differ by a factor of (11 / 9)^2, about 1.49.
# A gap is reported when it exceeds the threshold by more than ROUNDING (1e-12), and the former absolute
# threshold of 0.1 had the same allowance, so what is guaranteed is that every pair the former threshold
# reported, more than 0.1 + 1e-12 apart in probability, is still reported: near [0.45, 0.55] such a
# pair's log-odds gap exceeds the default by at least about 4e-12. A pair less than 1e-12 beyond 0.1
# apart may go unreported, as it did before, and raising the threshold by about 3e-12 or more drops
# some pairs the former threshold reported. Pass a threshold suited to the text.
DEFAULT_CLAIM_THRESHOLD = 2 * math.log(0.55 / 0.45)


def _log_odds_gap(stated: float, computed: float) -> float:
    """Return logit(stated) - logit(computed), infinite when exactly one side is certain.

    Args:
        stated: The stated probability.
        computed: The computed probability.

    Returns:
        The signed gap in natural log-odds; 0 when the two are within 1e-12 of each other.
    """
    if abs(stated - computed) <= ROUNDING:
        return 0.0
    if stated in (0.0, 1.0) or computed in (0.0, 1.0):
        return math.inf if stated > computed else -math.inf
    return math.log(stated / (1 - stated)) - math.log(computed / (1 - computed))


def _claim_threshold(value: object) -> float:
    """Validate a threshold on a log-odds gap.

    Args:
        value: The candidate threshold.

    Returns:
        The threshold as a float.

    Raises:
        ValidationError: If ``value`` is not a finite real number >= 0.
    """
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value < math.inf:
        msg = f"threshold: must be a finite number >= 0, got {value!r}"
        raise ValidationError(msg)
    return float(value)


def claims(
    graph: Graph,
    network: Network | None = None,
    *,
    evidence: Mapping[str, bool] | None = None,
    threshold: float = DEFAULT_CLAIM_THRESHOLD,
    engine: Engine | None = None,
) -> list[Finding]:
    """Compare each node's ``stated`` credence with the probability its premises give it.

    The computed value is the point answer ``P(X | evidence)`` of the network, ``P(X)`` when there
    is no evidence. A node whose variable is observed is skipped: its value is then the observation,
    not something its premises deliver. The two are compared in
    log-odds, ``logit(p) = ln(p / (1 - p))``, so that a gap counts by the factor between the odds
    rather than by the difference between the probabilities: 0.07 against 0.001 is far apart, 0.5
    against 0.569 is not. When the stated mean exceeds the computed value by more than ``threshold``
    the text asserts more confidence than its own argument delivers (an overclaim); when it falls
    short by more than ``threshold``, less (an underclaim). A stated or computed value of exactly 0
    or 1 is infinitely far from any other. Nodes with a ``stated`` credence that are not inference
    variables have nothing to compare against and are skipped.

    Args:
        graph: The graph.
        network: The network compiled from ``graph``; compiled here when omitted.
        evidence: Node ids and their observed values; none by default.
        threshold: The largest gap that is not reported, in natural log-odds: ``ln(k)`` reports
            odds that differ by more than a factor of ``k``. A gap within 1e-12 of it counts as equal
            to it, so rounding cannot tip a finding either way. The default, ``2 ln(11 / 9)``, about
            0.401, is a convention chosen for this package rather than a value the model fixes: it
            is the log-odds gap between 0.45 and 0.55, the infimum of the gaps between probabilities
            more than 0.1 apart, which such pairs approach arbitrarily closely but never reach. With
            the same 1e-12 allowance, every pair more than 0.1 + 1e-12 apart in probability, which
            is every pair an absolute threshold of 0.1 would report, is reported.
        engine: The inference engine; variable elimination by default.

    Returns:
        One finding per node whose stated and computed credences differ by more than ``threshold``,
        in insertion order. Its value is the signed gap in probability, stated minus computed, which
        stays finite when the log-odds gap is not. Its details are ``stated`` and ``computed``, in
        probability, and ``threshold_log_odds``, the threshold applied, in natural log-odds.

    Raises:
        ValidationError: If ``threshold`` is not a finite number >= 0, or an evidence node is not an
            inference variable.
        CompileError: If ``network`` is omitted and ``graph`` does not compile.
        ZeroProbabilityError: If the evidence and the ``exclusive`` constraints have probability zero.
    """
    limit = _claim_threshold(threshold)
    network = network if network is not None else compile_graph(graph)
    engine = engine or VariableElimination()
    given = Query({}, evidence or {}).evidence
    observed = {network.index(node_id) for node_id in given}
    if given:
        engine.query(network, Query({}, given))  # raises when the evidence has probability zero
    variables = {member for variable in network.variables for member in variable.members}
    suffix = f" given {render_evidence(given)}" if given else ""
    findings: list[Finding] = []
    for node in graph:
        if node.stated is None or node.id not in variables or network.index(node.id) in observed:
            continue
        stated = node.stated.mean
        computed = engine.query(network, Query({node.id: True}, given))
        gap = stated - computed
        if abs(_log_odds_gap(stated, computed)) <= limit + ROUNDING:
            continue
        diagnostic, verb = (OVERCLAIM, "more") if gap > 0 else (UNDERCLAIM, "less")
        findings.append(
            Finding(
                id=f"{diagnostic}:{node.id}",
                diagnostic=diagnostic,
                message=(
                    f"node {node.id!r} is stated at {stated:.3g} but its premises give {computed:.3g}{suffix}: "
                    f"the text asserts {abs(gap):.3g} {verb} confidence than its argument delivers"
                ),
                nodes=(node.id,),
                value=gap,
                details={"stated": stated, "computed": computed, "threshold_log_odds": limit},
            )
        )
    return findings
