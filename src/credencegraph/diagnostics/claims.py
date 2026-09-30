"""Consistency: the credence a source states against the credence its own argument delivers."""

from __future__ import annotations

from credencegraph.core.graph import Graph
from credencegraph.diagnostics.records import OVERCLAIM, ROUNDING, UNDERCLAIM, Finding, probability_threshold
from credencegraph.inference.elimination import VariableElimination
from credencegraph.inference.engine import Engine, Query
from credencegraph.semantics.compiler import compile_graph
from credencegraph.semantics.network import Network

# A convention of this package, not something the model determines: a gap of 0.1 in probability is
# where a stated credence starts to read as a different claim. Pass a threshold suited to the text.
DEFAULT_CLAIM_THRESHOLD = 0.1


def claims(
    graph: Graph,
    network: Network | None = None,
    *,
    threshold: float = DEFAULT_CLAIM_THRESHOLD,
    engine: Engine | None = None,
) -> list[Finding]:
    """Compare each node's ``stated`` credence with the probability its premises give it.

    The computed value is the point answer ``P(X)`` of the network. When the stated mean exceeds it
    by more than ``threshold`` the text asserts more confidence than its own argument delivers (an
    overclaim); when it falls short by more than ``threshold``, less (an underclaim). Nodes with a
    ``stated`` credence that are not inference variables have nothing to compare against and are
    skipped.

    Args:
        graph: The graph.
        network: The network compiled from ``graph``; compiled here when omitted.
        threshold: The largest gap, in probability, that is not reported; a gap within 1e-12 of it
            counts as equal to it, so rounding cannot tip a finding either way. The default, 0.1,
            is a convention chosen for this package rather than a value the model fixes.
        engine: The inference engine; variable elimination by default.

    Returns:
        One finding per node whose stated and computed credences differ by more than ``threshold``,
        in insertion order. Its value is the signed gap, stated minus computed.

    Raises:
        ValidationError: If ``threshold`` is not in [0, 1].
        CompileError: If ``network`` is omitted and ``graph`` does not compile.
        ZeroProbabilityError: If the ``exclusive`` constraints have probability zero.
    """
    limit = probability_threshold(threshold, "threshold")
    network = network if network is not None else compile_graph(graph)
    engine = engine or VariableElimination()
    variables = {member for variable in network.variables for member in variable.members}
    findings: list[Finding] = []
    for node in graph:
        if node.stated is None or node.id not in variables:
            continue
        stated = node.stated.mean
        computed = engine.query(network, Query({node.id: True}))
        gap = stated - computed
        if abs(gap) <= limit + ROUNDING:
            continue
        diagnostic, verb = (OVERCLAIM, "more") if gap > 0 else (UNDERCLAIM, "less")
        findings.append(
            Finding(
                id=f"{diagnostic}:{node.id}",
                diagnostic=diagnostic,
                message=(
                    f"node {node.id!r} is stated at {stated:.3g} but its premises give {computed:.3g}: "
                    f"the text asserts {abs(gap):.3g} {verb} confidence than its argument delivers"
                ),
                nodes=(node.id,),
                value=gap,
                details={"stated": stated, "computed": computed, "threshold": limit},
            )
        )
    return findings
