"""Weak points of a target: what its probability depends on, and what would change it most.

Every function here takes a compiled network and the id of a target node T, and asks about
``P(T = true | E)`` with the parameters at the values the network holds (their credence means for a
freshly compiled network). E is the evidence passed as ``evidence``, node ids and their observed
values; with none, the question is about ``P(T = true)``, the graph before anything is observed.

**Sensitivity.** Every joint probability of the network is multilinear in its parameters: each
``base`` and ``strength`` enters with degree at most one. With no ``exclusive`` constraint, ``P(T)``
is itself such a joint probability, a straight line in each parameter θ, so

    dP(T)/dθ = P(T | θ = 1) - P(T | θ = 0)

exactly, from two evaluations. Evidence E, and an ``exclusive`` constraint C, which is observed true
in every query, make ``P(T | E) = P(T, E, C) / P(E, C)`` a ratio of two such lines, N(θ) and D(θ).
The ratio is not multilinear in θ, so the two-point difference is then no longer the derivative. It
is taken instead from the quotient rule,

    dP(T | E)/dθ = (N'(θ) D(θ) - N(θ) D'(θ)) / D(θ)^2,  N' = N(1) - N(0),  D' = D(1) - D(0),

which is exact because N and D are each lines in θ: four evaluations at θ = 0 and θ = 1, and two at
the value θ holds. With no evidence and no constraint, D is 1 and this is the two-point difference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from credencegraph.core.credence import Credence
from credencegraph.core.errors import ValidationError
from credencegraph.diagnostics.records import (
    CRUX,
    ROUNDING,
    SENSITIVITY,
    SINGLE_POINT_OF_FAILURE,
    VALUE_OF_INFORMATION,
    Finding,
    probability_threshold,
    render_evidence,
)
from credencegraph.inference.elimination import VariableElimination
from credencegraph.inference.engine import Engine, Query
from credencegraph.inference.errors import ZeroProbabilityError
from credencegraph.semantics.network import PROPOSITION, Network, ParameterKey

# A convention of this package, not something the model determines: a premise whose failure leaves the
# target below a tenth of its own probability, an order of magnitude down, sinks it. The threshold is a
# fraction of P(target) rather than a probability, so that a target that is already improbable still has
# its premises ranked instead of every one of them reported. Pass a threshold suited to the question.
DEFAULT_FAILURE_THRESHOLD = 0.1


def _observed(network: Network, target: str, evidence: Mapping[str, bool] | None) -> Query:
    """Validate the evidence against the target.

    Args:
        network: The network.
        target: The id of the target node.
        evidence: Node ids and their observed values, or ``None`` for none.

    Returns:
        The query ``P(target = true | evidence)``.

    Raises:
        ValidationError: If ``evidence`` is not a mapping of node ids to bools, names a node that is
            not an inference variable, or observes the target, or a node merged with it by
            ``equivalent`` relations: every diagnostic of a target fixed by observation is trivial.
    """
    query = Query({target: True}, evidence or {})
    own = network.index(target)
    for node_id in query.evidence:
        if network.index(node_id) == own:
            merged = "" if node_id == target else ", which is equivalent to it,"
            msg = (
                f"evidence observes {node_id!r}{merged} so the target {target!r} is fixed and has no weak "
                "points; observe other nodes, or diagnose another target"
            )
            raise ValidationError(msg)
    return query


def _conditions(network: Network, evidence: Mapping[str, bool]) -> dict[int, int]:
    """Resolve the evidence and the ``exclusive`` constraints into one assignment of variables.

    Args:
        network: The network.
        evidence: Node ids and their observed values.

    Returns:
        Variable indices and values.

    Raises:
        ZeroProbabilityError: If the evidence gives nodes merged by ``equivalent`` relations
            different values, which no world satisfies.
    """
    conditions = dict.fromkeys(network.constraints, 1)
    for node_id, value in evidence.items():
        index = network.index(node_id)
        if conditions.get(index, int(value)) != int(value):
            msg = "the evidence gives equivalent nodes different values, which has probability zero"
            raise ZeroProbabilityError(msg)
        conditions[index] = int(value)
    return conditions


def _parts(network: Network, target: int, conditions: Mapping[int, int], engine: Engine) -> tuple[float, float]:
    """Split ``P(T | E)`` into the joint probabilities whose ratio it is.

    Args:
        network: The network.
        target: The target variable's index.
        conditions: The evidence and the ``exclusive`` constraints, by variable index.
        engine: The exact engine.

    Returns:
        ``P(T, E, C)`` and ``P(E, C)``, where C stands for every ``exclusive`` constraint observed
        true; ``P(E, C)`` is 1 when there is neither evidence nor constraint.
    """
    denominator = engine.probability(network, conditions) if conditions else 1.0
    return engine.probability(network, {**conditions, target: 1}), denominator


def derivatives(
    network: Network,
    target: str,
    *,
    evidence: Mapping[str, bool] | None = None,
    engine: Engine | None = None,
) -> dict[ParameterKey, float]:
    """Compute ``dP(target | evidence)/dθ`` exactly for every parameter θ of the network.

    See the module documentation for why this is exact. Each derivative is taken at the value the
    parameter currently holds, with every other parameter held at its own. With evidence the
    derivative is the quotient rule on two multilinear evaluations, not the two-point difference.

    Args:
        network: The network.
        target: The id of the target node.
        evidence: Node ids and their observed values; none by default.
        engine: The exact engine; variable elimination by default.

    Returns:
        The derivative of ``P(target = true | evidence)`` by parameter, in the order
        ``network.parameters`` lists them.

    Raises:
        ValidationError: If ``target`` or an evidence node is not an inference variable of the
            network, or the evidence observes the target.
        ZeroProbabilityError: If the evidence and the ``exclusive`` constraints have probability zero.
    """
    engine = engine or VariableElimination()
    query = _observed(network, target, evidence)
    index = network.index(target)
    conditions = _conditions(network, query.evidence)
    numerator, denominator = _parts(network, index, conditions, engine)
    if denominator <= 0.0:
        msg = "the evidence, together with the exclusive constraints, has probability zero"
        raise ZeroProbabilityError(msg)
    slopes: dict[ParameterKey, float] = {}
    for key in network.parameters:
        low_n, low_d = _parts(network.with_parameters({key: 0.0}), index, conditions, engine)
        high_n, high_d = _parts(network.with_parameters({key: 1.0}), index, conditions, engine)
        slopes[key] = ((high_n - low_n) * denominator - numerator * (high_d - low_d)) / denominator**2
    return slopes


def _subject(network: Network, key: ParameterKey) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Describe the node or relation that carries a parameter.

    Args:
        network: The network.
        key: The parameter.

    Returns:
        A phrase naming the parameter, the ids of the nodes it belongs to, and of the relation.
    """
    if key.kind == "base":
        return f"the base of {key.id!r}", (key.id,), ()
    for variable in network.variables:
        for link in variable.links:
            if link.relation == key.id:
                source = network.variables[link.parent].name
                return f"the strength of {link.type} relation {key.id!r}", (source, variable.name), (key.id,)
    return f"the strength of relation {key.id!r}", (), (key.id,)  # pragma: no cover - every strength has a link


def _sd(credence: Credence) -> float:
    """Return a credence's standard deviation.

    Args:
        credence: The credence.

    Returns:
        Its standard deviation; 0 for a ``Point``.
    """
    return math.sqrt(credence.variance)


def _ranked(findings: list[Finding]) -> list[Finding]:
    """Sort findings by the magnitude of their value, largest first; ties keep their order.

    Args:
        findings: The findings, each with a value.

    Returns:
        The sorted findings.
    """
    return sorted(findings, key=lambda finding: -abs(finding.value or 0.0))


def _probability(target: str, evidence: Mapping[str, bool] | None) -> str:
    """Write the probability a finding is about, for its message.

    Args:
        target: The id of the target node.
        evidence: Node ids and their observed values, or ``None`` for none.

    Returns:
        ``P('T')``, or ``P('T' | a=true, b=false)`` under evidence.
    """
    return f"P({target!r} | {render_evidence(evidence)})" if evidence else f"P({target!r})"


def sensitivity_findings(
    network: Network,
    target: str,
    slopes: Mapping[ParameterKey, float],
    *,
    evidence: Mapping[str, bool] | None = None,
) -> list[Finding]:
    """Turn derivatives into sensitivity findings.

    Args:
        network: The network.
        target: The id of the target node.
        slopes: The derivative of ``P(target | evidence)`` by parameter.
        evidence: The evidence the derivatives were taken under, named in the messages.

    Returns:
        One finding per parameter, largest magnitude first.
    """
    findings = []
    for key, slope in slopes.items():
        phrase, nodes, relations = _subject(network, key)
        findings.append(
            Finding(
                id=f"{SENSITIVITY}:{key.kind}:{key.id}",
                diagnostic=SENSITIVITY,
                message=f"{_probability(target, evidence)} changes by {slope:+.3g} per unit change in {phrase}",
                nodes=nodes,
                relations=relations,
                value=slope,
                details={"parameter_value": network.values[key]},
            )
        )
    return _ranked(findings)


def crux_findings(
    network: Network,
    target: str,
    slopes: Mapping[ParameterKey, float],
    *,
    evidence: Mapping[str, bool] | None = None,
) -> list[Finding]:
    """Turn derivatives into crux findings.

    Args:
        network: The network.
        target: The id of the target node.
        slopes: The derivative of ``P(target | evidence)`` by parameter.
        evidence: The evidence the derivatives were taken under, named in the messages.

    Returns:
        One finding per parameter, largest crux first.
    """
    findings = []
    for key, slope in slopes.items():
        phrase, nodes, relations = _subject(network, key)
        sd = _sd(network.parameters[key])
        score = abs(slope) * sd
        findings.append(
            Finding(
                id=f"{CRUX}:{key.kind}:{key.id}",
                diagnostic=CRUX,
                message=(
                    f"crux {score:.3g} for {phrase}: {_probability(target, evidence)} moves {abs(slope):.3g} per unit of it "
                    f"and its standard deviation is {sd:.3g}"
                ),
                nodes=nodes,
                relations=relations,
                value=score,
                details={"derivative": slope, "sd": sd},
            )
        )
    return _ranked(findings)


def sensitivity(
    network: Network,
    target: str,
    *,
    evidence: Mapping[str, bool] | None = None,
    engine: Engine | None = None,
) -> list[Finding]:
    """Report ``dP(target | evidence)/dθ`` for every parameter θ.

    Args:
        network: The network.
        target: The id of the target node.
        evidence: Node ids and their observed values; none by default.
        engine: The exact engine; variable elimination by default.

    Returns:
        One finding per parameter, largest magnitude first; its value is the signed derivative.

    Raises:
        ValidationError: If ``target`` or an evidence node is not an inference variable of the
            network, or the evidence observes the target.
        ZeroProbabilityError: If the evidence and the ``exclusive`` constraints have probability zero.
    """
    slopes = derivatives(network, target, evidence=evidence, engine=engine)
    return sensitivity_findings(network, target, slopes, evidence=evidence)


def crux(
    network: Network,
    target: str,
    *,
    evidence: Mapping[str, bool] | None = None,
    engine: Engine | None = None,
) -> list[Finding]:
    """Rank the parameters by ``|dP(target | evidence)/dθ| * sd(θ)``.

    A parameter is a crux when the target depends on it *and* it is uncertain: a steep derivative on
    a ``Point`` parameter, or a wide credence the target ignores, both score zero. This is the
    primary weak-point ranking. Under evidence, sd(θ) is still that of the parameter's own credence:
    the parameters are not updated on the evidence, as they are not for the uncertainty bands.

    Args:
        network: The network.
        target: The id of the target node.
        evidence: Node ids and their observed values; none by default.
        engine: The exact engine; variable elimination by default.

    Returns:
        One finding per parameter, largest crux first.

    Raises:
        ValidationError: If ``target`` or an evidence node is not an inference variable of the
            network, or the evidence observes the target.
        ZeroProbabilityError: If the evidence and the ``exclusive`` constraints have probability zero.
    """
    slopes = derivatives(network, target, evidence=evidence, engine=engine)
    return crux_findings(network, target, slopes, evidence=evidence)


def _others(network: Network, target: str, evidence: Mapping[str, bool]) -> list[tuple[str, tuple[str, ...]]]:
    """List the proposition variables other than the target's and the observed ones.

    Args:
        network: The network.
        target: The id of the target node.
        evidence: Node ids and their observed values.

    Returns:
        The name and member node ids of each, in network order.
    """
    skipped = {network.index(target)} | {network.index(node_id) for node_id in evidence}
    return [(v.name, v.members) for v in network.variables if v.kind == PROPOSITION and v.index not in skipped]


def single_points_of_failure(
    network: Network,
    target: str,
    *,
    evidence: Mapping[str, bool] | None = None,
    threshold: float = DEFAULT_FAILURE_THRESHOLD,
    engine: Engine | None = None,
) -> list[Finding]:
    """Find the variables Y whose failure alone sinks the target: ``P(target | do(Y = false), evidence)``.

    The intervention cuts the relations into Y and fixes it false, as in "suppose this premise is
    simply wrong"; the evidence is then conditioned on. A variable is reported when that probability
    is below ``threshold * P(target | evidence)``: the threshold is a fraction of the target's own
    probability, not a probability, so the same default separates the premises that sink the target
    from those that merely dent it whether the target starts at 0.9 or at 0.01. A threshold of 1
    reports every variable whose failure lowers the target at all; 0 reports none.

    Observed variables are not candidates: the evidence already says whether they hold. Nor is a
    variable whose failure the evidence rules out, where ``P(evidence | do(Y = false))`` is zero:
    given what was observed, Y did not fail.

    Args:
        network: The network.
        target: The id of the target node.
        evidence: Node ids and their observed values; none by default.
        threshold: The fraction of ``P(target | evidence)`` below which the target counts as failed;
            a probability within a relative 1e-12 of the cutoff counts as equal to it, so rounding
            cannot tip a finding however small ``P(target | evidence)`` is. The default, 0.1, an
            order of magnitude, is a convention chosen for this package rather than a value the model
            fixes.
        engine: The exact engine; variable elimination by default.

    Returns:
        One finding per single point of failure, lowest remaining probability first; its value is
        ``P(target | do(Y = false), evidence)``, and its details give ``P(target | evidence)`` as
        ``baseline`` and the fraction as ``threshold``.

    Raises:
        ValidationError: If ``threshold`` is not in [0, 1], ``target`` or an evidence node is not an
            inference variable, or the evidence observes the target.
        ZeroProbabilityError: If the evidence and the ``exclusive`` constraints have probability zero,
            or the constraints alone do under one of the interventions.
    """
    limit = probability_threshold(threshold, "threshold")
    engine = engine or VariableElimination()
    query = _observed(network, target, evidence)
    baseline = engine.query(network, query)
    # The cutoff is relative, so its rounding slack is too: an absolute one would exceed the cutoff
    # itself for a target below about 1e-12 and hide even a failure that takes the target to zero.
    cutoff = limit * baseline * (1.0 - ROUNDING)
    findings = []
    for name, members in _others(network, target, query.evidence):
        failed = network.intervene({name: False})
        try:
            value = engine.query(failed, query)
        except ZeroProbabilityError:
            if not query.evidence:
                raise
            # Raises when the constraints alone are impossible under the failure, as without evidence.
            engine.query(failed, Query({target: True}))
            continue
        if value >= cutoff:
            continue
        findings.append(
            Finding(
                id=f"{SINGLE_POINT_OF_FAILURE}:{name}",
                diagnostic=SINGLE_POINT_OF_FAILURE,
                message=(
                    f"if {name!r} is false, {_probability(target, query.evidence)} falls from {baseline:.3g} "
                    f"to {value:.3g}"
                ),
                nodes=members,
                value=value,
                details={"baseline": baseline, "threshold": limit},
            )
        )
    return sorted(findings, key=lambda finding: finding.value or 0.0)


def _mutual_information(table: list[list[float]]) -> float:
    """Compute the mutual information of a 2 x 2 joint distribution, in bits.

    Args:
        table: ``table[t][y] = P(T = t, Y = y)``.

    Returns:
        ``I(T; Y)``, clipped at zero against rounding.
    """
    rows = [table[t][0] + table[t][1] for t in (0, 1)]
    cols = [table[0][y] + table[1][y] for y in (0, 1)]
    information = 0.0
    for t in (0, 1):
        for y in (0, 1):
            joint = table[t][y]
            if joint > 0.0:
                information += joint * math.log2(joint / (rows[t] * cols[y]))
    return max(0.0, information)


def value_of_information(
    network: Network,
    target: str,
    *,
    evidence: Mapping[str, bool] | None = None,
    engine: Engine | None = None,
) -> list[Finding]:
    """Rank the other variables Y by the mutual information ``I(target; Y | evidence)``, in bits.

    ``I(T; Y | E)`` is the expected reduction in the entropy of T from learning whether Y is true,
    once E is known, so the top of the ranking is what would most change the target if it were
    resolved next, and is the first thing to investigate. It is at most 1 bit, the entropy of a
    binary variable. Observed variables are left out: learning them again tells nothing.

    The unit, bits (logarithms to base 2), is a choice of this package; the ranking does not depend
    on it. Multiply by ``ln 2`` for nats.

    Args:
        network: The network.
        target: The id of the target node.
        evidence: Node ids and their observed values; none by default.
        engine: The exact engine; variable elimination by default.

    Returns:
        One finding per other unobserved proposition variable, most informative first. Its details
        give ``P(target | evidence)`` as ``p_target`` and, for each value of Y with positive
        probability, ``P(target | Y, evidence)`` as ``p_target_given_true`` or ``_false``.

    Raises:
        ValidationError: If ``target`` or an evidence node is not an inference variable of the
            network, or the evidence observes the target.
        ZeroProbabilityError: If the evidence and the ``exclusive`` constraints have probability zero.
    """
    engine = engine or VariableElimination()
    given = _observed(network, target, evidence).evidence
    findings = []
    for name, members in _others(network, target, given):
        table = [
            [engine.query(network, Query({target: bool(t), name: bool(y)}, given)) for y in (0, 1)] for t in (0, 1)
        ]
        bits = _mutual_information(table)
        details = {"p_target": table[1][0] + table[1][1]}
        for y, label in ((1, "true"), (0, "false")):
            p_y = table[0][y] + table[1][y]
            if p_y > 0.0:
                details[f"p_target_given_{label}"] = table[1][y] / p_y
        findings.append(
            Finding(
                id=f"{VALUE_OF_INFORMATION}:{name}",
                diagnostic=VALUE_OF_INFORMATION,
                message=(
                    f"learning whether {name!r} is true would give {bits:.3g} bits of information about "
                    f"{target!r}{' given ' + render_evidence(given) if given else ''}"
                ),
                nodes=members,
                value=bits,
                details=details,
            )
        )
    return sorted(findings, key=lambda finding: -(finding.value or 0.0))
