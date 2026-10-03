"""Weak points of a target: what its probability depends on, and what would change it most.

Every function here takes a compiled network and the id of a target node T, and asks about
``P(T = true)`` with the parameters at the values the network holds (their credence means for a
freshly compiled network).

**Sensitivity.** Every joint probability of the network is multilinear in its parameters: each
``base`` and ``strength`` enters with degree at most one. With no ``exclusive`` constraint, ``P(T)``
is itself such a joint probability, a straight line in each parameter θ, so

    dP(T)/dθ = P(T | θ = 1) - P(T | θ = 0)

exactly, from two evaluations. An ``exclusive`` constraint C is observed true in every query, which
makes ``P(T) = P(T, C) / P(C)`` a ratio of two such lines, and the two-point difference is then no
longer the derivative. The derivative is taken instead from the quotient rule, with numerator and
denominator each evaluated at θ = 0 and θ = 1; with no constraint the denominator is 1 and this is
the two-point difference above.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from credencegraph.core.credence import Credence
from credencegraph.diagnostics.records import (
    CRUX,
    ROUNDING,
    SENSITIVITY,
    SINGLE_POINT_OF_FAILURE,
    VALUE_OF_INFORMATION,
    Finding,
    probability_threshold,
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


def _parts(network: Network, target: int, engine: Engine) -> tuple[float, float]:
    """Split ``P(T)`` into the joint probabilities whose ratio it is.

    Args:
        network: The network.
        target: The target variable's index.
        engine: The exact engine.

    Returns:
        ``P(T, C)`` and ``P(C)``, where C stands for every ``exclusive`` constraint observed true;
        ``P(C)`` is 1 when there is no constraint.
    """
    constraints = dict.fromkeys(network.constraints, 1)
    denominator = engine.probability(network, constraints) if constraints else 1.0
    return engine.probability(network, {**constraints, target: 1}), denominator


def derivatives(network: Network, target: str, *, engine: Engine | None = None) -> dict[ParameterKey, float]:
    """Compute ``dP(target)/dθ`` exactly for every parameter θ of the network.

    See the module documentation for why this is exact. Each derivative is taken at the value the
    parameter currently holds, with every other parameter held at its own.

    Args:
        network: The network.
        target: The id of the target node.
        engine: The exact engine; variable elimination by default.

    Returns:
        The derivative of ``P(target = true)`` by parameter, in the order ``network.parameters``
        lists them.

    Raises:
        ValidationError: If ``target`` is not an inference variable of the network.
        ZeroProbabilityError: If the ``exclusive`` constraints have probability zero.
    """
    engine = engine or VariableElimination()
    index = network.index(target)
    numerator, denominator = _parts(network, index, engine)
    if denominator <= 0.0:
        msg = "the exclusive constraints have probability zero"
        raise ZeroProbabilityError(msg)
    slopes: dict[ParameterKey, float] = {}
    for key in network.parameters:
        low_n, low_d = _parts(network.with_parameters({key: 0.0}), index, engine)
        high_n, high_d = _parts(network.with_parameters({key: 1.0}), index, engine)
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


def sensitivity_findings(network: Network, target: str, slopes: Mapping[ParameterKey, float]) -> list[Finding]:
    """Turn derivatives into sensitivity findings.

    Args:
        network: The network.
        target: The id of the target node.
        slopes: The derivative of ``P(target)`` by parameter.

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
                message=f"P({target!r}) changes by {slope:+.3g} per unit change in {phrase}",
                nodes=nodes,
                relations=relations,
                value=slope,
                details={"parameter_value": network.values[key]},
            )
        )
    return _ranked(findings)


def crux_findings(network: Network, target: str, slopes: Mapping[ParameterKey, float]) -> list[Finding]:
    """Turn derivatives into crux findings.

    Args:
        network: The network.
        target: The id of the target node.
        slopes: The derivative of ``P(target)`` by parameter.

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
                    f"crux {score:.3g} for {phrase}: P({target!r}) moves {abs(slope):.3g} per unit of it "
                    f"and its standard deviation is {sd:.3g}"
                ),
                nodes=nodes,
                relations=relations,
                value=score,
                details={"derivative": slope, "sd": sd},
            )
        )
    return _ranked(findings)


def sensitivity(network: Network, target: str, *, engine: Engine | None = None) -> list[Finding]:
    """Report ``dP(target)/dθ`` for every parameter θ.

    Args:
        network: The network.
        target: The id of the target node.
        engine: The exact engine; variable elimination by default.

    Returns:
        One finding per parameter, largest magnitude first; its value is the signed derivative.

    Raises:
        ValidationError: If ``target`` is not an inference variable of the network.
        ZeroProbabilityError: If the ``exclusive`` constraints have probability zero.
    """
    return sensitivity_findings(network, target, derivatives(network, target, engine=engine))


def crux(network: Network, target: str, *, engine: Engine | None = None) -> list[Finding]:
    """Rank the parameters by ``|dP(target)/dθ| * sd(θ)``.

    A parameter is a crux when the target depends on it *and* it is uncertain: a steep derivative on
    a ``Point`` parameter, or a wide credence the target ignores, both score zero. This is the
    primary weak-point ranking.

    Args:
        network: The network.
        target: The id of the target node.
        engine: The exact engine; variable elimination by default.

    Returns:
        One finding per parameter, largest crux first.

    Raises:
        ValidationError: If ``target`` is not an inference variable of the network.
        ZeroProbabilityError: If the ``exclusive`` constraints have probability zero.
    """
    return crux_findings(network, target, derivatives(network, target, engine=engine))


def _others(network: Network, target: str) -> list[tuple[str, tuple[str, ...]]]:
    """List the proposition variables other than the target's.

    Args:
        network: The network.
        target: The id of the target node.

    Returns:
        The name and member node ids of each, in network order.
    """
    own = network.index(target)
    return [(v.name, v.members) for v in network.variables if v.kind == PROPOSITION and v.index != own]


def single_points_of_failure(
    network: Network,
    target: str,
    *,
    threshold: float = DEFAULT_FAILURE_THRESHOLD,
    engine: Engine | None = None,
) -> list[Finding]:
    """Find the variables Y whose failure alone sinks the target: ``P(target | do(Y = false))``.

    The intervention cuts the relations into Y and fixes it false, as in "suppose this premise is
    simply wrong". A variable is reported when that probability is below ``threshold * P(target)``:
    the threshold is a fraction of the target's own probability, not a probability, so the same
    default separates the premises that sink the target from those that merely dent it whether the
    target starts at 0.9 or at 0.01. A threshold of 1 reports every variable whose failure lowers the
    target at all; 0 reports none.

    Args:
        network: The network.
        target: The id of the target node.
        threshold: The fraction of ``P(target)`` below which the target counts as failed; a
            probability within 1e-12 of ``threshold * P(target)`` counts as equal to it, so rounding
            cannot tip a finding. The default, 0.1, an order of magnitude, is a convention chosen for
            this package rather than a value the model fixes.
        engine: The exact engine; variable elimination by default.

    Returns:
        One finding per single point of failure, lowest remaining probability first; its value is
        ``P(target | do(Y = false))``, and its details give ``P(target)`` as ``baseline`` and the
        fraction as ``threshold``.

    Raises:
        ValidationError: If ``threshold`` is not in [0, 1], or ``target`` is not an inference variable.
        ZeroProbabilityError: If the ``exclusive`` constraints have probability zero, with or without
            one of the interventions.
    """
    limit = probability_threshold(threshold, "threshold")
    engine = engine or VariableElimination()
    query = Query({target: True})
    baseline = engine.query(network, query)
    cutoff = limit * baseline
    findings = []
    for name, members in _others(network, target):
        value = engine.query(network.intervene({name: False}), query)
        if value >= cutoff - ROUNDING:
            continue
        findings.append(
            Finding(
                id=f"{SINGLE_POINT_OF_FAILURE}:{name}",
                diagnostic=SINGLE_POINT_OF_FAILURE,
                message=f"if {name!r} is false, P({target!r}) falls from {baseline:.3g} to {value:.3g}",
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


def value_of_information(network: Network, target: str, *, engine: Engine | None = None) -> list[Finding]:
    """Rank the other variables Y by the mutual information ``I(target; Y)``, in bits.

    ``I(T; Y)`` is the expected reduction in the entropy of T from learning whether Y is true, so
    the top of the ranking is what would most change the target if it were resolved, and is the
    first thing to investigate. It is at most 1 bit, the entropy of a binary variable.

    The unit, bits (logarithms to base 2), is a choice of this package; the ranking does not depend
    on it. Multiply by ``ln 2`` for nats.

    Args:
        network: The network.
        target: The id of the target node.
        engine: The exact engine; variable elimination by default.

    Returns:
        One finding per other proposition variable, most informative first.

    Raises:
        ValidationError: If ``target`` is not an inference variable of the network.
        ZeroProbabilityError: If the ``exclusive`` constraints have probability zero.
    """
    engine = engine or VariableElimination()
    findings = []
    for name, members in _others(network, target):
        table = [[engine.query(network, Query({target: bool(t), name: bool(y)})) for y in (0, 1)] for t in (0, 1)]
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
                message=f"learning whether {name!r} is true would give {bits:.3g} bits of information about {target!r}",
                nodes=members,
                value=bits,
                details=details,
            )
        )
    return sorted(findings, key=lambda finding: -(finding.value or 0.0))
