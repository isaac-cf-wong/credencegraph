---
title: Diagnostics
description:
    Structural checks, stated-versus-computed consistency, and the weak points
    of a target proposition.
---

Every diagnostic returns a list of `Finding` records. A finding has an `id`
(`<diagnostic>:<subject>`, for example `crux:strength:r2`), the ids of the
`nodes` and `relations` involved, a headline `value`, the numbers behind it in
`details`, and a one-line `message`. `Finding.to_dict()` gives its JSON form.

| Function                   | Reports                                                                           |
| -------------------------- | --------------------------------------------------------------------------------- |
| `missing_parameters`       | An inference variable without a `base`                                            |
| `unanchored`               | An inference variable with no source and no inferential parent                    |
| `claims`                   | \|logit(stated) − logit(computed)\| above a threshold (default 2 ln(11/9) ≈ 0.40) |
| `sensitivity`              | ∂P(T)/∂θ for every parameter θ                                                    |
| `crux`                     | \|∂P(T)/∂θ\| · sd(θ): T depends on θ _and_ θ is uncertain                         |
| `single_points_of_failure` | Y with P(T \| do(Y = false)) below a fraction of P(T) (0.1)                       |
| `failure_impact`           | Every Y ranked by P(T \| do(Y = false)) / P(T), the threshold marking the line    |
| `value_of_information`     | The mutual information I(T; Y) in bits                                            |

When no parameter T depends on is uncertain, as in a graph of plain numbers,
every crux is zero; `crux` then returns a single finding `crux:<target>`, with
no value, that says so instead of ranking the zeros. Whether T depends on a
parameter is decided from the graph, by d-separation, so every engine gives the
same answer.

`diagnose(graph, target)` runs them all. If a parameter is missing the graph
cannot be compiled, and the report stops after the structural checks.

## Under evidence

Every inference diagnostic, and `diagnose`, takes `evidence`, a mapping of node
ids to observed values, written E below. Without it, or with `{}`, each answers
for the graph before anything is observed, exactly as before evidence existed.
With it:

| Diagnostic                 | Under evidence E                                                 |
| -------------------------- | ---------------------------------------------------------------- |
| `claims`                   | `stated` against P(X \| E); observed variables are skipped       |
| `sensitivity`              | ∂P(T \| E)/∂θ, by the quotient rule below                        |
| `crux`                     | \|∂P(T \| E)/∂θ\| · sd(θ), sd(θ) of the parameter's own credence |
| `single_points_of_failure` | Y with P(T \| do(Y = false), E) below a fraction of P(T \| E)    |
| `failure_impact`           | P(T \| do(Y = false), E) / P(T \| E), over the same candidates   |
| `value_of_information`     | I(T; Y \| E) in bits, over the unobserved variables Y            |

The target may not be observed, directly or through an equivalent node, since
its diagnostics would all be trivial; that is a `ValidationError`. Evidence of
probability zero, including equivalent nodes observed at different values, is a
`ZeroProbabilityError`. A single point of failure is not reported for a variable
whose failure the evidence rules out, P(E | do(Y = false)) = 0: given what was
observed, Y did not fail. Crux keeps sd(θ) from the parameter's credence as
written; the parameters are not updated on the evidence, as they are not for the
uncertainty bands of a query.

**Sensitivity under evidence is not a two-point difference.** Write C for the
`exclusive` constraints, observed in every query, and

N(θ) = P(T, E, C) and D(θ) = P(E, C),

so that P(T | E) = N(θ) / D(θ). Each of N and D is a joint probability of the
network, so each is a straight line in θ: N(θ) = N(0) + θ N′ with N′ = N(1) −
N(0), and likewise D. Their ratio is not a line, so P(T | E, θ = 1) − P(T | E, θ
= 0) is not its derivative; the quotient rule is exact instead:

∂P(T | E)/∂θ = (N′ D(θ) − N(θ) D′) / D(θ)²,

from N and D at θ = 0, at θ = 1, and at the value θ holds. The fast two-point
identity ∂P(T)/∂θ = P(T | θ = 1) − P(T | θ = 0) may be used only when D does not
depend on θ: with no evidence and no `exclusive` constraint, where D = 1. For
one observation o of a hypothesis H with base h, P(o | H) = L and P(o | not H) =
b, P(H | o) = hL / (hL + (1 − h) b) runs from 0 at h = 0 to 1 at h = 1, a
difference of 1, while its derivative at h = 0.3, L = 0.8, b = 0.2 is Lb / (hL +
(1 − h) b)² = 1.108.

**Sensitivity is exact.** Every joint probability of the network is multilinear
in the parameters, so without `exclusive` constraints or evidence P(T) is a
straight line in each θ and ∂P(T)/∂θ = P(T | θ = 1) − P(T | θ = 0). An
`exclusive` constraint C is observed in every query, which makes P(T) = P(T, C)
/ P(C) a ratio of two such lines; the two-point difference is then not the
derivative, and the quotient rule is used instead, as under evidence above.

**Claims are compared in log-odds but reported in probability.** A `claims`
finding's `value` is the signed gap stated − computed, and its `details` hold
`stated` and `computed`, all probabilities; the threshold it was tested against
is in natural log-odds and is named for its unit, `threshold_log_odds`, so the
two scales cannot be confused in the JSON report.

**Crux is the primary weak-point ranking.** A steep derivative on a parameter
known exactly, or a wide credence the target ignores, both score zero.

**Single points of failure** use intervention, not conditioning: "suppose this
premise is simply wrong". The threshold is a fraction of the target's own
probability, not a probability: by default a variable is reported when its
failure leaves P(T) below a tenth of what it was. An already improbable target
therefore still has its premises ranked, rather than every premise whose failure
lowers it reported; a threshold of 1 reports all of those.

**A premise behind one `requires` of strength r crosses the threshold only if r
\> 1 − threshold.** Suppose the only directed path from Y to T is one `requires`
relation Y → T of strength r, with no evidence and no `exclusive` constraint.
P(T = 1 | parents) is Y's necessity gate, 1 − r when Y is false and 1 when it is
true, times a factor A that does not involve Y. Failing Y leaves T's other
parents as they were, so P(T | do(Y = false)) = (1 − r) E[A], while P(T) ≤ E[A]
because the gate is at most 1:

P(T | do(Y = false)) / P(T) ≥ 1 − r.

When Y shares no ancestor with T's other parents the ratio is exactly (1 − r) /
((1 − r) + r p), with p = P(Y). At the default threshold of 0.1 such a premise
is never a single point of failure when r ≤ 0.9, however probable it is, so an
empty list says that no `requires` is stronger than 0.9, not that nothing is
fatal.

**Failure impact ranks what lies below the line.** `failure_impact` gives, for
every candidate of `single_points_of_failure`, the ratio P(T | do(Y = false), E)
/ P(T | E) as its `value`, lowest first, with P(T | E) as `baseline`, P(T | do(Y
= false), E) as `p_target_if_false` and the threshold in its `details`. The
threshold does not filter it: `single_point_of_failure` is true exactly for the
variables `single_points_of_failure` reports. The premises can therefore be
ranked by how much their failure dents the target without rerunning at another
threshold. When P(T | E) is zero the ratio is undefined and there are no
findings.

Parameters are treated as independent, so the most common source of
overconfidence in a hand-built argument is a missing common cause: two premises
that share an unstated assumption. Model the assumption as a proposition of its
own with a relation into each premise, and it will show up in these rankings.

<!-- prettier-ignore-start -->

::: credencegraph.diagnostics
    options:
        show_root_heading: false
        heading_level: 2
        inherited_members: true
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

<!-- prettier-ignore-end -->
