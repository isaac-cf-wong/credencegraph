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

| Function                   | Reports                                                        |
| -------------------------- | -------------------------------------------------------------- |
| `missing_parameters`       | An inference variable without a `base`                         |
| `unanchored`               | An inference variable with no source and no inferential parent |
| `claims`                   | \|stated − computed\| above a threshold (default 0.1)          |
| `sensitivity`              | ∂P(T)/∂θ for every parameter θ                                 |
| `crux`                     | \|∂P(T)/∂θ\| · sd(θ): T depends on θ _and_ θ is uncertain      |
| `single_points_of_failure` | Y with P(T \| do(Y = false)) below a fraction of P(T) (0.1)    |
| `value_of_information`     | The mutual information I(T; Y) in bits                         |

`diagnose(graph, target)` runs them all. If a parameter is missing the graph
cannot be compiled, and the report stops after the structural checks.

**Sensitivity is exact.** Every joint probability of the network is multilinear
in the parameters, so without `exclusive` constraints P(T) is a straight line in
each θ and ∂P(T)/∂θ = P(T | θ = 1) − P(T | θ = 0). An `exclusive` constraint C
is observed in every query, which makes P(T) = P(T, C) / P(C) a ratio of two
such lines; the two-point difference is then not the derivative, and the
quotient rule is used instead, again from the values at θ = 0 and θ = 1.

**Crux is the primary weak-point ranking.** A steep derivative on a parameter
known exactly, or a wide credence the target ignores, both score zero.

**Single points of failure** use intervention, not conditioning: "suppose this
premise is simply wrong". The threshold is a fraction of the target's own
probability, not a probability: by default a variable is reported when its
failure leaves P(T) below a tenth of what it was. An already improbable target
therefore still has its premises ranked, rather than every premise whose failure
lowers it reported; a threshold of 1 reports all of those.

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
