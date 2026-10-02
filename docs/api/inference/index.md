---
title: Inference
description:
    Exact engines, the marginal, joint, conditional and interventional queries,
    and their uncertainty bands.
---

Four query functions answer questions about a compiled network, each returning
an `Answer`:

| Query                                       | Meaning                                                    |
| ------------------------------------------- | ---------------------------------------------------------- |
| `marginal(network, "x")`                    | P(x)                                                       |
| `joint(network, {"x": True, "y": False})`   | P(x and not y)                                             |
| `conditional(network, "x", {"y": True})`    | P(x \| y)                                                  |
| `intervene(network, "x", set={"y": False})` | P(x) after cutting the relations into y and fixing y false |

Conditioning on a node also updates what the node rests on; intervening does
not. "What does the conclusion rest on?" is an interventional question.

Two exact engines are provided:

- `VariableElimination`, the default, eliminates variables in a min-fill order.
  If an intermediate factor would exceed `max_factor_size` entries (default
  2²²), the query fails with `ProblemTooLargeError`; it never falls back to an
  approximation on its own.
- `Enumeration` sums over every world. It is exponential in the number of free
  variables (limited to 20 by default) and serves as the independent reference
  in tests.

## Parameter uncertainty

Every `base` and `strength` is treated as independent of the others. An `Answer`
carries:

- `point`: the probability with every parameter at its credence mean. Because
  each parameter enters a joint probability with degree at most one, this is
  exactly the predictive probability E[P(A, E)] / E[P(E)] over the parameters'
  distributions, computed without sampling.
- `band`: the 5%, 50% and 95% quantiles (`q05`, `q50`, `q95`) of the exact
  answer over Monte Carlo draws of the `Beta` parameters. It shows how far the
  answer would move if the inputs were different.
- `mean_over_draws`: the average of those per-draw answers. The draws do not
  update the parameters on the evidence, so this generally differs from `point`
  and is reported under its own name.
- `draws`: the number of draws behind the band.

The band is omitted (`band` and `mean_over_draws` are `None`) when every
parameter is a `Point`, or when `draws=0` is passed. Queries make
`DEFAULT_DRAWS` (1000) draws by default; pass `rng` (a seed or a
`numpy.random.Generator`) for reproducible bands.

Overriding a `Beta` parameter with `Network.with_parameters` moves `point` but
not the band. The point answer uses the values the network holds, while every
`Beta` parameter is drawn from its stored credence, including one that has been
set to another value; a network whose `Beta` parameters have all been set still
gets a band. To hold a parameter fixed in the band, give it a `Point` credence in
the graph instead. Overriding a parameter whose credence is a `Point` moves the
band as well as `point`, because its overridden value is kept in every draw.

Parameters are independent by construction, so correlated inputs (two
measurements sharing an unknown bias, say) cannot be expressed as correlated
strengths. Model the shared cause as a proposition of its own, such as "the
instrument is calibrated", with a relation into each.

<!-- prettier-ignore-start -->

::: credencegraph.inference
    options:
        show_root_heading: false
        heading_level: 2
        inherited_members: true
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

<!-- prettier-ignore-end -->
