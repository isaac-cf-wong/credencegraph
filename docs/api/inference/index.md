---
title: Inference
description:
    Exact engines and the marginal, joint, conditional and interventional
    queries.
---

Four query functions answer questions about a compiled network, each returning
an `Answer` with the point probability:

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
