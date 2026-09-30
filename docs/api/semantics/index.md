---
title: Semantics
description:
    Conditional probability tables and compiling a graph into a network.
---

Every inference variable is a binary proposition. A node X with `requires`
parents R, `supports` parents S and `refutes` parents F, of strengths r, s and
f, and base b, is true with probability

$$
P(X = 1 \mid \text{parents}) = N \cdot O \cdot I,
$$

where

- $N = \prod_{i \in R,\ R_i = 0} (1 - r_i)$ is the necessity gate (noisy-AND),
- $O = 1 - (1 - b) \prod_{j \in S,\ S_j = 1} (1 - s_j)$ is the sufficiency term
  (noisy-OR with leak b),
- $I = \prod_{k \in F,\ F_k = 1} (1 - f_k)$ is the inhibition by refuters.

So b is the probability of X when every required premise holds and no support or
refuter is active; with no parents it is the prior.

`compile_graph` turns a `Graph` into an immutable `Network`:

- nodes joined by `equivalent` relations are merged into one variable, and their
  bases must agree;
- each `exclusive(A, B)` relation becomes a constraint variable C with $P(C = 1
  \mid A, B) = 0$ when both are true and 1 otherwise, observed as $C = 1$ in
  every query, which changes the marginals of A and B;
- an inference variable without a base is a compile error: there is no default
  credence.

A compiled network holds every parameter at its credence mean. Because each
parameter enters the joint distribution with degree at most one, the answer at
the means is the predictive ratio $E[P(A, E)] / E[P(E)]$, where both
expectations are taken over independent parameter distributions and the
`exclusive` constraints count as evidence on both sides. This is not the average
of $P(A \mid E)$ over the parameters, which generally differs.

<!-- prettier-ignore-start -->

::: credencegraph.semantics
    options:
        show_root_heading: false
        heading_level: 2
        inherited_members: true
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

<!-- prettier-ignore-end -->
