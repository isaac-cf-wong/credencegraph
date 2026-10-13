---
title: Rubrics
description:
    Rubric files, the anchored, typed and assessed checks, faithfulness,
    coverage, and the split and annotate edits.
---

A rubric declares the node types of a graph built from a document's own text,
the form each type carries and what each type must rest on. `load_rubric` reads
one from TOML; `check_graph(graph, rubric, level)` returns every violation as a
`Finding` whose `diagnostic` is its code; `coverage` reports how far
classification has got; `split_node` and `annotate_node` return a new graph with
the edit applied. The contract they implement is
[Verbatim ingestion](../../verbatim-ingestion.md).

<!-- prettier-ignore-start -->

::: credencegraph.rubric
    options:
        show_root_heading: false
        heading_level: 2
        inherited_members: true
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

<!-- prettier-ignore-end -->
