---
title: Core Data Model
description: Graph, nodes, relations, credences and their JSON form.
---

<!-- prettier-ignore-start -->

::: credencegraph.core
    options:
        show_root_heading: false
        heading_level: 2
        inherited_members: true
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

<!-- prettier-ignore-end -->

## JSON format

A graph serialises to a JSON document with `"format": "credencegraph"` and an
integer `"version"` (currently `1`). The structure and types are described by a
published JSON Schema, available from `credencegraph.core.json_schema()` and
shipped in the package as `credencegraph/core/schema/graph-1.schema.json`.

The schema covers structure only. Unique ids, relations that name existing
nodes, and the absence of cycles among `requires`, `supports` and `refutes`
relations are checked when a document is loaded, because a schema cannot express
them.

A credence is a bare number `p` (a point value with `0 <= p <= 1`) or an object
`{"alpha": a, "beta": b}` with `a, b > 0`.
