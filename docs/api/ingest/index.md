---
title: Ingestion
description:
    The LaTeX reader, and the graph of document nodes built from a document's
    chunks.
---

Installed with `pip install 'credencegraph[ingest]'`. `read_latex(path)` reads a
`.tex` document, and the files it inputs, into `Chunk` objects located by file
and line; `chunks_to_graph(chunks, document)` turns them into a graph of
anchored document nodes with `cites` relations. The `ingest` command does both;
see [Ingesting a LaTeX document](../../cli.md#ingesting-a-latex-document).

<!-- prettier-ignore-start -->

::: credencegraph.ingest
    options:
        show_root_heading: false
        heading_level: 2
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

::: credencegraph.ingest.latex
    options:
        show_root_heading: false
        heading_level: 2
        members:
            - read_latex
            - LatexError
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

<!-- prettier-ignore-end -->
