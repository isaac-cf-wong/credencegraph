---
title: Ingestion
description:
    The LaTeX and PDF readers, and the graph of document nodes built from a
    document's chunks.
---

Installed with `pip install 'credencegraph[ingest]'`. `read_latex(path)` reads a
`.tex` document, and the files it inputs, into `Chunk` objects located by file
and line; `read_pdf(path)` reads a `.pdf` document into chunks located by page
and box, and `analyse_pdf(path)` returns its structure as well;
`chunks_to_graph(chunks, document)` turns chunks into a graph of anchored
document nodes with `cites` relations. The `ingest` command does both; see
[Ingesting a document](../../cli.md#ingesting-a-document).

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

::: credencegraph.ingest.pdf
    options:
        show_root_heading: false
        heading_level: 2
        members:
            - read_pdf
            - analyse_pdf
            - render_region
            - PdfDocument
            - Line
            - Block
            - Glyph
            - Heading
            - Caption
            - Equation
            - Reference
            - Citation
            - PdfError
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

<!-- prettier-ignore-end -->
