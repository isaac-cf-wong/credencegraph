---
title: Verbatim-chunk ingestion (design)
description:
    The contract for building a graph from a document's own text — node and edge
    kinds, the rubric format, validation levels, coverage, re-anchoring, the
    command line, and which parts belong to the core and which to the ingestion
    layer.
---

> **Design, not yet implemented.** This page is the contract that the
> verbatim-chunk features will be built against. None of the commands, options,
> rules or error codes it introduces exist in the current release. Everything it
> says about the engine as it is today is marked as such and cites the code.

## Why

When an analyst writes every node's statement in their own words, the graph
drifts from its source in ways the graph cannot detect: a "design criterion"
becomes a "detection threshold", "shown for long segments" becomes "shown". The
original words are not in the graph, so nothing can compare against them.

Verbatim-chunk ingestion builds the graph from the document's own text. Every
node that comes from the document carries its exact text, where it is, and a
digest of it. The analyst's work is then to _classify_ that text — give each
chunk one type from a rubric, fill in a restricted form, and link it to what it
rests on — never to restate it. What the analyst adds that the document does not
say, such as an implicit premise or a check, lives in nodes of a separate,
visibly marked origin.

## What the engine provides today

The contract is built on the current data model without changing the file
format. These are the facts it relies on (paths are under `src/credencegraph/`):

- `Node.kind` is a free label the engine attaches no meaning to
  (`core/node.py:25`). The rubric type of a node is stored there.
- `Node.statement` is optional text (`core/node.py:26`, `:36`, `:46`), and
  `SourceAnchor` already has `document`, `locator`, `quote` and `digest`
  (`core/anchor.py:28-49`). Only `document` is required; `locator`, `quote` and
  `digest` default to `None`, and the engine checks only that each one given is
  a non-empty string (`core/anchor.py:39-49`). It defines no digest algorithm
  and does not compare `quote` with `statement`.
- `attributes` on nodes and relations is open, frozen JSON (`core/node.py:40`,
  `core/relation.py:64`, `core/attributes.py:56-83`). Every field this contract
  adds to a node lives there.
- A relation type outside `requires`, `supports`, `refutes`, `equivalent` and
  `exclusive` is an annotation that inference ignores
  (`core/relation.py:19-38`). `requires`, `supports` and `refutes` must carry a
  strength (`core/relation.py:72-80`), and their source is the premise and their
  target the conclusion (`diagnostics/structure.py:79`).
- The inference variables are the nodes that have a `base` or are an endpoint of
  an inferential relation (`semantics/compiler.py:60-73`). A chunk that is only
  typed and annotated is not a variable and needs no `base`.
- The file format has no graph-level metadata: a document holds exactly
  `format`, `version`, `nodes` and `relations` (`core/serialization.py:26`), and
  the JSON Schema forbids other keys. Nothing in this contract needs a new
  format version.
- `credencegraph init PATH` already exists and creates an empty graph file
  (`cli/edit.py:37-55`), and `credencegraph check PATH` already exists and
  checks that a graph parses and compiles (`cli/diagnose.py:267-293`). The new
  commands below are named and shaped so that both keep their current meaning.
- A node with no source and no inferential parents is reported as `unanchored`
  (`diagnostics/structure.py:66-92`). Implicit premises are meant to show up
  there.

## Node kinds

Every node of an ingested graph has an **origin**, stored as
`attributes.origin`, and a **type**, stored as `kind`.

- `document`: A chunk of the ingested document, or a part of one. Text: verbatim
  and equal to its anchor's `quote`, except a field child, which has none.
- `analyst`: An implicit premise: something the argument needs that the document
  does not state. Text: the analyst's words; no source.
- `evidence`: A check, an external source or a judgement bearing on another
  node. Text: the analyst's words; any sources.

### Document nodes

`ingest` creates one document node per **chunk**. The chunk units are:

- `sentence`: Sentence of running prose, including the abstract. Inline
  mathematics stays inside the text (as LaTeX when the input is LaTeX).
- `equation`: Display equation or multi-line display (`equation`, `align`, …).
  One node per display, not per line; its label, if any, is `attributes.label`.
- `caption`: Figure or table caption.
- `footnote`: Footnote.
- `reference`: Bibliography entry.

Titles and section headings are not nodes; the section a chunk sits in is part
of its locator. A document node has:

- `id`: `s-NNN` for sentences, footnotes and captions, `e-NNN` for equations,
  `r-NNN` for references, numbered in document order with zero padding to the
  width the document needs. Ids are only assigned once: re-anchoring keeps them
  (see [Revisions](#revisions-and-re-anchoring)).
- `statement`: the chunk's text, [normalised](#text-normalisation-and-digest).
- `sources`: exactly one anchor, with `document` the document id, `locator` the
  chunk's location, `quote` equal to `statement`, and `digest` the digest of
  `quote`.
- `kind`: `unassigned`, unless the rubric gives the unit an
  [initial type](#rubric-format).
- `attributes`: `origin = "document"`, `unit`, and, once typed, `form` and
  `assessment`.

**Locators** are `;`-separated `key=value` pairs:

- LaTeX input: `file=<path relative to the main file>;lines=<first>-<last>`, for
  example `file=sec/method.tex;lines=41-42`.
- PDF input: `page=<n>;bbox=<x0>,<y0>,<x1>,<y1>` in PDF points, one
  `page`/`bbox` pair per page the chunk spans, separated by `;`.
- A span child appends `;chars=<start>-<end>`: the half-open character range of
  its text within its parent's `statement`.

A citation inside a chunk becomes an annotation relation `cites` from the chunk
to the reference node it names.

### Text normalisation and digest

Text is normalised once, by the reader, before it is stored: Unicode NFC, every
run of whitespace (including line breaks) replaced by a single space, and
leading and trailing whitespace removed. The stored `statement` and `quote` are
the normalised text.

The digest is `sha256:` followed by the lower-case hexadecimal SHA-256 of the
UTF-8 bytes of the normalised text. The normalisation and digest functions live
in the core and are the only implementation: the readers call them, and the
checks recompute with them.

### Types: one purpose per node

Each typed node has exactly one type from the rubric. Two types are built in and
may not be declared by a rubric:

- `unassigned`: not yet classified. Allowed only on document nodes.
- `compound`: a document chunk that does more than one thing. It keeps its text
  and anchor but carries no `form`, no `assessment` and no `base`, and is an
  endpoint of no inferential relation. Its **children** carry the content.

A **child** is a document node joined to its compound parent by one annotation
relation `part-of` (child → parent), with id `<parent id>a`, `<parent id>b`, …
in the order created. It is one of:

- a **span child**: its `statement` and `quote` are a contiguous substring of
  the parent's `statement`, its locator is the parent's plus `chars=…`, and its
  digest is the digest of the span; or
- a **field child**: `statement` is null, its single anchor is a copy of the
  parent's (so it is anchored, and its form is checked against the parent's
  text), and it has a `form`. It has no text of its own, so the rule that a
  statement equals its quote does not apply to it.

A child never carries free text. A compound has at least two children, and a
child is never itself a compound.

### Forms

Where the rubric asks for one, a typed node carries `attributes.form`: an object
whose fields are declared by the rubric for that type. Every field has one of
these field types:

- `verbatim`: string. Faithfulness rule: a substring of the node's text.
- `number`: number. Faithfulness rule: appears as a numeral in the node's text.
- `quantity`: `{"value": number, "unit": string}`. Faithfulness rule: the value
  as a numeral and the unit as a token in the text.
- `eqref`: string, an equation label or number. Faithfulness rule: referenced in
  the text.
- `enum`: one of the strings the rubric lists. Faithfulness rule: none: the
  rubric's vocabulary, not the document's.
- `bool`: `true` or `false`. Faithfulness rule: none.
- `scope`: `{"<parameter>": [low, high]}`, numbers or `null` for open.
  Faithfulness rule: none; see below.

There is no free-text field type. A normalised proposition such as "estimator
consistent" is either a `verbatim` field quoting the text ("the estimator is
consistent") or an `enum`.

**Faithfulness.** "The node's text" is its `statement`, or its parent's for a
field child. Every `number`, `quantity` and `eqref` value, and every `verbatim`
string, must be found in it:

- **Numerals** recognised in the text are integers and decimals with an optional
  sign, scientific forms `1.2e-3`, `1.2×10^-3`, `1.2 \times 10^{-3}`, digit
  groups of three separated by a comma or a thin space (`1,000`), and a numeral
  followed by `%`, which contributes both its value and its value divided
  by 100. A form number matches a numeral that parses to the same float.
- **Units** match a whitespace- or punctuation-delimited token of the text
  exactly; inside LaTeX, the argument of `\mathrm{…}`, `\text{…}` and `\si{…}`
  counts as a token.
- **Equation references** match `\ref{v}`, `\eqref{v}`, `\cref{v}` or `\Cref{v}`
  for a label `v`, or `(v)` for a printed number `v`.

`scope` is exempt. Scope errors are mostly about scope the text does not state
("in all tested conditions"), so the scope is the analyst's reading of the
node's domain. It is checked against what the node rests on by the scope
diagnostics, not against the text. Its parameters are declared by the rubric.

### Analyst and evidence nodes

An **implicit premise** has `origin = "analyst"`, a type the rubric allows for
analyst nodes, a `statement` in the analyst's words, and **no sources**. Because
it has no source, it is reported as `unanchored` unless something supports it,
which is the intended signal: an assumption the document never wrote down.

An **evidence** node has `origin = "evidence"`, a type the rubric allows for
evidence, a `statement`, and any sources (an external paper, a code commit, a
dataset). It is joined to the node it bears on by a relation from the evidence
node to that node: `supports` or `refutes` with a strength, or the annotation
`assesses` when it should not enter inference. How a check result is
parameterised is outside this contract.

Neither kind has a `form`: forms and faithfulness apply to document nodes only.
Edge rules and assessments apply to every origin. Neither kind is ever
`unassigned` or `compound`.

## Edge kinds

- `part-of` (annotation), child → compound: the child is part of the compound's
  content.
- `cites` (annotation), document chunk → reference: the chunk cites that
  bibliography entry.
- `requires` or `supports` (inferential, with a strength), supporting node → the
  node resting on it: what a node rests on. The type's edge rule names which of
  the two.
- `supports` or `refutes` (inferential, with a strength) or `assesses`
  (annotation), evidence node → the node it bears on: evidence for or against
  that node.

What a node rests on is recorded **only** as relations, never also inside its
form, so there is one source of truth for the argument's structure. A compound
is an endpoint of no relation other than incoming `part-of` and outgoing
`cites`.

## Rubric format

A rubric is a TOML file. The core reads it with the standard library
(`tomllib`); it adds no dependency.

```toml
[rubric]
format = 1                 # the rubric format version this file is written in
name = "methods-paper"     # identifies the rubric in reports
version = "2026.1"         # the rubric's own version, chosen by its author

# Parameters that `scope` fields may range over.
[parameters.sample_rate]
unit = "Hz"
domain = [0, null]         # [low, high]; null is open

[parameters.segment_length]
unit = "s"
domain = [0, null]

# Types given to chunks by `ingest`; any unit not listed starts `unassigned`.
[initial]
reference = "reference"

[types.reference]
description = "A bibliography entry."

[types.method]
description = "What was done: a procedure, test or dataset."
form.kind = { type = "enum", values = ["test", "simulation", "dataset", "procedure"], required = true }
form.scope = { type = "scope" }

[types.result]
description = "What was found."
assess = true
base = "beta:8,2"
form.quantity = { type = "verbatim", required = true }
form.trend = { type = "enum", values = ["vanishes", "increases", "decreases", "bounded"] }
form.value = { type = "quantity" }
form.scope = { type = "scope" }

[types.claim]
description = "A conclusion the document asserts."
assess = true
base = "beta:5,5"
form.shape = { type = "enum", values = ["universal", "existential", "comparative", "bound"], required = true }
form.statement = { type = "verbatim", required = true }
form.scope = { type = "scope" }
rests_on = { types = ["result", "method", "derivation", "literature_claim", "assumption"], min = 1, relation = "requires", strength = "beta:9,1" }

[types.assumption]
description = "A premise taken without support, stated or implicit."
origins = ["document", "analyst"]
base = "beta:8,2"

[types.check]
description = "A check run on another node."
origins = ["evidence"]
```

- `rubric.format`: Rubric format version; `1` for this contract. A reader
  refuses a version it does not know.
- `rubric.name`, `rubric.version`: Free strings identifying the rubric, echoed
  in every report.
- `parameters.<name>.unit`, `domain`: A parameter `scope` fields may name;
  `domain` is `[low, high]`, `null` for open.
- `initial.<unit>`: The type `ingest` gives chunks of that unit; the default is
  `unassigned`.
- `types.<name>`: A node type. The name may not be `unassigned` or `compound`.
- `description`: Shown by `coverage` and in errors.
- `origins`: The origins the type may be used with; default `["document"]`.
- `assess`: Whether nodes of this type need an assessment at the `assessed`
  level; default `false`.
- `base`: The base `annotate` gives a node when it assigns this type and the
  node has none, in the credence syntax of the command line (`0.3`, `beta:8,2`).
- `form.<field>`: A form field: `type` (a field type above), `required` (default
  `false`), and `values` for `enum`.
- `rests_on`: The edge rule: at least `min` relations of type `relation`
  (`requires` or `supports`) into the node from nodes whose type is in `types`.
  `strength` is what `annotate --rests-on` gives the relation it creates.

Unknown keys are an error, so a misspelt rule is not silently ignored. A rubric
is data: its priors are the author's, and the public package ships only the
example rubric used in its tests and documentation.

## Validation levels

`check GRAPH --rubric RUBRIC --level LEVEL` checks a graph against a rubric.
Each level includes the ones before it. Every violation is reported, not only
the first; each is a finding with a stable `diagnostic` code, an id
`<code>:<node or relation id>`, and the nodes and relations involved, in the
same record as the existing diagnostics.

**`anchored`** — the graph is structurally sound; untyped nodes are allowed.
This is what `ingest` must produce.

- `origin-invalid`: `attributes.origin` is missing or not `document`, `analyst`
  or `evidence`.
- `unknown-type`: `kind` is neither built in nor declared by the rubric.
- `origin-not-allowed`: The type is used with an origin outside its `origins`,
  or `unassigned`/`compound` on a non-document node.
- `anchor-invalid`: A document node has other than one source, or its source
  lacks a locator, quote or digest.
- `anchor-mismatch`: A chunk's or span child's `statement` differs from its
  `quote`. Field children are excluded: their `statement` is null and their
  anchor is their parent's, whose `quote` must equal the parent's `statement`.
- `digest-mismatch`: A `digest` is not the digest of its `quote`.
- `analyst-has-source`: An analyst node has a source.
- `part-of-invalid`: A child has other than one `part-of`, or its parent is not
  a compound, or it is itself a compound.
- `span-not-in-parent`: A span child's text is not at its `chars` range in its
  parent's text.
- `field-child-invalid`: A field child has a `statement`, no `form`, or an
  anchor that is not its parent's.
- `compound-children`: A compound has fewer than two children.
- `compound-has-content`: A compound has a `form`, an `assessment` or a `base`,
  or is an endpoint of a relation other than incoming `part-of` or outgoing
  `cites`.

**`typed`** — every node has a type and the attributes its type requires.

- `untyped`: `kind` is `unassigned`.
- `form-missing-field`: A required form field is absent.
- `form-invalid-field`: A form field is undeclared, or its value does not have
  its field type.
- `form-not-in-text`: A `number`, `quantity`, `eqref` or `verbatim` value is not
  found in the node's text (faithfulness).
- `rests-on-missing`: Fewer than `min` relations satisfy the type's edge rule.

**`assessed`** — additionally, every node of a type with `assess = true` has a
verdict backed by evidence, or says why it was not assessed, and the graph
compiles for inference.

The assessment is `attributes.assessment`, one of:

```json
{ "verdict": "holds" }
{ "verdict": "fails" }
{ "verdict": "undetermined" }
{ "verdict": "not_assessed", "reason": "outside the scope of this review" }
```

- `unassessed`: A node of an `assess = true` type has no `assessment`.
- `no-evidence`: A `holds`, `fails` or `undetermined` verdict with no relation
  into the node from an evidence node.
- `missing-reason`: A `not_assessed` assessment without a non-empty `reason`.
- `compile-error`: The graph does not compile; the details are those of today's
  `compile-error`.

Which evidence backs a verdict is read from the relations, so the assessment
holds only the verdict.

Without `--rubric` and `--level`, `check` behaves exactly as it does today. With
`--rubric` and no `--level`, the level is `typed`. `--level` without `--rubric`
is an `invalid-argument`. At `anchored` and `typed` the graph is not compiled,
because an unassessed graph has no bases yet.

## Coverage

`coverage GRAPH --rubric RUBRIC` reports how far the classification has got. It
never fails on a valid graph; it is a report, not a gate. With `--json` it
prints:

```json
{
    "command": "coverage",
    "path": "paper.json",
    "rubric": { "name": "methods-paper", "version": "2026.1" },
    "nodes": 214,
    "by_origin": { "document": 205, "analyst": 6, "evidence": 3 },
    "by_type": {
        "unassigned": { "count": 40, "share": 0.195 },
        "claim": { "count": 12, "share": 0.059 },
        "compound": { "count": 9, "share": 0.044 }
    },
    "untyped": ["s-101", "s-102"],
    "unexamined": ["s-031"],
    "not_assessed": {
        "count": 2,
        "reasons": { "s-044": "outside the scope of this review" }
    }
}
```

- `share` is the type's count over the document nodes, children included and
  compounds included; analyst and evidence nodes are counted in `by_origin`
  only.
- `untyped` lists the `unassigned` nodes, `unexamined` the nodes of
  `assess = true` types with no `assessment`, both in graph order.
- `not_assessed` counts the explicit `not_assessed` verdicts with their reasons.

## Revisions and re-anchoring

A revised document is ingested afresh into a new graph, then re-anchored against
the old one:

```bash
credencegraph ingest paper-v2.tex v2.json --rubric methods.toml --document paper
credencegraph reanchor paper.json v2.json --out paper-v2.json
```

`reanchor OLD NEW --out MERGED` aligns the document chunks of `OLD` with those
of `NEW` (both top-level chunks only, children follow their parent) as two
sequences of digests, with a longest-matching-block alignment in document order
(the standard library's `difflib.SequenceMatcher`, with automatic junk detection
off). It then writes `MERGED`:

- **Kept** (an equal block): the old node is kept whole — id, type, form,
  assessment, children — with the new locator.
- **Changed** (old and new chunks in a replaced block, paired in order): the
  node keeps the old id, takes the new text, locator and digest, keeps its type
  and form, and its assessment moves to `attributes.previous` together with the
  old text and digest. It therefore fails `assessed` until it is re-assessed,
  and any form value no longer in its text fails `typed`. Children whose span
  still occurs in the new text are re-located; the others are dropped.
- **New** (an inserted chunk, or the unpaired surplus of a replaced block):
  added as in `NEW`, with the next unused number of its id prefix in `OLD`.
- **Removed** (a deleted chunk, or the unpaired surplus of a replaced block):
  dropped, with its children.

Analyst and evidence nodes are carried over unchanged. A relation is kept when
both its endpoints survive; the others are dropped. `NEW` must have the same
document id as `OLD`, or the command fails. `MERGED` may not be `OLD`'s path
unless `--force` is passed, so nothing is lost by default. The report lists
every kept, changed, new and removed node id and every dropped relation id.

## Command line

Ingestion layer:

- `ingest SOURCE GRAPH [--rubric RUBRIC] [--document ID] [--format latex|pdf] [--force]`
  writes a graph of document nodes for `SOURCE`.

Core:

- `reanchor OLD NEW --out MERGED [--force]` carries the work on `OLD` over to
  `NEW`, a fresh ingestion of a revised document.
- `check GRAPH --rubric RUBRIC --level anchored|typed|assessed`: the existing
  `check`, with two new options.
- `coverage GRAPH --rubric RUBRIC` prints the coverage report.
- `split GRAPH NODE --at "<verbatim clause>"…` makes `NODE` a compound with one
  span child per `--at`; `split GRAPH NODE --fields` adds one field child.
- `annotate GRAPH NODE --rubric RUBRIC [--type T] [--set FIELD=VALUE]… [--rests-on ID]… [--verdict V | --not-assessed REASON]`
  types, fills in, links and assesses a node.
- `add-node GRAPH --origin analyst|evidence …`: the existing `add-node`, with a
  new option for implicit premises and evidence.

What the commands guarantee:

- `ingest` is named so as not to change `init`, which creates an empty graph.
  `--format` defaults from the extension (`.tex`, `.pdf`). `--document` sets the
  anchors' document id and defaults to `SOURCE` as given. Without `--rubric`
  every chunk starts `unassigned`. The written graph passes
  `check --level anchored`.
- `split --at` takes clauses that occur exactly once in the node's text; each
  must occur, or the command fails and the file is unchanged. Splitting a node
  that is already a compound adds children to it. A node with a `form`, an
  `assessment` or a `base` cannot be split; `annotate NODE --type unassigned`
  removes all three along with its type.
- `annotate` validates each write: `--type` must be declared and allowed for the
  node's origin, `--set` names a declared field and its value (parsed as JSON,
  else taken as a string) must have the field type and pass faithfulness.
  Missing required fields are allowed, since they are filled one at a time;
  `check --level typed` reports them. Assigning a type with a `base` sets that
  base when the node has none. `--rests-on ID` adds the edge rule's relation
  from `ID` with the rule's strength. `--verdict` sets `holds`, `fails` or
  `undetermined`.
- Every command follows the existing conventions: `--json` output, exit status 1
  on failure (`cli/common.py:1-7`), and a file that is replaced in one step only
  when the write succeeds (`write_graph`, `cli/common.py:317-345`).

New error codes: `invalid-rubric` (the rubric file is malformed or of an unknown
`format`), `rubric-violation` (`check --level` found violations;
`details.violations` lists them all), and `missing-extra` (an ingestion command
was run without the ingestion extra installed; the hint names the install
command).

## Core and ingestion

**Core** — in the `credencegraph` package, with no new dependency:

- text normalisation and the digest;
- rubric loading and validation (`tomllib`);
- the `anchored`, `typed` and `assessed` checks, faithfulness and coverage;
- `split`, `annotate`, `reanchor`, the `check` and `add-node` options;
- the diagnostics that later read `scope`.

None of these knows where a graph came from; they apply as well to a graph
written by hand with verbatim anchors.

**Ingestion** — a subpackage `credencegraph.ingest`, installed with
`pip install credencegraph[ingest]`:

- the LaTeX reader and the PDF reader, which turn a document into chunks with
  locators;
- the `ingest` command.

The core never imports `credencegraph.ingest`. The command line registers
`ingest` but imports the readers only when it runs, so a core installation
starts without them and reports `missing-extra` if `ingest` is used. Every
dependency of the extra must have a licence compatible with this package's
BSD-3-Clause licence; copyleft PDF libraries such as PyMuPDF (AGPL) are
excluded. Given the same input file and the same version of the package, a
reader produces the same chunks, which re-anchoring relies on.

Not part of this contract: how chunks are found inside a document (sentence
splitting, macro expansion, PDF layout analysis), units of measure beyond the
token match above, the scope diagnostics, how check results are parameterised,
and any merging of chunks a reader split wrongly.

## Example

An invented methods paper. The rubric is the one above. After `ingest`, typing
and one split, the relevant part of the graph is:

```json
{
    "id": "s-012",
    "kind": "compound",
    "statement": "We show that the bias vanishes for long segments, which implies that the estimator is consistent.",
    "sources": [
        {
            "document": "paper",
            "locator": "file=sec/results.tex;lines=12-13",
            "quote": "We show that the bias vanishes for long segments, which implies that the estimator is consistent.",
            "digest": "sha256:…"
        }
    ],
    "attributes": { "origin": "document", "unit": "sentence" }
}
```

with two span children, joined to `s-012` by `part-of`:

```json
[
    {
        "id": "s-012a",
        "kind": "result",
        "statement": "We show that the bias vanishes for long segments",
        "sources": [
            {
                "document": "paper",
                "locator": "file=sec/results.tex;lines=12-13;chars=0-48",
                "quote": "We show that the bias vanishes for long segments",
                "digest": "sha256:…"
            }
        ],
        "base": { "alpha": 8, "beta": 2 },
        "attributes": {
            "origin": "document",
            "unit": "sentence",
            "form": { "quantity": "the bias", "trend": "vanishes" }
        }
    },
    {
        "id": "s-012b",
        "kind": "claim",
        "statement": "the estimator is consistent",
        "sources": [
            {
                "document": "paper",
                "locator": "file=sec/results.tex;lines=12-13;chars=69-96",
                "quote": "the estimator is consistent",
                "digest": "sha256:…"
            }
        ],
        "base": { "alpha": 5, "beta": 5 },
        "attributes": {
            "origin": "document",
            "unit": "sentence",
            "form": {
                "shape": "universal",
                "statement": "the estimator is consistent"
            }
        }
    }
]
```

and a relation `s-012a --requires--> s-012b` with strength Beta(9, 1), added by
`annotate s-012b --rests-on s-012a`. The bias "vanishes" is an `enum`, not
`value: 0`: the text contains no zero, so a number would fail faithfulness.

A second claim and the method it rests on:

```text
s-019  method  "We tested the filter on simulated data sampled between 10 and 100 Hz."
       form: {kind: test, scope: {sample_rate: [10, 100]}}
s-031  claim   "Our filter recovers signals with amplitude above 0.1 in all tested conditions."
       form: {shape: universal, statement: "Our filter recovers signals with amplitude above 0.1",
              scope: {sample_rate: [1, 1000]}}
       s-019 --requires--> s-031
```

Both pass `typed`. The claim's scope, [1, 1000] Hz, is the analyst's reading of
"all tested conditions" and is exempt from faithfulness; it exceeds the [10,
100] Hz its method covers, which is what the scope diagnostics will flag.
Setting `form.value = {"value": 0.2, "unit": "Hz"}` on the result `s-012a` would
fail `typed` with `form-not-in-text`: neither 0.2 nor `Hz` is in its text.
