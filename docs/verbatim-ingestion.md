---
title: Verbatim-chunk ingestion
description:
    The contract for building a graph from a document's own text — node and edge
    kinds, the rubric format, validation levels, coverage, re-anchoring, the
    command line, and which parts belong to the core and which to the ingestion
    layer.
---

> **Partly implemented.** This page is the contract that the verbatim-chunk
> features are built against. The core part now exists, in
> `credencegraph.rubric`: text normalisation and the digest, rubric files, the
> `anchored`, `typed` and `assessed` checks with faithfulness, coverage, and the
> `check --rubric --level`, `coverage`, `split` and `annotate` commands. The
> ingestion layer has the LaTeX reader, the PDF reader and `ingest` with
> `--format`. Not yet implemented: `reanchor` and `add-node --origin`. The
> section on what the engine provided before this contract describes it as it
> was and cites that code.

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
  One node per display, not per line; its label, if any, is `attributes.label`
  (in a PDF, its printed equation number).
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

- `unassigned`: not yet classified. Allowed only on document nodes. A node is
  reset to it with `annotate NODE --type unassigned`.
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
  sign (`+`, `-` or `−`, not directly after a letter, digit or dot, so the
  hyphen in `5-7` or `s-012` is not a minus), scientific forms `1.2e-3`,
  `1.2×10^-3`, `1.2 \times 10^{-3}`, digit groups of three separated by a comma
  or a thin space (`1,000`), and a numeral followed by `%`, which contributes
  both its value and its value divided by 100. A form number matches a numeral
  that parses to the same float.
- **Units** match a whitespace- or punctuation-delimited token of the text
  exactly; inside LaTeX, the argument of `\mathrm{…}`, `\text{…}` and `\si{…}`
  counts as a token. A unit needs a delimiter only on a side where it starts or
  ends with a letter or digit, so the `%` of `1.1%` is a token. The operators
  `/`, `^`, `·`, `⋅`, `×` and `*` join symbols into one compound unit, and the
  LaTeX subscript operator `_` joins a symbol to its subscript, so none of them
  delimits a token: a unit `m` is not found in `3 m/s`, `3 m^2`, `3 N×m`,
  `3 N*m` or `3 m_e`. Write the compound.
- **Equation references** match `\ref{v}`, `\eqref{v}`, `\cref{v}` or `\Cref{v}`
  for a label `v`, or `(v)` for a printed number `v`.

`scope` is exempt. Scope errors are mostly about scope the text does not state
("in all tested conditions"), so the scope is the analyst's reading of the
node's domain. It is checked against what the node rests on by the scope
diagnostics, not against the text. Its parameters are declared by the rubric.

**Scopes.** An interval `[low, high]` holds both its ends; `null` is an open
end, so `[10, null]` is every value of 10 or more and `[3, 3]` the single
value 3. A node's scope on a parameter is the intersection of the intervals its
`scope` fields give it. A parameter a node names in no `scope` field is not part
of its scope: the node makes no claim about it, so it is not checked there and,
as something another node rests on, restricts nothing there. A node with no
`scope` field, which includes every analyst and evidence node, declares no
scope. What a node rests on covers, on a parameter, the intersection of the
scopes they declare there; when those scopes do not overlap, the intersection is
empty and covers nothing. A node whose own `scope` fields do not overlap on a
parameter declares an empty scope there, which claims no value and so lies
within any domain and anything it rests on.

**Units.** A `quantity` unit is read as SI symbols, each with an optional SI
prefix and integer power, joined by `*`, `·`, `⋅`, `×` or a space, with `/`
dividing by everything after it: `Hz`, `kHz`, `m/s`, `m s^-1`, `kg m^2/s^2`.
Besides the SI base and derived units it knows `eV`, `L`, `pc`, `yr`, `min`,
`h`, `d`, `au`, `ly`, `rad`, `sr`, `%` and `1`. Two units agree when they are
the same text or have the same dimension, so `Hz` agrees with `kHz` and `1/s`
but not with `s`. Two different units of which either cannot be read do not
agree, since nothing shows that they do.

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

The other two inferential kinds, `equivalent` and `exclusive`, have no role in
this contract: `ingest` must not create them (a requirement on the ingestion
layer, which is not implemented yet), `annotate` never creates them, and an edge
rule cannot name them, since its `relation` is `requires` or `supports`. An
ingested graph may still carry them, added by the analyst with `add-relation`,
and the validation levels treat them like any relation outside the list above:
`anchored` rejects one only when an endpoint is a compound
(`compound-has-content`), and `typed` neither requires nor counts them, so they
can never satisfy a type's edge rule. At `assessed` they enter inference as
usual, so nodes joined by `equivalent` must not carry conflicting bases
(`compile-error`), and one from an evidence node counts as the relation that
backs a verdict.

## Rubric format

A rubric is a TOML file. The core reads it with the standard library
(`tomllib`); it adds no dependency.

```toml
# An example rubric for a methods paper. It is the rubric used in the documentation and the tests
# of credencegraph; its priors are illustrative, not a recommendation.

[rubric]
format = 2             # the rubric format version this file is written in
name = "methods-paper" # identifies the rubric in reports
version = "2026.2"     # the rubric's own version: changes whenever any value in the file changes

# How the values were obtained. These are illustrative, fitted against nothing. `status` is "example" or
# "calibrated"; `scoring` is the proper scoring rule the values were fitted with, "brier" or "log".
[calibration]
status = "example"
scoring = "log"
against = "nothing: the values are illustrative"
date = 2026-10-10

# Parameters that `scope` fields may range over. TOML has no null, so an open end is -inf or inf.
[parameters.sample_rate]
unit = "Hz"
domain = [0, inf]

[parameters.segment_length]
unit = "s"
domain = [0, inf]

# Types given to chunks by ingestion; any unit not listed starts `unassigned`.
[initial]
reference = "reference"

[types.reference]
description = "A bibliography entry."

[types.method]
description = "What was done: a procedure, test or dataset."
base = "beta:9,1"
form.kind = { type = "enum", values = [
  "test",
  "simulation",
  "dataset",
  "procedure",
], required = true }
form.scope = { type = "scope" }

[types.result]
description = "What was found."
assess = true
base = "beta:8,2"
form.quantity = { type = "verbatim", required = true }
form.trend = { type = "enum", values = [
  "vanishes",
  "increases",
  "decreases",
  "bounded",
] }
form.value = { type = "quantity" }
form.scope = { type = "scope" }

[types.derivation]
description = "A step of mathematics the document carries out."
assess = true
base = "beta:9,1"
form.equation = { type = "eqref" }

[types.literature_claim]
description = "A finding the document takes from a cited work."
base = "beta:8,2"
form.statement = { type = "verbatim", required = true }

[types.claim]
description = "A conclusion the document asserts."
assess = true
base = "beta:5,5"
form.shape = { type = "enum", values = [
  "universal",
  "existential",
  "comparative",
  "bound",
], required = true }
form.statement = { type = "verbatim", required = true }
form.bound = { type = "number" }
form.scope = { type = "scope" }
rests_on = { types = [
  "result",
  "method",
  "derivation",
  "literature_claim",
  "assumption",
], min = 1, relation = "requires", strength = "beta:9,1" }

[types.assumption]
description = "A premise taken without support, stated or implicit."
origins = ["document", "analyst"]
base = "beta:8,2"

[types.check]
description = "A check run on another node, of a kind not listed under [checks]."
origins = ["evidence"]

# Check types. A check name is the type of the evidence nodes the check produces, so it may not also be
# declared under [types]. `targets` are the declared types a check may be run on; `sensitivity` is
# P(pass | target true) and `false_pass` is P(pass | target false), and the mean of `sensitivity` must exceed
# the mean of `false_pass`. `models` lists the exact model ids the values were fitted with, and is empty for a
# check with no model in the loop.
[checks.recompute-numbers]
description = "Re-evaluate an arithmetic relation between numbers the text reports."
targets = ["result", "claim"]
sensitivity = "beta:19,1"
false_pass = "beta:1,9"
models = []

[checks.citation-support]
description = "Judge whether the cited source supports the citing sentence."
targets = ["literature_claim"]
sensitivity = "beta:8,2"
false_pass = "beta:2,8"
models = ["example-model-2026-01-01"]

# The implicit-premise pass.
[premises]
base = "beta:2,2"      # the prior of an implicit premise
strength = "beta:9,1"  # the strength of its requires relation into the conclusion
for = ["claim"]        # the conclusion types the pass runs on
max_per_conclusion = 3

# Hedge phrases and the credence each carries; compared after text normalisation and case-folding.
[hedges]
"suggests" = "beta:6,4"
"may" = "beta:5,5"
"we show" = "beta:9,1"
```

- `rubric.format`: Rubric format version: `1` or `2`. Format 2 adds the
  `calibration`, `checks`, `premises` and `hedges` tables below; a format-1 file
  may not use them, and a reader refuses a format it does not know.
- `rubric.name`, `rubric.version`: Free strings identifying the rubric, echoed
  in every report. `rubric.format` is the structure; `rubric.version` is the
  values, and changes whenever any value in the file changes.
- `calibration` (format 2): How the values were obtained. `status` is `example`
  (illustrative values) or `calibrated`; `scoring` is the proper scoring rule
  they were fitted with, `brier` or `log`; `against` describes the outcome set
  they were fitted on; `date` is a TOML date. All four are required.
- `parameters.<name>.unit`, `domain`: A parameter `scope` fields may name;
  `domain` is `[low, high]`, open by default. TOML has no null, so an open end
  is written `-inf` or `inf`.
- `initial.<unit>`: The type `ingest` gives chunks of that unit; the default is
  `unassigned`.
- `types.<name>`: A node type. The name may not be `unassigned` or `compound`.
- `description`: Shown by `coverage` and in errors; default empty.
- `origins`: The origins the type may be used with; default `["document"]`.
- `assess`: Whether nodes of this type need an assessment at the `assessed`
  level; default `false`.
- `base`: The base `annotate` gives a node when it assigns this type and the
  node has none, in the credence syntax of the command line (`0.3`, `beta:8,2`).
- `form.<field>`: A form field: `type` (a field type above), `required` (default
  `false`), and `values` for `enum`.
- `rests_on`: The edge rule: at least `min` (default 1) relations of type
  `relation` (`requires` or `supports`) into the node from nodes whose type is
  in `types`, each of which must be a declared type. `strength` is what
  `annotate --rests-on` gives the relation it creates. `types`, `relation` and
  `strength` are required.
- `checks.<name>` (format 2): A check type. Its name is the type of the evidence
  nodes the check produces, so it may not also be declared under `types`; its
  origins are `["evidence"]`. `targets` lists the declared types it may be run
  on. `sensitivity` is `P(pass | target true)` and `false_pass` is
  `P(pass | target false)`, in the credence syntax above; the mean of
  `sensitivity` must exceed the mean of `false_pass`, since a check no more
  likely to pass on a true target than a false one is no evidence. `models`
  lists the exact model ids the values were fitted with, and is empty for a
  check with no model in the loop. `description` is optional; the other four
  keys are required.
- `premises` (format 2): The implicit-premise pass. `base` is the prior of an
  implicit premise, `strength` the strength of its `requires` relation into the
  conclusion, `for` the declared conclusion types the pass runs on, and
  `max_per_conclusion` (an integer of 1 or more) how many premises one
  conclusion may receive. All four are required.
- `hedges` (format 2): A hedge phrase and the credence it carries. Phrases are
  compared after the [text normalisation](#text-normalisation-and-digest) and
  case-folding, so two keys that differ only in case or whitespace are an error.

Unknown keys are an error, so a misspelt rule is not silently ignored. A rubric
is data: its priors are the author's, and the public package ships only the
example rubric used in its tests and documentation, the one above, which is also
the file
[`docs/examples/methods-paper.toml`](https://github.com/isaac-cf-wong/credencegraph/blob/main/docs/examples/methods-paper.toml).
Every type a node can rest on has a `base`, so that a classified graph can
compile at the `assessed` level.

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
- `out-of-domain`: A node's scope reaches outside a parameter's declared
  `domain`, such as a sample rate scoped at `[-3, -3]` against a domain of
  `[0, inf]`.
- `scope-exceeds-support`: A node's scope on a parameter is not contained in the
  intersection of the scopes of what it rests on there, such as a claim scoped
  to sample rates `[1, 1000]` Hz resting on a result scoped to `[10, 100]` Hz.
- `unit-mismatch`: A `quantity` field of a node and the field of the same name
  on something it rests on have units of different dimensions, such as `Hz` and
  `s`.

The last three read what a node **rests on**: every `requires` relation into it,
and the relations its type's edge rule counts. The scope and unit checks use
only form fields that have their field type; the others are already
`form-invalid-field`.

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

**`--rubric-credences`** — opt-in rules, added to any level, for a graph whose
credences come from the rubric alone: every `base` and `strength` is a rubric
value, and the only thing that can raise a credence above its node's base is an
observed check. They read the rubric's format-2 tables. A hand-built graph is
unaffected unless the flag is passed.

A **check node** is an evidence node whose type is a check type of the rubric
and that carries a check record, an object, in `attributes.check`; the record's
`result` is `passed`, `failed` or `error`. A **check edge** is a `supports`
relation from a document or analyst node to a check node: the check was run on
that node. The check node's `base` is its type's `false_pass` and the check
edge's strength is `1 - (1 - E[sensitivity]) / (1 - E[false_pass])`, so a pass
has probability `E[false_pass]` when the checked node is false and
`E[sensitivity]` when it is true.

- `credence-not-rubric`: A document or analyst node has a `base` other than its
  type's, or an analyst node one other than `premises.base` when the rubric has
  `[premises]`; a `requires` between document or analyst nodes has a strength
  other than `premises.strength` from an analyst node (when the rubric has
  `[premises]`), or else its target type's edge-rule strength (an edge rule of
  `supports`, or none, gives no strength to match); a check node has a `base`
  other than its type's `false_pass`; or a check edge has a strength other than
  the one above. A node without a `base` is not reported here.
- `relation-not-allowed`: An inferential relation touching a document, analyst
  or check node that is neither a `requires` from a document or analyst node to
  another, nor a check edge. This covers `supports` and `refutes` between
  document or analyst nodes, any `equivalent` or `exclusive` touching one, any
  relation from an evidence node to one, and any relation out of a check node.
- `check-unobserved-invalid`: `attributes.observed` is on a node that is not a
  check node; or a check node's `attributes.observed` disagrees with its result:
  it must be `true` for `passed`, `false` for `failed`, and absent for `error`;
  or the result is none of these.

With only `requires` parents, a node's probability is its base times factors of
at most 1, and a check node is a child that changes nothing until it is
observed. So on a graph that passes these rules, before anything is observed, no
document or analyst node has a point credence above the mean of its base.

## Coverage

`coverage GRAPH --rubric RUBRIC` reports how far the classification has got. It
never fails on a valid graph; it is a report, not a gate. With `--json` it
prints:

```json
{
    "command": "coverage",
    "path": "paper.json",
    "rubric": { "name": "methods-paper", "version": "2026.2" },
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
- `annotate` validates each write: `--type` must be declared, or be the built-in
  `unassigned`, which needs no declaration and resets the node; either way it
  must be allowed for the node's origin. `--set` names a declared field and its
  value (parsed as JSON, else taken as a string) must have the field type and
  pass faithfulness. Missing required fields are allowed, since they are filled
  one at a time; `check --level typed` reports them. Assigning a type with a
  `base` sets that base when the node has none. `--rests-on ID` adds the edge
  rule's relation from `ID` with the rule's strength. `--verdict` sets `holds`,
  `fails` or `undetermined`.
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
splitting, macro expansion, PDF layout analysis), converting values between
units of measure (only their dimensions are compared), how check results are
parameterised, and any merging of chunks a reader split wrongly.

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

The method passes `typed`; the claim fails it with `scope-exceeds-support`. Its
scope, [1, 1000] Hz, is the analyst's reading of "all tested conditions" and is
exempt from faithfulness, but it exceeds the [10, 100] Hz its method covers.
Setting `form.value = {"value": 0.2, "unit": "Hz"}` on the result `s-012a` would
fail `typed` with `form-not-in-text`: neither 0.2 nor `Hz` is in its text.
