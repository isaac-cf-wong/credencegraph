---
title: Command line
description:
    Build, check, query and diagnose a graph file from the shell, with JSON
    output on every command.
---

The `credencegraph` command works on a graph file in the package's JSON format.
Every write goes through the same validation as the Python API, and the file is
replaced only when the write succeeds, so a rejected command leaves it exactly
as it was.

```bash
credencegraph init g.json
credencegraph add-node g.json --id calibrated --base 0.9
credencegraph add-node g.json --id signal --base 0.05 --source doi:10.0000/example#p4
credencegraph add-node g.json --id claim --base 0.1 --stated 0.5
credencegraph relate g.json calibrated signal --type requires --strength 1
credencegraph relate g.json signal claim --type supports --strength beta:8,2
credencegraph check g.json
credencegraph query g.json marginal claim                          # P(claim) = 0.1324
credencegraph query g.json conditional calibrated --given claim=true
credencegraph query g.json intervene claim --set calibrated=false  # 0.1
credencegraph diagnose g.json --target claim
```

## Commands

| Command    | Does                                                                                             |
| ---------- | ------------------------------------------------------------------------------------------------ |
| `init`     | Create an empty graph file; `--force` replaces an existing one.                                  |
| `add-node` | Add a node: `--id`, and optionally `--statement`, `--kind`, `--base`, `--stated`, `--source`.    |
| `relate`   | Add a relation `SOURCE TARGET --type TYPE`, with `--strength` where the type needs one.          |
| `query`    | `marginal`, `joint`, `conditional` or `intervene`, with `--given`, `--set`, `--draws`, `--seed`. |
| `diagnose` | The findings of every diagnostic; with `--target`, the weak points of that node too.             |
| `check`    | Whether the file is a valid graph that compiles; a graph that does not is a `compile-error`.     |
| `version`  | The installed version.                                                                           |

**Credences** are written as a bare probability, `0.3`, or as a Beta
distribution, `beta:ALPHA,BETA`, such as `beta:8,2`.

**Sources** are `DOCUMENT` or `DOCUMENT#LOCATOR`, split at the last `#`, so
`https://example.org/paper#sec2` has the locator `sec2`. Repeat `--source` for
several.

**Query targets and evidence** are `NODE=true` or `NODE=false`; a bare `NODE`
target means true. A `marginal` query takes one target and a `joint` query
several. Any kind of query accepts `--given`; only `intervene` accepts `--set`.
The relation id defaults to `SOURCE-TYPE-TARGET`; pass `--id` to add a second
relation of the same type between the same nodes.

A relation type outside `requires`, `supports`, `refutes`, `equivalent` and
`exclusive` is an annotation, which inference ignores. A type close to an
inferential one, such as `support`, is still stored, but the command warns.

## JSON output

With `--json`, a command prints exactly one JSON object on stdout, on success
and on failure. A success echoes the command and its result:

```json
{
    "command": "query",
    "path": "g.json",
    "kind": "marginal",
    "target": { "claim": true },
    "given": {},
    "set": {},
    "seed": 0,
    "point": 0.1324,
    "band": { "q05": 0.1235, "q50": 0.1335, "q95": 0.1387 },
    "mean_over_draws": 0.1326,
    "draws": 1000
}
```

A failure exits with status 1 and prints an `error` object instead:

```json
{
    "command": "relate",
    "error": {
        "code": "cycle",
        "message": "relation 'claim-supports-signal' would create a cycle: claim --supports[claim-supports-signal]--> signal --supports[signal-supports-claim]--> claim",
        "hint": "requires, supports and refutes relations must not form a cycle; drop or reverse one of the relations named",
        "details": {
            "cycle": ["claim", "signal", "claim"],
            "relations": ["claim-supports-signal", "signal-supports-claim"]
        }
    }
}
```

The `code` is stable and meant for programs; the `message` and `hint` are for
people. The codes are `file-exists`, `file-not-found`, `io-error`,
`invalid-graph`, `invalid-argument`, `duplicate-id`, `unknown-node`, `cycle`,
`compile-error`, `zero-probability` and `problem-too-large`.

A graph that cannot be compiled for inference is a `compile-error`, whichever
command meets it: an inference variable without a base, equivalent nodes with
different bases, or a cycle that appears once equivalent nodes are merged. The
`details` name the nodes: `errors` holds a `missing-parameter` finding per node
without a base, and `conflicting_bases` each pair of disagreeing equivalent
nodes. When `check` fails this way, `details` also carries the whole verdict:
the `path`, the `nodes` and `relations` counts and the `warnings`. A passing
`check` prints `{"command", "path", "nodes", "relations", "warnings"}`, where
the warnings are the unanchored variables, which do not fail the check.

A `zero-probability` error means the evidence of a query has probability zero,
so the query has no answer. The command works out the cause:

- If the query fails without its `--given` and `--set` values, or none were
  passed, as with `diagnose`, the cause is the graph: no world satisfies every
  `exclusive` relation, which needs a base or strength of exactly 0 or 1.
- Otherwise the passed values are the cause, and everything the error names is
  established by running the same command, of the same kind, with that one thing
  changed. The message names a flag when the query fails with that flag's values
  alone, or both flags when only their values together fail. The hint names an
  `exclusive` relation, in `details.exclusive_relation`, only when the passed
  values make both of its propositions true and removing the relation makes the
  command succeed; otherwise it mentions exclusive relations only when the named
  values succeed without them.
- The remedies name only changes that were tried and make the command succeed.
  Dropping any one of `details.drop_any_one_of` does, or, when no single value
  does, dropping all of `details.drop_all_of`: the first set of the smallest
  size that does, of which there may be others, and none of whose values can be
  kept. A drop that leaves the command ill-formed, such as an `intervene` query
  without `--set`, is not offered. Moving any one of `details.move_any_one_of`
  off 0 and 1 does too, or all of `details.move_all_of`; those parameters are
  listed with their values in `details.extreme_parameters`, as `base:NODE` or
  `strength:RELATION`, and a base or strength at 0 or 1 that does not bear on
  the query is not listed. A move is tried at 0.5, which stands for every other
  value: in exact arithmetic, whether a query has probability zero depends only
  on which bases and strengths are exactly 0 or 1. A base or strength within
  rounding error of 0 or 1, such as 0.9999999999999999, can round a probability
  to zero, and is not listed as one to move.
- The search is bounded. Drops, moves and the removal of one exclusive relation
  are tried separately, never combined. Every single drop and every single move
  is tried; when none succeeds, sets of two, three and so on are tried, up to
  256 sets for the drops and 256 for the moves. Each list is empty when no such
  change was found. When none is, the hint says so and that combined changes
  were not tried. `details.search_tries` gives, for `drop` and `move`, the
  number of `single` changes and of sets of `several` tried, and, when that
  search stopped at 256 sets, `stopped_at`: the size of the first set it left
  untried. Such a search is named in `details.search_truncated`, and the hint
  says what it tried and that a set not tried may still make the command
  succeed.

A `problem-too-large` error means exact inference would build an intermediate
factor above the limit, in table entries; `details.required` is the size of the
first such factor and `details.limit` the limit. `query` and `diagnose` take the
limit as `--max-factor-size` (default 2²²), so passing at least
`details.required` lets that factor through, though a later one may need more.
The search for remedies to a `zero-probability` error runs under the same limit.

Values that contradict each other outright are refused before any inference, as
an `invalid-argument`: two equivalent nodes given different values by `--given`
or by `--set`, and a `--given` value that contradicts the `--set` value of the
same proposition.

Without `--json`, a success prints a short summary and a failure prints
`error: …` and `hint: …` on stderr. Malformed command lines, such as a missing
required option, are reported by the argument parser with status 2 in both
modes.
