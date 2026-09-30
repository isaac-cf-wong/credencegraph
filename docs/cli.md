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

| Command    | Does                                                                                     |
| ---------- | ---------------------------------------------------------------------------------------- |
| `init`     | Create an empty graph file; `--force` replaces an existing one.                          |
| `add-node` | Add a node: `--id`, and optionally `--statement`, `--kind`, `--base`, `--stated`, `--source`. |
| `relate`   | Add a relation `SOURCE TARGET --type TYPE`, with `--strength` where the type needs one.  |
| `query`    | `marginal`, `joint`, `conditional` or `intervene`, with `--given`, `--set`, `--draws`, `--seed`. |
| `diagnose` | The findings of every diagnostic; with `--target`, the weak points of that node too.     |
| `check`    | Whether the file is a valid graph that compiles; exits with status 1 when it is not.     |
| `version`  | The installed version.                                                                   |

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
`compile-error`, `zero-probability`, `problem-too-large` and `inference-error`.

Without `--json`, a success prints a short summary and a failure prints
`error: …` and `hint: …` on stderr. Malformed command lines, such as a missing
required option, are reported by the argument parser with status 2 in both
modes.
