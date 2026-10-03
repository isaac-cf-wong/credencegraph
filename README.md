# credencegraph

[![Python CI](https://github.com/isaac-cf-wong/credencegraph/actions/workflows/ci.yml/badge.svg)](https://github.com/isaac-cf-wong/credencegraph/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/isaac-cf-wong/credencegraph/main.svg)](https://results.pre-commit.ci/latest/github/isaac-cf-wong/credencegraph/main)
[![Documentation Status](https://github.com/isaac-cf-wong/credencegraph/actions/workflows/documentation.yml/badge.svg)](https://isaac-cf-wong.github.io/credencegraph/)
[![codecov](https://codecov.io/gh/isaac-cf-wong/credencegraph/graph/badge.svg)](https://codecov.io/gh/isaac-cf-wong/credencegraph)
[![PyPI Version](https://img.shields.io/pypi/v/credencegraph)](https://pypi.org/project/credencegraph/)
[![Python Versions](https://img.shields.io/pypi/pyversions/credencegraph)](https://pypi.org/project/credencegraph/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

**credencegraph is a neutral engine for graphs of propositions with credences.**

Scientific statements are rarely certain. Instead of proving a statement,
credencegraph records how strongly each proposition is believed _given_ the
propositions it rests on, and computes what follows:

- the credence of any proposition, conjunction, or conditional query;
- how that credence changes when a premise is assumed true, false, or wrong;
- which premises and links a conclusion depends on most — its **weak points**.

Each node keeps the original text it came from alongside the proposition it
states. Relations are either **inferential** (`requires`, `supports`, `refutes`,
`equivalent`, `exclusive`), with a fixed probabilistic meaning, or
**annotations** (any other type), which are stored and queried but ignored by
inference. The engine has no built-in notion of authors, projects, or workflow;
those are expressible as ordinary nodes and annotation relations.

> **Status: experimental.** The package is under active development and stays on
> 0.x. Any minor release may change the API.

## Installation

```bash
pip install credencegraph
```

Requires Python 3.13+.

## Example

```python
from credencegraph.core import Beta, Graph, Node, Relation
from credencegraph.inference import conditional, intervene, marginal
from credencegraph.semantics import compile_graph

graph = Graph()
graph.add_node(Node("calibrated", statement="The instrument is calibrated.", base=0.9))
graph.add_node(Node("signal", statement="The signal is real.", base=0.05))
graph.add_node(Node("claim", statement="The effect exists.", base=0.1))
graph.add_relation(Relation("r1", "requires", "calibrated", "signal", strength=1.0))
graph.add_relation(Relation("r2", "supports", "signal", "claim", strength=Beta(8, 2)))
network = compile_graph(graph)

marginal(network, "claim").point  # 0.1324
conditional(network, "calibrated", {"claim": True}).point  # 0.9245
intervene(network, "claim", set={"calibrated": False}).point  # 0.1

answer = marginal(network, "claim", rng=0)
answer.band  # Band(q05=0.1235, q50=0.1335, q95=0.1387), over draws of Beta(8, 2)
answer.mean_over_draws  # 0.1326, the average of the per-draw answers
```

Conditioning on the claim raises the credence that the instrument is calibrated;
intervening on calibration asks what the claim rests on without that premise.
The point answer uses the credence means; the band shows how far the answer
moves when the uncertain parameters are drawn from their credences, and is
omitted when every parameter is a plain number.

## Diagnostics

```python
from credencegraph.diagnostics import diagnose

for finding in diagnose(graph, "claim"):
    print(finding.id, finding.message)
```

Each finding is a record with an id, the nodes and relations involved, a value
and a one-line explanation (`finding.to_dict()` is its JSON form). For the graph
above it reports, among others, that `calibrated` is an assumption with no
source, ranks `r2` as the crux — the only uncertain input the claim depends on —
and ranks `signal` as the premise most worth resolving. A node's `stated`
credence, the confidence its source asserts, is compared with the credence its
premises deliver in log-odds, and a gap above 2 ln(11/9) ≈ 0.40 is reported as
an overclaim or underclaim. That value is the infimum of the log-odds gaps
between probabilities more than 0.1 apart, approached arbitrarily closely but
never reached, so every pair that an absolute gap of 0.1 would report is
reported too, up to a rounding allowance of 1e-12.

## Command line

The same graph can be built and queried from the shell. Every command takes
`--json` and then prints a single JSON object, including on failure, where it
carries a stable error `code`, a message and a hint.

```bash
credencegraph init g.json
credencegraph add-node g.json --id calibrated --base 0.9
credencegraph add-node g.json --id signal --base 0.05
credencegraph add-node g.json --id claim --base 0.1
credencegraph relate g.json calibrated signal --type requires --strength 1
credencegraph relate g.json signal claim --type supports --strength beta:8,2
credencegraph check g.json
credencegraph query g.json marginal claim --json
credencegraph query g.json intervene claim --set calibrated=false
credencegraph diagnose g.json --target claim --json
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
