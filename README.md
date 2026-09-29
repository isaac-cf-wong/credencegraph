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

## License

BSD 3-Clause. See [LICENSE](LICENSE).
