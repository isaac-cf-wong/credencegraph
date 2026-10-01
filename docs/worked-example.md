---
title: Worked example
description:
    Encoding the argument of a paper by hand, reading its weak points, and two
    modelling choices every encoding meets.
---

This page encodes the argument of one paper by hand with the
[command line](cli.md), runs the diagnostics on its headline claim, and
discusses what they say. Two modelling choices come up in any such encoding —
which way an observation points, and how support and refutation of the same
proposition combine — and they are worked through first on small graphs whose
answers can be checked by hand.

## How the encoding is built

1. **One node per proposition the argument uses**, with the sentence it came
   from as `--statement` and its place in the paper as `--source`, such as
   `doi:10.0000/example#sec4`. A proposition the paper states as a measured
   result gets the credence the measurement supports; one the paper assumes gets
   a wide Beta, so that the diagnostics can tell "0.5 because nobody knows" from
   "0.5 because it was measured".
2. **The headline claim's own confidence as `--stated`**, when the paper gives
   one ("we exclude … at 95% confidence"), so that `diagnose` can compare it
   with what the encoded argument delivers.
3. **One relation per inferential step**: `requires` for a condition the claim
   cannot do without, `supports` and `refutes` for reasons for and against.
4. **A common parent for shared causes.** Two results that share an instrument,
   a calibration or a model are not independent. Their shared dependence is a
   node of its own, such as "the calibration is correct", that both require;
   leaving it out makes the claim look better supported than it is.

Then `credencegraph check` validates the file and
`credencegraph diagnose g.json --target CLAIM` reports the weak points.

## Which way an observation points

Relations read from premise to conclusion: `E --supports--> H` says E is a
reason for H. Empirical evidence is often stated the other way, as how likely
the observation is under the hypothesis and under its negation, P(obs | H) and
P(obs | ¬H). That direction is already expressible, with the hypothesis as the
premise and the observation as the conclusion:

- when P(obs | H) ≥ P(obs | ¬H), `H --supports--> obs` with the observation's
  base set to P(obs | ¬H) and strength 1 − (1 − P(obs | H)) / (1 − P(obs | ¬H));
- when P(obs | H) < P(obs | ¬H), `H --refutes--> obs` with the base set to P(obs
  | ¬H) and strength 1 − P(obs | H) / P(obs | ¬H).

For a hypothesis with prior 0.3 and two independent observations, each with
P(obs | H) = 0.8 and P(obs | ¬H) = 0.2, the base is 0.2 and the strength 0.75:

```bash
credencegraph init obs.json
credencegraph add-node obs.json --id H --base 0.3
credencegraph add-node obs.json --id o1 --base 0.2
credencegraph add-node obs.json --id o2 --base 0.2
credencegraph relate obs.json H o1 --type supports --strength 0.75
credencegraph relate obs.json H o2 --type supports --strength 0.75
credencegraph query obs.json conditional H --given o1=true                # 0.6316
credencegraph query obs.json conditional H --given o1=true --given o2=true  # 0.8727
credencegraph query obs.json conditional H --given o1=true --given o2=false # 0.3
```

These are Bayes' theorem with a likelihood ratio of 4 per observation: 0.3 · 0.8
/ (0.3 · 0.8 + 0.7 · 0.2) = 0.6316, and with both observations 0.192 / 0.220 =
0.8727. One observation for and one against cancel, leaving the prior. The two
observations are independent given H, which is what two measurements of the same
quantity usually are.

The cost is in the diagnostics. They describe the graph before anything is
observed, so `diagnose obs.json --target H` reports P(H) = 0.3, sensitivities of
zero for both observations' parameters, and would compare a stated credence for
H with the prior 0.3 rather than with the posterior, because the observations
only move H once they are conditioned on. What it does report is the value of
information: learning `o1` would give 0.236 bits about `H`. An argument encoded
in the premise-to-claim direction, `o1 --supports--> H`, is diagnosed directly,
but reads differently: a negative observation then returns H to its base rather
than lowering it below the prior, and several observations combine as
independent reasons rather than as repeated measurements.

## How support and refutation combine

Reasons for a proposition combine as a noisy-OR and reasons against it as
independent vetoes: P(H) = (1 − (1 − b) ∏(1 − s)) · ∏(1 − f) over the active
supports s and refuters f, with base b. A strong refuter therefore wins against
an equally strong support:

```bash
credencegraph init veto.json
credencegraph add-node veto.json --id h --base 0.1
credencegraph add-node veto.json --id e --base 1
credencegraph add-node veto.json --id d --base 1
credencegraph relate veto.json e h --type supports --strength 0.9
credencegraph relate veto.json d h --type refutes --strength 0.9
credencegraph query veto.json marginal h                     # 0.091
credencegraph query veto.json intervene h --set d=false      # 0.91, support alone
credencegraph query veto.json intervene h --set e=false      # 0.01, refuter alone
```

The support alone takes h from 0.1 to 0.91 and the refuter alone to 0.01, and
together they give 0.91 · 0.1 = 0.091. A model that adds the two reasons as
weights of evidence in log-odds instead gives logit⁻¹(logit 0.1 + (logit 0.91 −
logit 0.1) + (logit 0.01 − logit 0.1)) = 0.479, close to where it started. Which
is right depends on what the refuter means: a flaw that invalidates the claim
whatever else holds is a veto, while a competing measurement pulling the other
way is closer to the log-odds reading. An encoding meets this whenever a claim
has substantial evidence on both sides.

## The paper

_To be added: the paper encoded by hand, its graph, the diagnostics on its
headline claim and what they say about its weak points._
