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
2. **The source's own credence as `--stated`**, when it says how probable it
   takes the claim to be, so that `diagnose` can compare that with what the
   encoded argument delivers. A significance or a confidence level is not such a
   credence: it is a probability of the data under a model, not of the claim.
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

## The paper: the OPERA neutrino velocity

In September 2011 the OPERA collaboration reported that muon neutrinos sent from
CERN to the Gran Sasso laboratory, 730 km away, arrived about 60 ns earlier than
light would have ([arXiv:1109.4897v1](https://arxiv.org/abs/1109.4897v1)): a 6.0
sigma effect, (v − c)/c ≈ 2.5 × 10⁻⁵. The paper itself claims the measurement,
not new physics, and asks for independent checks. In 2012 the collaboration
traced the effect to its timing chain, a badly connected optical fibre among
others, and the corrected result was consistent with the speed of light. The
paper suits a worked example because its argument is short, the evidence against
the headline reading was strong and public at the time, and the weak point is
now known.

### The encoding

The graph has eight propositions. The measurement, `early`, requires three
premises: a correctly calibrated `timing` chain, a known `baseline` and an
unbiased statistical `extraction` of the delay. A rerun with short proton
bunches, `bunched`, supports the extraction, because it times each event
directly instead of fitting a waveform; it says nothing about the timing chain,
which it shares with the main measurement, so it is not a reason for `timing`.
The claim that the neutrinos are faster than light, `faster`, has a base of
0.001, its credence without the measurement; the measurement supports it, and
two published arguments refute it: the SN1987A neutrinos, which bound the speed
at much lower energy, and the energy that superluminal neutrinos would lose by
emitting electron–positron pairs
([arXiv:1109.6562](https://arxiv.org/abs/1109.6562)). The 6.0 sigma is a
significance: how improbable so large a delay would be if the true delay were
zero and the paper's error model held. It is not a probability that the early
arrival is real, and the paper states no such probability, so `early` has no
`stated` credence.

Every base and strength is one reader's judgement as of late 2011, not a number
from the paper; the Beta credences mark the ones that reader was least sure of.
The script builds the graph, queries the claim and diagnoses it:

```bash
--8<-- "docs/examples/opera.sh"
```

### What the diagnostics say

`P(faster) = 0.0716`. It factors into the measurement, P(early) = 0.876, the
support it gives, 1 − 0.999 · (1 − 0.95 · 0.876) = 0.833 without the refuters,
and the two refuters, which leave 1 − 0.99 · 0.7 = 0.307 and 1 − 0.9 · 0.8 =
0.28 of it: 0.833 · 0.307 · 0.28 = 0.0716.

- **The measurement is only as good as its premises.** P(early) = 0.876. The
  significance is computed within an error model whose systematic budget takes
  the timing chain, the baseline and the extraction as right; the graph keeps
  the chance that one of them is wrong, and the timing chain is the largest of
  those, as it turned out to be.
- **The crux is the refuters.** The two refuter strengths lead the ranking, with
  cruxes of 0.032 and 0.028, followed by `timing` at 0.0068. The claim depends
  most on how far the low-energy SN1987A bound and the pair-emission argument
  carry over to 17 GeV neutrinos, and that is where the reader was least sure.
- **Three premises are single points of failure**: the measurement, the baseline
  and the timing chain. The failure threshold is a fraction of the claim's own
  probability, so the default of 0.1 asks which premises would take it down by
  an order of magnitude. If the timing chain is wrong, the claim falls to 0.004,
  0.056 of what it was. A wrong extraction leaves 0.103 of it, just above the
  line, and a failed bunched rerun 0.93.
- **The bunched-beam rerun is worth almost nothing for the claim**: 1.5 × 10⁻⁵
  bits, against 0.009 bits for the timing chain. It checks the extraction, which
  was not in doubt, and cannot check the timing, which was.

### What the encoding shows about the model

The paper's evidence arrives in the premise-to-claim direction. The one
observation, the bunched rerun, is a reason for one premise, so the
likelihood-direction encoding of the previous sections was not needed.

The combination rule decides the answer. Under the veto, the refuters hold the
claim at 0.082 even if the measurement is taken as certainly right. Adding the
same reasons as weights of evidence in log-odds, each weight read from what the
reason does alone, gives P(faster) = 0.497 instead. A refuter alone takes the
claim from 0.001 to 0.0003 or 0.0002, a weight of only −1.2 and −1.6, while the
measurement alone takes it to 0.95, a weight of +9.9, so the measurement wins.
The veto gives the answer most physicists gave at the time; read this way,
log-odds does not, and its weights would have to be elicited some other way.
