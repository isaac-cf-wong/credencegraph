---
title: Reading the diagnostics
description:
    What the diagnostics measure, why equal credences rank claims by counting
    premises, how to read an overclaim, how to encode a null result, and what
    the structure of an argument shows on its own.
---

Every finding that involves a probability is conditional on the credences the
graph supplies. The diagnostics measure the encoding: what the premises deliver
if they hold as often as their bases say. They do not know more about the
premises than the encoding does. This page works through what that means on
small graphs whose answers can be checked by hand, using the
[command line](cli.md) as the [worked example](worked-example.md) does. Each
`diagnose` run below also lists every premise as `unanchored`, since none has a
`--source`; those lines are left out.

## Equal credences rank claims by counting premises

Two claims, each stated at 0.9 by their source, rest on premises that are all
given the same default credence, Beta(8, 2), with mean 0.8. `A` requires two of
them, `B` three:

```bash
credencegraph init rank.json
credencegraph add-node rank.json --id a1 --base beta:8,2
credencegraph add-node rank.json --id a2 --base beta:8,2
credencegraph add-node rank.json --id b1 --base beta:8,2
credencegraph add-node rank.json --id b2 --base beta:8,2
credencegraph add-node rank.json --id b3 --base beta:8,2
credencegraph add-node rank.json --id A --base 1 --stated 0.9
credencegraph add-node rank.json --id B --base 1 --stated 0.9
credencegraph relate rank.json a1 A --type requires --strength 1
credencegraph relate rank.json a2 A --type requires --strength 1
credencegraph relate rank.json b1 B --type requires --strength 1
credencegraph relate rank.json b2 B --type requires --strength 1
credencegraph relate rank.json b3 B --type requires --strength 1
credencegraph query rank.json marginal A   # 0.64
credencegraph query rank.json marginal B   # 0.512
```

These are 0.8² and 0.8³. When every premise has the same credence, the only
thing that differs between two claims is the structure, so the ranking is a
count of premises: a claim that requires k of them gets 0.8ᵏ, and the one with
fewer comes first. A ranking made this way can agree with an expert's judgement
of the claims, but so does counting, and the agreement says nothing about the
premises.

The information in a ranking comes from credences that differ. Suppose a reader
doubts `A`'s two premises, at 0.6, and trusts `B`'s three, at 0.97. The same
graph with those bases, `informed.json`, gives P(A) = 0.6² = 0.36 and P(B) =
0.97³ = 0.912673, and the ranking reverses.

## An overclaim flag is not a verdict

`diagnose` compares each `stated` credence with the probability the premises
deliver, in log-odds, and reports a gap above 2 ln(11/9) ≈ 0.401 as an overclaim
or an underclaim. Under the default credences above both claims are flagged:

```bash
credencegraph diagnose rank.json
# overclaim:A: node 'A' is stated at 0.9 but its premises give 0.64: ...
# overclaim:B: node 'B' is stated at 0.9 but its premises give 0.512: ...
```

That is the usual outcome, not a finding about the source. With premises at 0.8,
a claim that requires two of them, stated at 0.8, is already 0.81 apart in
log-odds, twice the threshold, and more premises or a higher stated credence
only widen the gap. Under default credences, almost every claim that rests on
more than one premise is flagged, including well-supported ones: `B`, which the
informed credences above put at 0.913, is flagged here. The flag alone says
little.

The informative part is the size of the gap, which the finding carries. Its
`value` is the gap in probability, stated minus computed: 0.26 for `A` and 0.388
for `B`. Its `details` hold `stated` and `computed`, from which the log-odds gap
the threshold is applied to follows, here 1.62 for `A` and 2.15 for `B`, and
`threshold_log_odds`, the threshold that was applied. Under equal credences and
equal stated credences these sizes too are ordered by the number of premises.
Under the informed credences only `A` is flagged, with a log-odds gap of 2.77,
while `B`'s stated 0.9 lies 0.15 below its computed 0.913 in log-odds and is not
reported. Read an overclaim as "the text asserts this much more than its
premises deliver _at these credences_", compare the sizes across claims, and
check whether the gap survives credences you would defend.

## Encoding a null result

`requires` is a noisy AND: every required premise that fails lowers the claim.
That is right for a claim that needs each premise, and wrong for a claim some of
whose premises, if they fail, make it more robust. A null result is the common
case. "The search found no signal" needs the search to have been able to see a
signal at the level claimed, but a detection threshold that is too lax, too
eager to report candidates, does not weaken a non-detection: nothing passed even
a permissive cut.

Encoding both as `requires` lets the lax threshold pull the claim down:

```bash
credencegraph init null-requires.json
credencegraph add-node null-requires.json --id sensitive --base 0.9
credencegraph add-node null-requires.json --id strict --base 0.6
credencegraph add-node null-requires.json --id null --base 0.95
credencegraph relate null-requires.json sensitive null --type requires --strength 1
credencegraph relate null-requires.json strict null --type requires --strength 1
credencegraph query null-requires.json marginal null                         # 0.513
credencegraph query null-requires.json intervene null --set strict=false     # 0
credencegraph diagnose null-requires.json --target null
```

Here `sensitive` is "the search would have found a signal at the claimed level"
and `strict` is "the detection threshold is strict enough to keep false alarms
out". The claim gets 0.95 · 0.9 · 0.6 = 0.513, and a lax threshold sets it to 0.
The diagnostics then point at the wrong thing: `strict` is a single point of
failure and tops the value of information, 0.641 bits against 0.112 for
`sensitive`. Strictness matters to a claimed detection, where false alarms are
the risk; for a non-detection, it is not a condition of the claim.

The null needs only what would have let it see the signal, and that is what it
should require:

```bash
credencegraph init null.json
credencegraph add-node null.json --id sensitive --base 0.9
credencegraph add-node null.json --id null --base 0.95
credencegraph relate null.json sensitive null --type requires --strength 1
credencegraph query null.json marginal null                          # 0.855
credencegraph add-node null.json --id lax --base 0.6
credencegraph relate null.json lax null --type supports --strength 0.5
credencegraph query null.json marginal null                          # 0.8685
credencegraph query null.json intervene null --set lax=false         # 0.855
```

When a premise would make the null more convincing, such as `lax`, "the
threshold was low enough that marginal candidates would have passed", it is a
reason for the claim and enters as `supports`: if it holds, the claim rises, to
0.9 · (1 − 0.05 · 0.5) = 0.8775; if it fails, the claim keeps 0.95 · 0.9 =
0.855, what it had without it. The 0.8685 the marginal query reports is the
average of the two over `lax`'s own credence, 0.6 · 0.8775 + 0.4 · 0.855. A
premise that does not bear on the null either way is best left out.

The test for each premise of a null or upper-limit claim is which way its
failure moves the claim. A failure that could hide a real signal, such as a
search less sensitive than stated, a noise level underestimated or a part of the
parameter space not covered, makes an upper limit too tight: that premise is
required. A failure that only makes the limit looser than it need be, such as a
conservative noise model, leaves "the quantity is below L" true, and does not
belong under `requires`.

## What the structure shows on its own

Some findings do not depend on the credences being right, because they come from
the shape of the argument. Here two results share a calibration, `cal`, and each
also needs a measurement of its own; both results support `claim` and `claim-b`,
and the source itself states a caveat that tells against `claim-b`. Every
premise has the same default credence:

```bash
credencegraph init map.json
credencegraph add-node map.json --id cal --base beta:8,2
credencegraph add-node map.json --id m1 --base beta:8,2
credencegraph add-node map.json --id m2 --base beta:8,2
credencegraph add-node map.json --id caveat --base beta:8,2
credencegraph add-node map.json --id r1 --base 1
credencegraph add-node map.json --id r2 --base 1
credencegraph add-node map.json --id claim --base 0.1
credencegraph add-node map.json --id claim-b --base 0.1
credencegraph relate map.json cal r1 --type requires --strength 1
credencegraph relate map.json m1 r1 --type requires --strength 1
credencegraph relate map.json cal r2 --type requires --strength 1
credencegraph relate map.json m2 r2 --type requires --strength 1
credencegraph relate map.json r1 claim --type supports --strength 0.9
credencegraph relate map.json r2 claim --type supports --strength 0.9
credencegraph relate map.json r1 claim-b --type supports --strength 0.9
credencegraph relate map.json r2 claim-b --type supports --strength 0.9
credencegraph relate map.json caveat claim-b --type refutes --strength 0.5
credencegraph query map.json marginal claim     # 0.763552
credencegraph query map.json marginal claim-b   # 0.458131
credencegraph diagnose map.json --target claim
```

- **A source's own contrary statements, entered as `refutes`, move a claim
  down.** `claim-b` has the same support as `claim` and loses to the caveat,
  0.763552 · (1 − 0.8 · 0.5) = 0.458131. The refuter's strength sets how far it
  falls; that it falls, below a claim with the same support, comes from the
  text.
- **A premise shared by several paths tops crux and value of information.** If
  `cal` fails, both results fail together, so P(claim) moves by 0.829 per unit
  of its base against 0.181 for `m1` or `m2`. Its crux is 0.1 against 0.0219,
  and learning it would give 0.401 bits about the claim against 0.0195. This
  holds at equal credences, because it is a fact about which paths share what.

This is the use the engine supports best from structure alone: mapping an
argument and finding what to check first. Grading the claims needs credences
that someone is prepared to defend, and every grade the diagnostics give is
conditional on them.
