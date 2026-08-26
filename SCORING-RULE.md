# Pre-committed scoring rule for P2 and P6

Written 2026-08-26 while `secondary2` was at 77/144 points — i.e. **before the data that
scores these predictions existed**. Committed so the comparison cannot be chosen after seeing
the numbers. Three times in this project I read a result the way I expected it to come out
(the LR boundary, the "worst cell" error, the four-variable spacing confound), so the rule
gets fixed first.

## The comparison

Solver **Heun** throughout, matching the headline tier. Seeds **{0, 1, 2}** — the intersection
available across all three spacings. Aggregation **IQM with bootstrap CI**, as registered.
Spacings compared: `karras`, `u`, `t`. NFE = actual spend.

"Low NFE" for both predictions means the **NFE ≤ 8 points**, which for Heun are the runs that
actually spend **3 and 7** network evaluations.

## P2 — "Flow matching wins at NFE ≤ 8 regardless of backbone"

**Primary score: at common Karras spacing**, because the design doc names Heun+Karras as the
headline and treats spacing as a separately swept rung. P2 is a claim about the objective, so
it is scored with the sampler held fixed.

- RIGHT if flow beats ε in **both** backbones at every NFE ≤ 8 point.
- WRONG if ε beats flow in both.
- MIXED if the backbones disagree — reported as mixed, not rounded to either.

CI overlap at a given NFE is reported as a **tie** at that point, never silently resolved.

**Secondary reading, reported alongside and never instead**: each objective at its own best
spacing among the three. This answers "deploy each method as its practitioners would" rather
than "fix the sampler and swap the objective". If the two readings disagree, **that disagreement
is the finding** and both get published.

## P6 — "Much of any low-NFE flow win is spacing, not objective: uniform-u on the ε arms recovers ≥ half the gap"

P6 has an explicit precondition: a low-NFE flow win must exist at common spacing.

1. If flow does **not** win at common Karras at NFE ≤ 8 → **premise void**. Score it that way
   and report the measured spacing effect separately rather than reinterpreting P6 into a
   prediction it did not make.
2. If flow **does** win, let `gap = FMD(ε@karras) − FMD(flow@karras)` at that NFE. P6 is RIGHT
   if `FMD(ε@u)` closes **≥ half** of `gap`, i.e. `FMD(ε@karras) − FMD(ε@u) ≥ 0.5 · gap`.
   Evaluated per backbone; both must hold for an unqualified RIGHT.

## Seed handling

Scored on **all three seeds**, no exclusions — these are objective and sampler questions, and
dropping seeds is a deviation that needs its own justification.

`dit/eps` seeds 2 and 4 carry the saturation collapse, and with n=3 the IQM trims nothing, so
`dit/eps` numbers in this tier are dominated by seed 2. That is stated as a **limitation of the
`dit/eps` cell**, and a seed-excluded variant may be reported as an explicitly-flagged footnote.
It does not replace the primary score.

## What would make me distrust the result

- Any spacing showing an identical value across all four cells at a given NFE (as happens at
  NFE 2, where a 2-point schedule has the same endpoints for every spacing) — that is a
  degenerate point, not evidence, and is excluded from scoring.
- `secondary2` finishing with fewer than 144 points, which would mean a partial tier.
