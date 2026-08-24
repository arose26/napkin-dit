# napkin-dit

**UNet vs DiT × ε-prediction vs flow matching — a 2×2 at matched parameters and matched NFE.**

SD 1.5 / 2.1 are UNet + ε-prediction. Flux and Ideogram are DiT + flow matching. The field
changed both variables at roughly the same moment, so every public comparison confounds them:

> At 2.81M parameters and equal network evaluations, **which of the two changes is actually
> doing the work — and does either transfer down to napkin scale?**

Four cells, five seeds each, one shared implementation. Everything inherits
[napkin-diffusion](https://github.com/arose26/napkin-diffusion) (t07): the cosine schedule,
the σ-space solvers, Karras spacing, the NFE accounting, and FMD against the pinned full
MNIST test set.

*Results pending — the sweep is running. Predictions below were registered before the first
model was trained; see [`Series 4 Design`](#pre-registered-predictions).*

## The thing that makes this comparison clean

Write both objectives in the σ parameterisation, where `x̃ = x₀ + σ·ε`:

```
DDPM:  x_t = √ᾱ·x̃,                σ = √((1−ᾱ)/ᾱ)
flow:  x_u = (1−u)x₀ + u·ε   →    x_u/(1−u) = x₀ + (u/(1−u))·ε
```

Same `x̃`, with `σ = u/(1−u)`. Both objectives induce **the identical probability-flow ODE**,
`dx̃/dσ = ε`. Asserted numerically at five noise levels: agreement to **3.0e-07** relative.

So flow matching and ε-prediction are not different generative processes. At a matched
sampler they differ only in **which noise levels training visits, how the loss weights them,
and whether the net emits ε or `v = ε − x₀`**. That has two consequences that shape the whole
repo:

1. **There is exactly one sampler in this file**, reached through a common `eps_hat(x̃, σ)`
   adapter. Two samplers would have confounded the objective axis with the implementer of
   its sampler.
2. **The canonical "flow matching sampler" is a step placement, not an objective.** Euler
   uniform-in-`u` is a specific σ spacing, and t07 already measured placement alone to be
   worth up to 2× at matched NFE. So spacing is swept across all four cells rather than
   riding along with the objective.

## The 2×2

|  | ε-prediction (DDPM, cosine VP) | flow matching (linear interpolant, v-target) |
|---|---|---|
| **UNet** 32/64/128ch, attn at 8px | `unet/eps` — the SD-1.5 corner | `unet/flow` |
| **DiT** patch 2, d=164, L=6, adaLN-zero | `dit/eps` | `dit/flow` — the Flux corner |

**Parameters: 2,813,057 (UNet) vs 2,813,496 (DiT) — 0.02% apart**, asserted in `selfcheck`
rather than eyeballed. Worth searching for: the neighbouring widths land at −4.0% (d=160) and
+4.2% (d=168), so "roughly matched" would have been a 4% capacity gift to one arm.

Held fixed and stated as design constants, not rungs: dataset (MNIST padded to 32×32),
parameter count, training steps, batch size, EMA decay, optimizer, warmup, the time-embedding
scale (both arms condition on `[0,1000]`; the flow arm passes `1000·u`), and the metric CNN.

### Matched params is *not* matched wall-clock

Measured, same machine, same batch, GPU-resident data: **UNet 14.7 steps/s, DiT 8.7 steps/s.**
The DiT costs **~1.7×** per step for the same capacity, because 256 tokens of full attention at
every layer is more FLOPs than a conv stack with attention only at 8px.

"Matched params" is a fair axis for a capacity question and an unfair one for a compute
question — at matched wall-clock the UNet would get 1.7× the training steps. This repo names
params + NFE as its axes and reports the step-time ratio next to the result instead of
pretending the two framings agree.

## Fair axis

**NFE (network evaluations), not sampler steps** — inherited from t07, where it was the whole
point. Heun costs 2 NFE per step and gets no free 2×; `selfcheck` asserts the accounting
(Heun at `nfe=20` spends 19). Grid: **2, 4, 8, 16, 32, 64**, denser at the bottom than t07's
because t07's crossover sat between 5 and 10.

Quality is **FMD** — Fréchet distance in the 64-d feature space of a small MNIST CNN, against
the **full** 10,000-image test set, never a prefix. (t07 lost an afternoon to that: the two
halves of the MNIST test set are measurably different populations, so slicing `[:n]` changes
the reference *distribution* along with its size.) **These numbers are comparable within this
repo only, never against published FIDs.**

Samplers swept: `euler` / `heun` × `karras` / `uniform-u` / `t` spacing. Karras + Heun is the
headline (all 5 seeds); the rest is a secondary tier at 3 seeds.

## Pre-registered predictions

Registered in `Series 4 Design - napkin-dit.md` before the first model was trained. Scored
honestly on arrival, prediction by prediction — wrong predictions in an instructive direction
are the most publishable thing here.

| # | Prediction | Result |
|---|---|---|
| **P1** | UNet wins on sample quality at every NFE ≥ 16 (conv locality beats learned mixing when data is small) | pending |
| **P2** | Flow matching wins at NFE ≤ 8 **regardless of backbone** — same sign in both rows | pending |
| **P3** | The two changes are **independent**: no interaction at any NFE | pending |
| **P4** | **Only one** of the two changes transfers down to napkin scale | pending |
| **P5** | Because the arms share an ODE, the objective effect **shrinks as NFE grows** and is ≤ the seed band at 64 NFE | pending |
| **P6** | Much of any low-NFE flow-matching win is **spacing, not objective**: `uniform-u` on the ε arms recovers ≥ half the gap | pending |
| **P7** | The DiT is worse or tied **everywhere** — 2.81M params over 256 tokens on 60k images is where attention has nothing to learn from | pending |

## Selfcheck — the test suite

```bash
python3 napkin_dit.py selfcheck    # ~4 min
python3 test_probe.py              # ~1 s, no GPU
```

| assertion | measured |
|---|---|
| DiT at patch=1 with attention off **is** a per-pixel MLP: perturbing pixel (i,j) changes output (i,j) and nothing else | total leak **exactly 0.0**, against 1.0e-02 with attention on |
| flow matching and DDPM agree on the marginal variance schedule | 3.0e-07 rel |
| flow `eps_hat` returns exact ε given a ground-truth v net | < 1e-4 rel |
| `ancestral == DDIM(η=1)` (t07, re-asserted since the sampler was rewritten) | 3.4e-06 rel |
| σ-space Euler `==` x-space DDIM(η=0) | 2.7e-04 rel |
| params matched | 0.02% |
| one fixed batch overfittable | see below |
| FMD(X,X) ≈ 0, and grows under a mean shift | ✓ |

The overfit check deviates from its registered spec and says so. The design asked for ~1e-15;
an MSE of 1e-15 on targets of order 1 means a per-element error of 3e-8, which is float32
epsilon — unreachable by any correct implementation. Measured floors instead, and the assert
is **relative to each cell's own step-0 loss**, because the flow arms start at ~1.9 (their
target `v = ε − x₀` has variance `1 + Var(x₀)`) against the ε arms' ~1.0. An absolute
threshold shared across the objective axis would have been biased against flow matching
before a single sample was drawn.

## Run it

```bash
./run.sh                    # every phase, resumable, writes .done.<phase> with its rc
```

`NPAR=4 ./run.sh` for a wide machine — but measure first, because the bottleneck moves: one
process saturates a 6GB laptop 4050 (NPAR=2 is *worse* than serial), while a 2-vCPU Colab T4
is launch-bound and wants 4-wide. Numbers are in `run.sh` next to the knob.

Individual phases:

```bash
python3 napkin_dit.py probe                                        # LR per backbone
python3 napkin_dit.py train --backbone dit --objective flow --seed 0
python3 napkin_dit.py sweep --tier headline
python3 napkin_dit.py agg   --tier headline                        # IQM + bootstrap CI + rank stability
python3 napkin_dit.py gif
```

One result file per (cell, seed, solver, spacing, NFE); **existence = done**, so any
interruption costs one run and any machine can pick up the remainder.

## Tuning, and the confound it would otherwise be

A naively-tuned transformer losing to a well-tuned UNet is the classic rigged 2×2 — and P1
predicts exactly that outcome, so the LR must not be the thing that produces it. `probe`
sweeps LR ∈ {1e-4, 2e-4, 5e-4} × {UNet, DiT} at ¼ length and picks **per backbone** (not per
cell, to keep the objective axis clean), ranking on a 200-step tail mean rather than one noisy
final minibatch. Chosen LRs are printed here as design constants once measured.

## Statistics

5 seeds per cell. **IQM with bootstrap CIs**, ties reported as ties. **Rank stability** —
P(best@k == best@5) over random seed subsets — at both ends of the NFE grid; if it saturates
below 1, the published object is a tied *set*, not a winner. All FMD at n=10,000 generated
samples against the full test set, no cross-n comparisons.

## What's deliberately not here

No CIFAR, no ImageNet, no latent space, no text conditioning, no classifier-free guidance, no
config system, no package layout. **No claim about DiTs at scale** — P7 exists precisely
because this measures the napkin end of the curve, and the honest headline for a DiT loss here
is a scale caveat, not "DiTs don't work".

See [INSIGHTS.md](INSIGHTS.md) for what actually broke, written as it happened.
