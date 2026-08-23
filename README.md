# napkin-dit

**UNet vs DiT × ε-prediction vs flow matching — a 2×2 at matched parameters and matched NFE.**

SD 1.5 / 2.1 are UNet + ε-prediction. Flux and Ideogram are DiT + flow matching. The field
replaced the backbone and replaced the objective at roughly the same moment, so every public
comparison moves both variables at once and none of them can tell you which change did the
work. This repo changes one at a time, at **2.81M parameters** and on a fair **NFE** axis.

|  | ε-prediction (DDPM, cosine VP) | flow matching (linear interpolant, v-target) |
|---|---|---|
| **UNet** 2,813,057 params | `unet/eps` — the SD-1.5 corner | `unet/flow` |
| **DiT** 2,813,496 params | `dit/eps` | `dit/flow` — the Flux corner |

4 cells × 5 seeds. Everything inherits [napkin-diffusion](https://github.com/arose26/napkin-diffusion)
(t07): the cosine schedule, the σ-space solvers, Karras spacing, the NFE accounting, FMD, and
the pinned full-test-set reference.

> **Status: sweep running.** The design and every prediction below were registered *before*
> the first model was trained — see
> [`Series 4 Design - napkin-dit.md`](https://github.com/arose26/napkin-dit) in the series
> notes and the "Registered predictions" section. Results land here when the sweep finishes;
> nothing in this table has been seen yet.

## The observation that makes this a clean experiment

Write both objectives in the σ parameterisation, where `x̃ = x₀ + σ·ε`:

```
DDPM:  x_t = √ᾱ · x̃                     σ = √((1−ᾱ)/ᾱ)
flow:  x_u = (1−u)x₀ + u·ε   →   x_u/(1−u) = x₀ + (u/(1−u))·ε
```

Same `x̃`. **Same probability-flow ODE**, `dx̃/dσ = ε`, up to a time reparameterisation
(`σ = u/(1−u)`) and a scaling. Selfcheck asserts this numerically — the two marginals agree
to **3.0e-07** relative.

So the headline consequence, stated before any result exists:

> At a matched sampler, "flow matching vs ε-prediction" is **entirely a training-time
> difference** — which noise levels get sampled, how the loss weights them, and whether the
> net emits ε or `v = ε − x₀`. It is not a different generative process.

That is why there is exactly **one sampler** in this file, reached through a common
`eps_hat(x̃, σ)` adapter. Two samplers would have confounded the objective axis with the
implementer of its sampler.

### …and why the "flow matching sampler" is a swept rung, not a passenger

The canonical FM sampler is Euler uniform-in-`u`, which in σ-space is a **step placement** —
and t07 already measured placement alone to be worth up to 2× at matched NFE. Leaving it
bundled with the objective would have handed flow matching a free variable. Spacing is
therefore swept across all four cells:

| spacing | what it is | whose native choice |
|---|---|---|
| `karras` | ρ=7 power law in σ | the fair common ground — **headline** |
| `u` | uniform in `u = σ/(1+σ)` | flow matching |
| `t` | uniform in the DDPM timestep index | DDPM / DDIM |

Solvers: `euler` (1 NFE/step) and `heun` (2 NFE/step, minus one on the last). NFE grid
**2, 4, 8, 16, 32, 64** — denser at the bottom than t07's, because t07's crossover sat
between 5 and 10.

## Registered predictions

Kole's pre-registered guess was: *at napkin scale the UNet wins on sample quality (conv
locality beats learned mixing when data is small), while flow matching wins at NFE ≤ 10
regardless of backbone — the two changes are independent, and only one transfers down.*
Split into scoreable claims, plus four I added at design time so the honest scoring covers
what the design implies as well as what was guessed:

- **P1** UNet beats DiT on FMD at every NFE ≥ 16.
- **P2** Flow matching wins at NFE ≤ 8 in **both** backbone rows.
- **P3** No interaction: the sign of each main effect is independent of the other axis.
- **P4** Exactly one of the two main effects favours the modern choice.
- **P5** The objective effect **shrinks with NFE** and is inside the seed band at NFE 64. If
  flow matching still wins at 64 NFE, this is wrong and the effect is training-side, not
  sampling-side.
- **P6** Much of any low-NFE flow-matching win is **spacing**: `u` spacing on the ε-pred arms
  recovers ≥ half the gap.
- **P7** (null-risk) The DiT is worse or tied *everywhere*, not just at high NFE. 2.81M
  params over 256 tokens on 60k images is where attention has nothing to learn from. If so
  the honest headline is a scale caveat, not "DiTs don't work".

## Selfcheck — the test suite

```bash
python3 napkin_dit.py selfcheck
```

| check | what it proves | measured |
|---|---|---|
| param match | the axis is capacity, not size | 2,813,057 vs 2,813,496 — **0.02%** |
| DiT degeneracy | `patch=1`, attention off ⇒ a **per-pixel MLP**: one input pixel influences its own output pixel and no other | total leak **exactly 0.0**, vs 1.0e-02 with attention on |
| schedule agreement | flow matching and DDPM share the marginal variance schedule | **3.0e-07** rel |
| adapter identity | the ε adapter is the raw net plus the documented scaling and nothing else | inside the device's own cuDNN nondeterminism floor |
| flow algebra | `eps_hat` recovers ε exactly from a ground-truth `v` net | < 1e-4 |
| t07 identities | `ancestral == DDIM(η=1)`; σ-space Euler `==` x-space DDIM(η=0); NFE accounting | 3.6e-06, 2.7e-04, exact |
| GPU dataset | the on-device dataset **is** the DataLoader's bytes; `batches()` is a drop-last epoch pass | exact equality, index coverage |
| fixed-batch overfit | forward process and target agree well enough to be driven to zero | ratio 2.3e-05 … 1.3e-04 of step-0 loss |

Two of these were registered by Kole and both taught something before any model trained —
the degeneracy check needed the *total* leak rather than the peak (one perturbed token out of
1024 dilutes below any sane per-pixel threshold), and the registered `~1e-15` overfit target
is arithmetically unreachable in float32. Both are written up in
[INSIGHTS.md](INSIGHTS.md).

## Run it

```bash
python3 napkin_dit.py selfcheck                     # ~10 min, no training needed
python3 napkin_dit.py probe                         # LR per backbone, 1/6 length
python3 napkin_dit.py train                         # 4 cells x 5 seeds
python3 napkin_dit.py sweep --tier headline         # heun x karras, 5 seeds
python3 napkin_dit.py agg   --tier headline         # IQM + bootstrap CI + rank stability
python3 napkin_dit.py gif                           # 2x2 contact sheet
```

`./run.sh` drives all of it as resumable phases. Every sweep point is one file under
`out/res/` and every checkpoint one file under `out/ckpt/` — **existence means done**, so a
crashed session, a reaped VM or a machine switch costs at most the runs in flight.

## Design constants — the things every arm inherits

Stated rather than buried, because any line all four cells share is silently part of the
experimental design:

- **Time conditioning is always on the [0, 1000] scale.** The ε arm passes the integer
  timestep; the flow arm passes `1000·u`. Identical embedding module for both, so time
  conditioning is not one of the things the objective axis changes.
- **Both arms start sampling at σ_max ≈ 91.7** (`T_START`, the largest `t` whose `ᾱ` is still
  ≥ 1e-4), so the initial noise level and therefore the NFE axis are matched.
- **The x₀ clip is on** during sampling for both arms — load-bearing, not cosmetic, at these
  step sizes. t07 has the details.
- **One LR per backbone**, chosen by the `probe` command, not per cell — a naively-tuned
  transformer losing to a tuned UNet is the classic rigged 2×2, and P1 predicts exactly that
  outcome, so the LR must not be what produces it. 500-step linear warmup for every arm.
- **Matched params is not matched wall-clock.** The DiT costs ~1.75× per step for the same
  capacity (256 tokens of full attention at every layer vs a conv stack with attention only
  at 8px). Reported next to the result rather than pretending the two framings agree.

## What's deliberately not here

No CIFAR, no ImageNet, no latent space, no text conditioning, no classifier-free guidance, no
config system, no package layout. **No claim about DiTs at scale** — P7 exists precisely
because this measures the napkin end of the curve, and says so. FMD is a Fréchet distance in
the 64-d feature space of a small MNIST CNN: comparable **within this repo only**, never
against published FIDs.
