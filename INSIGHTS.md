# INSIGHTS

Written as it happened, not reconstructed afterwards. The entries that cost debugging time
are worth more than the results table (Research Metastrategy #20).

---

## Build phase (2026-08-23)

### The parameter match is exact, not fudged

`DiT(d=164, depth=6, heads=4, patch=2)` = **2,813,496** params against the UNet's
**2,813,057** — 0.02% apart, and the selfcheck asserts `<2%` rather than trusting a
comment. Worth searching for: the neighbouring widths land at −4.0% (d=160) and +4.2%
(d=168), so "roughly matched" would have been a 4% capacity gift to one arm. Head dim is
41, which is ugly and completely fine — SDPA does not care.

Depth/width trades at fixed params, for the record: d=224/L=3 is −12%, d=136/L=9 is +8%,
d=120/L=12 is +17%. The 6-block shape is the one that lands on the UNet.

### Matched params is NOT matched wall-clock, and that is a real caveat

Measured on the laptop 4050 (shared with series 3, so absolute numbers are inflated but the
ratio is not): **UNet 7.0 steps/s, DiT 4.0 steps/s** at identical batch and param count.
The DiT costs **1.75x** per step for the same capacity, because 256 tokens of full
attention at every layer is more FLOPs than a conv stack with attention only at 8px.

This matters for how the headline is allowed to be phrased. "Matched params" is a fair axis
for a capacity question and an *unfair* one for a compute question — at matched wall-clock
the UNet would get 1.75x the training steps. The repo names params + NFE as its axes and
reports the step-time ratio next to the result rather than pretending the two framings
agree (Metastrategy #2).

### A max-based assert called a healthy net broken

The registered selfcheck was: a DiT at patch=1 with attention disabled must reduce to a
per-pixel MLP. First implementation perturbed one input pixel and asserted that the
**max** change at any *other* output pixel was <1e-5 with attention off and >1e-4 with
attention on. The off case passed. The on case **failed** — "attention ON but the DiT is
still per-pixel".

Attention was fine. With patch=1 there are 1024 tokens, so perturbing one of them shifts
each *other* token by roughly 1/1024 of the signal: measured per-pixel max leak 4.3e-6
against an on-pixel response of 1.4e-1. The max is a useless discriminator at that
dilution. The **sum** is not: 2.5e-3 with attention on versus **exactly 0.0** with it off —
bit-exact isolation, not merely small.

The fix was to re-derive what the property actually is (total influence, not peak
influence), not to loosen the tolerance. Metastrategy #7, and it cost about ten minutes
only because the assert was written before any model was trained.

### The registered 1e-15 overfit target is arithmetically impossible

Kole's selfcheck spec asked for one fixed batch overfittable to ~1e-15. An MSE of 1e-15 on
targets of order 1 means a per-element error of ~3e-8, which is float32 epsilon — the
number cannot be reached in this precision by any correct implementation.

Measured floors instead, 1500 steps at lr 2e-3 with cosine decay, fixed (t, eps) draw:

| cell | final | step-0 | ratio |
|---|---|---|---|
| unet/eps  | 2.1e-05 | 1.12 | 1.8e-05 |
| unet/flow | 1.1e-04 | 1.88 | 5.6e-05 |
| dit/eps   | 1.2e-04 | 1.00 | 1.2e-04 |
| dit/flow  | 4.9e-05 | 1.94 | 2.5e-05 |

Two real effects sit in that spread, and neither is a bug:

1. The **flow arms start at ~1.9 and the eps arms at ~1.0**, because the flow target is
   `v = eps - x0` whose variance is `1 + Var(x0)` while the eps target has variance 1. Any
   absolute threshold shared across the objective axis is therefore biased against flow
   matching before a single sample is drawn. The assert is now **relative** to each cell's
   own step-0 loss, which is the scale-free version of the claim under test.
2. The **DiT memorises a fixed batch ~50-100x slower than the UNet** at equal params
   (unet/eps 1.4e-6 at 1500 steps in an earlier longer run vs dit/eps 1.0e-4). Registered
   as evidence for P7 before the real training started: whatever the sweep says, the DiT
   is the slower learner here.

Deviation from the registered spec is reported rather than quietly relaxed.

### Both objectives share one ODE — so the sampler must be shared code

`x_u/(1-u) = x0 + (u/(1-u))*eps` is the same `x~ = x0 + sigma*eps` that DDPM's
`x_t/sqrt(alpha_bar)` gives, with `sigma = u/(1-u)`. Asserted numerically at five noise
levels: agreement to **3.0e-07** relative. So flow matching and eps-prediction are not
different generative processes at all; at a matched sampler they differ only in which
noise levels training visits, how the loss weights them, and whether the net emits eps
or v.

Consequence for the design: there is exactly ONE sampler in this file, reached through a
common `eps_hat(x~, sigma)` adapter. Writing two samplers would have confounded the
objective axis with the implementer of its sampler (Metastrategy #4), and it would have
hidden the fact that the canonical "flow matching sampler" (Euler uniform-in-u) is a step
*placement* — which t07 already measured to be worth up to 2x on its own. Placement is
therefore its own swept rung across all four cells, not a passenger on the objective.

The inherited t07 identities still hold in the rewritten harness: `ancestral == DDIM(eta=1)`
to 3.4e-06, sigma-space Euler `==` x-space DDIM(eta=0) to 2.7e-04.

### The DataLoader was the wrong tool for a 246MB dataset

Padded MNIST is 60000x1x32x32 float32 = **246MB**. It was sitting behind a
`DataLoader(num_workers=2)` purely because that is what the t07 harness inherited from every
PyTorch tutorial. The venue made the cost visible: **Colab free tier gives 2 vCPU**, and the
sweep runs several training processes concurrently on one GPU, so N processes x 2 worker
processes contend for two cores while the GPU waits.

The dataset now lives on the device and batches come from `torch.randperm` on-GPU — same
sampling law as `DataLoader(shuffle=True, drop_last=True)`, asserted in selfcheck against
the DataLoader's own bytes and against index coverage over one epoch. Metastrategy #15 says
match the machine to the bottleneck; the corollary is that the bottleneck of a napkin-scale
experiment is rarely the arithmetic.

### Moving the data on-device was worth 2.2x; the "obvious" next optimisation was worth 0

Measured on the laptop 4050, same machine, same batch, warmup discarded:

| | DataLoader | GPU-resident data |
|---|---|---|
| UNet | 7.0 steps/s | **15.4 steps/s** |
| DiT | 4.0 steps/s | **8.7 steps/s** |

So the tutorial-inherited DataLoader was costing **more than half the throughput** of a
napkin-scale run. That is the whole "match the machine to the bottleneck" lesson in one row.

Which made the next hypothesis look obvious. The EMA update is a Python loop over
`state_dict()`, two in-place ops per tensor — **180 tensors for the UNet**, so ~360 extra
kernel launches per step. On the T4 the 2 vCPUs were measurably pinned at the launch ceiling
(4 concurrent workers at ~43% of a core each = 2 cores fully consumed), so fusing the loop
with `torch._foreach_mul_`/`_foreach_add_` should have been a large, three-line win.

On the **4050** it was worth **1.06x** (UNet) and **1.02x** (DiT), and the ceiling on any EMA
optimisation whatsoever — measured by deleting the update entirely — is 1.07x / 1.02x.
Forward and backward dominate there; the launch count is a red herring.

**Scope that claim honestly, because the first draft of this entry did not.** It said the
hypothesis was refuted, full stop. It was refuted *on the 4050* — a machine with more cores
and a much faster GPU, i.e. a completely different CPU-to-GPU ratio from a 2-vCPU T4. Launch
overhead is by definition a larger share of the step on the slower-CPU box, so the
launch-bound argument predicts the fusion matters **more** on the T4, not less, and the 4050
number cannot rule that out. This is the same cross-hardware error as the tolerance entry
below, made by the same person two hours later, in a file that already warned about it.

So: the fusion is not written, on the narrower ground that nothing measured shows it helping
and unmeasured code that moves no number is not neutral (Metastrategy #18) — a fused EMA
would imply to the next reader that EMA cost was load-bearing. The T4 case is **open**, and
settling it costs a fifth process on two cores, which would perturb the run it is trying to
speed up. Deliberately left unresolved rather than answered with the wrong machine's number.

### The LR probe picked the edge of its own grid, which is not a pick

The probe ran 58 min and returned `{"unet": 5e-4, "dit": 5e-4}` — both at the **maximum** of
the grid `{1e-4, 2e-4, 5e-4}`. A boundary solution means the optimum was never bracketed.

The trend is what makes it dangerous rather than merely untidy. Tail-mean loss at 1/4 length,
improvement per grid step:

| cell | 1e-4 | 2e-4 | 5e-4 | last step |
|---|---|---|---|---|
| unet/eps | 0.0381 | 0.0346 | 0.0323 | −6.6% |
| unet/flow | 0.1512 | 0.1447 | 0.1400 | −3.2% |
| dit/eps | 0.0645 | 0.0634 | 0.0421 | **−33.6%** |
| dit/flow | 0.2371 | 0.1838 | 0.1625 | **−11.6%** |

The **UNet had flattened and the DiT had not.** So the DiT's optimum lay outside the grid
while the UNet's sat near its own — training both at 5e-4 would have under-tuned one arm and
not the other. That is precisely the rigged 2×2 the probe exists to prevent, and it would have
produced **the pre-registered result** (P1: "UNet wins at napkin scale") as a tuning artifact.
A confirmed prediction is the worst possible place for this bug to hide, because nothing about
the output would have looked wrong.

Caught with 0/20 checkpoints written, at a cost of ~2 minutes. Three changes, not one:

- the grid extends to `{1e-4, 2e-4, 5e-4, 1e-3, 2e-3}` so it can bracket;
- `cmd_probe` **refuses to write `lr.json`** when any backbone picks the grid maximum, unless
  `--allow-boundary-lr` says so deliberately — a silent boundary pick is now impossible;
- non-finite losses map to `+inf` before ranking, because 2e-3 may well diverge and `nan`
  sorts unpredictably, which could have let a *diverged* run win the ranking outright.

The general lesson, and it is not about learning rates: **a hyperparameter search that returns
an endpoint has not selected anything, it has told you the grid was wrong.** Assert that, don't
read past it. The first version printed the winner and moved on.

### The registered story was the mirror image of the measured one

Three of seven pre-registered predictions came back decisively wrong, and the fourth was right
for a reason the data refutes. Registered: *the UNet wins on quality, flow matching wins at low
NFE.* Measured: **the DiT wins at low NFE, flow matching wins at high NFE.** Both axes
inverted.

The DiT's low-NFE win is 4.6× (ε) and 4.9× (flow) at 7 NFE, and the mechanism was visible in
one contact sheet: **the UNet leaves background speckle at 7 NFE and the DiT does not** — 82%
of DiT pixels pinned at ±1 against the UNet's 58%. This is the payoff for the one-sampler
design. Both arms run identical solver code through the same `eps_hat(x̃, σ)` adapter, so the
difference cannot be a sampler difference, and I did not have to argue that — it is structural.
Had I written two samplers, this result would have been uninterpretable.

The registered rationale for P1 was "conv locality beats learned mixing when data is small".
Getting the *answer* roughly right at high NFE while the *mechanism* is refuted at low NFE is
the outcome that would have been easiest to miss if I had scored predictions as a single
right/wrong bit rather than checking the reason.

### I nearly published a confound-driven correction to my own confound-driven result

The spacing tier landed and appeared to invert the low-NFE headline: under Euler with each
objective on its preferred spacing, flow matching beats ε-prediction at 8 NFE in both backbones,
where the published Heun+Karras headline said the opposite. I drafted a revision announcing that
my own P2 score was wrong and that spacing had been masking a flow-matching win.

Second-opinion review killed it. Between the published headline and the spacing tier, **four
things differ**: solver (Heun → Euler), NFE (7 → 8), aggregation (IQM over 5 seeds → median over
3), and seed set (I dropped `dit/eps`'s two collapsed seeds). Attributing the inversion to
spacing alone is precisely the one-variable-at-a-time failure this entire repo exists to attack —
committed by me, in a correction to a result I had published an hour earlier, in a project whose
stated purpose is separating two changes the field moved together.

What survives is narrower and still worth having. **Within the Euler tier** — matched solver,
NFE, seeds and aggregation — ε-prediction prefers Karras spacing and flow matching prefers
uniform-in-`u`, consistently in both backbones at every NFE, and the objective ranking inverts
between the two choices. That licenses "the low-NFE objective ranking is not identifiable without
naming the spacing." It does **not** license "the Heun headline was a spacing artifact."

The arm that would settle it is Heun × three spacings at matched seeds — which is the
`secondary2` tier, and it was already running while I was drafting the claim it would have
tested. The right move was to wait forty minutes.

Two lessons, and the second is the uncomfortable one. Sweeping the spacing across all four cells
was correct and load-bearing: had it ridden along with the objective, the headline would have
silently conflated "flow matching" with "flow matching's native sampler" and I would never have
known. And: **knowing the failure mode by name confers no immunity.** I had written the warning
about inherited design constants into this file days earlier.

### I published a claim my own confidence intervals refuted, in the same table

The headline README said "at 7 NFE, ε-prediction beats flow matching in **both** backbones
(47.66 vs 57.01 for the DiT; 219.41 vs 280.09 for the UNet)", and scored P2 as *wrong and
backwards* on that basis.

The CIs were printed in the same table, two columns over: `unet/eps` [166.32, 273.40] against
`unet/flow` [193.80, 357.53], and `dit/eps` [33.30, 86.16] against `dit/flow` [45.38, 78.39].
Both overlap. Both were ties. I had computed the intervals, printed them, and then read the
point estimates.

This is the same failure as the earlier "worst cell" error — an assertion about a table sitting
in front of me — and it is exactly what the repo's own methodology section exists to prevent.
Reporting ties as ties is the *stated reason* for running 5 seeds and bootstrapping in the first
place; having the machinery and then not consulting it is worse than not having it, because the
intervals lend the wrong claim an air of rigour.

The correction: P2 is **MIXED**, both backbones tie at the only low-NFE point the scoring rule
admits. The DiT's *backbone* advantage at the same NFE survives the same check with room to
spare — neither column's intervals come close to overlapping — so the headline finding stands
and only the objective half of it was wrong.

### A confirmed prediction is the most dangerous place for a bug

Worth stating on its own, because it nearly happened twice.

The LR-grid boundary bug would have produced P1 ("UNet wins") as a **tuning artifact** — the DiT
under-tuned, the UNet not. The output would have matched the registration exactly, and nothing
about it would have looked wrong. I only found it because the probe printed its own grid and the
argmin sat at the edge.

Then the second-opinion review of my *scoring* caught two factual errors in the writeup: I wrote
that DiT+flow was "the worst of the four cells at high NFE" when the table plainly showed it
second-best, and I claimed clean sign-independence for P3 when the backbone effect flips once at
NFE 15. Both were assertions about a table that was directly in front of me.

The pattern in both: **the error was invisible precisely because the conclusion was the one I
expected.** Verification has to be cheapest where confidence is highest, which is the opposite
of where it naturally goes.

### `dit/eps` fails by desaturating, and the loss cannot see it

Per-seed FMD at 63 NFE for `dit/eps`: `1.69, 1.72, 129.80, 1.64, 13.82`. Two of five seeds
degraded. The three healthy ones land at ~1.65 and **beat `unet/eps`'s 3.71–5.12**, so what
reads as a UNet win in the ε column is a *reliability* gap, not a capability gap — and IQM over
5 seeds reports the reliability, which is the correct thing for it to report.

The failure mode is a **saturation collapse**: correct digit shapes with compressed dynamic
range, 5.8% of pixels pinned at ±1 against a healthy seed's 41%. It is obvious in a contact
sheet and invisible in the loss — the collapsed seed had the **lowest final training loss of its
entire cell** (0.0284 against a cell mean of 0.0310).

That is the cleanest demonstration in this project of why the artifact gets eyeballed before the
aggregate is believed. A ranking built on training loss would have called seed 2 the *best* of
the five.

Open, and **left open deliberately**: whether this is a property of DiT+ε or of DiT+ε *at the
probe-selected 2e-3*. One follow-up arm settles it — retrain seeds 2 and 4 at 5e-4 and see
whether the collapse disappears, about an hour of compute. It was registered here before being
run, so the answer could not be reinterpreted after the fact; then it was **cut for budget**.

The honest consequence is that the README's claim had to be weakened to match what was actually
measured — "seed-unstable at 2e-3" rather than "seed-unstable" — because the stronger reading is
the one the missing control would have tested. Cutting an experiment is allowed; letting the
claim keep the scope the experiment would have earned is not.

### Four defects lined up so that a total failure reported success

The worst hour of the project, and none of the four was individually subtle.

I widened the LR grid with a scripted `str.replace`. The anchor I pattern-matched was
`default=[1e-4, 2e-4, 5e-4, 1e-3]`; the file actually said `default=[1e-4, 2e-4, 5e-4]`. **I
asserted on the `run.sh` edit in the same commit and not on this one**, so it silently matched
nothing. The `--allow-boundary-lr` flag lived in the same replacement string, so it was never
registered either — while the *code that reads* `a.allow_boundary_lr` went in fine, from a
different, asserted edit.

What followed, in order:

1. Every probe invocation reached `if at_edge and not a.allow_boundary_lr:` and died with
   `AttributeError`. All 10 shards — **after** correctly training and writing their JSON — and
   the final reduce too.
2. `probe_all` wrapped each shard in `... || echo "FAILED probe $1 $2"`. It printed `FAILED`
   ten times and **returned 0**, because `echo` succeeds.
3. The reduce crashed, so it wrote no `lr.json`.
4. `probe_all`'s success test was `[ -f out/lr.json ]` — and a **stale `lr.json` from the
   previous probe run** was still sitting there. Existence passed. Phase `rc=0`.

So a phase in which every single unit of work crashed announced success, and training launched
on learning rates chosen by a *different, narrower grid* — 5e-4 for both arms, the exact
unequal under-tuning the widening was meant to fix. `driver.log` said `probe rc=0`. Ten lines
of `FAILED` sat in a log nobody was grepping.

The tell was in the data, not the logs: `lr.json` said 5e-4 while the shard files said 2e-3 was
the argmin **in every cell**. Two artifacts of the same phase disagreeing is what a stale marker
looks like from outside.

Fixes, all four:

- **Assert every scripted edit.** An unasserted `str.replace` that matches nothing is a silent
  no-op that leaves the file *looking* edited. Every edit in the repair commit ends with an
  `assert` on the anchor going in and the result coming out, and the fix was verified by
  running `--help` and reading the flag back, not by believing the diff.
- `probe_all` propagates shard failure (`|| { echo ...; return 1; }`) instead of `|| echo`.
- **`rm -f out/lr.json` before the reduce**, so the existence test can only be satisfied by
  *this* run. Metastrategy #28 — a completion marker must be impossible to find anywhere but
  this run's output — applied to an artifact rather than to a log string.
- A one-point grid has no boundary, so single-LR shards stop emitting false refusals.

Metastrategy #20 says almost every bug was caught by an assert or a printed distribution and
never by rereading code. This one is the counter-example that proves the rule: it was caught by
two *printed artifacts disagreeing with each other*. The assert that would have caught it
earlier is the one I skipped writing.

### A tolerance calibrated on one GPU is not a tolerance

The selfcheck asserts that the eps adapter is the raw network plus the documented scaling
and nothing else — `assert worst_e < 1e-6`, which passed at **3.0e-07** on the laptop 4050.
It **failed on the Colab T4 at 2.62e-06**, on identical code and identical inputs.

Nothing was wrong. cuDNN selects convolution algorithms nondeterministically, and how much
that costs is a property of the hardware, not of the code: calling the same net twice on
bit-identical input differs by ~0 on the 4050 and ~2.6e-06 on the T4. The threshold had
quietly encoded "this GPU" into the experiment.

Fix, and the general shape of the fix: **measure the floor instead of guessing it.** The
check now calls the raw net twice on the same input, takes that as the device's
nondeterminism floor, and requires the adapter to sit inside a small multiple of it. The
property under test was always "the adapter applies no transformation", never "two forward
passes are bit-identical" — a real transformation would be orders of magnitude above the
floor, which is exactly the discrimination a constant threshold cannot make portably.

This is the failure mode Metastrategy #7 warns about, and it is worth noticing which way it
went: the assert fired on a **healthy** system, and the temptation was to bump 1e-6 to 1e-5
and move on. That would have left a number in the file that means nothing on the next
machine.

### Ops

- Passed a **fabricated SSH public key** to `create-pod` (invented the base64 rather than
  reading `~/.ssh/*.pub`), so the pod was unreachable and had to be destroyed and
  recreated. Two minutes and $0.02. Read the key file; never type a key.
- The laptop 4050 is occupied by series 3 (`napkin-nemesis`, ~3.6GB of 6GB). Local sweep
  estimate was **~15 hours** of training alone, fighting a live experiment for SMs.
- Reached for a rented 4090 first. That was over-applying a standing "rent freely" rule to a
  job that does not need it: this is a 2.8M-param net on MNIST, i.e. a T4-class workload, and
  Colab was available. Two of three rented pods never ran at all — the first because a
  **fabricated SSH public key** was passed to `create-pod` (invented base64 instead of
  reading `~/.ssh/*.pub`), the second because the host driver reported CUDA 12.4 while the
  image demanded >=12.8, so the container crash-looped every 16 seconds while still billing.
  Both were diagnosed only by reading the pod's own system log, which said exactly that. The
  lesson is not "runpod is bad"; it is that the venue decision deserved measuring the job
  first (Metastrategy #16) and that a standing rule is not a substitute for sizing.
- **`pgrep -f` returned the xargs wrapper shells, not the workers.** `py-spy dump --pid` on
  the first match failed with "Failed to find python version from target process" because
  pids 10621-10623 were bash wrappers sitting at 0s CPU while the real python processes were
  10624-10629. Metastrategy #26 word for word. The reliable list of workers was
  `nvidia-smi --query-compute-apps=pid,used_memory`, which by construction only names
  processes that actually hold GPU memory.
- **The optimal concurrency inverted between two machines, and so did the bottleneck.**
  Measured properly on each box (unet, 250 steps after warmup, GPU-resident data):

  | | per-proc | aggregate |
  |---|---|---|
  | 4050, NPAR=1 | 14.7 st/s | **14.7** |
  | 4050, NPAR=2 | 6.4 st/s | 12.9 — *worse than one process* |
  | T4, 4 workers | ~3.2 st/s | ~11, and the 2 vCPUs pinned at 172% of 200% |

  One process already saturates the 6GB 4050, so a second is worse than useless; on the
  2-vCPU T4 the GPU was idle-ish and the **CPU launch path** was the ceiling, so 4-wide
  helped. Same code, opposite bottleneck. `NPAR` therefore defaults to 1 with the numbers
  written into `run.sh` next to it, because the next machine will invert it again.

  This also closes the EMA question left open above, in the direction that makes the earlier
  rescoping correct rather than merely cautious: fusing kernel launches *cannot* help a
  GPU-bound box, which is exactly why it measured 1.06x on the 4050. The T4's launch-bound
  regime was genuinely a different regime, and refusing to rule on it from 4050 numbers was
  the right call — the two boxes do not even share a bottleneck.

- **I measured a crash and reported it as a throughput curve.** The first concurrency sweep
  printed a beautiful monotone result — 39 st/s at NPAR=1 rising to 682 st/s at NPAR=4 —
  because the timing script lived in a scratch directory, could not `import napkin_dit`, and
  every worker exited 1 in milliseconds. I was measuring how fast python can fail to start.
  Nothing in the harness caught it: the loop timed wall clock and divided by an assumed step
  count, so a faster crash looked like a faster machine. The fix is the rule from
  Metastrategy #27 applied to benchmarks and not just to watchers — **check the exit code and
  parse a number the work itself printed, before dividing anything by it.** The corrected
  version reads each worker's rc file and its stdout, and would have refused to print.
- **`until [ -z "$(nvidia-smi --query-compute-apps=pid ...)" ]` hung for five minutes.**
  Waiting for *zero GPU processes globally* is a condition another project can prevent from
  ever becoming true — an unrelated `napkin_gap` process (series 2's ClawStreet query) was
  holding a CUDA context, so the loop could never exit even though every napkin-dit process
  had already died. A watcher's condition must be about **my** pids or **my** artifacts, never
  a global the rest of the machine also writes to.
- **A second experiment on the same card cost 13x, and no watcher could see it.** `napkin_skips.py`
  started training on the same 6GB GPU mid-sweep. Combined VRAM hit 5641/6141 MiB and NFE-63
  sweep points went from **384s to over 5200s**. WSL spills to shared host memory rather than
  OOMing, so there was no crash, no error, and no failed phase — `supervise.sh` retries on
  failure and this was not a failure. The only signal was two artifacts disagreeing again: a
  worker at 100% CPU and no result file in 88 minutes.
- **The resume watcher's threshold was set to the wrong quantity, and nearly undid the fix.**
  First version waited for "enough free VRAM for us" (2600 MiB). But the other job alone leaves
  **2961 MiB free**, which clears 2600 — so it would have resumed straight back into the
  contention it existed to avoid. It was caught at 1/3 checks, seconds from firing. The
  threshold has to *separate the states* (other-job-present 2961 vs idle 4173), not measure our
  own appetite. Sizing a threshold by what you need, rather than by what distinguishes the
  cases, is a distinct and easy mistake.
- **`pgrep -f` killed the shell that issued it. Third time.** `kill -TERM $(pgrep -f
  '[r]esume-when-free.sh')` exited 144 because the bracket trick only disguises the *pattern* —
  the same command line also contained the plain string `resume-when-free.sh` in later
  arguments, and `-f` matches the whole line. Documented in this very file, twice, and walked
  into anyway. The only reliable habit is to not match processes by name at all: use the process
  group, a pid captured at launch, or read `/proc/*/cmdline` directly.
- **A silent phase is an unreadable phase.** The probe shards run with `log_every=10**9`, so
  for twenty minutes the only progress signal was `probe=0/6` — which cannot distinguish slow
  from wedged, exactly the state Metastrategy #31 says you most need to detect. What did
  distinguish them was accumulated CPU time per pid (546-570s across 21 min of wall clock,
  i.e. ~43% of a core each), which is a two-sample progress measurement rather than a
  liveness check.
- `phase()` in `run.sh` was tested against a job that exits 0 and a job that exits 7 before
  being trusted, and against a re-run to confirm a completed phase is skipped. rc=7 is
  captured and distinguished from rc=0 (Metastrategy #33).
