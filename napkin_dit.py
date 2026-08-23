"""napkin-dit: UNet vs DiT x eps-prediction vs flow matching, a 2x2 at matched params and NFE.

SD 1.5/2.1 are UNet + eps-pred. Flux and Ideogram are DiT + flow matching. The field
changed both variables at roughly the same time, so the public evidence confounds them.
This file changes one at a time, at ~2.81M params and on a fair NFE axis.

The observation that makes the comparison clean rather than a vibes contest: write both
objectives in the sigma parameterisation, where x~ = x0 + sigma*eps.

    DDPM:  x_t = sqrt(ab)*x~,          sigma = sqrt((1-ab)/ab)
    flow:  x_u = (1-u)*x0 + u*eps  ->  x_u/(1-u) = x0 + (u/(1-u))*eps

Same x~. Same probability-flow ODE, dx~/dsigma = eps. The two objectives differ ONLY in
training: which noise levels are drawn, how the loss weights them, and whether the net
emits eps or v = eps - x0. So both arms share ONE sampler through a common eps_hat(x~,
sigma) adapter, and the objective flag touches the training loop and the adapter only.

Everything here inherits napkin-diffusion (t07): the cosine schedule, the sigma-space
solvers, karras spacing, the NFE accounting, FMD, and the pinned full-test-set reference.

Usage:
    python3 napkin_dit.py selfcheck
    python3 napkin_dit.py probe                       # LR per backbone, 1/6 length
    python3 napkin_dit.py train --backbone dit --objective flow --seed 0
    python3 napkin_dit.py sweep  --tier headline
    python3 napkin_dit.py agg
    python3 napkin_dit.py gif
"""
import argparse, itertools, json, math, pathlib, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = pathlib.Path(__file__).parent / "out"
T = 1000

BACKBONES = ("unet", "dit")
OBJECTIVES = ("eps", "flow")

# ---------------------------------------------------------------- noise schedule

def cosine_alpha_bar(T=T, s=0.008):
    """Nichol & Dhariwal cosine schedule, via betas clipped at 0.999. (t07, unchanged)"""
    t = torch.linspace(0, T, T + 1)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0)


AB = cosine_alpha_bar().to(DEV)
SIG = ((1 - AB) / AB).sqrt()                 # sigma(t), ascending in t
# Largest t whose alpha_bar is still >= 1e-4 (sigma ~ 91.7). t07 established that starting
# at t=999 (sigma ~ 2e4) is a step no 2nd-order solver survives. Both arms start here, so
# the initial noise level -- and therefore the NFE axis -- is matched across the 2x2.
T_START = int((AB >= 1e-4).nonzero().max())
SIG_MAX, SIG_MIN = SIG[T_START].item(), SIG[0].item()


def q_sample(x0, t, noise):
    ab = AB[t].view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

# ------------------------------------------------------------------ UNet (t07)

class TimeEmb(nn.Module):
    """Shared by both backbones. Time is ALWAYS passed on the [0, 1000] scale: the eps arm
    passes the integer timestep, the flow arm passes 1000*u. Identical embedding for both
    arms means the time conditioning is not one of the things the objective axis changes."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.SiLU(), nn.Linear(d * 4, d * 4))

    def forward(self, t):
        half = self.d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).permute(1, 0, 3, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(h.transpose(1, 2).reshape(B, C, H, W))


class UNet(nn.Module):
    """32 -> 64 -> 128 channels at 32/16/8 px, self-attention at 8px. 2.81M params."""

    def __init__(self, ch=(32, 64, 128), tdim=32):
        super().__init__()
        c1, c2, c3 = ch
        td = tdim * 4
        self.temb = TimeEmb(tdim)
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.d1a, self.d1b = ResBlock(c1, c1, td), ResBlock(c1, c1, td)
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2a, self.d2b = ResBlock(c1, c2, td), ResBlock(c2, c2, td)
        self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.d3a, self.d3b = ResBlock(c2, c3, td), ResBlock(c3, c3, td)
        self.at3 = Attn(c3)
        self.mid1, self.midat, self.mid2 = ResBlock(c3, c3, td), Attn(c3), ResBlock(c3, c3, td)
        self.u3a, self.u3b = ResBlock(c3 * 2, c3, td), ResBlock(c3, c3, td)
        self.up3 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.u2a, self.u2b = ResBlock(c2 * 2, c2, td), ResBlock(c2, c2, td)
        self.up2 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.u1a, self.u1b = ResBlock(c1 * 2, c1, td), ResBlock(c1, c1, td)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))

    def forward(self, x, t):
        e = self.temb(t)
        h0 = self.inp(x)
        h1 = self.d1b(self.d1a(h0, e), e)
        h2 = self.d2b(self.d2a(self.down1(h1), e), e)
        h3 = self.at3(self.d3b(self.d3a(self.down2(h2), e), e))
        m = self.mid2(self.midat(self.mid1(h3, e)), e)
        u = self.u3b(self.u3a(torch.cat([m, h3], 1), e), e)
        u = self.u2b(self.u2a(torch.cat([self.up3(u), h2], 1), e), e)
        u = self.u1b(self.u1a(torch.cat([self.up2(u), h1], 1), e), e)
        return self.out(u)

# ------------------------------------------------------------------------- DiT

class DiTBlock(nn.Module):
    """Peebles & Xie adaLN-zero block. `attn=False` removes token mixing entirely, which
    is what makes the patch=1 degeneracy selfcheck possible."""

    def __init__(self, d, heads, td, attn=True):
        super().__init__()
        self.h, self.use_attn = heads, attn
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv, self.proj = nn.Linear(d, d * 3), nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(approximate="tanh"),
                                 nn.Linear(d * 4, d))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(td, d * 6))
        nn.init.zeros_(self.mod[1].weight); nn.init.zeros_(self.mod[1].bias)   # adaLN-ZERO

    def forward(self, x, c):
        B, N, D = x.shape
        s1, g1, a1, s2, g2, a2 = self.mod(c).chunk(6, -1)
        if self.use_attn:
            h = self.n1(x) * (1 + s1[:, None]) + g1[:, None]
            q, k, v = self.qkv(h).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
            o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
            x = x + a1[:, None] * self.proj(o)
        h = self.n2(x) * (1 + s2[:, None]) + g2[:, None]
        return x + a2[:, None] * self.mlp(h)


class DiT(nn.Module):
    """Patchify -> L adaLN-zero transformer blocks -> unpatchify. Sized to match the UNet.

    d=164/L=6/patch=2 lands at 2,813,496 params against the UNet's 2,813,057 -- 0.02%
    apart, asserted in selfcheck rather than eyeballed. Patch 2 on a 32x32 image is 256
    tokens, so attention runs over the whole image at every layer -- the structural
    opposite of the UNet's attention-at-8px-only, which is the point of the axis."""

    def __init__(self, d=164, depth=6, heads=4, patch=2, tdim=32, attn=True, img=32, cin=1):
        super().__init__()
        self.p, self.img, self.cin, self.d = patch, img, cin, d
        n = (img // patch) ** 2
        td = tdim * 4
        self.temb = TimeEmb(tdim)
        self.embed = nn.Linear(cin * patch * patch, d)
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.blocks = nn.ModuleList([DiTBlock(d, heads, td, attn) for _ in range(depth)])
        self.nf = nn.LayerNorm(d, elementwise_affine=False)
        self.modf = nn.Sequential(nn.SiLU(), nn.Linear(td, d * 2))
        self.head = nn.Linear(d, cin * patch * patch)
        for m in (self.modf[1], self.head):                     # zero-init the output path
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t):
        B, C, H, W = x.shape
        p, g = self.p, self.img // self.p
        tok = x.reshape(B, C, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C * p * p)
        h = self.embed(tok) + self.pos
        c = self.temb(t)
        for blk in self.blocks:
            h = blk(h, c)
        s, gain = self.modf(c).chunk(2, -1)
        h = self.head(self.nf(h) * (1 + s[:, None]) + gain[:, None])
        return (h.reshape(B, g, g, C, p, p).permute(0, 3, 1, 4, 2, 5)
                 .reshape(B, C, H, W))


def build(backbone, **kw):
    return (UNet() if backbone == "unet" else DiT(**kw)).to(DEV)

# ------------------------------------------------------- objectives and adapters
# Both objectives are read through one interface: eps_hat(x_tilde, sigma). Everything
# downstream (all samplers) sees only that, so the sampler cannot differ between arms.


def sigma_to_t(sigma):
    """Nearest training timestep for a continuous sigma (eps arm's time conditioning)."""
    s = torch.as_tensor(sigma, device=DEV).reshape(-1).contiguous()
    return torch.searchsorted(SIG, s).clamp(max=T - 1)


class Denoiser:
    """Wraps a net + objective into eps_hat(x~, sigma). Also owns the training target, so
    the forward process and its inverse can never drift apart."""

    def __init__(self, net, objective):
        self.net, self.obj = net, objective

    # --- training ---------------------------------------------------------------
    def loss(self, x0, gen=None):
        n = x0.shape[0]
        eps = torch.randn(x0.shape, device=x0.device, generator=gen)
        if self.obj == "eps":
            t = torch.randint(0, T, (n,), device=x0.device, generator=gen)
            return F.mse_loss(self.net(q_sample(x0, t, eps), t.float()), eps)
        u = torch.rand(n, device=x0.device, generator=gen).view(-1, 1, 1, 1)
        xu = (1 - u) * x0 + u * eps
        return F.mse_loss(self.net(xu, u.view(-1) * T), eps - x0)     # v-target

    # --- sampling ---------------------------------------------------------------
    def eps_hat(self, xt, sigma):
        """xt is x~ = x0 + sigma*eps. Returns the model's eps at that noise level."""
        n = xt.shape[0]
        if self.obj == "eps":
            ab = 1.0 / (1.0 + sigma ** 2)
            t = sigma_to_t(sigma).float().expand(n)
            return self.net(xt * ab ** 0.5, t)
        u = sigma / (1.0 + sigma)                    # inverse of sigma = u/(1-u)
        xu = xt * (1 - u)
        v = self.net(xu, torch.full((n,), u * T, device=xt.device))
        return xu + (1 - u) * v                      # eps = x_u + (1-u)*v

# --------------------------------------------------------------------- samplers
# In sigma coordinates the probability-flow ODE is dx~/dsigma = eps, so deterministic
# DDIM is *literally* Euler and heun is its 2nd-order sibling. (t07's footnote.)


def _timesteps(steps):
    return torch.linspace(T_START, 0, steps).round().long().to(DEV)


def sigma_schedule(steps, spacing):
    """Which noise levels to visit. Returns sigmas[steps+1], ending at exactly 0.

    "t"        uniform in the training timestep index -- the native DDPM/DDIM spacing.
    "karras"   rho=7 power law, uniform in sigma^(1/rho) (Karras et al. 2022).
    "u"        uniform in u = sigma/(1+sigma) -- the native FLOW MATCHING spacing.

    "u" is here because the canonical FM sampler is Euler uniform-in-u, which is a step
    *placement*, not an objective. t07 measured placement alone to be worth up to 2x at
    matched NFE, so leaving it bundled with the objective would have handed flow matching
    a free variable. It is swept across all four cells instead.
    """
    if spacing == "karras":
        rho = 7.0
        i = torch.arange(steps, dtype=torch.float64)
        sig = (SIG_MAX ** (1 / rho) + i / (steps - 1)
               * (SIG_MIN ** (1 / rho) - SIG_MAX ** (1 / rho))) ** rho
        sig = sig.float().to(DEV)
    elif spacing == "u":
        umax, umin = SIG_MAX / (1 + SIG_MAX), SIG_MIN / (1 + SIG_MIN)
        u = torch.linspace(umax, umin, steps, dtype=torch.float64).to(DEV)
        sig = (u / (1 - u)).float()
    else:
        sig = SIG[_timesteps(steps)]
    return torch.cat([sig, torch.zeros(1, device=DEV)])


@torch.no_grad()
def sample(den, n, solver="heun", nfe=16, seed=0, spacing="karras", clamp=True, track=None):
    """Shared by every cell of the 2x2. Returns (images in [-1,1], nfe_used)."""
    g = torch.Generator(DEV).manual_seed(seed)
    steps = max(2, (nfe + 1) // 2) if solver == "heun" else max(2, nfe)
    sig = sigma_schedule(steps, spacing)
    xt = torch.randn(n, 1, 32, 32, device=DEV, generator=g) * (1 + sig[0] ** 2).sqrt()
    used = 0

    def d_at(xt, i):
        """eps at sigma[i], with the x0 clip. The clip is load-bearing, not cosmetic:
        the first step has dsigma ~ -90, and one imperfect eps scaled by that is noise."""
        e = den.eps_hat(xt, sig[i].item())
        if clamp:
            x0 = (xt - sig[i] * e).clamp(-1, 1)
            e = (xt - x0) / sig[i]
        return e

    for i in range(steps):
        dsig = sig[i + 1] - sig[i]
        d1 = d_at(xt, i); used += 1
        if solver == "heun" and sig[i + 1] > 0:
            d2 = d_at(xt + dsig * d1, i + 1); used += 1
            xt = xt + dsig * 0.5 * (d1 + d2)
        else:
            xt = xt + dsig * d1
        if track is not None:
            track.append((xt / (1 + sig[i + 1] ** 2).sqrt()).clamp(-1, 1))
    return (xt.clamp(-1, 1) if clamp else xt), used


@torch.no_grad()
def sample_ancestral(net, n, nfe=20, seed=0, clamp=True):
    """DDPM posterior sampling in x-space. eps arm only; kept so the inherited t07
    identity asserts (ancestral == DDIM(eta=1)) still have something to compare against."""
    g = torch.Generator(DEV).manual_seed(seed)
    ts = _timesteps(nfe)
    x = torch.randn(n, 1, 32, 32, device=DEV, generator=g)
    for i, t in enumerate(ts):
        ab_t = AB[t]
        ab_p = AB[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=DEV)
        eps = net(x, t.repeat(n).float())
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clamp:
            x0 = x0.clamp(-1, 1)
        a_s = ab_t / ab_p
        b_s = 1 - a_s
        mean = (ab_p.sqrt() * b_s / (1 - ab_t)) * x0 + (a_s.sqrt() * (1 - ab_p) / (1 - ab_t)) * x
        if i + 1 < len(ts):
            var = b_s * (1 - ab_p) / (1 - ab_t)
            x = mean + var.sqrt() * torch.randn(x.shape, device=DEV, generator=g)
        else:
            x = mean
    return x


@torch.no_grad()
def sample_ddim_eta(net, n, eta, nfe, seed=0, clamp=True):
    """Classic x-space DDIM with tunable eta. selfcheck only."""
    g = torch.Generator(DEV).manual_seed(seed)
    ts = _timesteps(nfe)
    ab = torch.cat([AB[ts], torch.ones(1, device=DEV)])
    x = torch.randn(n, 1, 32, 32, device=DEV, generator=g)
    for i, t in enumerate(ts):
        ab_t, ab_p = ab[i], ab[i + 1]
        eps = net(x, t.repeat(n).float())
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clamp:
            x0 = x0.clamp(-1, 1)
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        s = eta * ((1 - ab_p) / (1 - ab_t) * (1 - ab_t / ab_p)).sqrt()
        x = ab_p.sqrt() * x0 + (1 - ab_p - s ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(ts):
            x = x + s * torch.randn(x.shape, device=DEV, generator=g)
    return x

# ------------------------------------------------------------------------- data

def loader(bs, train=True, fashion=False, shuffle=True):
    """Only used to build the GPU-resident cache and to train the metric CNN (needs labels)."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Pad(2),
                             transforms.Normalize((0.5,), (0.5,))])
    ds = (datasets.FashionMNIST if fashion else datasets.MNIST)(
        OUT / "data", train=train, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=2)


def gpu_dataset(fashion=False, train=True):
    """The whole split resident on the GPU, cached to disk after the first build.

    MNIST padded to 32x32 is 245MB in float32, so there is no reason for it to live behind a
    DataLoader -- and a real reason for it not to: the sweep runs 5 training processes
    concurrently on one GPU, and Colab's free tier gives 2 vCPU. Five runs x two worker
    processes starves on CPU while the GPU idles. Profiling the bottleneck rather than the
    marketing is Metastrategy #15; measured speedup is in INSIGHTS.md.
    """
    cache = OUT / f"gpu-{'fashion' if fashion else 'mnist'}-{'train' if train else 'test'}.pt"
    if not cache.exists():
        OUT.mkdir(exist_ok=True)
        x = torch.cat([b for b, _ in loader(1000, train, fashion, shuffle=False)])
        torch.save(x, cache.with_suffix(".tmp")); cache.with_suffix(".tmp").rename(cache)
    return torch.load(cache, map_location=DEV, weights_only=True)


def batches(data, bs, gen):
    """Epoch-shuffled minibatches, drop_last, entirely on-device. Same sampling law as
    DataLoader(shuffle=True, drop_last=True) -- just without leaving the GPU."""
    n = data.shape[0]
    while True:
        perm = torch.randperm(n, device=DEV, generator=gen)
        for i in range(0, n - bs + 1, bs):
            yield data[perm[i:i + bs]]

# ------------------------------------------------------------------------- FMD
# Frechet distance in the feature space of a small MNIST CNN. NOT FID: comparable within
# this repo only. (t07, unchanged -- including the reference-set pinning.)

class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(64, 10)

    def forward(self, x, feat=False):
        f = self.body(x)
        return f if feat else self.head(f)


def train_clf(fashion=False, epochs=2):
    p = OUT / "clf.pt"
    clf = Clf().to(DEV)
    if p.exists():
        clf.load_state_dict(torch.load(p)); return clf.eval()
    torch.manual_seed(0)
    clf = Clf().to(DEV)
    opt = torch.optim.AdamW(clf.parameters(), 1e-3)
    for _ in range(epochs):
        for x, y in loader(256, True, fashion):
            loss = F.cross_entropy(clf(x.to(DEV)), y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(clf.state_dict(), p.with_suffix(".tmp"))     # atomic: sweep runs sharded
    p.with_suffix(".tmp").rename(p)
    return clf.eval()


@torch.no_grad()
def feats(clf, imgs, bs=500):
    return torch.cat([clf(imgs[i:i + bs].to(DEV), feat=True).double()
                      for i in range(0, len(imgs), bs)])


def frechet(f1, f2):
    """tr((S1^.5 S2 S1^.5)^.5) via eigvalsh. eigvals(S1@S2) goes complex on noisy
    finite-sample covariances -- see t07's INSIGHTS."""
    m1, m2 = f1.mean(0), f2.mean(0)
    s1, s2 = torch.cov(f1.T), torch.cov(f2.T)
    ev, V = torch.linalg.eigh(s1)
    s1h = V @ torch.diag(ev.clamp(min=0).sqrt()) @ V.T
    inner = torch.linalg.eigvalsh(s1h @ s2 @ s1h).clamp(min=0).sqrt().sum()
    return ((m1 - m2) ** 2).sum().item() + (s1.trace() + s2.trace() - 2 * inner).item()

# --------------------------------------------------------------------- training

def ckpt_path(backbone, objective, seed):
    return OUT / "ckpt" / f"{backbone}-{objective}-s{seed}.pt"


def train_one(backbone, objective, seed, steps, lr, bs=128, warmup=500, fashion=False,
              log_every=500, save=True):
    """One cell, one seed. Identical loop for all four cells -- the only branches are
    inside Denoiser.loss and the backbone constructor (metastrategy #4)."""
    p = ckpt_path(backbone, objective, seed)
    if save and p.exists():
        print(f"skip {p.name} (exists)"); return p
    torch.manual_seed(seed)
    net = build(backbone)
    den = Denoiser(net, objective)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    opt = torch.optim.AdamW(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    g = torch.Generator(DEV).manual_seed(seed)
    src = batches(gpu_dataset(fashion), bs, g)
    t0, step, last = time.time(), 0, float("nan")
    while step < steps:
        x = next(src)
        with torch.amp.autocast("cuda", enabled=DEV == "cuda"):
            loss = den.loss(x)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                ema[k].mul_(0.999).add_(v.detach(), alpha=0.001) \
                    if v.dtype.is_floating_point else ema[k].copy_(v)
        step += 1; last = loss.item()
        if step % log_every == 0:
            print(f"{backbone}/{objective} s{seed} step {step}/{steps} "
                  f"loss {last:.4f} {time.time()-t0:.0f}s", flush=True)
    if save:
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ema": ema, "backbone": backbone, "objective": objective,
                    "seed": seed, "steps": steps, "lr": lr}, p)
        print("saved", p, f"({time.time()-t0:.0f}s)", flush=True)
    return last


def load_den(backbone, objective, seed):
    d = torch.load(ckpt_path(backbone, objective, seed), map_location=DEV)
    net = build(backbone)
    net.load_state_dict(d["ema"])
    return Denoiser(net.eval(), objective)

# --------------------------------------------------------------------- commands

def cmd_train(a):
    lrs = json.loads((OUT / "lr.json").read_text()) if (OUT / "lr.json").exists() else {}
    for b, o, s in itertools.product(a.backbone, a.objective, a.seed):
        train_one(b, o, s, a.steps, a.lr or lrs.get(b, 2e-4), a.bs,
                  fashion=a.dataset == "fashion")


def cmd_probe(a):
    """LR per backbone at 1/6 length. A naively-tuned transformer losing to a tuned UNet is
    the classic rigged 2x2, and P1 predicts exactly that outcome -- so the LR must not be
    what produces it. Picked per BACKBONE (not per cell) to keep the objective axis clean."""
    res, steps = {}, max(200, a.steps // 6)
    for b in a.backbone:
        for lr in a.lrs:
            ls = [train_one(b, o, 0, steps, lr, a.bs, save=False, log_every=10**9)
                  for o in a.objective]
            # losses are not comparable across objectives (different targets), so rank each
            # objective's LR separately and score a backbone's LR by its mean rank.
            res.setdefault(b, {})[lr] = ls
            print(f"probe {b:4s} lr={lr:<7g} " +
                  " ".join(f"{o}={v:.4f}" for o, v in zip(a.objective, ls)), flush=True)
    best = {}
    for b, d in res.items():
        rank = {lr: 0 for lr in d}
        for j in range(len(a.objective)):
            for i, lr in enumerate(sorted(d, key=lambda k: d[k][j])):
                rank[lr] += i
        best[b] = min(rank, key=rank.get)
    (OUT / "lr.json").write_text(json.dumps(best, indent=2))
    (OUT / "probe.json").write_text(json.dumps(res, indent=2))
    print("chosen LRs:", best)


TIERS = {  # (solvers, spacings, seeds)
    "headline":  (["heun"], ["karras"], [0, 1, 2, 3, 4]),
    "secondary": (["euler"], ["karras", "u", "t"], [0, 1, 2]),
    "secondary2": (["heun"], ["u", "t"], [0, 1, 2]),
}


def res_path(b, o, s, solver, spacing, nfe):
    return OUT / "res" / f"{b}-{o}-s{s}-{solver}-{spacing}-n{nfe}.json"


def cmd_sweep(a):
    """One result file per (cell, seed, solver, spacing, nfe). Existence = done, so this is
    interruptible and any machine can pick up the remainder (metastrategy #14)."""
    (OUT / "res").mkdir(parents=True, exist_ok=True)
    clf = train_clf(a.dataset == "fashion")
    # Reference is ALWAYS the full test set, never a prefix of length a.n: the two halves of
    # the MNIST test set are measurably different populations (t07's afternoon).
    fr = feats(clf, gpu_dataset(a.dataset == "fashion", train=False))
    solvers, spacings, seeds = TIERS[a.tier]
    seeds = [x for x in seeds if x in a.seed]        # --seed shards the sweep across procs
    for b, o, s in itertools.product(a.backbone, a.objective, seeds):
        if not ckpt_path(b, o, s).exists():
            print(f"missing ckpt {b}/{o}/s{s} -- skipping"); continue
        todo = [(sv, sp, n) for sv, sp, n in itertools.product(solvers, spacings, a.nfe)
                if not res_path(b, o, s, sv, sp, n).exists()]
        if not todo:
            continue
        den = load_den(b, o, s)
        for sv, sp, nfe in todo:
            t0 = time.time()
            imgs, used = [], 0
            for i in range(0, a.n, 500):
                xb, used = sample(den, min(500, a.n - i), sv, nfe, seed=i, spacing=sp)
                imgs.append(xb.cpu())
            d = frechet(fr, feats(clf, torch.cat(imgs)))
            rec = {"backbone": b, "objective": o, "seed": s, "solver": sv, "spacing": sp,
                   "nfe_req": nfe, "nfe": used, "fmd": d, "n": a.n, "secs": time.time() - t0}
            res_path(b, o, s, sv, sp, nfe).write_text(json.dumps(rec))
            print(f"{b}/{o} s{s} {sv}/{sp} nfe={used:3d} FMD={d:9.3f} "
                  f"({rec['secs']:.0f}s)", flush=True)

# ------------------------------------------------------------------ aggregation

def iqm(v):
    v = sorted(v)
    k = len(v) // 4
    w = v[k:len(v) - k] or v
    return sum(w) / len(w)


def boot_ci(v, reps=2000, seed=0):
    if len(v) < 2:
        return (v[0], v[0]) if v else (float("nan"),) * 2
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(v, dtype=torch.double)
    idx = torch.randint(len(v), (reps, len(v)), generator=g)
    b = sorted(iqm(t[i].tolist()) for i in idx)
    return b[int(.025 * reps)], b[int(.975 * reps)]


def load_res():
    return [json.loads(p.read_text()) for p in sorted((OUT / "res").glob("*.json"))]


def cmd_agg(a):
    rows = [r for r in load_res() if r["solver"] in TIERS[a.tier][0]
            and r["spacing"] in TIERS[a.tier][1]]
    if not rows:
        print("no results for tier", a.tier); return
    cells, out = sorted({(r["backbone"], r["objective"]) for r in rows}), {}
    for sv, sp in sorted({(r["solver"], r["spacing"]) for r in rows}):
        print(f"\n=== solver={sv} spacing={sp}   FMD, IQM over seeds [95% bootstrap CI] ===")
        nfes = sorted({r["nfe"] for r in rows if r["solver"] == sv and r["spacing"] == sp})
        print(f"{'NFE':>5} " + " ".join(f"{b+'/'+o:>26}" for b, o in cells))
        for nfe in nfes:
            line, best = f"{nfe:>5} ", None
            vals = {}
            for b, o in cells:
                v = [r["fmd"] for r in rows if (r["backbone"], r["objective"]) == (b, o)
                     and r["solver"] == sv and r["spacing"] == sp and r["nfe"] == nfe]
                vals[(b, o)] = v
                out.setdefault(f"{b}/{o}|{sv}|{sp}", []).append(
                    {"nfe": nfe, "iqm": iqm(v) if v else None, "n_seeds": len(v),
                     "ci": boot_ci(v) if v else None, "raw": v})
                line += f"{(f'{iqm(v):.2f} [{boot_ci(v)[0]:.2f},{boot_ci(v)[1]:.2f}]' if v else '-'):>27}"
            print(line)
            # rank stability: how often does a k-seed subset name the same winner as all 5?
            if nfe == nfes[-1] and all(len(v) >= 3 for v in vals.values()):
                print("       rank-stability P(best@k == best@N): " + rank_stability(vals))
    (OUT / f"agg-{a.tier}.json").write_text(json.dumps(out, indent=2))
    plot(out, a.tier)


def rank_stability(vals, reps=1000):
    """P(the winner named by k random seeds == the winner named by all of them)."""
    import random
    rnd = random.Random(0)
    N = min(len(v) for v in vals.values())
    ref = min(vals, key=lambda c: iqm(vals[c][:N]))
    outs = []
    for k in range(1, N + 1):
        hit = sum(min(vals, key=lambda c: iqm([vals[c][i] for i in idx])) == ref
                  for idx in (rnd.sample(range(N), k) for _ in range(reps)))
        outs.append(f"k={k}:{hit/reps:.2f}")
    return " ".join(outs)


def plot(out, tier):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(out)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for k, mark in zip(keys, itertools.cycle("osd^v<>")):
        pts = [p for p in out[k] if p["iqm"] is not None]
        if not pts:
            continue
        x = [p["nfe"] for p in pts]
        ax.plot(x, [p["iqm"] for p in pts], marker=mark, label=k)
        ax.fill_between(x, [p["ci"][0] for p in pts], [p["ci"][1] for p in pts], alpha=.15)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("network evaluations (NFE)"); ax.set_ylabel("FMD (lower is better)")
    ax.set_title(f"UNet vs DiT x eps vs flow, matched 2.81M params ({tier})")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / f"ablation-{tier}.png")
    print("\nwrote", OUT / f"ablation-{tier}.png")


def cmd_gif(a):
    """2x2 contact sheet: the same seed denoising in all four cells, side by side."""
    import imageio.v2 as imageio
    from torchvision.utils import make_grid
    cells = [(b, o) for b in BACKBONES for o in OBJECTIVES]
    tracks = {}
    for b, o in cells:
        fr = []
        sample(load_den(b, o, 0), 16, a.solver, a.gif_nfe, seed=1, spacing=a.spacing, track=fr)
        tracks[(b, o)] = fr
    L = max(len(v) for v in tracks.values())
    frames = []
    for i in range(L):
        row = torch.cat([tracks[c][min(i, len(tracks[c]) - 1)].cpu() for c in cells])
        frames.append(make_grid(row, nrow=16, normalize=True, value_range=(-1, 1))
                      .mul(255).byte().permute(1, 2, 0).numpy())
    frames += [frames[-1]] * 8
    imageio.mimsave(OUT / "denoise2x2.gif", frames, duration=0.12, loop=0)
    imageio.imwrite(OUT / "samples2x2.png", frames[-1])
    print("wrote", OUT / "denoise2x2.gif", "(rows top->bottom:", cells, ")")

# -------------------------------------------------------------------- selfcheck

def cmd_selfcheck(a):
    OUT.mkdir(exist_ok=True)
    ab = AB.cpu()
    assert (ab.diff() < 0).all(), "alpha_bar must be strictly decreasing"
    assert ab[0] > 0.99 and ab[-1] < 0.01, f"endpoints off: {ab[0]:.4f} {ab[-1]:.5f}"

    # -- 5. matched params, asserted rather than eyeballed -----------------------
    pu = sum(p.numel() for p in UNet().parameters())
    pd = sum(p.numel() for p in DiT().parameters())
    assert abs(pd - pu) / pu < 0.02, f"params not matched: unet {pu} vs dit {pd}"

    # -- 1. DiT degeneracy: patch=1, no attention => a per-pixel MLP --------------
    # If token mixing is off and each token is one pixel, perturbing input pixel (i,j)
    # must change output pixel (i,j) and NOTHING else. Stronger than permutation
    # equivariance and it survives the positional embedding.
    torch.manual_seed(0)
    deg = DiT(d=64, depth=2, heads=4, patch=1, attn=False).to(DEV).eval()
    for m in deg.modules():                     # undo adaLN-zero so the probe isn't trivial
        if isinstance(m, nn.Linear) and m.weight.abs().sum() == 0:
            nn.init.normal_(m.weight, std=.05); nn.init.normal_(m.bias, std=.05)
    with torch.no_grad():
        x = torch.randn(1, 1, 32, 32, device=DEV)
        t = torch.tensor([500.0], device=DEV)
        y0 = deg(x, t)
        leak, on = 0.0, 1e9
        for (i, j) in [(0, 0), (5, 17), (16, 16), (31, 31)]:
            xp = x.clone(); xp[0, 0, i, j] += 1.0
            d = (deg(xp, t) - y0).abs()[0, 0]
            on = min(on, d[i, j].item())
            d[i, j] = 0
            leak = max(leak, d.sum().item())        # TOTAL leak, summed over 1023 pixels
        assert on > 1e-3, f"patch=1 no-attn DiT barely responds to its own pixel: {on:.2e}"
        assert leak < 1e-9, f"patch=1 no-attn DiT is not per-pixel: total leak {leak:.2e}"
    # ...and turning attention back on must break exactly that property. Sum the leak
    # rather than maxing it: one perturbed token out of 1024 shifts each *other* token by
    # ~1/1024 of the signal, so the per-pixel max (4e-6) is a useless discriminator while
    # the total (2e-3, against an exact 0.0 above) is unambiguous.
    torch.manual_seed(0)
    mix = DiT(d=64, depth=2, heads=4, patch=1, attn=True).to(DEV).eval()
    for m in mix.modules():
        if isinstance(m, nn.Linear) and m.weight.abs().sum() == 0:
            nn.init.normal_(m.weight, std=.05); nn.init.normal_(m.bias, std=.05)
    with torch.no_grad():
        y0 = mix(x, t); xp = x.clone(); xp[0, 0, 16, 16] += 1.0
        d = (mix(xp, t) - y0).abs()[0, 0]; d[16, 16] = 0
        mixleak = d.sum().item()
        assert mixleak > 1e-4, "attention ON but the DiT is still per-pixel -- attn is dead"

    # -- 2. flow matching and DDPM agree on the marginal variance schedule --------
    x0 = torch.randn(64, 1, 32, 32, device=DEV)
    eps = torch.randn_like(x0)
    worst_x, worst_e, nd = 0.0, 0.0, 0.0
    torch.manual_seed(0)
    net = build("unet")
    dn_e, dn_f = Denoiser(net, "eps"), Denoiser(net, "flow")
    for t in (50, 200, 500, 800, T_START):
        s = SIG[t].item()
        u = s / (1 + s)
        xt_ddpm = q_sample(x0, torch.full((64,), t, device=DEV), eps) / AB[t].sqrt()
        xt_flow = ((1 - u) * x0 + u * eps) / (1 - u)
        worst_x = max(worst_x, ((xt_ddpm - xt_flow).abs().max()
                                / xt_ddpm.abs().max()).item())
        # ...and the eps adapter must be the raw net plus the documented scaling, nothing
        # else. The tolerance CANNOT be a constant: cuDNN picks convolution algorithms
        # nondeterministically, and how much that costs is hardware-dependent -- 3e-07 on a
        # laptop 4050, 2.6e-06 on a T4, which failed a 1e-6 threshold calibrated on the
        # former. So measure the floor on THIS device (call the raw net twice on identical
        # input) and require the adapter to sit inside it. An independently measured
        # baseline beats a magic number (Metastrategy #7: re-derive the property).
        with torch.no_grad():
            inp = xt_ddpm * (1 / (1 + s ** 2)) ** .5
            tt = torch.full((64,), float(sigma_to_t(s).item()), device=DEV)
            r1, r2 = net(inp, tt), net(inp, tt)
            nd = max(nd, ((r1 - r2).abs().max() / r1.abs().max()).item())
            got = dn_e.eps_hat(xt_ddpm, s)
        worst_e = max(worst_e, ((r1 - got).abs().max() / r1.abs().max()).item())
    assert worst_x < 1e-4, f"FM and DDPM marginals disagree, rel err {worst_x:.2e}"
    assert worst_e <= max(1e-7, 4 * nd), \
        f"eps adapter is not the raw net: rel err {worst_e:.2e} vs a same-input " \
        f"nondeterminism floor of {nd:.2e} on this device"
    # the flow adapter's algebra: given a net that emits the TRUE v, eps_hat must be exact
    class TrueV(nn.Module):
        def forward(self, xu, t):
            return eps - x0                                   # v = eps - x0, exactly
    err_v = 0.0
    for s in (0.1, 1.0, 10.0, SIG_MAX):
        u = s / (1 + s)
        got = Denoiser(TrueV(), "flow").eps_hat(x0 + s * eps, s)
        err_v = max(err_v, ((got - eps).abs().max() / eps.abs().max()).item())
    assert err_v < 1e-4, f"flow eps_hat != eps under a ground-truth v net, {err_v:.2e}"

    # -- 4. inherited t07 identities (the sampler is shared, so re-assert them) ----
    torch.manual_seed(0)
    m = build("unet").eval()
    dn = Denoiser(m, "eps")
    a1 = sample_ancestral(m, 4, nfe=20, seed=7, clamp=False)
    a2 = sample_ddim_eta(m, 4, eta=1.0, nfe=20, seed=7, clamp=False)
    err = ((a1 - a2).abs().max() / a1.abs().max()).item()
    assert err < 1e-3, f"ancestral != ddim(eta=1), max rel err {err}"
    e1, _ = sample(dn, 4, "euler", 20, seed=7, spacing="t", clamp=False)
    e2 = sample_ddim_eta(m, 4, eta=0.0, nfe=20, seed=7, clamp=False)
    err2 = ((e1 - e2).abs().max() / e1.abs().max()).item()
    assert err2 < 1e-3, f"sigma-space Euler != x-space DDIM, max rel err {err2}"
    _, n_h = sample(dn, 2, "heun", 20, seed=0)
    _, n_e = sample(dn, 2, "euler", 20, seed=0)
    assert (n_e, n_h) == (20, 19), f"NFE accounting off: euler {n_e}, heun {n_h}"

    for sp in ("t", "karras", "u"):
        sg = sigma_schedule(30, sp)
        assert (sg.diff() < 0).all(), f"{sp} sigmas must be strictly decreasing"
        assert sg[-1] == 0 and len(sg) == 31, f"{sp} schedule must end at sigma=0"
        assert abs(sg[0] - SIG_MAX) < 1e-3, f"{sp} must start at sigma_max"
    st, sk = sigma_schedule(30, "t"), sigma_schedule(30, "karras")
    assert sk.diff().abs().max() < st.diff().abs().max() / 4, \
        "karras should avoid t-uniform's enormous first sigma step"

    f = torch.randn(500, 64, device=DEV).double()
    assert abs(frechet(f, f.clone())) < 1e-6, "FMD of a set with itself must be ~0"
    assert frechet(f, f + 3) > 8, "FMD must grow with a mean shift"

    # -- 6. the GPU-resident dataset must be the DataLoader's data, not merely shaped like it
    for train, n in ((True, 60000), (False, 10000)):
        d = gpu_dataset(train=train)
        assert d.shape == (n, 1, 32, 32), f"gpu_dataset(train={train}) shape {tuple(d.shape)}"
        assert d.min() >= -1 - 1e-6 and d.max() <= 1 + 1e-6, "dataset outside [-1,1]"
        ref = torch.cat([b for b, _ in loader(2000, train, False, shuffle=False)][:3]).to(DEV)
        assert torch.equal(d[:len(ref)], ref), f"gpu_dataset(train={train}) != the DataLoader"
    # ...and `batches` must be an epoch-shuffled drop_last pass: every index exactly once.
    idx = torch.arange(10, device=DEV, dtype=torch.float).view(10, 1, 1, 1)
    it = batches(idx, 3, torch.Generator(DEV).manual_seed(0))
    ep = torch.cat([next(it).flatten() for _ in range(3)]).long()
    assert len(ep) == 9 and len(ep.unique()) == 9, f"batches() repeats or drops: {ep.tolist()}"
    nxt = next(it).flatten().long()                  # the 4th batch must reshuffle, not wrap
    assert len(nxt) == 3, "batches() must always yield a full batch (drop_last)"

    # -- 3. one fixed batch, both backbones x both objectives, driven to the floor --
    # DEVIATION FROM THE REGISTERED SPEC, stated rather than quietly relaxed. The design
    # doc registered "overfittable to ~1e-15". That is unreachable: an MSE of 1e-15 means
    # a per-element error of 3e-8 on targets of order 1, which is float32 epsilon. Measured
    # floors at 1500 steps (lr 2e-3, cosine): unet/eps 1.4e-6, dit/eps 1.0e-4, unet/flow
    # 2.2e-4, dit/flow 4.3e-5. Two real effects sit in that spread and neither is a bug:
    # the flow arms carry a target of variance ~1.35 (v = eps - x0) against the eps arms'
    # 1.0, and the DiT memorises a fixed batch ~100x slower than the UNet. So the assert is
    # RELATIVE -- final loss against the same cell's step-0 loss -- which is the scale-free
    # version of the claim actually under test: the forward process and its target agree
    # well enough that the pair can be driven to zero.
    x = gpu_dataset()[:32]
    floors = {}
    for b, o in itertools.product(BACKBONES, OBJECTIVES):
        torch.manual_seed(0)
        net = build(b)
        den = Denoiser(net, o)
        opt = torch.optim.AdamW(net.parameters(), 2e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.overfit_steps)
        first = None
        for _ in range(a.overfit_steps):
            gg = torch.Generator(DEV).manual_seed(0)      # SAME (t, eps) draw every step
            loss = den.loss(x, gen=gg)
            first = first if first is not None else loss.item()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        rel = loss.item() / first
        floors[f"{b}/{o}"] = (loss.item(), first, rel)
        assert rel < a.overfit_tol, \
            f"{b}/{o} cannot overfit one fixed batch: {loss.item():.3e} " \
            f"= {rel:.1e} of its step-0 loss {first:.3e}"
    print("overfit floors (final, step0, ratio): "
          + "  ".join(f"{k} {v[0]:.1e}/{v[1]:.2f}={v[2]:.1e}" for k, v in floors.items()))
    print(f"selfcheck OK  params unet={pu} dit={pd} ({abs(pd-pu)/pu*100:.1f}% apart), "
          f"per-pixel leak {leak:.1e} (vs {mixleak:.1e} with attention on), "
          f"FM/DDPM marginal err {worst_x:.1e}, eps adapter {worst_e:.1e} "
          f"(device nondeterminism floor {nd:.1e}), "
          f"eta=1 err {err:.1e}, euler err {err2:.1e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selfcheck", "probe", "train", "sweep", "agg", "gif"])
    p.add_argument("--backbone", nargs="+", default=list(BACKBONES), choices=BACKBONES)
    p.add_argument("--objective", nargs="+", default=list(OBJECTIVES), choices=OBJECTIVES)
    p.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    p.add_argument("--steps", type=int, default=14000)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=None, help="override out/lr.json")
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 2e-4, 5e-4, 1e-3])
    p.add_argument("--n", type=int, default=10000, help="samples per sweep point")
    p.add_argument("--nfe", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--tier", default="headline", choices=list(TIERS))
    p.add_argument("--solver", default="heun", choices=["euler", "heun"])
    p.add_argument("--spacing", default="karras", choices=["t", "karras", "u"])
    p.add_argument("--gif-nfe", type=int, default=16)
    p.add_argument("--overfit-steps", type=int, default=1500)
    p.add_argument("--overfit-tol", type=float, default=1e-3,
               help="final/step-0 loss ratio the fixed-batch overfit must reach")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
