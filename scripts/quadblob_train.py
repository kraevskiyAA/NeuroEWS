"""Third illustrative 2D density pair, replacing Spirals (whose chirality-
flip signal turned out to be intrinsically weak -- see spirals_v2_train.py,
a much bigger model barely moved delta2). This one is a different visual
and structural story from both EWI (mixture rotation) and Blob->Ring
(radial hollowing):

  p0: four isotropic Gaussian clusters at the corners of a square,
      (+-2, +-2), std=0.35 each -- a scattered, four-cluster crowd.
  p1: a single isotropic Gaussian blob at the origin, std=1.2 -- the
      crowd coalesces into one cluster.

Both are smooth (no hard edges/uniform boxes -- unlike the checkerboard
attempt, which a 3000-step budget could not learn well), so this should
train reliably, same as EWI's already-multimodal mixture.

Architecture: v-prediction, LayerNorm-residual (ewi/heart/blobring family),
3000 steps.
"""
import time, math, pickle, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

t_script0 = time.time()
torch.manual_seed(42)
device = torch.device('cpu')
print(f'device: {device}')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, '..', 'checkpoints', 'quadblob')
os.makedirs(CKPT_DIR, exist_ok=True)

CORNER = 2.0
BLOB_STD = 0.35
CENTER_STD = 1.2

def sample_p0(n):
    """Four blobs at the corners of a square."""
    signs = torch.randint(0, 2, (n, 2)).float() * 2 - 1   # +-1
    centers = signs * CORNER
    return centers + torch.randn(n, 2) * BLOB_STD

def sample_p1(n):
    """Single blob at the center."""
    return torch.randn(n, 2) * CENTER_STD

def generate_series(B, L, t_burn=1):
    tau = torch.randint(t_burn + 1, L - t_burn, (B,))
    t_idx = torch.arange(L)[None, :]
    before = (t_idx < tau[:, None]).float()[:, :, None]
    x0_all = sample_p0(B * L).reshape(B, L, 2)
    x1_all = sample_p1(B * L).reshape(B, L, 2)
    return before * x0_all + (1 - before) * x1_all, tau

class NoiseSchedule:
    def __init__(self, T=500, beta_min=0.0001, beta_max=0.02):
        self.T = T
        self.betas = torch.linspace(beta_min, beta_max, T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)
        self.mus = self.alpha_bars.sqrt()
        self.sigmas = (1 - self.alpha_bars).sqrt()
    def get(self, t):
        if t.dim() > 1: return self.mus[t], self.sigmas[t]
        return self.mus[t, None], self.sigmas[t, None]
    def to(self, dev):
        for a in ('betas', 'alphas', 'alpha_bars', 'mus', 'sigmas'):
            setattr(self, a, getattr(self, a).to(dev))
        return self

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer('freqs', freqs)
    def forward(self, t):
        x = t.float().unsqueeze(-1) * self.freqs
        return torch.cat([x.sin(), x.cos()], dim=-1)

class HistoryEncoder(nn.Module):
    def __init__(self, d_obs=2, hidden=64):
        super().__init__()
        self.gru = nn.GRU(d_obs, hidden, batch_first=True)
        self.hidden_dim = hidden
    def encode_sequence(self, x):
        B = x.shape[0]
        out, _ = self.gru(x)
        zeros = torch.zeros(B, 1, self.hidden_dim, device=x.device)
        return torch.cat([zeros, out[:, :-1, :]], dim=1)
    def encode_prefix(self, x):
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        _, h = self.gru(x)
        return h.squeeze(0)

class ConditionalDenoiser(nn.Module):
    def __init__(self, d=2, hidden=128, context_dim=64, n_layers=4):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(hidden)
        self.input_proj = nn.Linear(d, hidden)
        self.time_proj = nn.Linear(hidden, hidden)
        self.film_proj = nn.Linear(context_dim, 2 * hidden * n_layers)
        self.n_layers = n_layers
        self.hidden = hidden
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.output_norm = nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, d)
    def forward(self, x, t, ctx):
        h = self.input_proj(x) + self.time_proj(self.time_emb(t))
        film = self.film_proj(ctx)
        scales, shifts = film.chunk(2, dim=-1)
        scales = scales.reshape(-1, self.n_layers, self.hidden)
        shifts = shifts.reshape(-1, self.n_layers, self.hidden)
        for i, layer in enumerate(self.layers):
            h_block = F.silu(layer(self.norms[i](h)))
            h_block = h_block * (1 + scales[:, i, :]) + shifts[:, i, :]
            h = h + h_block
        return self.output_proj(self.output_norm(h))

D, L, T_BURN = 2, 256, 50
HIDDEN_ENC, HIDDEN_DEN, N_LAYERS = 64, 128, 4
T_DIFF, BETA_MIN, BETA_MAX = 500, 0.0001, 0.02
BATCH_SERIES, N_STEPS = 32, 3000
LR_MAX, LR_MIN, WARMUP_STEPS, EMA_DECAY = 3e-4, 3e-6, 100, 0.995

encoder = HistoryEncoder(d_obs=D, hidden=HIDDEN_ENC).to(device)
denoiser = ConditionalDenoiser(d=D, hidden=HIDDEN_DEN, context_dim=HIDDEN_ENC, n_layers=N_LAYERS).to(device)
sched = NoiseSchedule(T_DIFF, BETA_MIN, BETA_MAX).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(denoiser.parameters()), lr=LR_MAX)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(WARMUP_STEPS, 1)
    prog = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return (LR_MIN + (LR_MAX - LR_MIN) * cos) / LR_MAX

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
ema_encoder = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
ema_denoiser = {k: v.detach().clone() for k, v in denoiser.state_dict().items()}

@torch.no_grad()
def ema_update(shadow, model, decay):
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            shadow[k].mul_(decay).add_(v, alpha=1 - decay)
        else:
            shadow[k].copy_(v)

print('Training (v-prediction, EMA, cosine LR)...')
t0 = time.time()
losses = []
for step in range(N_STEPS):
    x, _ = generate_series(BATCH_SERIES, L, t_burn=T_BURN)
    x = x.to(device)
    ctx = encoder.encode_sequence(x)
    B = x.shape[0]
    x_flat = x.reshape(B * L, D)
    ctx_flat = ctx.reshape(B * L, HIDDEN_ENC)
    s = torch.randint(0, T_DIFF, (B * L,), device=device)
    eps = torch.randn(B * L, D, device=device)
    mu_s, sigma_s = sched.get(s)
    x_noisy = mu_s * x_flat + sigma_s * eps
    v_target = mu_s * eps - sigma_s * x_flat
    v_hat = denoiser(x_noisy, s, ctx_flat)
    loss = F.mse_loss(v_hat, v_target)
    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
    ema_update(ema_encoder, encoder, EMA_DECAY)
    ema_update(ema_denoiser, denoiser, EMA_DECAY)
    losses.append(loss.item())
    if (step + 1) % 500 == 0:
        print(f'  step {step+1:4d}/{N_STEPS}  loss: {np.mean(losses[-100:]):.4f}  elapsed: {time.time()-t0:.0f}s')

encoder.load_state_dict(ema_encoder)
denoiser.load_state_dict(ema_denoiser)
print(f'Training done in {time.time()-t0:.0f}s.')

torch.save(encoder.state_dict(), f'{CKPT_DIR}/encoder.pt')
torch.save(denoiser.state_dict(), f'{CKPT_DIR}/denoiser.pt')

@torch.no_grad()
def ddim_encode_batch(X0, ctx, n_steps, clip_x=8.0):
    encoder.eval(); denoiser.eval()
    N = X0.shape[0]
    x = X0.float().to(device)
    ctx_exp = ctx.unsqueeze(0).expand(N, -1)
    idx = torch.linspace(0, sched.T - 1, n_steps + 1).long().to(device)
    for k in range(n_steps):
        t_c, t_n = idx[k], idx[k + 1]
        t_vec = torch.full((N,), t_c.item(), dtype=torch.long, device=device)
        v_hat = denoiser(x, t_vec, ctx_exp)
        mu_c, sig_c = sched.mus[t_c], sched.sigmas[t_c]
        mu_n, sig_n = sched.mus[t_n], sched.sigmas[t_n]
        x0_pred = mu_c * x - sig_c * v_hat
        eps_pred = sig_c * x + mu_c * v_hat
        x = (mu_n * x0_pred + sig_n * eps_pred).clamp(-clip_x, clip_x)
    return x.cpu().numpy()

N_DDIM = sched.T - 1
torch.manual_seed(0)
N_WARMUP_KL = 30
x_warmup_kl = sample_p0(N_WARMUP_KL).to(device)
with torch.no_grad():
    h_fixed_kl = encoder.encode_prefix(x_warmup_kl.unsqueeze(0)).squeeze(0)
np.save(f'{CKPT_DIR}/h_fixed_kl.npy', h_fixed_kl.numpy())
print('DDIM / PF-ODE map ready.')

W_MMD, SIGMA_MMD = 25, np.sqrt(D)

from scipy.spatial.distance import cdist
def mmd2_vs_gaussian(Z_win, sigma=SIGMA_MMD, d=D):
    C = (sigma**2 / (sigma**2 + 2)) ** (d / 2)
    h_B = (sigma**2 / (sigma**2 + 1)) ** (d / 2)
    sq = cdist(Z_win, Z_win, 'sqeuclidean')
    termA = np.exp(-sq / (2 * sigma**2)).mean()
    termB = 2 * (h_B * np.exp(-np.sum(Z_win**2, 1) / (2*(sigma**2+1)))).mean()
    return termA - termB + C

C_CONST = (SIGMA_MMD**2 / (SIGMA_MMD**2 + 2)) ** (D / 2)
MU0 = (1 - C_CONST) / W_MMD

rng = np.random.default_rng(0)
torch.manual_seed(99)
N_POOL_P1 = 3000
Z_p1_pool = ddim_encode_batch(sample_p1(N_POOL_P1), torch.from_numpy(np.load(f'{CKPT_DIR}/h_fixed_kl.npy')), N_DDIM)
N_PILOT = 500
mmd2_p1_list = [mmd2_vs_gaussian(Z_p1_pool[rng.choice(N_POOL_P1, W_MMD, replace=False)]) for _ in range(N_PILOT)]
S_p1 = np.array(mmd2_p1_list) - MU0
print(f"pilot p1: delta2={S_p1.mean():.5f} v1={S_p1.std():.5f}")

with open(f'{CKPT_DIR}/meta.pkl', 'wb') as f:
    pickle.dump(dict(D=D, W_MMD=W_MMD, SIGMA_MMD=SIGMA_MMD, N_DDIM=N_DDIM,
                      T_DIFF=T_DIFF, BETA_MIN=BETA_MIN, BETA_MAX=BETA_MAX,
                      HIDDEN_ENC=HIDDEN_ENC, HIDDEN_DEN=HIDDEN_DEN, N_LAYERS=N_LAYERS,
                      S_p1=S_p1), f)
print(f'Checkpoint saved to {CKPT_DIR}')
print(f'Total script time: {time.time()-t_script0:.0f}s')
