"""Resume training the MNIST 0->1 conditional-diffusion + GRU model from
the existing checkpoints/mnist/ weights (currently the 30,000-step run)
for N_STEPS_NEW further steps, extending the cosine LR schedule to a
virtual total of N_STEPS_DONE + N_STEPS_NEW steps rather than restarting
it (so the LR picks up near where the 30k run left off, instead of
spiking back to LR_MAX). Optimizer/EMA state from the previous run was
not checkpointed, so Adam moments restart at zero and the EMA shadow is
re-seeded from the loaded (already-EMA) weights -- a standard, acceptable
simplification for this kind of exploratory continuation.

Progress is printed with flush=True (and the script should be invoked
with `python3 -u`) specifically because the previous run's fully-buffered
stdout (redirected to a file) produced no visible progress until process
exit -- fixed here for real-time monitoring.
"""
import time, math, pickle, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from scipy.spatial.distance import cdist

t_script0 = time.time()
torch.manual_seed(4242)
device = torch.device('cpu')
print(f'device: {device}', flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, '..', 'checkpoints', 'mnist')

N_STEPS_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
with open(f'{CKPT_DIR}/meta.pkl', 'rb') as f:
    _meta_prev = pickle.load(f)
N_STEPS_DONE = _meta_prev['N_STEPS']
N_STEPS_TOTAL = N_STEPS_DONE + N_STEPS_NEW
print(f'Resuming from {N_STEPS_DONE} steps, training {N_STEPS_NEW} more '
      f'(virtual total {N_STEPS_TOTAL})', flush=True)

D_SIZE = 16
D_IMG = D_SIZE ** 2

mnist = datasets.MNIST(root=os.path.join(SCRIPT_DIR, '..', 'data'), train=True, download=True,
                        transform=transforms.ToTensor())
imgs_0_raw = mnist.data[mnist.targets == 0].float() / 255.0
imgs_1_raw = mnist.data[mnist.targets == 1].float() / 255.0

def _resize(t):
    return F.interpolate(t.unsqueeze(1), size=(D_SIZE, D_SIZE), mode='bilinear', align_corners=False).squeeze(1)

imgs_0 = (_resize(imgs_0_raw) * 2 - 1).reshape(-1, D_IMG)
imgs_1 = (_resize(imgs_1_raw) * 2 - 1).reshape(-1, D_IMG)

def sample_p0(n): return imgs_0[torch.randint(len(imgs_0), (n,))]
def sample_p1(n): return imgs_1[torch.randint(len(imgs_1), (n,))]

L, T_BURN = 125, 20

def generate_series(B, L, t_burn=1):
    tau = torch.randint(t_burn + 1, L - t_burn, (B,))
    t_idx = torch.arange(L)[None, :]
    before = (t_idx < tau[:, None]).float()[:, :, None]
    x0_all = sample_p0(B * L).reshape(B, L, D_IMG)
    x1_all = sample_p1(B * L).reshape(B, L, D_IMG)
    return before * x0_all + (1 - before) * x1_all, tau

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer('freqs', freqs)
    def forward(self, t):
        x = t.float().unsqueeze(-1) * self.freqs
        return torch.cat([x.sin(), x.cos()], dim=-1)

class ImageProjection(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_IMG, 256), nn.SiLU(), nn.Linear(256, embed_dim))
    def forward(self, x): return self.net(x)

class HistoryEncoder(nn.Module):
    def __init__(self, embed_dim=32, hidden=128):
        super().__init__()
        self.proj = ImageProjection(embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.hidden_dim = hidden
    def _embed(self, x):
        B, Lx, Dx = x.shape
        return self.proj(x.reshape(B * Lx, Dx)).reshape(B, Lx, -1)
    def encode_sequence(self, x):
        B = x.shape[0]
        out, _ = self.gru(self._embed(x))
        zeros = torch.zeros(B, 1, self.hidden_dim, device=x.device)
        return torch.cat([zeros, out[:, :-1]], dim=1)
    def encode_prefix(self, x):
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        _, h = self.gru(self._embed(x))
        return h.squeeze(0)

class ConditionalDenoiser(nn.Module):
    def __init__(self, d=D_IMG, hidden=256, context_dim=128, n_layers=4, t_dim=128):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(t_dim)
        self.input_proj = nn.Linear(d + t_dim + context_dim, hidden)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.output_norm = nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, d)
    def forward(self, x, t, ctx):
        h = self.input_proj(torch.cat([x, self.time_emb(t), ctx], dim=-1))
        for norm, layer in zip(self.norms, self.layers):
            h = h + F.silu(layer(norm(h)))
        return self.output_proj(self.output_norm(h))

EMBED_DIM, HIDDEN_ENC = 64, 256
T_DIFF = 1000
BATCH_SERIES = 64
LR_MAX, LR_MIN, WARMUP_STEPS, EMA_DECAY = 3e-4, 3e-6, 1000, 0.999

betas = torch.linspace(1e-4, 0.02, T_DIFF).to(device)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, 0)
mus = alpha_bars.sqrt()
sigmas = (1 - alpha_bars).sqrt()

encoder = HistoryEncoder(EMBED_DIM, HIDDEN_ENC).to(device)
denoiser = ConditionalDenoiser(D_IMG, context_dim=HIDDEN_ENC).to(device)
encoder.load_state_dict(torch.load(f'{CKPT_DIR}/encoder.pt', map_location=device))
denoiser.load_state_dict(torch.load(f'{CKPT_DIR}/denoiser.pt', map_location=device))
print('Loaded existing weights.', flush=True)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(denoiser.parameters()), lr=LR_MAX)

def lr_lambda_abs(step):
    """step here is the ABSOLUTE step index into the virtual N_STEPS_TOTAL
    schedule (continues the same cosine curve a single N_STEPS_TOTAL run
    would have used, rather than restarting it)."""
    if step < WARMUP_STEPS:
        return step / max(WARMUP_STEPS, 1)
    prog = (step - WARMUP_STEPS) / max(N_STEPS_TOTAL - WARMUP_STEPS, 1)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return (LR_MIN + (LR_MAX - LR_MIN) * cos) / LR_MAX

ema_encoder = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
ema_denoiser = {k: v.detach().clone() for k, v in denoiser.state_dict().items()}

@torch.no_grad()
def ema_update(shadow, model, decay):
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            shadow[k].mul_(decay).add_(v, alpha=1 - decay)
        else:
            shadow[k].copy_(v)

print(f'Training {N_STEPS_NEW} more steps (no CFG dropout, matching the '
      f'existing checkpoint)...', flush=True)
t0 = time.time()
losses = []
for step_idx in range(N_STEPS_NEW):
    abs_step = N_STEPS_DONE + step_idx
    lr_scale = lr_lambda_abs(abs_step)
    for g in optimizer.param_groups:
        g['lr'] = lr_scale * LR_MAX

    x, _ = generate_series(BATCH_SERIES, L, t_burn=T_BURN)
    x = x.to(device)
    ctx = encoder.encode_sequence(x)
    B = x.shape[0]
    x_flat = x.reshape(B * L, D_IMG)
    ctx_flat = ctx.reshape(B * L, HIDDEN_ENC)
    t = torch.randint(0, T_DIFF, (B * L,), device=device)
    eps = torch.randn_like(x_flat)
    xt = mus[t, None] * x_flat + sigmas[t, None] * eps
    v_target = mus[t, None] * eps - sigmas[t, None] * x_flat
    loss = F.mse_loss(denoiser(xt, t, ctx_flat), v_target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    ema_update(ema_encoder, encoder, EMA_DECAY)
    ema_update(ema_denoiser, denoiser, EMA_DECAY)
    losses.append(loss.item())
    if (step_idx + 1) % 500 == 0:
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'  step {N_STEPS_DONE + step_idx + 1:6d}/{N_STEPS_TOTAL}  '
              f'loss: {np.mean(losses[-200:]):.4f}  lr: {cur_lr:.2e}  '
              f'elapsed: {time.time()-t0:.0f}s', flush=True)

encoder.load_state_dict(ema_encoder)
denoiser.load_state_dict(ema_denoiser)
print(f'Additional training done in {time.time()-t0:.0f}s.', flush=True)

torch.save(encoder.state_dict(), f'{CKPT_DIR}/encoder.pt')
torch.save(denoiser.state_dict(), f'{CKPT_DIR}/denoiser.pt')

CLIP_X = 8.0
N_DDIM = 100

@torch.no_grad()
def ddim_encode_batch(X0, ctx, n_steps=N_DDIM, clip_x=CLIP_X):
    encoder.eval(); denoiser.eval()
    N = X0.shape[0]
    x = X0.float().to(device)
    ctx_exp = ctx.unsqueeze(0).expand(N, -1)
    idx = torch.linspace(0, T_DIFF - 1, n_steps + 1).long().to(device)
    for k in range(n_steps):
        t_c, t_n = idx[k], idx[k + 1]
        t_vec = torch.full((N,), t_c.item(), dtype=torch.long, device=device)
        v_hat = denoiser(x, t_vec, ctx_exp)
        mu_c, sig_c = mus[t_c], sigmas[t_c]
        mu_n, sig_n = mus[t_n], sigmas[t_n]
        x0_pred = mu_c * x - sig_c * v_hat
        eps_pred = sig_c * x + mu_c * v_hat
        x = (mu_n * x0_pred + sig_n * eps_pred).clamp(-clip_x, clip_x)
    return x.cpu().numpy()

torch.manual_seed(0)
N_WARMUP_KL = T_BURN
x_warmup_kl = sample_p0(N_WARMUP_KL).to(device)
with torch.no_grad():
    h_fixed_kl = encoder.encode_prefix(x_warmup_kl.unsqueeze(0)).squeeze(0)
np.save(f'{CKPT_DIR}/h_fixed_kl.npy', h_fixed_kl.numpy())
print('DDIM / PF-ODE map ready.', flush=True)

torch.manual_seed(777)
Z_check = ddim_encode_batch(sample_p0(500), h_fixed_kl)
print(f'Sanity: real p0 latents -- mean(|per-dim mean|)={np.abs(Z_check.mean(axis=0)).mean():.4f} '
      f'(want ~0), mean(per-dim var)={Z_check.var(axis=0).mean():.4f} (want ~1)', flush=True)

W_MMD, SIGMA_MMD = T_BURN, np.sqrt(D_IMG)

def mmd2_vs_gaussian(Z_win, sigma=SIGMA_MMD, d=D_IMG):
    C = (sigma**2 / (sigma**2 + 2)) ** (d / 2)
    h_B = (sigma**2 / (sigma**2 + 1)) ** (d / 2)
    sq = cdist(Z_win, Z_win, 'sqeuclidean')
    termA = np.exp(-sq / (2 * sigma**2)).mean()
    termB = 2 * (h_B * np.exp(-np.sum(Z_win**2, 1) / (2 * (sigma**2 + 1)))).mean()
    return termA - termB + C

C_CONST = (SIGMA_MMD**2 / (SIGMA_MMD**2 + 2)) ** (D_IMG / 2)
MU0 = (1 - C_CONST) / W_MMD

rng = np.random.default_rng(0)
N_POOL_P1 = 1500
Z_p1_pool = ddim_encode_batch(sample_p1(N_POOL_P1), h_fixed_kl)
N_PILOT = 400
mmd2_p1_list = [mmd2_vs_gaussian(Z_p1_pool[rng.choice(N_POOL_P1, W_MMD, replace=False)]) for _ in range(N_PILOT)]
S_p1 = np.array(mmd2_p1_list) - MU0
print(f'pilot p1: delta2={S_p1.mean():.5f}  v1={S_p1.std():.5f}', flush=True)

with open(f'{CKPT_DIR}/meta.pkl', 'wb') as f:
    pickle.dump(dict(
        D=D_IMG, W_MMD=W_MMD, SIGMA_MMD=SIGMA_MMD, N_DDIM=N_DDIM, T_DIFF=T_DIFF,
        D_SIZE=D_SIZE, D_IMG=D_IMG, L=L, T_BURN=T_BURN, EMBED_DIM=EMBED_DIM,
        HIDDEN_ENC=HIDDEN_ENC, N_STEPS=N_STEPS_TOTAL, S_p1=S_p1,
    ), f)

print(f'Checkpoint saved to {CKPT_DIR}  (N_STEPS now {N_STEPS_TOTAL})', flush=True)
print(f'Total script time: {time.time()-t_script0:.0f}s', flush=True)
