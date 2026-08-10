"""Monte-Carlo operating-characteristics study for MNIST digit 0 -> 1 (real
data). See mc_ewi.py / mc_lib.py for the shared harness and equal-footing
rule; see compare_mnist.py for the single-trajectory version this extends
to 1000 independent trials with tau ~ Uniform(W_MMD, L). Uses the same
reduced-budget (6,000-step) checkpoint as the paper's real-data section.
"""
import pickle, time, os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from sr_lib import Detector
from baselines_lib import (GenericSRDetector, median_heuristic_sigma,
                            make_two_sample_mmd_stat, mean_embedding_stat,
                            RollingNormalizedSRDetector, RawCUSUMDetector)
import mc_lib

CKPT_DIR = os.path.join(SCRIPT_DIR, '..', 'checkpoints', 'mnist')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device('cpu')

with open(f'{CKPT_DIR}/meta.pkl', 'rb') as f:
    C = pickle.load(f)
D = C['D']; W_MMD = C['W_MMD']; SIGMA_MMD = C['SIGMA_MMD']; N_DDIM = C['N_DDIM']
T_DIFF = C['T_DIFF']; D_SIZE = C['D_SIZE']; D_IMG = C['D_IMG']
L = C['L']; T_BURN = C['T_BURN']; EMBED_DIM = C['EMBED_DIM']; HIDDEN_ENC = C['HIDDEN_ENC']
S_p1_saved = C['S_p1']

betas = torch.linspace(1e-4, 0.02, T_DIFF).to(device)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, 0)
mus = alpha_bars.sqrt()
sigmas = (1 - alpha_bars).sqrt()

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

encoder = HistoryEncoder(EMBED_DIM, HIDDEN_ENC).to(device)
denoiser = ConditionalDenoiser(D_IMG, context_dim=HIDDEN_ENC).to(device)
encoder.load_state_dict(torch.load(f'{CKPT_DIR}/encoder.pt', map_location=device))
denoiser.load_state_dict(torch.load(f'{CKPT_DIR}/denoiser.pt', map_location=device))
encoder.eval(); denoiser.eval()
h_fixed = torch.from_numpy(np.load(f'{CKPT_DIR}/h_fixed_kl.npy'))

NAME, SLUG = 'MNIST digit 0 -> 1', 'mnist_0_to_1'

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
def sample_p1_np(n): return sample_p1(n).numpy()

CLIP_X = 8.0
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

def ddim(X0, ctx):
    return ddim_encode_batch(X0, ctx)

TAU_VIS, L_VIS = 62, 125
HORIZON = L_VIS - W_MMD
P_FA = 0.05
N_POOL = 1500
N_TRIALS = 1000
t_all0 = time.time()

torch.manual_seed(0)
burnin_sample = sample_p0(W_MMD).numpy()

print('=== Calibrating Ours (exact null) ===')
det_ours = Detector(D, W_MMD, SIGMA_MMD)
delta2_ours, v1_ours, alpha_ours = det_ours.fit_pilot(S_p1_saved)
torch.manual_seed(99)
Z_p1_pool = ddim(sample_p1(N_POOL), h_fixed)
t0 = time.time()
logA_ours = det_ours.calibrate_fast(HORIZON, n_sim=250, p_fa=P_FA)
print(f'  pilot delta2={delta2_ours:.5f}  log(1+A*)={logA_ours:.3f}  ({time.time()-t0:.0f}s)')

print('=== Calibrating A (self-norm, bootstrap of burn-in) ===')
sigma_A = median_heuristic_sigma(burnin_sample)
stat_A = make_two_sample_mmd_stat(burnin_sample, sigma_A)
det_A = RollingNormalizedSRDetector(roll_window=60, min_roll=10)
rng_seed_A = np.random.default_rng(1)
seed_u_A = [stat_A(burnin_sample[rng_seed_A.integers(0, W_MMD, W_MMD)]) for _ in range(20)]
det_A.seed(seed_u_A)
pilot_pool_A = sample_p1_np(N_POOL)
rng2 = np.random.default_rng(1)
S_p1_A_raw = np.array([stat_A(pilot_pool_A[rng2.choice(N_POOL, W_MMD, replace=False)]) for _ in range(500)])
delta2_A, v1_A, alpha_A = det_A.fit_pilot(S_p1_A_raw)
t0 = time.time()
logA_A = det_A.calibrate_bootstrap(burnin_sample, stat_A, W_MMD, HORIZON, n_sim=250, p_fa=P_FA)
print(f'  sigma={sigma_A:.3f}  pilot delta2(std)={delta2_A:.3f}  log(1+A*)={logA_A:.3f}  ({time.time()-t0:.0f}s)')

print('=== Calibrating B (exact null, mean-embedding) ===')
null_sampler_B = lambda rng_, w: rng_.standard_normal((w, D))
det_B = GenericSRDetector(mean_embedding_stat, null_sampler_B, W_MMD, rng_seed=2, n_null=1500)
rng3 = np.random.default_rng(2)
S_p1_B = np.array([mean_embedding_stat(Z_p1_pool[rng3.choice(N_POOL, W_MMD, replace=False)]) - det_B.MU0
                   for _ in range(500)])
delta2_B, v1_B, alpha_B = det_B.fit_pilot(S_p1_B)
t0 = time.time()
logA_B = det_B.calibrate(HORIZON, n_sim=250, p_fa=P_FA)
print(f'  pilot delta2={delta2_B:.5f}  log(1+A*)={logA_B:.3f}  ({time.time()-t0:.0f}s)')

print('=== Calibrating C (chi2 asymptotic null) ===')
det_C = RawCUSUMDetector(burnin_sample, W_MMD, D)
t0 = time.time()
h_C = det_C.calibrate_chi2(HORIZON, n_sim=2000, p_fa=P_FA)
print(f'  slack k={det_C.k:.4f}  threshold h*={h_C:.3f}  ({time.time()-t0:.0f}s)')

print(f'\n=== Generating {N_TRIALS} trials, tau ~ Uniform({W_MMD}, {L_VIS}) ===')
t0 = time.time()
hist_batch, taus = mc_lib.generate_trials(sample_p0, sample_p1, N_TRIALS, L_VIS, D,
                                           tau_min=W_MMD, tau_max=L_VIS, seed=1000)
print(f'  generated  ({time.time()-t0:.0f}s)')

t0 = time.time()
Z_batch = mc_lib.encode_batch(hist_batch, h_fixed, ddim, chunk=8000)
print(f'  encoded ({time.time()-t0:.0f}s)')

t0 = time.time()
tau_hat_ours = det_ours.run_batch(Z_batch, logA_ours)
print(f'  Ours run_batch ({time.time()-t0:.0f}s)')

t0 = time.time()
tau_hat_A = det_A.run_batch(hist_batch, burnin_sample, sigma_A, W_MMD, logA_A)
print(f'  A run_batch ({time.time()-t0:.0f}s)')

t0 = time.time()
tau_hat_B = det_B.run_batch(Z_batch, logA_B)
print(f'  B run_batch ({time.time()-t0:.0f}s)')

t0 = time.time()
tau_hat_C = det_C.run_batch(hist_batch, h_C)
print(f'  C run_batch ({time.time()-t0:.0f}s)')

metrics = dict(
    ours=mc_lib.compute_metrics(tau_hat_ours, taus),
    A=mc_lib.compute_metrics(tau_hat_A, taus),
    B=mc_lib.compute_metrics(tau_hat_B, taus),
    C=mc_lib.compute_metrics(tau_hat_C, taus),
)
print()
mc_lib.print_metrics_table([
    ('Ours (PF-ODE + MMD$^2$ + SR)', metrics['ours']),
    ('A: raw MMD$^2$, self-norm + SR', metrics['A']),
    ('B: PF-ODE mean-embedding + SR', metrics['B']),
    ('C: raw Hotelling + CUSUM (chi2 null)', metrics['C']),
])

out = dict(name=NAME, slug=SLUG, D=D, W_MMD=W_MMD, L=L_VIS, p_fa=P_FA,
           n_trials=N_TRIALS, tau_min=W_MMD, tau_max=L_VIS,
           thresholds=dict(ours=logA_ours, A=logA_A, B=logA_B, C=h_C),
           metrics=metrics)
out_path = os.path.join(RESULTS_DIR, f'mc_{SLUG}.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)
print(f'\nSaved {out_path}')
print(f'Total time: {time.time()-t_all0:.0f}s')
