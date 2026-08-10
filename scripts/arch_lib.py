"""Shared 2D-observation architecture pieces (NoiseSchedule, embeddings,
GRU history encoder, both denoiser variants) + a generic DDIM/PF-ODE batch
encoder that handles both v-prediction and eps-prediction models, so the
per-dataset detect scripts don't redefine these."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class HistoryEncoder2D(nn.Module):
    """GRU history encoder for raw 2D observations (ewi / spirals / heart)."""
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


class DenoiserResidualLN(nn.Module):
    """v-prediction, FiLM, pre-norm residual w/ LayerNorm (ewi, heart)."""
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


class DenoiserPlainFiLM(nn.Module):
    """eps-prediction, FiLM, NO norm / NO residual skip (spirals)."""
    def __init__(self, d=2, hidden=128, context_dim=64, n_layers=4):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(hidden)
        self.input_proj = nn.Linear(d, hidden)
        self.time_proj = nn.Linear(hidden, hidden)
        self.film_proj = nn.Linear(context_dim, 2 * hidden * n_layers)
        self.n_layers = n_layers
        self.hidden = hidden
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden, d)
    def forward(self, x, t, ctx):
        h = self.input_proj(x) + self.time_proj(self.time_emb(t))
        film = self.film_proj(ctx)
        scales, shifts = film.chunk(2, dim=-1)
        scales = scales.reshape(-1, self.n_layers, self.hidden)
        shifts = shifts.reshape(-1, self.n_layers, self.hidden)
        for i, layer in enumerate(self.layers):
            h = F.silu(layer(h))
            h = h * (1 + scales[:, i, :]) + shifts[:, i, :]
        return self.output_proj(h)


@torch.no_grad()
def ddim_encode_batch_v(X0, ctx, denoiser, sched, n_steps, device, clip_x=8.0):
    """v-prediction PF-ODE forward encode (ewi, heart, mnist)."""
    denoiser.eval()
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


@torch.no_grad()
def ddim_encode_batch_eps(X0, ctx, denoiser, sched, n_steps, device):
    """eps-prediction PF-ODE forward encode (spirals)."""
    denoiser.eval()
    N = X0.shape[0]
    x = X0.float().to(device)
    ctx_exp = ctx.unsqueeze(0).expand(N, -1)
    idx = torch.linspace(0, sched.T - 1, n_steps + 1).long().to(device)
    for k in range(n_steps):
        t_c, t_n = idx[k], idx[k + 1]
        t_vec = torch.full((N,), t_c.item(), dtype=torch.long, device=device)
        eps_hat = denoiser(x, t_vec, ctx_exp)
        mu_c, sig_c = sched.mus[t_c], sched.sigmas[t_c]
        mu_n, sig_n = sched.mus[t_n], sched.sigmas[t_n]
        x0_pred = (x - sig_c * eps_hat) / mu_c
        x = mu_n * x0_pred + sig_n * eps_hat
    return x.cpu().numpy()
