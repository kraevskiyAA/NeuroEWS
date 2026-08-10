"""Shared Monte-Carlo operating-characteristics harness, used by
mc_ewi.py / mc_quadblob.py / mc_blobring.py / mc_mnist.py.

For a given dataset, each of the four detectors (Ours, A, B, C) is
calibrated ONCE -- exactly the real deployment regime: a threshold is set
from a false-alarm budget against one observed burn-in, then used
repeatedly, not re-derived for every new episode. Given that fixed
calibration, n_trials independent test series are drawn with the
changepoint tau placed uniformly at random in [tau_min, tau_max) (never
before the burn-in window closes, per the problem's own assumption that
only the burn-in is known to be p0), and every method's ALREADY-CALIBRATED
threshold is run against all of them at once via the vectorised
`run_batch` methods on sr_lib.Detector / baselines_lib's three classes.

Metrics reported, matching the classical sequential-testing triple:
  - false alarm probability: P(the method fires strictly before tau)
  - mean detection delay:    E[tau_hat - tau | fired at or after tau]
  - miss fraction:           P(never fires by the end of the horizon)
"""
import numpy as np
import torch


def generate_trials(sample_p0, sample_p1, n_trials, L, D, tau_min, tau_max, seed=1000):
    """sample_p0/sample_p1: callables (n) -> torch.Tensor (n, D). Returns
    (hist_batch, taus): hist_batch (n_trials, L, D) float32 numpy, taus
    (n_trials,) int array uniform in [tau_min, tau_max)."""
    rng = np.random.default_rng(seed)
    taus = rng.integers(tau_min, tau_max, size=n_trials)
    p0_seq = sample_p0(n_trials * L).numpy().reshape(n_trials, L, D)
    p1_seq = sample_p1(n_trials * L).numpy().reshape(n_trials, L, D)
    t_idx = np.arange(L)[None, :]
    mask = t_idx < taus[:, None]                      # True while still pre-change
    hist_batch = np.where(mask[..., None], p0_seq, p1_seq).astype(np.float32)
    return hist_batch, taus


def encode_batch(hist_batch, ctx, ddim_fn, chunk=20000):
    """Batched PF-ODE encode of every (trial, t) observation at once,
    chunked only to bound peak activation memory -- ctx is the SAME fixed
    context for every sample (the frozen burn-in context, as everywhere
    else in this project), so this is embarrassingly parallel across both
    trials and time. Returns (n_trials, L, D) numpy."""
    n_trials, L, D = hist_batch.shape
    flat = torch.from_numpy(hist_batch.reshape(n_trials * L, D))
    outs = []
    for i in range(0, flat.shape[0], chunk):
        outs.append(ddim_fn(flat[i:i + chunk], ctx))
    Z_flat = np.concatenate(outs, axis=0)
    return Z_flat.reshape(n_trials, L, D)


def compute_metrics(tau_hat, taus):
    """tau_hat, taus: (n_trials,) int arrays; tau_hat == -1 means never
    fired within the horizon."""
    tau_hat = np.asarray(tau_hat)
    taus = np.asarray(taus)
    fired = tau_hat >= 0
    false_alarm = fired & (tau_hat < taus)
    hit = fired & (tau_hat >= taus)
    miss = ~fired
    n = len(taus)
    out = dict(
        n_trials=int(n),
        fa_prob=float(false_alarm.mean()),
        hit_frac=float(hit.mean()),
        miss_frac=float(miss.mean()),
        mean_delay=float((tau_hat[hit] - taus[hit]).mean()) if hit.any() else float('nan'),
        std_delay=float((tau_hat[hit] - taus[hit]).std()) if hit.any() else float('nan'),
    )
    return out


def print_metrics_table(name_metrics):
    """name_metrics: list of (label, metrics_dict)."""
    print(f"{'method':38s} {'P(false alarm)':>15s} {'P(miss)':>9s} {'P(hit)':>8s} "
          f"{'mean delay':>11s} {'std delay':>10s}")
    for label, m in name_metrics:
        md = f"{m['mean_delay']:.2f}" if m['mean_delay'] == m['mean_delay'] else 'n/a'
        sd = f"{m['std_delay']:.2f}" if m['std_delay'] == m['std_delay'] else 'n/a'
        print(f"{label:38s} {m['fa_prob']:15.3f} {m['miss_frac']:9.3f} {m['hit_frac']:8.3f} "
              f"{md:>11s} {sd:>10s}")
