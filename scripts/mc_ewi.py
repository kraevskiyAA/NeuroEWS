"""Monte-Carlo operating-characteristics study for the Gaussian mixture
rotation experiment: 1000 independent test series, changepoint tau ~
Uniform(W_MMD, L) (never before the burn-in window closes), all four
methods calibrated ONCE (same equal-footing rule as compare_ewi.py -- see
that file's docstring) and then run against every trial via the
vectorised run_batch methods. See mc_lib.py for the shared harness.
"""
import pickle, time, os, sys
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from arch_lib import HistoryEncoder2D, DenoiserResidualLN, NoiseSchedule, ddim_encode_batch_v
from sr_lib import Detector
from baselines_lib import (GenericSRDetector, median_heuristic_sigma,
                            make_two_sample_mmd_stat, mean_embedding_stat,
                            RollingNormalizedSRDetector, RawCUSUMDetector)
import mc_lib

CKPT_DIR = os.path.join(SCRIPT_DIR, '..', 'checkpoints', 'ewi')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device('cpu')

with open(f'{CKPT_DIR}/calib.pkl', 'rb') as f:
    C = pickle.load(f)
D, W_MMD, SIGMA_MMD = C['D'], C['W_MMD'], C['SIGMA_MMD']
HIDDEN_ENC, HIDDEN_DEN, N_LAYERS = C['HIDDEN_ENC'], C['HIDDEN_DEN'], C['N_LAYERS']
T_DIFF, BETA_MIN, BETA_MAX, N_DDIM = C['T_DIFF'], C['BETA_MIN'], C['BETA_MAX'], C['N_DDIM']

encoder = HistoryEncoder2D(d_obs=D, hidden=HIDDEN_ENC).to(device)
denoiser = DenoiserResidualLN(d=D, hidden=HIDDEN_DEN, context_dim=HIDDEN_ENC, n_layers=N_LAYERS).to(device)
sched = NoiseSchedule(T_DIFF, BETA_MIN, BETA_MAX).to(device)
encoder.load_state_dict(torch.load(f'{CKPT_DIR}/encoder.pt', map_location=device))
denoiser.load_state_dict(torch.load(f'{CKPT_DIR}/denoiser.pt', map_location=device))
encoder.eval(); denoiser.eval()
h_fixed_kl = torch.from_numpy(np.load(f'{CKPT_DIR}/h_fixed_kl.npy'))

NAME, SLUG = 'Gaussian mixture rotation', 'ewi_rotation'

def sample_p0(n):
    g1 = torch.randn(n, 2) + torch.tensor([-2., 0.])
    g2 = torch.randn(n, 2) + torch.tensor([2., 0.])
    m = torch.round(torch.rand(n, 1))
    return m * g1 + (1 - m) * g2

def sample_p1(n):
    g1 = torch.randn(n, 2) + torch.tensor([0., -2.])
    g2 = torch.randn(n, 2) + torch.tensor([0., 2.])
    m = torch.round(torch.rand(n, 1))
    return m * g1 + (1 - m) * g2

def sample_p1_np(n): return sample_p1(n).numpy()

def ddim(X0, ctx):
    return ddim_encode_batch_v(X0, ctx, denoiser, sched, N_DDIM, device)

TAU_VIS, L_VIS = 100, 200
HORIZON = L_VIS - W_MMD
P_FA = 0.05
N_TRIALS = 1000
t_all0 = time.time()

# ============================================================ Calibration (once) ==
torch.manual_seed(0)
burnin_sample = sample_p0(W_MMD).numpy()   # the one historical burn-in the detector is set up from

print('=== Calibrating Ours (exact null) ===')
det_ours = Detector(D, W_MMD, SIGMA_MMD)
torch.manual_seed(99)
Z_p1_pool = ddim(sample_p1(3000), h_fixed_kl)
rng = np.random.default_rng(0)
S_p1_ours = np.array([det_ours.mmd2(Z_p1_pool[rng.choice(3000, W_MMD, replace=False)]) for _ in range(500)])
delta2_ours, v1_ours, alpha_ours = det_ours.fit_pilot(S_p1_ours)
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
pilot_pool_A = sample_p1_np(3000)
rng2 = np.random.default_rng(1)
S_p1_A_raw = np.array([stat_A(pilot_pool_A[rng2.choice(3000, W_MMD, replace=False)]) for _ in range(500)])
delta2_A, v1_A, alpha_A = det_A.fit_pilot(S_p1_A_raw)
t0 = time.time()
logA_A = det_A.calibrate_bootstrap(burnin_sample, stat_A, W_MMD, HORIZON, n_sim=250, p_fa=P_FA)
print(f'  sigma={sigma_A:.3f}  pilot delta2(std)={delta2_A:.3f}  log(1+A*)={logA_A:.3f}  ({time.time()-t0:.0f}s)')

print('=== Calibrating B (exact null, mean-embedding) ===')
null_sampler_B = lambda rng_, w: rng_.standard_normal((w, D))
det_B = GenericSRDetector(mean_embedding_stat, null_sampler_B, W_MMD, rng_seed=2, n_null=1500)
rng3 = np.random.default_rng(2)
S_p1_B = np.array([mean_embedding_stat(Z_p1_pool[rng3.choice(3000, W_MMD, replace=False)]) - det_B.MU0
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

# ============================================================ 1000-trial run ======
print(f'\n=== Generating {N_TRIALS} trials, tau ~ Uniform({W_MMD}, {L_VIS}) ===')
t0 = time.time()
hist_batch, taus = mc_lib.generate_trials(sample_p0, sample_p1, N_TRIALS, L_VIS, D,
                                           tau_min=W_MMD, tau_max=L_VIS, seed=1000)
print(f'  generated  ({time.time()-t0:.0f}s)')

t0 = time.time()
Z_batch = mc_lib.encode_batch(hist_batch, h_fixed_kl, ddim)
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
