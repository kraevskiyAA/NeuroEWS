"""
Shared detection algorithm (Algorithm 1 in full_text.txt): one-sample MMD^2
vs N(0,I), Tartakovsky-Spivak mixture likelihood ratio, Shiryaev-Roberts
recursion in numerically-stable log space, and Approach-B (false-alarm
-probability-controlled) threshold calibration.

Used identically across all four dataset experiments (ewi, spirals,
ring->heart, mnist) so the combined figure compares apples to apples.
"""
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde, norm as sp_norm

LOG_LAMBDA_CAP = 15.0


def mmd2_vs_gaussian(Z_win, sigma, d):
    C = (sigma**2 / (sigma**2 + 2)) ** (d / 2)
    h_B = (sigma**2 / (sigma**2 + 1)) ** (d / 2)
    sq = cdist(Z_win, Z_win, 'sqeuclidean')
    termA = np.exp(-sq / (2 * sigma**2)).mean()
    termB = 2 * (h_B * np.exp(-np.sum(Z_win**2, 1) / (2 * (sigma**2 + 1)))).mean()
    return termA - termB + C


def softplus_stable(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


class Detector:
    """Bundles null calibration (g0 KDE) + Tartakovsky-Spivak mixture prior
    (alpha, v1) + the stable log-space SR recursion for one dataset."""

    def __init__(self, D, W_MMD, SIGMA_MMD, rng_seed=0, N_NULL=5000):
        self.D, self.W_MMD, self.SIGMA_MMD = D, W_MMD, SIGMA_MMD
        self.C_CONST = (SIGMA_MMD**2 / (SIGMA_MMD**2 + 2)) ** (D / 2)
        self.MU0 = (1 - self.C_CONST) / W_MMD
        rng = np.random.default_rng(rng_seed)
        self.null_S = np.array([
            mmd2_vs_gaussian(rng.standard_normal((W_MMD, D)), SIGMA_MMD, D)
            for _ in range(N_NULL)
        ]) - self.MU0
        self.p0_kde = gaussian_kde(self.null_S, bw_method='silverman')
        self.ALPHA_SR = None
        self.V1_SR = None
        self.DELTA2_EST = None

    def fit_pilot(self, S_p1):
        """S_p1: array of MMD^2-mu0 samples under H1 (pooled)."""
        self.DELTA2_EST = float(np.mean(S_p1))
        self.V1_SR = float(np.std(S_p1))
        self.ALPHA_SR = 1.0 / self.DELTA2_EST
        return self.DELTA2_EST, self.V1_SR, self.ALPHA_SR

    def mmd2(self, Z_win):
        return mmd2_vs_gaussian(Z_win, self.SIGMA_MMD, self.D) - self.MU0

    def sr_log_lambda(self, S):
        log_p0 = np.log(max(self.p0_kde(np.array([S]))[0], 1e-300))
        a, v1 = self.ALPHA_SR, self.V1_SR
        log_p1 = (np.log(a) - a * S + 0.5 * a**2 * v1**2
                  + sp_norm.logcdf((S - a * v1**2) / v1))
        return float(np.clip(log_p1 - log_p0, -LOG_LAMBDA_CAP, LOG_LAMBDA_CAP))

    def simulate_null_M_path(self, n_steps, rng_):
        Z = rng_.standard_normal((n_steps + self.W_MMD, self.D))
        M_path = np.empty(n_steps)
        M = 0.0
        for i in range(n_steps):
            t = i + self.W_MMD
            S = mmd2_vs_gaussian(Z[t - self.W_MMD:t], self.SIGMA_MMD, self.D) - self.MU0
            M = softplus_stable(M + self.sr_log_lambda(S))
            M_path[i] = M
        return M_path

    def calibrate(self, horizon, n_sim=1500, p_fa=0.05, seed=123):
        rng_cal = np.random.default_rng(seed)
        paths = np.stack([self.simulate_null_M_path(horizon, rng_cal) for _ in range(n_sim)])
        max_M = paths.max(axis=1)
        logA_star = float(np.quantile(max_M, 1 - p_fa))
        crossed = paths >= logA_star
        first_idx = np.where(crossed.any(axis=1), crossed.argmax(axis=1), horizon)
        arl = float(first_idx.mean())
        return logA_star, arl

    def calibrate_fast(self, horizon, n_sim=1500, p_fa=0.05, seed=123):
        """Numerically equivalent to calibrate(), but ~50-100x faster: the
        n_sim null trajectories are independent, so instead of looping
        n_sim*horizon times with one scipy.stats.gaussian_kde.__call__ per
        step (built for batch queries, not one point at a time), this
        batches all n_sim trajectories together and loops only over
        `horizon`, replacing the KDE call with a hand-rolled vectorised
        Gaussian-kernel sum using the exact same bandwidth scipy would use.
        Only logA_star is returned (no ARL -- add if needed)."""
        rng_cal = np.random.default_rng(seed)
        w, D, sigma = self.W_MMD, self.D, self.SIGMA_MMD
        a, v1 = self.ALPHA_SR, self.V1_SR
        null_S = self.null_S
        n_null = len(null_S)
        h = float(np.sqrt(self.p0_kde.covariance[0, 0]))  # exact scipy bandwidth
        C = (sigma**2 / (sigma**2 + 2)) ** (D / 2)
        h_B = (sigma**2 / (sigma**2 + 1)) ** (D / 2)

        Z = rng_cal.standard_normal((n_sim, horizon + w, D))
        M = np.zeros(n_sim)
        M_max = np.zeros(n_sim)
        for i in range(horizon):
            t = i + w
            Zwin = Z[:, t - w:t, :]                                   # (n_sim, w, D)
            sq_norms = np.sum(Zwin**2, axis=-1)                        # (n_sim, w)
            inner = np.einsum('swd,svd->swv', Zwin, Zwin)              # (n_sim, w, w)
            sq = sq_norms[:, :, None] + sq_norms[:, None, :] - 2 * inner
            termA = np.exp(-sq / (2 * sigma**2)).mean(axis=(1, 2))
            termB = 2 * (h_B * np.exp(-sq_norms / (2 * (sigma**2 + 1)))).mean(axis=-1)
            S = termA - termB + C - self.MU0                           # (n_sim,)

            diffs = (S[:, None] - null_S[None, :]) / h
            g0 = np.exp(-0.5 * diffs**2).sum(axis=1) / (n_null * h * math.sqrt(2 * math.pi))
            log_p0 = np.log(np.maximum(g0, 1e-300))
            log_p1 = (np.log(a) - a * S + 0.5 * a**2 * v1**2
                      + sp_norm.logcdf((S - a * v1**2) / v1))
            loglam = np.clip(log_p1 - log_p0, -LOG_LAMBDA_CAP, LOG_LAMBDA_CAP)
            M = np.maximum(M + loglam, 0.0) + np.log1p(np.exp(-np.abs(M + loglam)))
            M_max = np.maximum(M_max, M)

        return float(np.quantile(M_max, 1 - p_fa))

    def run_batch(self, Z_batch, logA_star):
        """Vectorised sibling of run_series/calibrate_fast for a Monte Carlo
        study over many independent test series at once: Z_batch is
        (n_trials, L, D) latent codes (already PF-ODE-encoded), the
        calibrated threshold is fixed and shared. Returns tau_hat: (n_trials,)
        int array of the first-crossing time index, -1 where the statistic
        never reached logA_star within L. Same closed-form math as
        calibrate_fast, just fed real trajectories instead of standard-normal
        null draws, and tracking first-crossing rather than the running max."""
        n_trials, L, D = Z_batch.shape
        w, sigma = self.W_MMD, self.SIGMA_MMD
        a, v1 = self.ALPHA_SR, self.V1_SR
        null_S = self.null_S
        n_null = len(null_S)
        h = float(np.sqrt(self.p0_kde.covariance[0, 0]))
        C = (sigma**2 / (sigma**2 + 2)) ** (D / 2)
        h_B = (sigma**2 / (sigma**2 + 1)) ** (D / 2)

        M = np.zeros(n_trials)
        tau_hat = np.full(n_trials, -1, dtype=np.int64)
        for t in range(w, L):
            Zwin = Z_batch[:, t - w:t, :]                               # (n, w, D)
            sq_norms = np.sum(Zwin**2, axis=-1)                          # (n, w)
            inner = np.einsum('swd,svd->swv', Zwin, Zwin)
            sq = sq_norms[:, :, None] + sq_norms[:, None, :] - 2 * inner
            termA = np.exp(-sq / (2 * sigma**2)).mean(axis=(1, 2))
            termB = 2 * (h_B * np.exp(-sq_norms / (2 * (sigma**2 + 1)))).mean(axis=-1)
            S = termA - termB + C - self.MU0                             # (n,)

            diffs = (S[:, None] - null_S[None, :]) / h
            g0 = np.exp(-0.5 * diffs**2).sum(axis=1) / (n_null * h * math.sqrt(2 * math.pi))
            log_p0 = np.log(np.maximum(g0, 1e-300))
            log_p1 = (np.log(a) - a * S + 0.5 * a**2 * v1**2
                      + sp_norm.logcdf((S - a * v1**2) / v1))
            loglam = np.clip(log_p1 - log_p0, -LOG_LAMBDA_CAP, LOG_LAMBDA_CAP)
            M = np.maximum(M + loglam, 0.0) + np.log1p(np.exp(-np.abs(M + loglam)))
            newly = (tau_hat < 0) & (M >= logA_star)
            tau_hat[newly] = t
        return tau_hat

    def run_series(self, Z, tau, logA_star):
        """Z: (L,D) latent codes for one test series. Returns S_tilde, M (=
        log(1+R_t)) arrays (NaN before window fills) and tau_hat (or None)."""
        L = Z.shape[0]
        S_tilde = np.full(L, np.nan)
        M_vals = np.full(L, np.nan)
        M = 0.0
        tau_hat = None
        for t in range(self.W_MMD, L):
            S = self.mmd2(Z[t - self.W_MMD:t])
            S_tilde[t] = S
            M = softplus_stable(M + self.sr_log_lambda(S))
            M_vals[t] = M
            if tau_hat is None and M >= logA_star:
                tau_hat = t
        return S_tilde, M_vals, tau_hat
