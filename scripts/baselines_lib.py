"""Baseline detectors compared against the full pipeline (PF-ODE encoder +
one-sample MMD^2 vs N(0,I) + Tartakovsky-Spivak mixture Shiryaev-Roberts,
implemented in sr_lib.Detector). Each baseline removes one or more pieces
of the full method, to isolate what actually drives detection performance:

  A. Two-sample MMD^2, self-normalised SR -- MMD^2 between the current
     window and a reference pool, computed directly on RAW observations,
     no PF-ODE encoder. Unlike "ours", this statistic's null law has no
     known closed form even asymptotically (same generalised-chi-squared
     situation the paper derives for MMD^2 in general -- just without the
     N(0,I) reference that lets "ours" evaluate it exactly). Standardised
     online via a rolling mean/std of its own observed history and
     approximated as N(0,1) -- an honest approximation, not a theorem.

  B. Mean-embedding SR -- SR on ||mean(Z_window)||^2, using the SAME
     PF-ODE-encoded latents as the full method. Under H0, Z ~ N(0,I)
     EXACTLY (the paper's own injectivity result), so this baseline's null
     is just as exactly known as "ours" -- it differs only in throwing
     away everything but the mean.

  C. Raw Hotelling statistic + CUSUM -- squared Mahalanobis distance of
     the raw window mean from the p0 mean. No PF-ODE, no MMD, no
     mixture-SR. Unlike A, this one DOES have a known asymptotic null:
     by the CLT the window mean is approximately Gaussian, so the
     Mahalanobis distance is approximately chi-squared(d) -- classical
     Hotelling's T^2 theory, no Monte Carlo needed for the threshold.

Equal-footing rule used throughout: nothing here ever draws fresh samples
from the *true* p0 generator. Any reference pool or null-calibration data
for A and C comes from the actually-observed burn-in segment of the
current series (or a bootstrap of it) -- exactly the same information
"ours" is allowed (Section~sec:problem: burn-in observations are known to
be from p0, nothing else is assumed known).
"""
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm as sp_norm, chi2 as sp_chi2, gaussian_kde

LOG_LAMBDA_CAP = 15.0


def softplus_stable(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


# ---------------------------------------------------------------------------
# Baseline B only: exact-null mixture-SR engine (identical in spirit to
# sr_lib.Detector -- kept here as a separate class only because its window
# statistic is different (mean-embedding, not full MMD^2), not because its
# null-calibration logic differs. Draws its null directly from N(0,I),
# exactly as "ours" does -- legitimate because that is a theorem, not an
# empirical assumption about data we don't have.
# ---------------------------------------------------------------------------
class GenericSRDetector:
    def __init__(self, stat_fn, null_sampler, w, rng_seed=0, n_null=5000):
        self.stat_fn = stat_fn
        self.null_sampler = null_sampler
        self.w = w
        rng = np.random.default_rng(rng_seed)
        raw_null = np.array([stat_fn(null_sampler(rng, w)) for _ in range(n_null)])
        self.MU0 = float(raw_null.mean())
        self.null_S = raw_null - self.MU0
        self.p0_kde = gaussian_kde(self.null_S, bw_method='silverman')
        self.ALPHA_SR = None
        self.V1_SR = None
        self.DELTA2_EST = None

    def centred_stat(self, window):
        return self.stat_fn(window) - self.MU0

    def fit_pilot(self, S_p1):
        self.DELTA2_EST = float(np.mean(S_p1))
        self.V1_SR = float(np.std(S_p1))
        self.ALPHA_SR = 1.0 / self.DELTA2_EST
        return self.DELTA2_EST, self.V1_SR, self.ALPHA_SR

    def sr_log_lambda(self, S):
        log_p0 = np.log(max(self.p0_kde(np.array([S]))[0], 1e-300))
        a, v1 = self.ALPHA_SR, self.V1_SR
        log_p1 = (np.log(a) - a * S + 0.5 * a**2 * v1**2
                  + sp_norm.logcdf((S - a * v1**2) / v1))
        return float(np.clip(log_p1 - log_p0, -LOG_LAMBDA_CAP, LOG_LAMBDA_CAP))

    def simulate_null_M_path(self, n_steps, rng_):
        M_path = np.empty(n_steps)
        M = 0.0
        for i in range(n_steps):
            win = self.null_sampler(rng_, self.w)
            S = self.stat_fn(win) - self.MU0
            M = softplus_stable(M + self.sr_log_lambda(S))
            M_path[i] = M
        return M_path

    def calibrate(self, horizon, n_sim=800, p_fa=0.05, seed=123):
        rng_cal = np.random.default_rng(seed)
        paths = np.stack([self.simulate_null_M_path(horizon, rng_cal) for _ in range(n_sim)])
        return float(np.quantile(paths.max(axis=1), 1 - p_fa))

    def run_windows(self, windows, logA_star):
        M = 0.0
        M_vals = np.empty(len(windows))
        tau_hat_rel = None
        for i, win in enumerate(windows):
            S = self.centred_stat(win)
            M = softplus_stable(M + self.sr_log_lambda(S))
            M_vals[i] = M
            if tau_hat_rel is None and M >= logA_star:
                tau_hat_rel = i
        return M_vals, tau_hat_rel

    def run_batch(self, Z_batch, logA_star):
        """Vectorised sibling of run_windows for a Monte Carlo study over
        many independent series at once (baseline B: mean-embedding stat
        only). Z_batch: (n_trials, L, D) PF-ODE latents. Returns tau_hat:
        (n_trials,) int array, -1 where never crossed logA_star."""
        n_trials, L, D = Z_batch.shape
        w = self.w
        a, v1 = self.ALPHA_SR, self.V1_SR
        null_S = self.null_S
        n_null = len(null_S)
        h = float(np.sqrt(self.p0_kde.covariance[0, 0]))

        M = np.zeros(n_trials)
        tau_hat = np.full(n_trials, -1, dtype=np.int64)
        for t in range(w, L):
            Zwin = Z_batch[:, t - w:t, :]
            m = Zwin.mean(axis=1)
            S = np.sum(m**2, axis=-1) - self.MU0

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


def mean_embedding_stat(Z_win):
    """||mean(Z_window)||^2. Under H0, Z ~ iid N(0,I_D) within the window,
    so mean(Z_window) ~ N(0, I_D/w) and this statistic ~ (1/w)*chi2(D)
    exactly -- a mean-shift-only analogue of the linear-kernel MMD row in
    Table~tab:kernels of the paper."""
    m = Z_win.mean(axis=0)
    return float(np.sum(m ** 2))


# ---------------------------------------------------------------------------
# Baseline A: two-sample MMD^2 on raw observations, self-normalised online
# (no known null law, no oracle access to p0 -- only the burn-in segment of
# THIS series, via bootstrap for calibration).
# ---------------------------------------------------------------------------
def median_heuristic_sigma(Y_ref):
    d = cdist(Y_ref, Y_ref, 'euclidean')
    iu = np.triu_indices_from(d, k=1)
    return float(np.median(d[iu]))


def make_two_sample_mmd_stat(Y_ref, sigma):
    """Y_ref: (m, d) reference pool -- must come from the burn-in segment of
    the series being monitored, never from an oracle p0 generator."""
    YY = cdist(Y_ref, Y_ref, 'sqeuclidean')
    termC = float(np.exp(-YY / (2 * sigma**2)).mean())

    def stat_fn(X_win):
        XX = cdist(X_win, X_win, 'sqeuclidean')
        XY = cdist(X_win, Y_ref, 'sqeuclidean')
        termA = np.exp(-XX / (2 * sigma**2)).mean()
        termB = np.exp(-XY / (2 * sigma**2)).mean()
        return float(termA - 2 * termB + termC)

    return stat_fn


class RollingNormalizedSRDetector:
    """For a raw-space statistic with NO known null law (not even
    asymptotically): standardise online via a trailing rolling mean/std of
    the statistic's OWN observed history, and approximate the standardised
    value's null distribution as N(0,1). This is a genuine approximation
    (the paper's own derivation shows MMD^2 is generically NOT
    asymptotically Gaussian even when centred correctly) -- it is exactly
    the price of not having "ours"'s exact reference law, made explicit
    rather than hidden behind a KDE fit to data we would not really have.

    Feeds the identical closed-form Tartakovsky-Spivak Gaussian mixture SR
    recursion used everywhere else, just with g0 = phi (standard normal
    density) instead of a KDE.
    """

    def __init__(self, roll_window=60, min_roll=10):
        self.roll_window = roll_window
        self.min_roll = min_roll
        self.buffer = []
        self.mu0 = None    # frozen burn-in baseline, used only to standardise pilot data
        self.sigma0 = None
        self.ALPHA_SR = None
        self.V1_SR = None
        self.DELTA2_EST = None

    def seed(self, burnin_u_values):
        """burnin_u_values: stat_fn evaluated on bootstrap-resampled
        sub-windows of the burn-in segment only."""
        self.buffer = list(burnin_u_values)
        self.mu0 = float(np.mean(burnin_u_values))
        self.sigma0 = float(np.std(burnin_u_values) + 1e-12)

    def _roll_mu_sigma(self):
        tail = self.buffer[-self.roll_window:]
        mu = float(np.mean(tail))
        sigma = float(np.std(tail) + 1e-12)
        return mu, sigma

    def standardize_online(self, u):
        """Online step: standardise u using the CURRENT rolling buffer
        (causal -- only past values), then append u to the buffer."""
        mu, sigma = self._roll_mu_sigma()
        s_std = (u - mu) / sigma
        self.buffer.append(u)
        return s_std

    def fit_pilot(self, S_p1_raw):
        """S_p1_raw: raw (unstandardised) stat_fn values from a pilot pool.
        Standardised against the FROZEN burn-in mu0/sigma0 (not a rolling
        window fit to the pilot batch itself), consistent with how ours/B
        pilot-calibrate against a fixed null reference."""
        S_p1_std = (np.asarray(S_p1_raw) - self.mu0) / self.sigma0
        self.DELTA2_EST = float(np.mean(S_p1_std))
        self.V1_SR = float(np.std(S_p1_std))
        self.ALPHA_SR = 1.0 / self.DELTA2_EST if self.DELTA2_EST > 0 else None
        return self.DELTA2_EST, self.V1_SR, self.ALPHA_SR

    def sr_log_lambda(self, s_std):
        a, v1 = self.ALPHA_SR, self.V1_SR
        log_p0 = -0.5 * s_std**2 - 0.5 * math.log(2 * math.pi)  # log phi(s_std)
        log_p1 = (math.log(a) - a * s_std + 0.5 * a**2 * v1**2
                  + sp_norm.logcdf((s_std - a * v1**2) / v1))
        return float(np.clip(log_p1 - log_p0, -LOG_LAMBDA_CAP, LOG_LAMBDA_CAP))

    def run_online(self, stat_fn, windows):
        """Genuine online run: rolling buffer grows/slides using the ACTUAL
        observed sequence of window statistics from this series, exactly as
        a real deployment would see them (no lookahead)."""
        M = 0.0
        M_vals = np.empty(len(windows))
        tau_hat_rel = None
        for i, win in enumerate(windows):
            u = stat_fn(win)
            s_std = self.standardize_online(u)
            if len(self.buffer) < self.min_roll:
                M_vals[i] = 0.0
                continue
            M = softplus_stable(M + self.sr_log_lambda(s_std))
            M_vals[i] = M
            if tau_hat_rel is None and M >= self._logA_star:
                tau_hat_rel = i
        return M_vals, tau_hat_rel

    def run_batch(self, hist_batch, Y_ref, sigma, w, logA_star):
        """Vectorised sibling of run_online for a Monte Carlo study over many
        independent trials at once. hist_batch: (n_trials, L, d) raw
        observations; Y_ref/sigma define the (fixed, burn-in-derived)
        two-sample MMD^2 statistic shared by every trial. Each trial keeps
        its OWN causal rolling buffer, seeded identically (from the same
        burn-in bootstrap used to fit self.buffer/mu0/sigma0) then diverging
        as soon as each trial's own post-seed data differs -- exactly the
        online behaviour of run_online, just batched."""
        n_trials, L, d = hist_batch.shape
        a, v1 = self.ALPHA_SR, self.V1_SR
        sq_Yref = np.sum(Y_ref**2, axis=-1)
        YY = sq_Yref[:, None] + sq_Yref[None, :] - 2 * np.einsum('id,jd->ij', Y_ref, Y_ref)
        termC = float(np.exp(-YY / (2 * sigma**2)).mean())

        n_seed = len(self.buffer)
        buf = np.empty((n_trials, n_seed + (L - w)))
        buf[:, :n_seed] = np.asarray(self.buffer)[None, :]

        M = np.zeros(n_trials)
        tau_hat = np.full(n_trials, -1, dtype=np.int64)
        col = n_seed
        for t in range(w, L):
            Xwin = hist_batch[:, t - w:t, :]                             # (n, w, d)
            sqX = np.sum(Xwin**2, axis=-1)                               # (n, w)

            innerXX = np.einsum('twd,tvd->twv', Xwin, Xwin)
            sqXX = sqX[:, :, None] + sqX[:, None, :] - 2 * innerXX
            termA = np.exp(-sqXX / (2 * sigma**2)).mean(axis=(1, 2))

            innerXY = np.einsum('twd,md->twm', Xwin, Y_ref)
            sqXY = sqX[:, :, None] + sq_Yref[None, None, :] - 2 * innerXY
            termB = np.exp(-sqXY / (2 * sigma**2)).mean(axis=(1, 2))

            u = termA - 2 * termB + termC                                # (n,)

            tail = buf[:, max(0, col - self.roll_window):col]
            mu = tail.mean(axis=1)
            sig = tail.std(axis=1) + 1e-12
            s_std = (u - mu) / sig
            buf[:, col] = u
            col += 1

            if min(col, self.roll_window) < self.min_roll:
                continue
            log_p0 = -0.5 * s_std**2 - 0.5 * math.log(2 * math.pi)
            log_p1 = (np.log(a) - a * s_std + 0.5 * a**2 * v1**2
                      + sp_norm.logcdf((s_std - a * v1**2) / v1))
            loglam = np.clip(log_p1 - log_p0, -LOG_LAMBDA_CAP, LOG_LAMBDA_CAP)
            M = np.maximum(M + loglam, 0.0) + np.log1p(np.exp(-np.abs(M + loglam)))
            newly = (tau_hat < 0) & (M >= logA_star)
            tau_hat[newly] = t
        return tau_hat

    # ---- Calibration: bootstrap the burn-in segment, never the true p0 ----
    def calibrate_bootstrap(self, burnin_sample, stat_fn, w, horizon, n_sim=250,
                             p_fa=0.05, seed=123):
        """burnin_sample: (T_burn, d) the ACTUAL observed burn-in segment of
        THIS series. Synthetic null sequences are built by resampling
        individual observations from it with replacement (iid model, so
        this is a valid bootstrap of p0) -- never a fresh draw from the true
        generator."""
        rng_cal = np.random.default_rng(seed)
        n_burn = len(burnin_sample)
        maxvals = np.empty(n_sim)
        for s in range(n_sim):
            det = RollingNormalizedSRDetector(self.roll_window, self.min_roll)
            seed_u = [stat_fn(burnin_sample[rng_cal.integers(0, n_burn, w)])
                      for _ in range(max(self.min_roll, 5))]
            det.seed(seed_u)
            det.ALPHA_SR, det.V1_SR = self.ALPHA_SR, self.V1_SR
            M = 0.0
            M_max = 0.0
            for i in range(horizon):
                boot_win = burnin_sample[rng_cal.integers(0, n_burn, w)]
                u = stat_fn(boot_win)
                s_std = det.standardize_online(u)
                if len(det.buffer) < det.min_roll:
                    continue
                M = softplus_stable(M + det.sr_log_lambda(s_std))
                M_max = max(M_max, M)
            maxvals[s] = M_max
        self._logA_star = float(np.quantile(maxvals, 1 - p_fa))
        return self._logA_star


# ---------------------------------------------------------------------------
# Baseline C: Hotelling's T^2 (squared Mahalanobis distance of the window
# mean from a burn-in-estimated p0 mean) + classical one-sided CUSUM. This
# statistic DOES have a known asymptotic null (chi-squared(d), by the CLT on
# the window mean) -- so unlike A, no rolling approximation is needed; the
# threshold comes from the closed-form chi2 distribution, not Monte Carlo.
# ---------------------------------------------------------------------------
class RawCUSUMDetector:
    def __init__(self, burnin_sample, w, d):
        """burnin_sample: (T_burn, d) the actual observed burn-in segment.

        Uses a DIAGONAL covariance (per-dimension variance only), not the
        full d x d matrix: with only T_burn ~ w observations available (the
        burn-in segment, never an oracle pool), a full covariance is
        singular whenever d > T_burn -- exactly the MNIST case (d=256,
        T_burn=20). A diagonal covariance stays well-posed for any d, and
        is if anything MORE in the spirit of "the simplest thing a
        practitioner reaches for first" than a full Mahalanobis distance."""
        self.w = w
        self.d = d
        self.mu0 = burnin_sample.mean(axis=0)
        var0 = burnin_sample.var(axis=0, ddof=1) + 1e-6
        self.inv_var0 = 1.0 / var0
        # Known asymptotic null: w * sum_j (Xbar_j-mu0_j)^2 / var0_j ~ chi2(d)
        # (per-dimension CLT + independence-of-coordinates approximation).
        # CUSUM slack is the null mean of chi2(d), i.e. d itself -- closed-form.
        self.k = float(d)

    def _window_stat(self, X_win):
        xbar = X_win.mean(axis=0) - self.mu0
        # scale by w so the statistic is asymptotically chi2(d) regardless of w
        return float(self.w * np.sum(xbar**2 * self.inv_var0))

    def calibrate_chi2(self, horizon, n_sim=2000, p_fa=0.05, seed=123):
        """Threshold for the running max of a CUSUM fed by iid chi2(d)
        increments (the statistic's known asymptotic null) -- closed-form
        input distribution, Monte Carlo only over the (trivial) CUSUM path,
        no data of any kind needed."""
        rng_cal = np.random.default_rng(seed)
        T = sp_chi2.rvs(self.d, size=(n_sim, horizon), random_state=rng_cal)
        g = np.zeros(n_sim)
        gmax = np.zeros(n_sim)
        for i in range(horizon):
            g = np.maximum(0.0, g + T[:, i] - self.k)
            gmax = np.maximum(gmax, g)
        return float(np.quantile(gmax, 1 - p_fa))

    def run_windows(self, windows, h_star):
        g = 0.0
        g_vals = np.empty(len(windows))
        tau_hat_rel = None
        for i, win in enumerate(windows):
            T = self._window_stat(win)
            g = max(0.0, g + T - self.k)
            g_vals[i] = g
            if tau_hat_rel is None and g >= h_star:
                tau_hat_rel = i
        return g_vals, tau_hat_rel

    def run_batch(self, hist_batch, h_star):
        """Vectorised sibling of run_windows for a Monte Carlo study over
        many independent trials at once. hist_batch: (n_trials, L, d) raw
        observations. Returns tau_hat: (n_trials,) int array, -1 where the
        CUSUM statistic never crossed h_star."""
        n_trials, L, d = hist_batch.shape
        g = np.zeros(n_trials)
        tau_hat = np.full(n_trials, -1, dtype=np.int64)
        for t in range(self.w, L):
            Xwin = hist_batch[:, t - self.w:t, :]
            xbar = Xwin.mean(axis=1) - self.mu0[None, :]
            T = self.w * np.sum(xbar**2 * self.inv_var0[None, :], axis=-1)
            g = np.maximum(0.0, g + T - self.k)
            newly = (tau_hat < 0) & (g >= h_star)
            tau_hat[newly] = t
        return tau_hat
