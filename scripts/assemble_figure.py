"""Assemble the combined cross-dataset detection figure (English labels) for
insertion into full_text.txt's Experiments section, AND save each dataset's
own single-column figure separately. MNIST is excluded here on purpose --
it belongs in a separate real-data section.
"""
import pickle, os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
OUT_COMBINED = os.path.join(FIGURES_DIR, 'fig_synthetic_experiments.png')

# (result file, panel title, output filename slug used for fig_{slug}.png)
datasets = [
    ('result_ewi.pkl',      'Gaussian mixture rotation',      'ewi_rotation'),
    ('result_quadblob.pkl', 'Four clusters → one cluster',    'four_clusters_to_one'),
    ('result_blobring.pkl', 'Blob → Ring',                    'blob_to_ring'),
]

loaded = []
for fn, title, slug in datasets:
    path = os.path.join(RESULTS_DIR, fn)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            loaded.append((title, slug, pickle.load(f)))
    else:
        print(f'skip (not ready): {fn}')

def draw_row0(ax0, title, R):
    if R['kind'] == 'scatter':
        p0e, p1e = R['p0_examples'], R['p1_examples']
        ax0.scatter(p0e[:, 0], p0e[:, 1], s=4, alpha=0.35, color='steelblue', label='$p_0$ (pre-change)')
        ax0.scatter(p1e[:, 0], p1e[:, 1], s=4, alpha=0.35, color='tomato', label='$p_1$ (post-change)')
        lim = max(np.abs(p0e).max(), np.abs(p1e).max()) * 1.15
        ax0.set_xlim(-lim, lim); ax0.set_ylim(-lim, lim); ax0.set_aspect('equal')
        ax0.axhline(0, color='gray', lw=0.4, alpha=0.4); ax0.axvline(0, color='gray', lw=0.4, alpha=0.4)
        ax0.legend(fontsize=8, loc='upper right', markerscale=2)
        ax0.set_xticks([]); ax0.set_yticks([])
    else:
        D_SIZE = R['D_SIZE']
        ex0, ex1 = R['example_0s'], R['example_1s']
        n_show = 4
        grid0 = np.concatenate([ex0[i].reshape(D_SIZE, D_SIZE) for i in range(n_show)], axis=1)
        grid1 = np.concatenate([ex1[i].reshape(D_SIZE, D_SIZE) for i in range(n_show)], axis=1)
        grid = np.concatenate([grid0, grid1], axis=0)
        grid = (grid.clip(-1, 1) + 1) / 2
        ax0.imshow(grid, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax0.axhline(D_SIZE - 0.5, color='red', lw=1.2)
        ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values():
            sp.set_visible(False)
    ax0.set_title(title, fontsize=12, fontweight='bold')

def draw_row1(ax1, R, legend=False):
    TAU, L, W_MMD = R['TAU'], R['L'], R['W_MMD']
    S_tilde, delta2 = R['S_tilde'], R['delta2']
    t_valid = np.arange(W_MMD, L)
    ax1.axvspan(W_MMD, TAU, alpha=0.07, color='steelblue', zorder=0)
    ax1.axvspan(TAU, L, alpha=0.07, color='tomato', zorder=0)
    ax1.plot(t_valid, S_tilde[W_MMD:], lw=1.1, color='darkorange', zorder=3,
              label=r'$\tilde S_t$')
    ax1.axhline(0, color='steelblue', ls=':', lw=1.1, alpha=0.9, zorder=2,
                label=r'$\mathbb{E}[\tilde S\mid H_0]=0$')
    ax1.axhline(delta2, color='tomato', ls='--', lw=1.1, alpha=0.9, zorder=2,
                label=fr'$\delta^2\approx{delta2:.4f}$')
    ax1.axvline(TAU, color='red', ls='--', lw=1.6, zorder=5, label=fr'$\tau={TAU}$')
    ax1.set_xlim(0, L - 1)
    if legend:
        ax1.set_ylabel(r'$\tilde S_t = \widehat{\mathrm{MMD}}^2_t - \mu_0$', fontsize=10)
    ax1.set_title(fr'$\delta^2 \approx {delta2:.4f}$   ($w={W_MMD}$)', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelsize=8)
    if legend:
        ax1.legend(fontsize=7.5, loc='upper left')

def draw_row2(ax2, R, legend=False):
    TAU, L, W_MMD = R['TAU'], R['L'], R['W_MMD']
    logR, tau_hat, logA_star = R['logR'], R['tau_hat'], R['logA_star']
    p_fa, horizon = R['p_fa'], R['horizon']
    t_valid = np.arange(W_MMD, L)
    ax2.axvspan(W_MMD, TAU, alpha=0.07, color='steelblue', zorder=0)
    ax2.axvspan(TAU, L, alpha=0.07, color='tomato', zorder=0)
    ax2.plot(t_valid, logR[W_MMD:], color='purple', lw=1.1, zorder=3,
              label=r'$\log(1+R_t)$')
    ax2.axhline(logA_star, color='red', ls='--', lw=1.6, alpha=0.9, zorder=2,
                label=fr'threshold $\log(1+A^*)={logA_star:.2f}$')
    ax2.axvline(TAU, color='red', ls='--', lw=1.6, zorder=5, label=fr'$\tau={TAU}$')
    delay_txt = 'not detected'
    if tau_hat is not None:
        ax2.axvline(tau_hat, color='green', ls=':', lw=1.6, zorder=5,
                    label=fr'$\hat\tau={tau_hat}$ (delay {tau_hat - TAU})')
        ax2.plot([tau_hat], [logR[tau_hat]], 'o', color='green', ms=8, zorder=6)
        delay_txt = f'delay = {tau_hat - TAU}'
    ax2.set_xlim(0, L - 1)
    ax2.set_xlabel('$t$', fontsize=10)
    if legend:
        ax2.set_ylabel(r'$\log(1+R_t)$', fontsize=10)
    ax2.set_title(f'{delay_txt}   ($P_{{FA}}\\leq{p_fa*100:.0f}\\%$ / {horizon}w)', fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelsize=8)
    if legend:
        ax2.legend(fontsize=7.5, loc='upper left')

# ---- Combined figure ----
n = len(loaded)
fig, axes = plt.subplots(3, n, figsize=(5.2 * n, 10.5))
if n == 1:
    axes = axes.reshape(3, 1)
for j, (title, slug, R) in enumerate(loaded):
    draw_row0(axes[0, j], title, R)
    draw_row1(axes[1, j], R, legend=True)
    draw_row2(axes[2, j], R, legend=True)

fig.text(0.005, 0.83, 'Example\nsamples', fontsize=10, rotation=90, va='center', ha='center')
fig.text(0.005, 0.50, 'Rolling MMD$^2$\n(PF-ODE latent)', fontsize=10, rotation=90, va='center', ha='center')
fig.text(0.005, 0.17, 'Shiryaev–Roberts\nstatistic', fontsize=10, rotation=90, va='center', ha='center')
plt.tight_layout(rect=[0.02, 0, 1, 1])
plt.savefig(OUT_COMBINED, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {OUT_COMBINED}  ({n} datasets)')

# ---- Individual per-dataset figures ----
for title, slug, R in loaded:
    fig, axes = plt.subplots(3, 1, figsize=(6, 11))
    draw_row0(axes[0], title, R)
    draw_row1(axes[1], R, legend=True)
    draw_row2(axes[2], R, legend=True)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f'fig_{slug}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')
