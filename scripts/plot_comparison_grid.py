"""Combine the four per-dataset comparison figures (Section see plot_comparisons.py)
into a single 4x4 grid: rows = datasets, columns = methods (Ours / A / B / C)."""
import pickle, os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = [
    ('compare_ewi_rotation.pkl', 'Gaussian mixture\nrotation'),
    ('compare_four_clusters_to_one.pkl', 'Four clusters\n→ one'),
    ('compare_blob_to_ring.pkl', 'Blob → Ring'),
    ('compare_mnist_0_to_1.pkl', 'MNIST\ndigit 0 → 1'),
]
METHODS = ['ours', 'A', 'B', 'C']
COLORS = dict(ours='purple', A='teal', B='darkorange', C='brown')
COL_TITLES = dict(
    ours='Ours\n(PF-ODE + MMD$^2$ + SR)',
    A='A: raw MMD$^2$\n(self-norm + SR)',
    B='B: mean-embedding\n(PF-ODE + SR)',
    C='C: Hotelling + CUSUM\n(chi$^2$ null)',
)

fig, axes = plt.subplots(4, 4, figsize=(18, 14))

for row, (pkl_name, row_label) in enumerate(DATASETS):
    pkl_path = os.path.join(RESULTS_DIR, pkl_name)
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    TAU, L, W_MMD = d['TAU'], d['L'], d['W_MMD']
    t_valid = np.arange(W_MMD, L)

    for col, key in enumerate(METHODS):
        ax = axes[row, col]
        r = d['results'][key]
        vals = r['M_vals']
        thr = r['logA']
        that = r['tau_hat']
        color = COLORS[key]

        ax.axvspan(W_MMD, TAU, alpha=0.06, color='steelblue', zorder=0)
        ax.axvspan(TAU, L, alpha=0.06, color='tomato', zorder=0)
        ax.plot(t_valid, vals, color=color, lw=1.1, zorder=3)
        if thr is not None and np.isfinite(thr):
            ax.axhline(thr, color='red', ls='--', lw=1.1, alpha=0.85, zorder=2)
        ax.axvline(TAU, color='red', ls='--', lw=1.1, zorder=4)

        if that is None:
            delay_txt = 'not detected'
            title_color = '#6b6572'
        else:
            ax.axvline(that, color='green', ls=':', lw=1.2, zorder=5)
            ax.plot([that], [vals[that - W_MMD]], 'o', color='green', ms=5, zorder=6)
            delay = that - TAU
            delay_txt = f'delay={delay:+d}'
            title_color = '#2e7d4f' if delay >= 0 else '#b23b3b'

        ax.set_title(delay_txt, fontsize=9, color=title_color, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)

        if row == 0:
            ax.annotate(COL_TITLES[key], xy=(0.5, 1.38), xycoords='axes fraction',
                        ha='center', va='bottom', fontsize=10.5, fontweight='bold')
        if col == 0:
            ax.set_ylabel(row_label, rotation=0, ha='right', va='center',
                           labelpad=8, fontsize=10.5, fontweight='bold')
        if row == 3:
            ax.set_xlabel('t', fontsize=8)

plt.tight_layout(rect=[0.03, 0, 1, 0.95])
fig.suptitle('Full method vs. baselines — all four experiments (equal-footing calibration)',
             fontsize=14, y=0.985)
out_path = os.path.join(FIGURES_DIR, 'fig_compare_grid.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved', out_path)
