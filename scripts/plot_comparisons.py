"""Load results/compare_*.pkl (produced by compare_ewi.py / compare_quadblob.py
/ compare_blobring.py / compare_mnist.py) and render the actual comparison
figures + a summary table, since the compare_*.py scripts themselves only
pickle the numbers (the plotting code lives in the notebooks' Section 11,
which was never executed)."""
import pickle, os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = [
    ('compare_ewi_rotation.pkl', 'fig_compare_ewi.png', 'Gaussian mixture rotation'),
    ('compare_four_clusters_to_one.pkl', 'fig_compare_quadblob.png', 'Four clusters → one cluster'),
    ('compare_blob_to_ring.pkl', 'fig_compare_blobring.png', 'Blob → Ring'),
    ('compare_mnist_0_to_1.pkl', 'fig_compare_mnist.png', 'MNIST digit 0 → 1'),
]

COLORS = dict(ours='purple', A='teal', B='darkorange', C='brown')
LABELS = dict(
    ours='Ours (PF-ODE + MMD$^2$ + SR, exact null)',
    A='A: raw MMD$^2$, self-norm + SR (approx. null)',
    B='B: PF-ODE mean-embedding + SR (exact null)',
    C='C: raw Hotelling + CUSUM (chi2 null)',
)

summary_rows = []

for pkl_name, fig_name, title in DATASETS:
    pkl_path = os.path.join(RESULTS_DIR, pkl_name)
    if not os.path.exists(pkl_path):
        print(f'skip (missing): {pkl_path}')
        continue
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    TAU, L, W_MMD = d['TAU'], d['L'], d['W_MMD']
    t_valid = np.arange(W_MMD, L)

    fig, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)
    for ax, key in zip(axes, ['ours', 'A', 'B', 'C']):
        r = d['results'][key]
        vals = r['M_vals']
        thr = r['logA']
        that = r['tau_hat']
        color = COLORS[key]
        ax.axvspan(W_MMD, TAU, alpha=0.06, color='steelblue', zorder=0)
        ax.axvspan(TAU, L, alpha=0.06, color='tomato', zorder=0)
        ax.plot(t_valid, vals, color=color, lw=1.2, zorder=3)
        if thr is not None and np.isfinite(thr):
            ax.axhline(thr, color='red', ls='--', lw=1.4, alpha=0.85, zorder=2,
                       label=f'threshold={thr:.2f}')
        ax.axvline(TAU, color='red', ls='--', lw=1.4, zorder=4, label=f'tau={TAU}')
        delay_txt = 'not detected'
        if that is not None:
            ax.axvline(that, color='green', ls=':', lw=1.4, zorder=5,
                       label=f'tau_hat={that} (delay {that - TAU})')
            ax.plot([that], [vals[that - W_MMD]], 'o', color='green', ms=7, zorder=6)
            delay_txt = f'delay={that - TAU}'
        ax.set_ylabel(LABELS[key], fontsize=9)
        ax.set_title(f'{LABELS[key]}   ({delay_txt})', fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('t')
    plt.suptitle(f'{title}: full method vs. baselines', fontsize=13, y=1.0)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, fig_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

    for key in ['ours', 'A', 'B', 'C']:
        r = d['results'][key]
        summary_rows.append(dict(dataset=title, method=r['name'], delta2=r['delta2'],
                                  threshold=r['logA'], tau=TAU, tau_hat=r['tau_hat'],
                                  delay=r['delay']))

# ---- Summary table (printed + saved as csv) ----
import csv
csv_path = os.path.join(RESULTS_DIR, 'compare_summary.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset', 'method', 'delta2', 'threshold', 'tau', 'tau_hat', 'delay'])
    w.writeheader()
    for row in summary_rows:
        w.writerow(row)
print(f'Saved {csv_path}')

print()
print(f"{'dataset':30s} {'method':38s} {'delta2':>8s} {'threshold':>10s} {'tau_hat':>8s} {'delay':>7s}")
for row in summary_rows:
    d2 = f"{row['delta2']:.4f}" if row['delta2'] is not None else 'n/a'
    thr = f"{row['threshold']:.2f}" if row['threshold'] is not None and np.isfinite(row['threshold']) else 'nan'
    print(f"{row['dataset']:30s} {row['method']:38s} {d2:>8s} {thr:>10s} "
          f"{str(row['tau_hat']):>8s} {str(row['delay']):>7s}")
