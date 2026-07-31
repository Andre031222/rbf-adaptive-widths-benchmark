"""Rigorous multi-model statistical comparison over multiple datasets
(Demsar, 2006): Friedman omnibus test, Nemenyi post-hoc with a critical-
difference diagram, and Holm-corrected Wilcoxon signed-rank tests for the
targeted claims. Reads results_benchmark.json; writes stats_full.json and
fig_cd_diagram.png.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures"

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, VIOLET, GREEN = "#2a78d6", "#eb6834", "#4a3aa7", "#008300"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "font.size": 11,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans",
})


def load_matrix():
    d = json.load(open(RESULTS / "results_benchmark.json"))
    models = d["models"]
    datasets = list(d["rmse"])
    M = pd.DataFrame({m: [d["rmse"][ds][m] for ds in datasets] for m in models},
                     index=datasets)
    return d, M


def holm(pairs):
    order = sorted(pairs, key=lambda kv: kv[1])
    k = len(order)
    out, prev = {}, 0.0
    for i, (name, p) in enumerate(order):
        adj = min(1.0, max(prev, (k - i) * p))
        out[name] = adj
        prev = adj
    return out


def cd_diagram(avg_ranks, nemenyi):
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    palette = {m: "#3a3a3a" for m in avg_ranks.index}
    sp.critical_difference_diagram(
        avg_ranks, nemenyi, ax=ax,
        label_props={"fontsize": 9.5},
        marker_props={"s": 55, "color": "#3a3a3a"},
        color_palette=palette)
    ax.set_title("Critical-difference diagram (Nemenyi, $\\alpha=0.05$) — "
                 "lower rank = better", fontsize=9.5, fontweight="bold",
                 color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_cd_diagram.png", bbox_inches="tight")
    plt.close(fig)


def main():
    d, M = load_matrix()
    models = list(M.columns)
    n_datasets = len(M)

    stat, p = friedmanchisquare(*[M[m].values for m in models])
    friedman = {"chi2": float(stat), "p": float(p), "k_models": len(models),
                "n_datasets": n_datasets}

    nemenyi = sp.posthoc_nemenyi_friedman(M.values)
    nemenyi.index = nemenyi.columns = models
    avg_ranks = M.rank(axis=1).mean(axis=0)          # lower = better
    cd_diagram(avg_ranks, nemenyi)

    raw = {}
    raw["RBF_vs_RBF-A"] = wilcoxon(M["RBF"], M["RBF-A"]).pvalue
    raw["RBF_vs_KRR"] = wilcoxon(M["RBF"], M["KRR"]).pvalue
    raw["RBF-A_vs_MLP"] = wilcoxon(M["RBF-A"], M["MLP"]).pvalue
    raw["RBF-A_vs_Ridge"] = wilcoxon(M["RBF-A"], M["Ridge"]).pvalue
    raw["KRR_vs_SVR"] = wilcoxon(M["KRR"], M["SVR"]).pvalue
    raw["KRR_vs_RF"] = wilcoxon(M["KRR"], M["RF"]).pvalue
    holm_adj = holm(list(raw.items()))

    out = {
        "friedman": friedman,
        "avg_ranks": {m: float(avg_ranks[m]) for m in models},
        "nemenyi_critical_pairs": {
            f"{a}|{b}": float(nemenyi.loc[a, b])
            for a in models for b in models if a < b},
        "wilcoxon_holm": {k: {"p_raw": float(raw[k]), "p_holm": float(holm_adj[k])}
                          for k in raw},
        "median_error_ratio_RBF_over_RBF-A":
            float(np.median(M["RBF"].values / M["RBF-A"].values)),
    }
    json.dump(out, open(RESULTS / "stats_full.json", "w"), indent=2)

    print(f"Friedman chi2={stat:.2f}, p={p:.2e} (k={len(models)}, N={n_datasets})")
    print("\nAverage ranks (lower=better):")
    for m in avg_ranks.sort_values().index:
        print(f"  {m:8s} {avg_ranks[m]:.2f}")
    print("\nWilcoxon (Holm-adjusted):")
    for k in raw:
        print(f"  {k:18s} p_raw={raw[k]:.4f}  p_holm={holm_adj[k]:.4f}")


if __name__ == "__main__":
    main()
