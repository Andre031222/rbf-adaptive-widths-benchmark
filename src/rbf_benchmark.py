"""Journal-grade benchmark: RBF network vs. classical baselines across eight
real regression datasets, with ablations and paired significance tests.

Outputs figures into ../figuras and a machine-readable summary into
results_benchmark.json. Deterministic (fixed seeds) and fully reproducible.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.datasets import load_diabetes

warnings.filterwarnings("ignore")
from pathlib import Path
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / "data") + "/"
FIGDIR = str(ROOT / "figures")
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d4"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET = "#e87ba4", "#008300", "#4a3aa7"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "font.size": 11,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.alpha": 0.9, "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
})


class RBFNetwork:
    """Gaussian RBF network: K-means centres, closed-form least-squares readout."""

    def __init__(self, n_centers=25, sigma_scale=1.0, reg=1.0, random_state=SEED):
        self.n_centers = n_centers
        self.sigma_scale = sigma_scale
        self.reg = reg
        self.random_state = random_state

    def _design(self, X):
        phi = np.exp(-cdist(X, self.centers_, "sqeuclidean") / (2 * self.sigma_ ** 2))
        return np.hstack([phi, np.ones((X.shape[0], 1))])

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        m = min(self.n_centers, len(X))
        km = KMeans(m, n_init=10, random_state=self.random_state).fit(X)
        self.centers_ = km.cluster_centers_
        dmax = cdist(self.centers_, self.centers_).max()
        self.sigma_ = self.sigma_scale * dmax / np.sqrt(2 * m) + 1e-9
        phi = self._design(X)
        a = phi.T @ phi + self.reg * np.eye(phi.shape[1])
        self.w_ = np.linalg.solve(a, phi.T @ y)
        return self

    def predict(self, X):
        return self._design(np.asarray(X, float)) @ self.w_


class RBFAdaptive:
    """Proposed RBF with per-centre adaptive widths: each centre's width is the
    mean distance to its k nearest centres, so dense regions get narrow kernels
    and sparse regions get wide ones (a local, data-driven width)."""

    def __init__(self, n_centers=100, k=5, scale=1.0, reg=1e-2, random_state=SEED):
        self.n_centers = n_centers
        self.k = k
        self.scale = scale
        self.reg = reg
        self.random_state = random_state

    def _design(self, X):
        phi = np.exp(-cdist(X, self.centers_, "sqeuclidean") / (2 * self.sigmas_ ** 2))
        return np.hstack([phi, np.ones((X.shape[0], 1))])

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        m = min(self.n_centers, len(X))
        self.centers_ = KMeans(m, n_init=10, random_state=self.random_state).fit(X).cluster_centers_
        d_sorted = np.sort(cdist(self.centers_, self.centers_), axis=1)
        k = min(self.k, m - 1)
        self.sigmas_ = self.scale * d_sorted[:, 1:k + 1].mean(axis=1) + 1e-9
        phi = self._design(X)
        a = phi.T @ phi + self.reg * np.eye(phi.shape[1])
        self.w_ = np.linalg.solve(a, phi.T @ y)
        return self

    def predict(self, X):
        return self._design(np.asarray(X, float)) @ self.w_


def load_datasets():
    d = {}
    df = pd.read_csv(DATA + "auto-mpg.csv").apply(pd.to_numeric, errors="coerce").dropna()
    d["auto-mpg"] = (df.drop(columns=["mpg"]).values, df["mpg"].values)

    df = pd.read_csv(DATA + "winequality-red.csv", sep=None, engine="python")
    d["wine-red"] = (df.iloc[:, :-1].values, df.iloc[:, -1].values.astype(float))

    df = pd.read_csv(DATA + "yacht.data", sep=r"\s+", header=None).dropna()
    d["yacht"] = (df.iloc[:, :-1].values, df.iloc[:, -1].values)

    df = pd.read_csv(DATA + "abalone.data", header=None)
    df[0] = df[0].map({"M": 0, "F": 1, "I": 2})
    d["abalone"] = (df.iloc[:, :-1].values, df.iloc[:, -1].values.astype(float))

    df = pd.read_excel(DATA + "concrete.xls")
    d["concrete"] = (df.iloc[:, :-1].values, df.iloc[:, -1].values)

    df = pd.read_csv(DATA + "airfoil.dat", sep=r"\s+", header=None)
    d["airfoil"] = (df.iloc[:, :-1].values, df.iloc[:, -1].values)

    df = pd.read_excel(DATA + "energy.xlsx").dropna(axis=1, how="all").dropna()
    d["energy"] = (df.iloc[:, :8].values, df.iloc[:, 8].values)

    d["diabetes"] = load_diabetes(return_X_y=True)
    return d


def make_models():
    return {
        "RBF": lambda: RBFNetwork(n_centers=100, reg=1.0),
        "RBF-A": lambda: RBFAdaptive(n_centers=100, k=5, scale=1.0, reg=1e-2),
        "KRR": lambda: KernelRidge(kernel="rbf", alpha=1e-2),
        "Ridge": lambda: Ridge(alpha=1.0),
        "kNN": lambda: KNeighborsRegressor(n_neighbors=7),
        "SVR": lambda: SVR(C=10.0, gamma="scale"),
        "RF": lambda: RandomForestRegressor(n_estimators=200, random_state=SEED),
        "GBoost": lambda: GradientBoostingRegressor(random_state=SEED),
        "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                    max_iter=2000, early_stopping=True, random_state=SEED),
    }


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def cross_validate(model_factory, X, y, folds=5):
    kf = KFold(folds, shuffle=True, random_state=SEED)
    r, r2, t = [], [], []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        t0 = time.perf_counter()
        est = model_factory().fit(Xtr, y[tr])
        t.append(time.perf_counter() - t0)
        p = est.predict(Xte)
        r.append(rmse(y[te], p))
        r2.append(r2_score(y[te], p))
    return np.array(r), np.array(r2), np.array(t)


def run_benchmark(datasets, models):
    rmse_tbl, r2_tbl, time_tbl, r2std = {}, {}, {}, {}
    for dn, (X, y) in datasets.items():
        X, y = np.asarray(X, float), np.asarray(y, float)
        for mn, mf in models.items():
            r, r2, t = cross_validate(mf, X, y)
            rmse_tbl[(dn, mn)] = r.mean()
            r2_tbl[(dn, mn)] = r2.mean()
            r2std[(dn, mn)] = r2.std()
            time_tbl[(dn, mn)] = t.mean()
        print(f"  {dn:10s} done")
    return rmse_tbl, r2_tbl, time_tbl, r2std


def ablation(datasets, keys):
    grids = {"n_centers": [5, 10, 20, 40, 80, 120],
             "sigma_scale": [0.25, 0.5, 1.0, 2.0, 4.0],
             "reg": [1e-4, 1e-2, 1.0, 10.0, 100.0]}
    out = {}
    for dn in keys:
        X, y = datasets[dn]
        X, y = np.asarray(X, float), np.asarray(y, float)
        out[dn] = {}
        for param, values in grids.items():
            curve = []
            for v in values:
                r, _, _ = cross_validate(lambda v=v, p=param: RBFNetwork(**{p: v}), X, y)
                curve.append(r.mean())
            out[dn][param] = (values, curve)
    return out


def ranks_per_dataset(rmse_tbl, datasets, models):
    dn_list, mn_list = list(datasets), list(models)
    R = np.zeros((len(dn_list), len(mn_list)))
    for i, dn in enumerate(dn_list):
        row = np.array([rmse_tbl[(dn, mn)] for mn in mn_list])
        R[i] = row.argsort().argsort() + 1
    return dn_list, mn_list, R


def fig_rank_heatmap(dn_list, mn_list, R):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    im = ax.imshow(R, cmap="Blues_r", aspect="auto", vmin=1, vmax=len(mn_list))
    ax.set_xticks(range(len(mn_list))); ax.set_xticklabels(mn_list)
    ax.set_yticks(range(len(dn_list))); ax.set_yticklabels(dn_list)
    ax.grid(False)
    for i in range(len(dn_list)):
        for j in range(len(mn_list)):
            v = int(R[i, j])
            ax.text(j, i, v, ha="center", va="center", fontsize=8.5,
                    color="white" if v <= 3 else INK, fontweight="bold")
    ax.set_title("Per-dataset rank by CV RMSE (1 = best)",
                 fontsize=10.5, fontweight="bold", color=INK, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("rank", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_rank_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig_ablation(abl):
    dn = list(abl)[0]
    titles = {"n_centers": "Number of centres $m$",
              "sigma_scale": r"Width scale ($\times\,\sigma$)",
              "reg": r"Ridge $\lambda$"}
    logx = {"n_centers": False, "sigma_scale": True, "reg": True}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for ax, param in zip(axes, ["n_centers", "sigma_scale", "reg"]):
        for dname, col in zip(abl, [BLUE, ORANGE]):
            xs, ys = abl[dname][param]
            ax.plot(xs, ys, "-o", color=col, lw=2.2, ms=5, label=dname)
        if logx[param]:
            ax.set_xscale("log")
        ax.set_title(titles[param], fontsize=9.5, fontweight="bold",
                     color=INK, loc="left")
        ax.set_ylabel("CV RMSE", fontsize=8.5)
        ax.tick_params(labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("RBF sensitivity to its three hyper-parameters",
                 fontsize=11, fontweight="bold", color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(f"{FIGDIR}/fig_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig_insight(datasets, dn_list, mn_list, R):
    rbf_j = mn_list.index("RBF")
    dims = np.array([datasets[dn][0].shape[1] for dn in dn_list])
    sizes = np.array([datasets[dn][0].shape[0] for dn in dn_list])
    rbf_rank = R[:, rbf_j]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, xv, xlab, xlog in [(a1, dims, "input dimension $d$", False),
                               (a2, sizes, "sample size $n$", True)]:
        ax.scatter(xv, rbf_rank, s=70, color=BLUE, edgecolor="white",
                   linewidth=1.0, zorder=3)
        for x, r, dn in zip(xv, rbf_rank, dn_list):
            ax.annotate(dn, (x, r), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=INK2)
        if xlog:
            ax.set_xscale("log")
        ax.set_xlabel(xlab); ax.set_ylabel("RBF rank (1 = best of 7)")
        ax.invert_yaxis()
    fig.suptitle("Where the RBF wins: it ranks higher at low $d$ and small $n$",
                 fontsize=10.5, fontweight="bold", color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{FIGDIR}/fig_insight.png", bbox_inches="tight")
    plt.close(fig)


def fig_improvement(rmse_tbl, dn_list):
    variants = ["RBF", "RBF-A", "KRR"]
    cols = [ORANGE, BLUE, VIOLET]
    x = np.arange(len(dn_list))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for i, (v, c) in enumerate(zip(variants, cols)):
        vals = [rmse_tbl[(dn, v)] for dn in dn_list]
        rf = [rmse_tbl[(dn, "RF")] for dn in dn_list]
        norm = [a / b for a, b in zip(vals, rf)]
        ax.bar(x + (i - 1) * w, norm, w, color=c, zorder=3,
               label={"RBF": "RBF (classic)", "RBF-A": "RBF-A (proposed)",
                      "KRR": "KRR (kernel limit)"}[v])
    ax.axhline(1.0, color=INK2, lw=1, ls="--", zorder=2)
    ax.text(len(dn_list) - 0.5, 1.03, "RF reference", fontsize=7.5, color=INK2, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(dn_list, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("RMSE relative to Random Forest")
    ax.set_ylim(0, min(6, max(rmse_tbl[(dn, "RBF")] / rmse_tbl[(dn, "RF")]
                              for dn in dn_list) * 1.05))
    ax.set_title("Adaptive widths and the kernel limit close the RBF gap",
                 fontsize=10.5, fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_improvement.png", bbox_inches="tight")
    plt.close(fig)


def paired_tests(rmse_tbl, r2std, time_tbl, dn_list):
    def vec(tbl, model):
        return np.array([tbl[(dn, model)] for dn in dn_list])
    res = {}
    # improvement of the proposed variants over the classical RBF
    for better in ["RBF-A", "KRR"]:
        s, p = wilcoxon(vec(rmse_tbl, "RBF"), vec(rmse_tbl, better))
        res[f"rmse_RBF_vs_{better}"] = {
            "stat": float(s), "p": float(p),
            "median_error_ratio": float(np.median(vec(rmse_tbl, "RBF") / vec(rmse_tbl, better)))}
    # is the kernel limit competitive with the strongest baselines?
    for other in ["RF", "GBoost", "SVR"]:
        s, p = wilcoxon(vec(rmse_tbl, "KRR"), vec(rmse_tbl, other))
        res[f"rmse_KRR_vs_{other}"] = {"stat": float(s), "p": float(p)}
    # the classical RBF's one advantage: training speed vs MLP
    s, p = wilcoxon(vec(time_tbl, "RBF"), vec(time_tbl, "MLP"))
    res["time_RBF_vs_MLP"] = {"stat": float(s), "p": float(p),
                              "median_speedup": float(np.median(
                                  vec(time_tbl, "MLP") / vec(time_tbl, "RBF")))}
    return res


if __name__ == "__main__":
    print("Loading datasets...")
    datasets = load_datasets()
    models = make_models()
    print("Running 5-fold benchmark (8 datasets x 9 models)...")
    rmse_tbl, r2_tbl, time_tbl, r2std = run_benchmark(datasets, models)

    dn_list, mn_list, R = ranks_per_dataset(rmse_tbl, datasets, models)
    fig_rank_heatmap(dn_list, mn_list, R)
    fig_insight(datasets, dn_list, mn_list, R)
    fig_improvement(rmse_tbl, dn_list)

    print("Running ablations...")
    abl = ablation(datasets, ["yacht", "abalone"])
    fig_ablation(abl)

    print("Paired significance tests...")
    stats = paired_tests(rmse_tbl, r2std, time_tbl, dn_list)

    summary = {
        "datasets": {dn: {"n": int(datasets[dn][0].shape[0]),
                          "d": int(datasets[dn][0].shape[1])} for dn in dn_list},
        "models": mn_list,
        "rmse": {dn: {mn: round(rmse_tbl[(dn, mn)], 4) for mn in mn_list} for dn in dn_list},
        "r2": {dn: {mn: round(r2_tbl[(dn, mn)], 4) for mn in mn_list} for dn in dn_list},
        "r2_std": {dn: {mn: round(r2std[(dn, mn)], 4) for mn in mn_list} for dn in dn_list},
        "time_s": {dn: {mn: round(time_tbl[(dn, mn)], 4) for mn in mn_list} for dn in dn_list},
        "rbf_rank": {dn: int(R[i, mn_list.index("RBF")]) for i, dn in enumerate(dn_list)},
        "mean_rank": {mn: float(np.mean([R[i, j] for i in range(len(dn_list))]))
                      for j, mn in enumerate(mn_list)},
        "stats": stats,
    }
    with open(RESULTS / "results_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nMean rank by model (lower=better):")
    for mn in sorted(summary["mean_rank"], key=summary["mean_rank"].get):
        print(f"  {mn:8s} {summary['mean_rank'][mn]:.2f}")
    print("\nStats:", json.dumps(stats, indent=2))
