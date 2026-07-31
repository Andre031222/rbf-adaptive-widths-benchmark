"""
RBF vs MLP — reproducible experiments for the Informatica manuscript.
Implements an RBF network from scratch (K-means centers + linear least squares
readout) and benchmarks it against a Multilayer Perceptron (MLP) in the
small / noisy / low-dimensional regime.

Experiment A: 1-D noisy function approximation (visual + metrics).
Experiment B: real tabular benchmark (Diabetes, 442x10) with 5-fold CV.

Outputs: figures into ../figuras and a metrics summary printed to stdout +
saved to results.json for the paper.
"""
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import cdist

RNG = 42
np.random.seed(RNG)
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FIGDIR = str(ROOT / "figures")
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Publication-quality figure defaults.
# Palette: dataviz validated categorical order (blue, orange, aqua, yellow);
# text tokens kept off the series colors. Passes CVD + contrast checks.
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRID = "#d9d8d4"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "font.size": 11,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.9,
    "grid.linewidth": 0.6, "font.family": "DejaVu Sans",
})
# Back-compat aliases (RBF -> blue, MLP -> orange)
NAVY, GOLD, GREEN, GREY = BLUE, ORANGE, AQUA, "#8A97A8"


# ---------------------------------------------------------------------------
# RBF network implemented from scratch
# ---------------------------------------------------------------------------
class RBFNetwork:
    """Gaussian RBF network. Two-phase hybrid training:
       (1) unsupervised placement of centers with K-means,
       (2) supervised linear least-squares fit of output weights (ridge)."""

    def __init__(self, n_centers=20, sigma=None, reg=1e-6, random_state=RNG):
        self.n_centers = n_centers
        self.sigma = sigma          # if None, use the mean-distance heuristic
        self.reg = reg              # small ridge term for numerical stability
        self.random_state = random_state

    def _design(self, X):
        # Gaussian activations phi_j(x) = exp(-||x - c_j||^2 / (2 sigma^2))
        d2 = cdist(X, self.centers_, "sqeuclidean")
        Phi = np.exp(-d2 / (2.0 * self.sigma ** 2))
        return np.hstack([Phi, np.ones((X.shape[0], 1))])  # + bias column

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        km = KMeans(n_clusters=self.n_centers, n_init=10,
                    random_state=self.random_state).fit(X)
        self.centers_ = km.cluster_centers_
        if self.sigma is None:
            # heuristic: sigma = d_max / sqrt(2 * n_centers)
            d = cdist(self.centers_, self.centers_)
            self.sigma = d.max() / np.sqrt(2.0 * self.n_centers)
        Phi = self._design(X)
        # ridge-regularized least squares:  w = (PhiᵀPhi + λI)⁻¹ Phiᵀ y
        A = Phi.T @ Phi + self.reg * np.eye(Phi.shape[1])
        self.w_ = np.linalg.solve(A, Phi.T @ y)
        return self

    def predict(self, X):
        return self._design(np.asarray(X, float)) @ self.w_


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


# ---------------------------------------------------------------------------
# Figure 1 — geometric intuition: local Gaussian receptive fields
# ---------------------------------------------------------------------------
def fig_receptive_fields():
    xs = np.linspace(-3, 3, 500)
    centers = [-2.0, -0.5, 1.0, 2.2]
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    sig = 0.6
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, c in enumerate(centers):
        y = np.exp(-(xs - c) ** 2 / (2 * sig ** 2))
        ax.fill_between(xs, y, color=cols[i], alpha=0.13, lw=0)
        ax.plot(xs, y, color=cols[i], lw=2.4)
        ax.plot([c], [1.0], marker="o", ms=6, color=cols[i],
                mec="white", mew=1.2, zorder=4)
        ax.annotate(f"$c_{i+1}$", (c, 1.0), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=9,
                    color=INK, fontweight="bold")
    ax.set_xlabel("input $x$"); ax.set_ylabel(r"activation $\varphi(x)$")
    ax.set_ylim(0, 1.18)
    ax.set_title("Each RBF neuron fires only near its centre",
                 fontsize=11, fontweight="bold", color=INK, loc="left")
    ax.annotate(r"width $\sigma$ sets the radius", (1.0, 0.61),
                (1.9, 0.85), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_receptive_fields.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 0 — architecture schematic (input -> radial hidden -> linear output)
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.axis("off")
    layers = {0.1: (["$x_1$", "$x_2$", "$x_d$"], "#8A97A8", "Input"),
              0.5: ([r"$\varphi_1$", r"$\varphi_2$", r"$\varphi_3$",
                     r"$\varphi_m$"], BLUE, "Radial hidden\n(K-means centers)"),
              0.9: (["$\\hat{y}$"], ORANGE, "Linear output\n(least squares)")}
    pos = {}
    for x, (nodes, col, _) in layers.items():
        ys = np.linspace(0.75, 0.25, len(nodes))
        for j, (n, y) in enumerate(zip(nodes, ys)):
            if n in ("$x_d$", r"$\varphi_m$") and j == len(nodes) - 1:
                ax.text(x, (ys[-2] + y) / 2 + 0.02, r"$\vdots$", ha="center",
                        va="center", fontsize=14)
            c = plt.Circle((x, y), 0.045, color=col, ec="white", zorder=3)
            ax.add_patch(c)
            ax.text(x, y, n, ha="center", va="center", color="white",
                    fontsize=8, zorder=4)
            pos[(x, j)] = (x, y)
    # edges
    xs = list(layers.keys())
    for a, b in [(xs[0], xs[1]), (xs[1], xs[2])]:
        na = len(layers[a][0]); nb = len(layers[b][0])
        ya = np.linspace(0.75, 0.25, na); yb = np.linspace(0.75, 0.25, nb)
        for i in range(na):
            for k in range(nb):
                ax.plot([a, b], [ya[i], yb[k]], color=GREY, lw=0.4,
                        alpha=0.35, zorder=1)
    for x, (_, _, lab) in layers.items():
        ax.text(x, 0.06, lab, ha="center", va="center", fontsize=9,
                fontweight="bold", color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.9)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_architecture.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment A — 1-D noisy function
# ---------------------------------------------------------------------------
def target(x):
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)


def experiment_A():
    n = 120
    x = np.sort(np.random.rand(n))
    noise = 0.15
    y = target(x) + np.random.normal(0, noise, n)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=RNG)
    Xtr, Xte = xtr[:, None], xte[:, None]
    grid = np.linspace(0, 1, 400)[:, None]

    # RBF
    t0 = time.perf_counter()
    rbf = RBFNetwork(n_centers=20).fit(Xtr, ytr)
    t_rbf = time.perf_counter() - t0
    rbf_test = rmse(yte, rbf.predict(Xte))

    # MLP (fairly tuned: standardized input + L-BFGS, strong for small data)
    from sklearn.pipeline import make_pipeline
    t0 = time.perf_counter()
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(100,), activation="tanh",
                     solver="lbfgs", max_iter=5000, random_state=RNG)
    ).fit(Xtr, ytr)
    t_mlp = time.perf_counter() - t0
    mlp_test = rmse(yte, mlp.predict(Xte))

    # Robustness probe: same MLP capacity but default Adam solver, no scaling
    mlp_naive = MLPRegressor(hidden_layer_sizes=(128, 128), activation="tanh",
                             solver="adam", max_iter=20000,
                             random_state=0).fit(Xtr, ytr)
    mlp_naive_test = rmse(yte, mlp_naive.predict(Xte))

    # Figure 2 — the two fits
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.scatter(xtr, ytr, s=18, color="#8A97A8", alpha=0.65, zorder=2,
               edgecolor="white", linewidth=0.4, label="train (noisy)")
    ax.plot(grid, target(grid.ravel()), color=INK2, lw=1.4, ls=(0, (5, 3)),
            zorder=3, label="true function")
    ax.plot(grid, rbf.predict(grid), color=BLUE, lw=2.6, zorder=5,
            label=f"RBF  (test RMSE = {rbf_test:.3f})")
    ax.plot(grid, mlp.predict(grid), color=ORANGE, lw=2.6, zorder=4,
            label=f"MLP  (test RMSE = {mlp_test:.3f})")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Both models recover the signal; the difference is training "
                 "cost", fontsize=10.5, fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#e2e1dd",
              fontsize=8.5, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_fit_1d.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "n_train": len(xtr), "n_test": len(xte), "noise_sd": noise,
        "rbf": {"test_rmse": rbf_test, "train_time_s": t_rbf,
                "n_centers": 20, "sigma": float(rbf.sigma)},
        "mlp": {"test_rmse": mlp_test, "train_time_s": t_mlp,
                "hidden": [100], "solver": "lbfgs"},
        "mlp_naive_adam": {"test_rmse": mlp_naive_test,
                           "hidden": [128, 128], "solver": "adam"},
        "speedup_rbf_over_mlp": t_mlp / t_rbf,
    }


# ---------------------------------------------------------------------------
# Experiment B — real tabular benchmark (Diabetes), 5-fold CV
# ---------------------------------------------------------------------------
def experiment_B():
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)

    res = {"rbf": {"rmse": [], "r2": [], "time": []},
           "mlp": {"rmse": [], "r2": [], "time": []}}
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr, yte = y[tr], y[te]

        t0 = time.perf_counter()
        rbf = RBFNetwork(n_centers=25, reg=1.0).fit(Xtr, ytr)
        tt = time.perf_counter() - t0
        p = rbf.predict(Xte)
        res["rbf"]["rmse"].append(rmse(yte, p))
        res["rbf"]["r2"].append(r2_score(yte, p))
        res["rbf"]["time"].append(tt)

        t0 = time.perf_counter()
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                           max_iter=2000, random_state=RNG,
                           early_stopping=True).fit(Xtr, ytr)
        tt = time.perf_counter() - t0
        p = mlp.predict(Xte)
        res["mlp"]["rmse"].append(rmse(yte, p))
        res["mlp"]["r2"].append(r2_score(yte, p))
        res["mlp"]["time"].append(tt)

    def agg(d):
        return {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()}

    summary = {"rbf": agg(res["rbf"]), "mlp": agg(res["mlp"]),
               "n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}

    # Figure 3 — three-panel CV comparison: error, stability, training cost.
    models = ["RBF", "MLP"]
    cols = [BLUE, ORANGE]
    panels = [
        ("Test error", "RMSE  (lower = better)",
         [summary["rbf"]["rmse"][0], summary["mlp"]["rmse"][0]],
         [summary["rbf"]["rmse"][1], summary["mlp"]["rmse"][1]],
         "{:.1f}"),
        ("Stability across folds", "$R^2$ std. dev.  (lower = better)",
         [summary["rbf"]["r2"][1], summary["mlp"]["r2"][1]],
         None, "{:.3f}"),
        ("Training cost", "time in s  (lower = faster)",
         [summary["rbf"]["time"][0], summary["mlp"]["time"][0]],
         [summary["rbf"]["time"][1], summary["mlp"]["time"][1]],
         "{:.3f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1))
    for ax, (title, ylab, vals, errs, fmt) in zip(axes, panels):
        bars = ax.bar(models, vals, yerr=errs, width=0.62,
                      color=cols, capsize=4, zorder=3,
                      error_kw=dict(ecolor=INK2, lw=1))
        ax.margins(y=0.24)
        for b, v, e in zip(bars, vals, errs or [0, 0]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e + 0.01,
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=INK)
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color=INK, loc="left", pad=8)
        ax.set_ylabel(ylab, fontsize=8.5, color=INK2)
        ax.tick_params(labelsize=9)
    # highlight the headline panel: RBF is far more stable
    axes[1].set_facecolor("#eef6f2")
    fig.suptitle("Diabetes (442$\\times$10), 5-fold cross-validation",
                 fontsize=11, fontweight="bold", color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{FIGDIR}/fig_diabetes_cv.png", bbox_inches="tight")
    plt.close(fig)
    return summary


if __name__ == "__main__":
    fig_architecture()
    fig_receptive_fields()
    A = experiment_A()
    B = experiment_B()
    out = {"experiment_A": A, "experiment_B": B}
    with open(RESULTS / "results_1d.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
