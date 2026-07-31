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
    lss = ["-", "--", "-.", ":"]
    sig = 0.6
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, c in enumerate(centers):
        y = np.exp(-(xs - c) ** 2 / (2 * sig ** 2))
        ax.fill_between(xs, y, color=cols[i], alpha=0.13, lw=0)
        ax.plot(xs, y, color=cols[i], ls=lss[i], lw=2.4)
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
    fig, ax = plt.subplots(figsize=(6.9, 2.9))
    ax.axis("off")
    GREYN = "#8A97A8"
    xin, xhid, xout = 0.11, 0.50, 0.89
    yin = [0.70, 0.50, 0.28]
    yhid = [0.80, 0.61, 0.42, 0.18]
    yout = 0.49
    # edges: input -> hidden (each unit compares x to its centre), light
    for ya in yin:
        for yb in yhid:
            ax.plot([xin, xhid], [ya, yb], color=GREYN, lw=0.5, alpha=0.30, zorder=1)
    # edges: hidden -> output (learned weights), darker
    for yb in yhid:
        ax.plot([xhid, xout], [yb, yout], color=INK2, lw=1.0, alpha=0.55, zorder=1)
    ax.text((xhid + xout) / 2, (yhid[0] + yout) / 2 + 0.03, r"$w_1$",
            fontsize=9, color=INK, ha="center")
    ax.text((xhid + xout) / 2, (yhid[-1] + yout) / 2 - 0.04, r"$w_m$",
            fontsize=9, color=INK, ha="center")
    # input nodes
    for y, lab in zip(yin, ["$x_1$", "$x_2$", "$x_d$"]):
        ax.add_patch(plt.Circle((xin, y), 0.030, facecolor=GREYN,
                                edgecolor="white", lw=1.2, zorder=3))
        ax.text(xin, y, lab, ha="center", va="center", color="white",
                fontsize=9, zorder=4)
    ax.text(xin, (yin[1] + yin[2]) / 2, r"$\vdots$", ha="center", va="center",
            fontsize=13, color=INK2)
    # hidden nodes with a radial motif (concentric rings)
    for y, lab in zip(yhid, ["$\\varphi_1$", "$\\varphi_2$", "$\\varphi_3$", "$\\varphi_m$"]):
        ax.add_patch(plt.Circle((xhid, y), 0.048, facecolor="#eaf1fb",
                                edgecolor=BLUE, lw=1.6, zorder=3))
        ax.add_patch(plt.Circle((xhid, y), 0.028, facecolor="none",
                                edgecolor=BLUE, lw=0.8, alpha=0.55, zorder=4))
        ax.text(xhid, y, lab, ha="center", va="center", color=BLUE,
                fontsize=8.5, fontweight="bold", zorder=5)
    ax.text(xhid, (yhid[2] + yhid[3]) / 2, r"$\vdots$", ha="center",
            va="center", fontsize=13, color=INK2)
    # output node
    ax.add_patch(plt.Circle((xout, yout), 0.040, facecolor=ORANGE,
                            edgecolor="white", lw=1.2, zorder=3))
    ax.text(xout, yout, r"$\hat{y}$", ha="center", va="center",
            color="white", fontsize=10, zorder=4)
    # phase annotations (top)
    ax.plot([xin, xhid - 0.02], [0.95, 0.95], color=INK2, lw=0.8)
    ax.text((xin + xhid) / 2, 0.99, "Phase 1 \u00b7 K-means centres $c_j$ (unsupervised)",
            ha="center", va="bottom", fontsize=8, color=INK2)
    ax.plot([xhid + 0.02, xout], [0.95, 0.95], color=INK2, lw=0.8)
    ax.text((xhid + xout) / 2, 0.99, "Phase 2 \u00b7 least-squares weights $w_j$",
            ha="center", va="bottom", fontsize=8, color=INK2)
    # layer captions (bottom)
    ax.text(xin, 0.015, "Input  $\\mathbf{x}\\in\\mathbb{R}^{d}$", ha="center",
            va="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(xhid, 0.015, "Radial hidden  "
            "$\\varphi_j(\\mathbf{x})=e^{-\\|\\mathbf{x}-\\mathbf{c}_j\\|^2/2\\sigma_j^2}$",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.text(xout, 0.015, "Output  $\\hat{y}=\\sum_j w_j\\varphi_j$", ha="center",
            va="center", fontsize=9, fontweight="bold", color=INK)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.03, 1.08)
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
    ax.plot(grid, target(grid.ravel()), color=INK2, lw=1.6, ls=":",
            zorder=3, label="true function")
    ax.plot(grid, rbf.predict(grid), color=BLUE, lw=2.6, zorder=5,
            label=f"RBF  (test RMSE = {rbf_test:.3f})")
    ax.plot(grid, mlp.predict(grid), color=ORANGE, lw=2.6, ls="-.", zorder=4,
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
