"""Toy demo: tune Lasso α + per-feature length-scales with DP-GIBO vs random search.

Self-contained — no real-world data required. Generates a small synthetic
regression problem, runs both methods, and saves a comparison figure in
NeurIPS style.

Usage:
    python -m lasso_example.run main
    python -m lasso_example.run main --num-iterations 5 --path-length 30 --mu 0.5
"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Sequence

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import Bounds
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Allow running as a script (python lasso_example/run.py) as well as a module.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from algorithms import DPGIBOConfig, run_dp_gibo, run_random_search
from utils.plotting import style


C_DP = "#1f77b4"
C_NONDP = "gray"
C_RAND = "#2ca02c"


# ----------------------------------------------------------------------------
# Lasso objective on a synthetic regression task
# ----------------------------------------------------------------------------


class LassoObjective:
    """Per-test-sample squared-error loss for Lasso.

    θ packs ``[log_lengthscales (n_features), log_alpha]``. Lasso is fit on
    ``X_train / lengthscales`` (per-feature rescaling acts like ARD), then
    evaluated on a held-out test set.
    """

    def __init__(
        self,
        n_features: int = 10,
        n_train: int = 50,
        n_test: int = 500,
        n_informative: int = 5,
        noise: float = 15.0,
        seed: int = 0,
    ):
        X, y = make_regression(
            n_samples=n_train + n_test,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=seed,
        )
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=n_train, test_size=n_test, random_state=seed
        )
        self._sX = StandardScaler().fit(X_tr)
        self._sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        self.X_train = self._sX.transform(X_tr)
        self.X_test = self._sX.transform(X_te)
        self.y_train = self._sy.transform(y_tr.reshape(-1, 1)).ravel()
        self.y_test = self._sy.transform(y_te.reshape(-1, 1)).ravel()

        self.n_features = n_features
        self.bounds = Bounds(
            lb=[-2.0] * n_features + [math.log(1e-4)],
            ub=[2.0] * n_features + [math.log(10.0)],
        )

    @property
    def d(self) -> int:
        return self.n_features + 1

    @property
    def n_users(self) -> int:
        return self.X_test.shape[0]

    def bounds_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        lo = torch.tensor(self.bounds.lb, dtype=torch.get_default_dtype())
        hi = torch.tensor(self.bounds.ub, dtype=torch.get_default_dtype())
        return lo, hi

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        theta_np = theta.detach().cpu().numpy()
        n = theta_np.shape[0]
        losses = torch.zeros(
            (self.X_test.shape[0], n), dtype=torch.float64, device=theta.device
        )
        for i, h in enumerate(theta_np):
            ls = np.exp(h[: self.n_features])
            alpha = float(np.exp(h[-1]))
            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(self.X_train / ls, self.y_train)
            residuals = self.y_test - model.predict(self.X_test / ls)
            losses[:, i] = torch.tensor(0.5 * residuals**2, dtype=torch.float64)
        return losses

    def mean_loss(self, theta: torch.Tensor) -> float:
        return self(theta).mean().item()

    def random_theta(self, n: int = 1, generator: torch.Generator | None = None) -> torch.Tensor:
        lo = torch.tensor(self.bounds.lb, dtype=torch.get_default_dtype())
        hi = torch.tensor(self.bounds.ub, dtype=torch.get_default_dtype())
        u = torch.rand(n, len(lo), generator=generator)
        return u * (hi - lo) + lo


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------


def _plot_comparison(
    dp_curves: np.ndarray,         # (n_runs, T+1)
    nondp_curves: np.ndarray,      # (n_runs, T+1)
    random_curves: np.ndarray,     # (n_runs, n_iter)
    bs: int,
    out_dir: pathlib.Path,
    extensions: Sequence[str],
    usetex: bool = False,
):
    """Save a side-by-side figure: convergence (left) + final-loss strip (right).

    DP-GIBO and GIBO show the loss at the current θ at each outer iteration
    (raw trajectory, so DP noise is visible). Random search shows running-best.
    """
    n_runs = dp_curves.shape[0]

    def _summary(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = curves.mean(axis=0)
        ci = 1.96 * curves.std(axis=0) / math.sqrt(max(n_runs, 1))
        return mean, ci

    mean_dp, ci_dp = _summary(dp_curves)
    mean_nondp, ci_nondp = _summary(nondp_curves)
    mean_rd, ci_rd = _summary(random_curves)

    x_outer = np.arange(0, dp_curves.shape[1]) * bs
    x_rd = np.arange(random_curves.shape[1])

    with plt.rc_context(style.neurips(usetex=usetex, rel_width=0.9, nrows=1.25, ncols=2)):
        fig, axes = plt.subplots(1, 2, squeeze=False, constrained_layout=True)

        # left: convergence curves
        ax = axes[0, 0]
        ax.plot(x_outer, mean_dp, lw=1, color=C_DP, label="DP-GIBO")
        ax.fill_between(x_outer, mean_dp - ci_dp, mean_dp + ci_dp, alpha=0.2, color=C_DP)
        ax.plot(x_outer, mean_nondp, lw=1, color=C_NONDP, label=r"GIBO ($\mu = \infty$)")
        ax.fill_between(x_outer, mean_nondp - ci_nondp, mean_nondp + ci_nondp, alpha=0.2, color=C_NONDP)
        ax.plot(x_rd, mean_rd, lw=1, color=C_RAND, label="Random search")
        ax.fill_between(x_rd, mean_rd - ci_rd, mean_rd + ci_rd, alpha=0.2, color=C_RAND)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Validation loss")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(frameon=False, loc="upper right", handlelength=1.2, handletextpad=0.4)

        # right: final-loss strip plot
        ax = axes[0, 1]
        rng = np.random.default_rng(0)
        for i, (finals, color) in enumerate(
            [
                (dp_curves[:, -1], C_DP),
                (nondp_curves[:, -1], C_NONDP),
                (random_curves[:, -1], C_RAND),
            ]
        ):
            x_base = np.full_like(finals, i, dtype=float)
            jitter = (rng.random(len(finals)) - 0.5) * 0.2
            ax.scatter(x_base + jitter, finals, color=color, alpha=0.5, s=18, edgecolors="none")
            ax.hlines(finals.mean(), i - 0.2, i + 0.2, color=color, lw=1.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["DP-GIBO", "GIBO", "Random"])
        ax.set_ylabel("Final loss")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        fig.align_labels()

        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in extensions:
            fig.savefig(
                out_dir / f"figure_lasso_dpbo_vs_random.{ext}",
                dpi=400,
                bbox_inches="tight",
                pad_inches=0.05,
            )
        plt.close(fig)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(
    n_features: int = 10,
    n_train: int = 50,
    n_test: int = 500,
    num_iterations: int = 5,
    path_length: int = 2,
    mu: float = 1.0,
    grad_clip: float = 0.5,
    learning_rate: float = 0.5,
    inducing_steps: int = 200,
    seed: int = 0,
    out_dir: str | None = None,
    extensions: Sequence[str] = ("pdf", "png"),
    usetex: bool = False,
):
    """Run DP-GIBO, GIBO (μ=∞), and random search on the toy Lasso task.

    Each of `num_iterations` seeds runs all three methods from a shared θ_0.
    Curves are saved as .npy and a side-by-side comparison figure is saved.

    Args:
        n_features: number of input features in the synthetic regression task.
        n_train / n_test: train / test split sizes.
        num_iterations: number of independent seeds (separate θ_0 per seed).
        path_length: T, number of DP-GIBO outer iterations.
        mu: μ-GDP parameter for the DP-GIBO arm (∞ used for the GIBO arm).
        grad_clip: per-user gradient clip norm B.
        learning_rate: AdaGrad base step size for DP-GIBO.
        inducing_steps: Adam steps per inducing-point optimisation.
        seed: base seed; run i uses seed + i.
        out_dir: where to write outputs (defaults to this folder).
        extensions: figure formats to save.
        usetex: use LaTeX for figure text (requires a TeX install).
    """
    out_path = pathlib.Path(out_dir) if out_dir else pathlib.Path(__file__).resolve().parent
    torch.manual_seed(seed)
    np.random.seed(seed)

    objective = LassoObjective(
        n_features=n_features, n_train=n_train, n_test=n_test, seed=seed
    )
    bounds = objective.bounds_torch()

    common = dict(
        d=objective.d,
        n_users=objective.n_users,
        path_length=path_length,
        grad_clip=grad_clip,
        learning_rate=learning_rate,
        inducing_steps=inducing_steps,
        seed=seed,
    )
    cfg_dp = DPGIBOConfig(mu=mu, **common)
    cfg_nondp = DPGIBOConfig(mu=10_000.0, **common)

    n_random_iter = path_length * cfg_dp.bs
    dp_curves = np.zeros((num_iterations, path_length + 1))
    nondp_curves = np.zeros((num_iterations, path_length + 1))
    random_curves = np.zeros((num_iterations, n_random_iter))

    for i in range(num_iterations):
        run_seed = seed + i
        torch.manual_seed(run_seed)

        theta_0 = objective.random_theta(n=1)
        dp_result = run_dp_gibo(objective, cfg_dp, theta_0, bounds=bounds)
        nondp_result = run_dp_gibo(objective, cfg_nondp, theta_0, bounds=bounds)
        rs_result = run_random_search(objective, n_iter=n_random_iter, seed=run_seed)

        dp_curves[i, :] = dp_result.losses
        nondp_curves[i, :] = nondp_result.losses
        random_curves[i, :] = rs_result.best_so_far

        print(
            f"[seed {run_seed}] DP-GIBO={dp_result.losses[-1]:.4f}  "
            f"GIBO={nondp_result.losses[-1]:.4f}  "
            f"random={rs_result.best_so_far[-1]:.4f}"
        )

    np.save(out_path / "lasso_dp_gibo_losses.npy", dp_curves)
    np.save(out_path / "lasso_gibo_losses.npy", nondp_curves)
    np.save(out_path / "lasso_random_losses.npy", random_curves)

    print()
    print(f"=== Mean final loss across {num_iterations} seeds ===")
    print(f"  DP-GIBO (μ={mu}): {dp_curves[:, -1].mean():.4f} ± {dp_curves[:, -1].std():.4f}")
    print(f"  GIBO    (μ=∞)  : {nondp_curves[:, -1].mean():.4f} ± {nondp_curves[:, -1].std():.4f}")
    print(f"  Random search  : {random_curves[:, -1].mean():.4f} ± {random_curves[:, -1].std():.4f}")

    _plot_comparison(
        dp_curves=dp_curves,
        nondp_curves=nondp_curves,
        random_curves=random_curves,
        bs=cfg_dp.bs,
        out_dir=out_path,
        extensions=extensions,
        usetex=usetex,
    )


if __name__ == "__main__":
    fire.Fire(main)
