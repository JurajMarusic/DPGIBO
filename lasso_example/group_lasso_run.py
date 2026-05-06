"""Toy demo: tune 5 group-lasso λ's with DP-GIBO vs GIBO vs random search.

Synthetic regression with 5 feature groups (each of `group_size` features).
Only the first ``n_informative_groups`` of them carry signal — the rest are
pure noise — so the optimum has *small* λ for informative groups and *large*
λ for noise groups. Random search has to find that pattern in a 5-D box.

Inner solver is plain ISTA (proximal gradient) for group-lasso:
    minimise  ‖y − Xw‖² / 2n  +  Σₖ λₖ ‖w_{Gₖ}‖₂

Usage:
    python -m lasso_example.group_lasso_run
    python -m lasso_example.group_lasso_run --num-iterations 5 --path-length 4 --mu 1.0
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Allow running as a script as well as a module.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from algorithms import DPGIBOConfig, run_dp_gibo, run_random_search
from utils.plotting import style

C_DP = "#1f77b4"
C_NONDP = "gray"
C_RAND = "#2ca02c"


# ----------------------------------------------------------------------------
# Group-lasso ISTA solver
# ----------------------------------------------------------------------------


def _solve_group_lasso(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[list[int]],
    lambdas: np.ndarray,
    L: float,
    n_iter: int = 200,
) -> np.ndarray:
    """Plain ISTA for group lasso. ``L`` is a Lipschitz constant of ∇(½‖Xw−y‖²/n)."""
    n, d = X.shape
    lr = 1.0 / L
    w = np.zeros(d)
    for _ in range(n_iter):
        grad = X.T @ (X @ w - y) / n
        w = w - lr * grad
        for k, group_idx in enumerate(groups):
            wk = w[group_idx]
            norm_wk = np.linalg.norm(wk)
            if norm_wk > 0:
                shrink = max(0.0, 1.0 - lr * lambdas[k] / norm_wk)
                w[group_idx] = shrink * wk
    return w


# ----------------------------------------------------------------------------
# Group-lasso objective on a synthetic regression task
# ----------------------------------------------------------------------------


class GroupLassoObjective:
    """Per-test-sample squared-error loss for group lasso.

    θ packs ``[log_lambda_1, ..., log_lambda_K]`` (one per feature group, K
    groups total). The objective fits group-lasso on the train split with the
    given λ's and reports per-test-sample squared errors.
    """

    def __init__(
        self,
        n_groups: int = 20,
        group_size: int = 5,
        n_informative_groups: int = 1,
        n_train: int = 5,
        n_test: int = 500,
        noise: float = 1.0,
        within_group_corr: float = 0.9,  # 0 = i.i.d., 1 = identical columns
        ista_iter: int = 200,
        seed: int = 1,
    ):
        self.n_groups = n_groups
        self.group_size = group_size
        self.ista_iter = ista_iter

        rng = np.random.default_rng(seed)
        n_features = n_groups * group_size
        n_total = n_train + n_test

        # Each group's features are noisy copies of a single latent factor.
        # corr(X[:, j], X[:, j']) ≈ within_group_corr for j, j' in the same group.
        a = within_group_corr**0.5
        b = (1.0 - within_group_corr) ** 0.5
        X = np.zeros((n_total, n_features))
        for k in range(n_groups):
            factor = rng.normal(size=(n_total, 1))
            idiosyncratic = rng.normal(size=(n_total, group_size))
            X[:, k * group_size : (k + 1) * group_size] = a * factor + b * idiosyncratic

        true_w = np.zeros(n_features)
        for k in range(n_informative_groups):
            true_w[k * group_size : (k + 1) * group_size] = rng.normal(size=group_size)
        y = X @ true_w + noise * rng.normal(size=n_total)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=n_train, test_size=n_test, random_state=seed
        )
        self._sX = StandardScaler().fit(X_tr)
        self._sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        self.X_train = self._sX.transform(X_tr)
        self.X_test = self._sX.transform(X_te)
        self.y_train = self._sy.transform(y_tr.reshape(-1, 1)).ravel()
        self.y_test = self._sy.transform(y_te.reshape(-1, 1)).ravel()

        self.groups = [
            list(range(k * group_size, (k + 1) * group_size)) for k in range(n_groups)
        ]
        # Lipschitz constant of ∇(½‖Xw−y‖²/n_train), cached once.
        self._L_train = max(
            np.linalg.norm(self.X_train, ord=2) ** 2 / self.X_train.shape[0], 1e-6
        )

        self.bounds = Bounds(
            lb=[math.log(1e-6)] * n_groups,
            ub=[math.log(1e0)] * n_groups,
        )

    @property
    def d(self) -> int:
        return self.n_groups

    @property
    def n_users(self) -> int:
        return self.X_test.shape[0]

    def bounds_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        lo = torch.tensor(self.bounds.lb, dtype=torch.get_default_dtype())
        hi = torch.tensor(self.bounds.ub, dtype=torch.get_default_dtype())
        return lo, hi

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        theta_np = theta.detach().cpu().numpy()
        n_theta = theta_np.shape[0]
        losses = torch.zeros(
            (self.X_test.shape[0], n_theta), dtype=torch.float64, device=theta.device
        )
        for i, log_lams in enumerate(theta_np):
            lambdas = np.exp(log_lams)
            w = _solve_group_lasso(
                self.X_train,
                self.y_train,
                self.groups,
                lambdas,
                L=self._L_train,
                n_iter=self.ista_iter,
            )
            residuals = self.y_test - self.X_test @ w
            losses[:, i] = torch.tensor(0.5 * residuals**2, dtype=torch.float64)
        return losses

    def mean_loss(self, theta: torch.Tensor) -> float:
        return self(theta).mean().item()

    def random_theta(
        self, n: int = 1, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        lo = torch.tensor(self.bounds.lb, dtype=torch.get_default_dtype())
        hi = torch.tensor(self.bounds.ub, dtype=torch.get_default_dtype())
        u = torch.rand(n, len(lo), generator=generator)
        return u * (hi - lo) + lo


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------


def _plot_comparison(
    dp_curves: np.ndarray,
    nondp_curves: np.ndarray,
    random_curves: np.ndarray,
    bs: int,
    out_dir: pathlib.Path,
    extensions: Sequence[str],
    name_suffix: str = "",
    usetex: bool = False,
):
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

    with plt.rc_context(
        style.neurips(usetex=usetex, rel_width=0.9, nrows=1.25, ncols=2)
    ):
        fig, axes = plt.subplots(1, 2, squeeze=False, constrained_layout=True)

        ax = axes[0, 0]
        ax.plot(x_outer, mean_dp, lw=1, color=C_DP, label="DP-GIBO")
        ax.fill_between(
            x_outer, mean_dp - ci_dp, mean_dp + ci_dp, alpha=0.2, color=C_DP
        )
        ax.plot(
            x_outer, mean_nondp, lw=1, color=C_NONDP, label=r"GIBO ($\mu = \infty$)"
        )
        ax.fill_between(
            x_outer,
            mean_nondp - ci_nondp,
            mean_nondp + ci_nondp,
            alpha=0.2,
            color=C_NONDP,
        )
        ax.plot(x_rd, mean_rd, lw=1, color=C_RAND, label="Random search")
        ax.fill_between(x_rd, mean_rd - ci_rd, mean_rd + ci_rd, alpha=0.2, color=C_RAND)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Validation loss")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(frameon=False, loc="upper right", handlelength=1.2, handletextpad=0.4)

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
            ax.scatter(
                x_base + jitter, finals, color=color, alpha=0.5, s=18, edgecolors="none"
            )
            ax.hlines(finals.mean(), i - 0.2, i + 0.2, color=color, lw=1.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["DP-GIBO", "GIBO", "Random"])
        ax.set_ylabel("Final loss")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        fig.align_labels()
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in extensions:
            fig.savefig(
                out_dir / f"figure_group_lasso_dpbo_vs_random{name_suffix}.{ext}",
                dpi=400,
                bbox_inches="tight",
                pad_inches=0.05,
            )
        plt.close(fig)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(
    n_groups: int = 2,
    group_size: int = 5,
    n_informative_groups: int = 1,
    n_train: int = 50,
    n_test: int = 150,
    num_iterations: int = 5,
    path_length: int = 20,
    batch_size: int | None = None,  # defaults to d+1 = n_groups+1
    mu: float = 1.0,
    grad_clip: float = 0.05,
    learning_rate: float = 1.0,
    kernel_lengthscale: float = 1.0,  # GP RBF lengthscale; box is ~11.5 wide per coord
    optimizer: str = "adagrad",  # "adagrad" | "adam" | "momentum"; all DP-safe
    inducing_steps: int = 100,
    ista_iter: int = 200,
    init_center: float = -3.0,  # GIBO / DP-GIBO init centred at this log-λ value
    init_half_width: float = 3.0,  # GIBO / DP-GIBO init θ_0 ∈ [center-h, center+h]^d
    seed: int = 0,
    verbose: bool = False,
    out_dir: str | None = None,
    extensions: Sequence[str] = ("pdf", "png"),
    usetex: bool = False,
):
    """Run DP-GIBO, GIBO (μ=∞), and random search on the group-lasso toy task.

    Args:
        n_groups: number of feature groups; θ has dimension n_groups.
        group_size: number of features per group.
        n_informative_groups: how many of the leading groups carry signal.
        n_train / n_test: train / test split sizes.
        num_iterations: number of independent seeds.
        path_length: T, number of DP-GIBO outer iterations.
        mu: μ-GDP parameter for the DP arm (μ=∞ used for the GIBO arm).
        grad_clip: per-user gradient clip norm B.
        learning_rate: AdaGrad base step size.
        inducing_steps: Adam steps per inducing-point optimisation.
        ista_iter: ISTA steps per group-lasso fit.
        seed: base seed; run i uses seed + i.
        out_dir: where to write outputs (defaults to this folder).
        extensions: figure formats to save.
        usetex: use LaTeX for figure text (requires a TeX install).
    """
    out_path = (
        pathlib.Path(out_dir) if out_dir else pathlib.Path(__file__).resolve().parent
    )
    torch.manual_seed(seed)
    np.random.seed(seed)

    objective = GroupLassoObjective(
        n_groups=n_groups,
        group_size=group_size,
        n_informative_groups=n_informative_groups,
        n_train=n_train,
        n_test=n_test,
        ista_iter=ista_iter,
        seed=seed,
    )
    bounds = objective.bounds_torch()

    common = dict(
        d=objective.d,
        n_users=objective.n_users,
        path_length=path_length,
        batch_size=batch_size,
        grad_clip=grad_clip,
        learning_rate=learning_rate,
        kernel_lengthscale=kernel_lengthscale,
        optimizer=optimizer,
        inducing_steps=inducing_steps,
        seed=seed,
    )
    cfg_dp = DPGIBOConfig(mu=mu, **common)
    cfg_nondp = DPGIBOConfig(mu=1_000_000.0, **common)

    n_random_iter = path_length * cfg_dp.bs
    dp_curves = np.zeros((num_iterations, path_length + 1))
    nondp_curves = np.zeros((num_iterations, path_length + 1))
    random_curves = np.zeros((num_iterations, n_random_iter))
    best_random: tuple[float, torch.Tensor, int] | None = None  # (loss, θ, seed)
    # Best GIBO run across seeds: (final_loss, start_θ, end_θ, start_loss, seed).
    best_nondp: tuple[float, torch.Tensor, torch.Tensor, float, int] | None = None

    for i in range(num_iterations):
        run_seed = seed + i
        torch.manual_seed(run_seed)

        # GIBO / DP-GIBO start in a small box around `init_center` (default 0,
        # i.e. λ ≈ 1) so the initial gradient isn't dominated by extreme λ
        # where the loss is flat. Random search still samples uniformly over
        # the full bounds. Set init_center < 0 to start at lower λ.
        theta_0 = init_center + (torch.rand(1, objective.d) * 2 - 1) * init_half_width
        if bounds is not None:
            theta_0 = torch.clamp(theta_0, min=bounds[0], max=bounds[1])
        if verbose:
            print(f"\n--- seed {run_seed}: DP-GIBO (μ={mu}) ---")
        dp_result = run_dp_gibo(
            objective, cfg_dp, theta_0, bounds=bounds, verbose=verbose
        )
        if verbose:
            print(f"--- seed {run_seed}: GIBO (μ=∞) ---")
        nondp_result = run_dp_gibo(
            objective, cfg_nondp, theta_0, bounds=bounds, verbose=verbose
        )
        rs_result = run_random_search(objective, n_iter=n_random_iter, seed=run_seed)

        dp_curves[i, :] = dp_result.losses
        nondp_curves[i, :] = nondp_result.losses
        random_curves[i, :] = rs_result.best_so_far

        rs_best = float(rs_result.best_so_far[-1])
        if best_random is None or rs_best < best_random[0]:
            best_random = (rs_best, rs_result.best_theta.clone(), run_seed)

        nondp_final = float(nondp_result.losses[-1])
        if best_nondp is None or nondp_final < best_nondp[0]:
            best_nondp = (
                nondp_final,
                nondp_result.theta_path[0].detach().clone(),
                nondp_result.theta_path[-1].detach().clone(),
                float(nondp_result.losses[0]),
                run_seed,
            )

        print(
            f"[seed {run_seed}] DP-GIBO={dp_result.losses[-1]:.4f}  "
            f"GIBO={nondp_result.losses[-1]:.4f}  "
            f"random={rs_result.best_so_far[-1]:.4f}"
        )

    name_suffix = f"_d{n_groups}"
    np.save(out_path / f"group_lasso_dp_gibo_losses{name_suffix}.npy", dp_curves)
    np.save(out_path / f"group_lasso_gibo_losses{name_suffix}.npy", nondp_curves)
    np.save(out_path / f"group_lasso_random_losses{name_suffix}.npy", random_curves)

    print()
    print(f"=== Mean final loss across {num_iterations} seeds ===")
    print(
        f"  DP-GIBO (μ={mu}): {dp_curves[:, -1].mean():.4f} ± {dp_curves[:, -1].std():.4f}"
    )
    print(
        f"  GIBO    (μ=∞)  : {nondp_curves[:, -1].mean():.4f} ± {nondp_curves[:, -1].std():.4f}"
    )
    print(
        f"  Random search  : {random_curves[:, -1].mean():.4f} ± {random_curves[:, -1].std():.4f}"
    )

    if best_random is not None:
        best_loss, best_theta, best_seed = best_random
        log_lambdas = best_theta.detach().cpu().numpy().ravel()
        lambdas = np.exp(log_lambdas)
        with np.printoptions(precision=4, suppress=True, linewidth=200):
            print()
            print(
                f"=== Best random-search θ (seed {best_seed}, loss {best_loss:.4f}) ==="
            )
            print(f"  log λ = {log_lambdas}")
            print(f"  λ     = {lambdas}")

    if best_nondp is not None:
        end_loss, start_theta, end_theta, start_loss, nd_seed = best_nondp
        start_log = start_theta.detach().cpu().numpy().ravel()
        end_log = end_theta.detach().cpu().numpy().ravel()
        with np.printoptions(precision=4, suppress=True, linewidth=200):
            print()
            print(f"=== Best GIBO (μ=∞) trajectory (seed {nd_seed}) ===")
            print(f"  start: loss={start_loss:.4f}")
            print(f"    log λ = {start_log}")
            print(f"    λ     = {np.exp(start_log)}")
            print(f"  end:   loss={end_loss:.4f}")
            print(f"    log λ = {end_log}")
            print(f"    λ     = {np.exp(end_log)}")
            print(f"  Δθ    = {end_log - start_log}")

    _plot_comparison(
        dp_curves=dp_curves,
        nondp_curves=nondp_curves,
        random_curves=random_curves,
        bs=cfg_dp.bs,
        out_dir=out_path,
        extensions=extensions,
        name_suffix=name_suffix,
        usetex=usetex,
    )


if __name__ == "__main__":
    fire.Fire(main)
