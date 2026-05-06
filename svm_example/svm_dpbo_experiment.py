"""DP-GIBO hyperparameter tuning of an SVR on the slice-localization CT dataset.

DP-GIBO (Differentially Private Gradient-Information Bayesian Optimization)
treats per-test-sample validation losses as per-user contributions and applies
the Gaussian mechanism (μ-GDP) to a per-user-clipped GP gradient estimate.
The outer loop is AdaGrad on the noised gradient.

Usage:
    python svm_dpbo_experiment.py main                 # run all (DP, non-DP, random)
    python svm_dpbo_experiment.py dp                   # only the DP-GIBO run
    python svm_dpbo_experiment.py nondp                # only the μ=∞ baseline (GIBO)
    python svm_dpbo_experiment.py random               # only the random-search baseline
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import warnings

import fire
import numpy as np
import pandas as pd
import torch
from scipy.optimize import Bounds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class DPGIBOConfig:
    """Hyperparameters for the DP-GIBO outer loop."""

    n_features: int = 19           # number of input features used from the CSV
    path_length: int = 30          # number of outer iterations T
    batch_size: int | None = None  # inducing-point batch size; defaults to d+1
    mu: float = 0.2                # μ-GDP parameter (∞ → non-private GIBO)
    grad_clip: float = 0.5         # per-user gradient clip norm (B)
    n_users: int = 9000            # number of "users" = test points in DP scaling
    closest_cache: int = 0         # how many old inducing points to retain (0 = none)
    kernel: str = "gaussian"
    learning_rate: float = 1.0
    inducing_steps: int = 1000     # Adam steps used to optimize inducing locations
    inducing_lr: float = 1e-2
    seed: int = 42
    device: str = "cpu"

    @property
    def d(self) -> int:
        return self.n_features + 3

    @property
    def bs(self) -> int:
        return self.batch_size if self.batch_size is not None else self.d + 1

    @property
    def sigma(self) -> float:
        """Gaussian-mechanism standard deviation under μ-GDP composition."""
        return 2.0 * self.grad_clip * math.sqrt(self.path_length) / (self.n_users * self.mu)


# ----------------------------------------------------------------------------
# Gaussian kernel and GP gradient posterior
# ----------------------------------------------------------------------------


def kernel_fn(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.exp(-(x1.unsqueeze(1) - x2.unsqueeze(0)).square().sum(dim=2) / 2)


def kernel_grad_fn(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return -(x1 - x2).T * kernel_fn(x1, x2)


def kernel_grad_grad_fn(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    k = kernel_fn(x1, x2)
    v = x1 - x2
    v_outer = torch.outer(v.squeeze(), v.squeeze())
    eye = torch.eye(x1.size(-1), device=x1.device, dtype=x1.dtype)
    return k * (-v_outer + eye)


def posterior_var_of_grad(
    x: torch.Tensor, z: torch.Tensor, jitter: float = 5e-6
) -> torch.Tensor:
    """Posterior covariance of ∇f(x) under a GP fit on inducing points z."""
    K_xx = kernel_grad_grad_fn(x, x)
    K_xz = kernel_grad_fn(x, z)
    K_zz = kernel_fn(z, z)
    L = torch.linalg.cholesky(K_zz + jitter * torch.eye(K_zz.size(-1), device=z.device))
    v = torch.cholesky_solve(K_xz.T, L)
    return K_xx - K_xz @ v


def posterior_mean_of_grad(
    x: torch.Tensor,
    z: torch.Tensor,
    f_values: torch.Tensor,
    jitter: float = 1e-9,
) -> torch.Tensor:
    """Posterior mean of ∇f(x) given function values f(z) at inducing points.

    Args:
        x: query point, shape (1, d).
        z: inducing locations, shape (n, d).
        f_values: function values at z, shape (n_users, n) — one row per user.

    Returns:
        Per-user gradient estimates with shape (n_users, d, 1).
    """
    K_xz = kernel_grad_fn(x, z)
    K_zz = kernel_fn(z, z)
    L = torch.linalg.cholesky(K_zz + jitter * torch.eye(K_zz.size(-1), device=z.device))
    f_values = f_values.to(dtype=L.dtype)
    v = torch.cholesky_solve(f_values.unsqueeze(-1), L)
    return K_xz @ v


def select_inducing_points(
    x: torch.Tensor,
    batch_size: int,
    D: torch.Tensor,
    num_iter: int = 1000,
    lr: float = 1e-2,
) -> torch.Tensor:
    """Pick `batch_size` inducing points minimising posterior gradient variance at x."""
    d = x.size(-1)
    z = x + 1e-5 / math.sqrt(d) * torch.randn(
        batch_size, d, dtype=x.dtype, device=x.device
    )
    z = z.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    for _ in range(num_iter):
        optimizer.zero_grad()
        z_combined = torch.cat([D, z], dim=0) if D.numel() else z
        loss = posterior_var_of_grad(x, z_combined).diag().sum()
        loss.backward()
        optimizer.step()
    return z.detach()


def find_closest_points(D: torch.Tensor, point: torch.Tensor, k: int) -> torch.Tensor:
    """Return the k inducing points in D nearest to `point` (Euclidean)."""
    if k <= 0 or D.numel() == 0:
        return torch.empty(0, point.shape[-1], device=D.device, dtype=D.dtype)
    k = min(k, D.shape[0])
    distances = torch.norm(D - point, dim=1)
    idx = torch.topk(distances, k, largest=False).indices
    return D[idx]


# ----------------------------------------------------------------------------
# SVR objective on the slice-localization CT dataset
# ----------------------------------------------------------------------------


class SVRObjective:
    """Per-test-sample SVR validation loss as a function of log-hyperparameters.

    The objective vector packs ``[log_lengthscales (n_features), log_eps, log_gamma, log_C]``.
    """

    def __init__(
        self,
        csv_path: str | pathlib.Path,
        n_features: int = 19,
        n_train: int = 1000,
        n_test: int = 9000,
        seed: int = 42,
    ):
        data = pd.read_csv(csv_path)
        X = data.iloc[:, 1 : n_features + 1].values
        y = data.iloc[:, -1].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=n_train, test_size=n_test, random_state=seed
        )
        self._scaler_X = StandardScaler().fit(X_tr)
        self._scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
        self.X_train = self._scaler_X.transform(X_tr)
        self.X_test = self._scaler_X.transform(X_te)
        self.y_train = self._scaler_y.transform(y_tr.reshape(-1, 1)).ravel()
        self.y_test = self._scaler_y.transform(y_te.reshape(-1, 1)).ravel()

        self.n_features = n_features
        self.bounds = Bounds(
            lb=[-3.0] * n_features + [math.log(0.01), math.log(0.1), math.log(0.01)],
            ub=[3.0] * n_features + [math.log(1.0), math.log(3.0), math.log(5.0)],
        )

    @property
    def d(self) -> int:
        return self.n_features + 3

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Return per-user squared-error losses, shape (n_users, n_theta)."""
        theta_np = theta.detach().cpu().numpy()
        n_theta = theta_np.shape[0]
        losses = torch.zeros(
            (self.X_test.shape[0], n_theta), dtype=torch.float64, device=theta.device
        )
        for i, h in enumerate(theta_np):
            log_ls = h[: self.n_features]
            log_eps, log_gamma, log_C = h[-3], h[-2], h[-1]
            ls = np.exp(log_ls)
            svr = SVR(
                kernel="rbf",
                epsilon=float(np.exp(log_eps)),
                gamma=float(np.exp(log_gamma)),
                C=float(np.exp(log_C)),
            )
            svr.fit(self.X_train / ls, self.y_train)
            residuals = self.y_test - svr.predict(self.X_test / ls)
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
# DP-GIBO outer loop
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class DPGIBOResult:
    losses: np.ndarray            # shape (T+1,)
    theta_path: list[torch.Tensor]
    trace_path: list[float]
    final_grad_norm: float


def run_dp_gibo(
    objective: SVRObjective,
    cfg: DPGIBOConfig,
    theta_0: torch.Tensor,
) -> DPGIBOResult:
    """Run T iterations of DP-GIBO from `theta_0`."""
    device = torch.device(cfg.device)
    sigma = cfg.sigma
    theta = theta_0.to(device).clone()

    losses = [objective.mean_loss(theta)]
    theta_path: list[torch.Tensor] = [theta[0].detach().clone()]
    trace_path: list[float] = []

    D = torch.empty(0, theta.shape[1], device=device)
    historical_grad = torch.zeros_like(theta)
    g_norm = torch.tensor(0.0)

    for _ in tqdm(range(cfg.path_length), desc=f"DP-GIBO (μ={cfg.mu})"):
        D_temp = (
            find_closest_points(D, theta[0], cfg.closest_cache).detach()
            if cfg.closest_cache > 0
            else torch.empty(0, theta.shape[1], device=device)
        )

        z = select_inducing_points(
            theta,
            batch_size=cfg.bs,
            D=D_temp,
            num_iter=cfg.inducing_steps,
            lr=cfg.inducing_lr,
        )
        D = torch.cat([D, z], dim=0)
        D_temp = torch.cat([D_temp, z], dim=0)

        f_values = objective(D_temp)
        per_user_grads = posterior_mean_of_grad(theta, D_temp, f_values).detach()

        # Per-user gradient clipping: each user's gradient ∥·∥ ≤ B.
        clip = torch.minimum(
            torch.tensor(1.0, device=device),
            cfg.grad_clip / torch.norm(per_user_grads, dim=1, keepdim=True),
        )
        clipped = per_user_grads * clip
        g = clipped.sum(dim=0) / cfg.n_users
        g_norm = g.norm()

        cov = posterior_var_of_grad(theta, D_temp)
        trace_path.append(cov.trace().item())

        # AdaGrad update on the privatised gradient.
        gradient = g.T + sigma * torch.randn(theta.shape[1], device=device)
        historical_grad = historical_grad + gradient**2
        adjusted_step = cfg.learning_rate / (torch.sqrt(historical_grad) + 1e-8)
        theta = torch.clamp(theta - adjusted_step * gradient, min=-5.0)

        losses.append(objective.mean_loss(theta))
        theta_path.append(theta[0].detach().clone())

    return DPGIBOResult(
        losses=np.asarray(losses),
        theta_path=theta_path,
        trace_path=trace_path,
        final_grad_norm=float(g_norm.item()),
    )


# ----------------------------------------------------------------------------
# Random-search baseline
# ----------------------------------------------------------------------------


def run_random_search(
    objective: SVRObjective,
    n_iter: int,
    seed: int = 0,
) -> np.ndarray:
    """Track the running-best validation loss over `n_iter` random hyperparameters."""
    gen = torch.Generator().manual_seed(seed)
    history = np.empty(n_iter)
    best = math.inf
    for t in tqdm(range(n_iter), desc="Random search"):
        loss = objective.mean_loss(objective.random_theta(n=1, generator=gen))
        best = min(best, loss)
        history[t] = best
    return history


# ----------------------------------------------------------------------------
# CLI entry points
# ----------------------------------------------------------------------------


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT
DEFAULT_CSV = DATA_DIR / "slice_localization_data.csv"


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _make_objective(csv_path: str, n_features: int, seed: int) -> SVRObjective:
    return SVRObjective(csv_path, n_features=n_features, seed=seed)


def main(
    csv_path: str = str(DEFAULT_CSV),
    out_dir: str = str(PROJECT_ROOT),
    num_iterations: int = 1,
    path_length: int = 30,
    mu: float = 0.2,
    n_features: int = 19,
    seed: int = 42,
    skip_random: bool = False,
):
    """Run DP-GIBO (μ given), GIBO (μ=∞), and (optionally) random search."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)

    objective = _make_objective(csv_path, n_features, seed)
    cfg_dp = DPGIBOConfig(n_features=n_features, path_length=path_length, mu=mu, seed=seed)
    cfg_nondp = DPGIBOConfig(n_features=n_features, path_length=path_length, mu=10_000.0, seed=seed)

    losses_dp = np.zeros((num_iterations, path_length + 1))
    losses_nondp = np.zeros((num_iterations, path_length + 1))
    losses_random = (
        np.zeros((num_iterations, path_length * cfg_dp.bs)) if not skip_random else None
    )
    last_paths: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None

    for i in range(num_iterations):
        theta_0 = objective.random_theta(n=1)

        result_dp = run_dp_gibo(objective, cfg_dp, theta_0)
        result_nondp = run_dp_gibo(objective, cfg_nondp, theta_0)

        losses_dp[i, :] = result_dp.losses
        losses_nondp[i, :] = result_nondp.losses
        last_paths = (result_dp.theta_path, result_nondp.theta_path)

        if not skip_random:
            losses_random[i, :] = run_random_search(
                objective, n_iter=path_length * cfg_dp.bs, seed=seed + i
            )

        print(f"[run {i}] DP-GIBO (μ={mu}) final loss: {result_dp.losses[-1]:.4f}")
        print(f"[run {i}] GIBO    (μ=∞)  final loss: {result_nondp.losses[-1]:.4f}")

    np.save(out / "losses_dp_gibo.npy", losses_dp)
    np.save(out / "losses_dp_gibo_inf.npy", losses_nondp)
    if losses_random is not None:
        np.save(out / "losses_random.npy", losses_random)

    # Per-coordinate trajectory dumps used by the plotting script.
    if last_paths is not None:
        path_dp, path_nondp = last_paths
        path_dp_t = torch.stack(path_dp).cpu().numpy()
        path_nondp_t = torch.stack(path_nondp).cpu().numpy()
        for elem, suffix in [(4, "1"), (n_features, "2")]:
            if elem < path_dp_t.shape[1]:
                np.save(out / f"dp-path{suffix}.npy", path_dp_t[:, elem])
                np.save(out / f"nondp-path{suffix}.npy", path_nondp_t[:, elem])


def dp(
    csv_path: str = str(DEFAULT_CSV),
    out_dir: str = str(PROJECT_ROOT),
    num_iterations: int = 1,
    path_length: int = 30,
    mu: float = 0.2,
    n_features: int = 19,
    seed: int = 42,
):
    """Run only the DP-GIBO arm and save losses_dp_gibo.npy."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)

    objective = _make_objective(csv_path, n_features, seed)
    cfg = DPGIBOConfig(n_features=n_features, path_length=path_length, mu=mu, seed=seed)

    losses = np.zeros((num_iterations, path_length + 1))
    for i in range(num_iterations):
        result = run_dp_gibo(objective, cfg, objective.random_theta(n=1))
        losses[i, :] = result.losses
        print(f"[run {i}] DP-GIBO (μ={mu}) final loss: {result.losses[-1]:.4f}")

    np.save(out / "losses_dp_gibo.npy", losses)


def nondp(
    csv_path: str = str(DEFAULT_CSV),
    out_dir: str = str(PROJECT_ROOT),
    num_iterations: int = 1,
    path_length: int = 30,
    n_features: int = 19,
    seed: int = 42,
):
    """Run only the GIBO (μ=∞) arm and save losses_dp_gibo_inf.npy."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)

    objective = _make_objective(csv_path, n_features, seed)
    cfg = DPGIBOConfig(n_features=n_features, path_length=path_length, mu=10_000.0, seed=seed)

    losses = np.zeros((num_iterations, path_length + 1))
    for i in range(num_iterations):
        result = run_dp_gibo(objective, cfg, objective.random_theta(n=1))
        losses[i, :] = result.losses
        print(f"[run {i}] GIBO (μ=∞) final loss: {result.losses[-1]:.4f}")

    np.save(out / "losses_dp_gibo_inf.npy", losses)


def random(
    csv_path: str = str(DEFAULT_CSV),
    out_dir: str = str(PROJECT_ROOT),
    num_iterations: int = 1,
    n_iter: int = 30 * 22,
    n_features: int = 19,
    seed: int = 42,
):
    """Run only the random-search baseline and save losses_random.npy."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)

    objective = _make_objective(csv_path, n_features, seed)
    losses = np.zeros((num_iterations, n_iter))
    for i in range(num_iterations):
        losses[i, :] = run_random_search(objective, n_iter=n_iter, seed=seed + i)
        print(f"[run {i}] random final best loss: {losses[i, -1]:.4f}")

    np.save(out / "losses_random.npy", losses)


if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
            "dp": dp,
            "nondp": nondp,
            "random": random,
        }
    )
