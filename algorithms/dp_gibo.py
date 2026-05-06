"""DP-GIBO: Differentially Private Gradient-Information Bayesian Optimization.

The outer loop is AdaGrad on a privatised gradient estimate; the gradient is
the posterior mean of ∇f under a Gaussian-process model fit to per-user
function values, after per-user clipping (norm ≤ B) and a Gaussian-mechanism
noise of σ = 2 B √T / (N μ) for μ-GDP composition over T iterations.

Any objective that exposes the duck-typed interface below can be used:

    obj(theta: Tensor[n, d]) -> Tensor[n_users, n]    per-user losses
    obj.mean_loss(theta) -> float                     scalar mean for tracking
    obj.random_theta(n=1, generator=None) -> Tensor   uniform draw inside bounds
"""

from __future__ import annotations

import dataclasses
import math
from typing import Protocol

import numpy as np
import torch
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class DPGIBOConfig:
    """Hyperparameters for the DP-GIBO outer loop."""

    d: int                         # objective dimension (length of θ)
    n_users: int                   # number of "users" (per-user records)
    path_length: int = 30          # number of outer iterations T
    batch_size: int | None = None  # inducing-point batch size; defaults to d+1
    mu: float = 0.5                # μ-GDP parameter (∞ → non-private GIBO)
    grad_clip: float = 0.5         # per-user gradient clip norm (B)
    closest_cache: int = 0         # how many old inducing points to retain (0 = none)
    kernel_lengthscale: float = 1.0  # GP RBF lengthscale; ~10–30% of search-box width
    learning_rate: float = 1.0
    optimizer: str = "adagrad"     # "adagrad" | "adam" | "momentum"; all DP-safe
    momentum_beta: float = 0.9     # heavy-ball momentum factor (used by "momentum")
    adam_beta1: float = 0.9        # Adam first-moment decay
    adam_beta2: float = 0.999      # Adam second-moment decay
    adam_eps: float = 1e-8
    inducing_steps: int = 1000     # Adam steps used to optimize inducing locations
    inducing_lr: float = 1e-2
    seed: int = 0
    device: str = "cpu"

    @property
    def bs(self) -> int:
        return self.batch_size if self.batch_size is not None else self.d + 1

    @property
    def sigma(self) -> float:
        """Gaussian-mechanism standard deviation under μ-GDP composition."""
        return 2.0 * self.grad_clip * math.sqrt(self.path_length) / (self.n_users * self.mu)


# ----------------------------------------------------------------------------
# Objective protocol
# ----------------------------------------------------------------------------


class Objective(Protocol):
    def __call__(self, theta: torch.Tensor) -> torch.Tensor: ...
    def mean_loss(self, theta: torch.Tensor) -> float: ...
    def random_theta(
        self, n: int = 1, generator: torch.Generator | None = None
    ) -> torch.Tensor: ...


# ----------------------------------------------------------------------------
# Gaussian kernel and GP gradient posterior
# ----------------------------------------------------------------------------


def kernel_fn(x1: torch.Tensor, x2: torch.Tensor, ell: float = 1.0) -> torch.Tensor:
    """Squared-exponential kernel ``exp(-‖x1-x2‖² / (2 ell²))``."""
    return torch.exp(-(x1.unsqueeze(1) - x2.unsqueeze(0)).square().sum(dim=2) / (2 * ell**2))


def kernel_grad_fn(x1: torch.Tensor, x2: torch.Tensor, ell: float = 1.0) -> torch.Tensor:
    """∂k/∂x1 with isotropic lengthscale `ell`."""
    return -((x1 - x2) / ell**2).T * kernel_fn(x1, x2, ell)


def kernel_grad_grad_fn(
    x1: torch.Tensor, x2: torch.Tensor, ell: float = 1.0
) -> torch.Tensor:
    """∂²k/(∂x1 ∂x2) with isotropic lengthscale `ell` (Cov of gradient process)."""
    k = kernel_fn(x1, x2, ell)
    v = x1 - x2
    v_outer = torch.outer(v.squeeze(), v.squeeze())
    eye = torch.eye(x1.size(-1), device=x1.device, dtype=x1.dtype)
    return (k / ell**2) * (eye - v_outer / ell**2)


def posterior_var_of_grad(
    x: torch.Tensor, z: torch.Tensor, jitter: float = 5e-6, ell: float = 1.0
) -> torch.Tensor:
    """Posterior covariance of ∇f(x) given inducing locations z."""
    K_xx = kernel_grad_grad_fn(x, x, ell)
    K_xz = kernel_grad_fn(x, z, ell)
    K_zz = kernel_fn(z, z, ell)
    L = torch.linalg.cholesky(K_zz + jitter * torch.eye(K_zz.size(-1), device=z.device))
    v = torch.cholesky_solve(K_xz.T, L)
    return K_xx - K_xz @ v


def posterior_mean_of_grad(
    x: torch.Tensor,
    z: torch.Tensor,
    f_values: torch.Tensor,
    jitter: float = 1e-9,
    ell: float = 1.0,
) -> torch.Tensor:
    """Posterior mean of ∇f(x) given function values f(z) at inducing points.

    Args:
        x: query point, shape (1, d).
        z: inducing locations, shape (n, d).
        f_values: function values at z, shape (n_users, n) — one row per user.
        ell: isotropic kernel lengthscale.

    Returns:
        Per-user gradient estimates with shape (n_users, d, 1).
    """
    K_xz = kernel_grad_fn(x, z, ell)
    K_zz = kernel_fn(z, z, ell)
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
    ell: float = 1.0,
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
        loss = posterior_var_of_grad(x, z_combined, ell=ell).diag().sum()
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
# DP-GIBO outer loop
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class DPGIBOResult:
    losses: np.ndarray             # shape (T+1,)
    theta_path: list[torch.Tensor]
    trace_path: list[float]
    final_grad_norm: float


def run_dp_gibo(
    objective: Objective,
    cfg: DPGIBOConfig,
    theta_0: torch.Tensor,
    bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    progress: bool = True,
    verbose: bool = False,
) -> DPGIBOResult:
    """Run T iterations of DP-GIBO from `theta_0`.

    Args:
        objective: per-user loss function (see :class:`Objective`).
        cfg: algorithm hyperparameters.
        theta_0: initial point, shape (1, d).
        bounds: optional ``(lo, hi)`` tensors of shape (d,); if given, θ is
            clamped to this box after every step (prevents AdaGrad runaway).
        progress: show a tqdm progress bar.
        verbose: print per-iteration diagnostics (loss, |g|, |update|, |Δθ|,
            cov-trace). Useful for debugging slow / non-converging runs.
    """
    device = torch.device(cfg.device)
    sigma = cfg.sigma
    theta = theta_0.to(device).clone()
    initial_theta = theta.detach().clone()
    if bounds is not None:
        lo = bounds[0].to(device=device, dtype=theta.dtype)
        hi = bounds[1].to(device=device, dtype=theta.dtype)
    else:
        lo = hi = None

    losses = [objective.mean_loss(theta)]
    theta_path: list[torch.Tensor] = [theta[0].detach().clone()]
    trace_path: list[float] = []

    D = torch.empty(0, theta.shape[1], device=device)
    g_norm = torch.tensor(0.0)

    # Optimizer state. All update rules below post-process the *already-
    # privatised* gradient, so μ-DP is preserved by post-processing immunity.
    historical_grad = torch.zeros_like(theta)        # AdaGrad
    momentum_buf = torch.zeros_like(theta)           # heavy-ball momentum
    adam_m = torch.zeros_like(theta)                 # Adam 1st moment
    adam_v = torch.zeros_like(theta)                 # Adam 2nd moment
    step_count = 0

    iterator = range(cfg.path_length)
    if progress:
        iterator = tqdm(iterator, desc=f"DP-GIBO (μ={cfg.mu})")

    for _ in iterator:
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
            ell=cfg.kernel_lengthscale,
        )
        D = torch.cat([D, z], dim=0)
        D_temp = torch.cat([D_temp, z], dim=0)

        f_values = objective(D_temp)
        per_user_grads = posterior_mean_of_grad(
            theta, D_temp, f_values, ell=cfg.kernel_lengthscale
        ).detach()

        # Per-user clipping: each user's gradient ∥·∥ ≤ B.
        clip = torch.minimum(
            torch.tensor(1.0, device=device),
            cfg.grad_clip / torch.norm(per_user_grads, dim=1, keepdim=True),
        )
        clipped = per_user_grads * clip
        g = clipped.sum(dim=0) / cfg.n_users
        g_norm = g.norm()
        per_user_norm = per_user_grads.norm(dim=1).mean()
        print(
            f"  iter {step_count + 1:3d}: "
            f"|g|={g_norm.item():.3e}  "
            f"⟨|g_i|⟩={per_user_norm.item():.3e}  "
            f"σ_noise={sigma:.3e}"
        )

        cov = posterior_var_of_grad(theta, D_temp, ell=cfg.kernel_lengthscale)
        trace_path.append(cov.trace().item())

        # Privatised gradient. Anything below this line is post-processing
        # of a μ-DP release, so the DP guarantee is preserved.
        gradient = g.T + sigma * torch.randn(theta.shape[1], device=device)
        step_count += 1

        if cfg.optimizer == "adagrad":
            historical_grad = historical_grad + gradient**2
            update = cfg.learning_rate * gradient / (torch.sqrt(historical_grad) + 1e-8)
        elif cfg.optimizer == "momentum":
            momentum_buf = cfg.momentum_beta * momentum_buf + gradient
            update = cfg.learning_rate * momentum_buf
        elif cfg.optimizer == "adam":
            adam_m = cfg.adam_beta1 * adam_m + (1 - cfg.adam_beta1) * gradient
            adam_v = cfg.adam_beta2 * adam_v + (1 - cfg.adam_beta2) * gradient**2
            m_hat = adam_m / (1 - cfg.adam_beta1**step_count)
            v_hat = adam_v / (1 - cfg.adam_beta2**step_count)
            update = cfg.learning_rate * m_hat / (torch.sqrt(v_hat) + cfg.adam_eps)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")

        theta = theta - update
        if lo is not None:
            theta = torch.clamp(theta, min=lo, max=hi)

        losses.append(objective.mean_loss(theta))
        theta_path.append(theta[0].detach().clone())

        if verbose:
            with np.printoptions(precision=3, suppress=True, sign="+", linewidth=200):
                g_np = g.detach().cpu().numpy().ravel()
                upd_np = update.detach().cpu().numpy().ravel()
                theta_np = theta.detach().cpu().numpy().ravel()
                print(
                    f"  iter {step_count:3d}: "
                    f"loss {losses[-2]:.4f} → {losses[-1]:.4f}  "
                    f"|g|={g.norm().item():.3e}  "
                    f"|noise|={(sigma * math.sqrt(theta.shape[1])):.3e}  "
                    f"|upd|={update.norm().item():.3e}  "
                    f"|Δθ|={(theta - initial_theta).norm().item():.3f}  "
                    f"trace={trace_path[-1]:.3e}"
                )
                print(f"           g   = {g_np}")
                print(f"           upd = {upd_np}")
                print(f"           θ   = {theta_np}")

    return DPGIBOResult(
        losses=np.asarray(losses),
        theta_path=theta_path,
        trace_path=trace_path,
        final_grad_norm=float(g_norm.item()),
    )
