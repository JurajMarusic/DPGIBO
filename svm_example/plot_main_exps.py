"""Plot main experiments figures for the sDPBO NeurIPS submission.

Produces two side-by-side figures:
  - figure_main_ablations: 2x2 grid of hyperparameter sweeps (ε, μ, σ).
  - figure_main_trajectories: 2x1 grid of validation-loss curves and
    hyperparameter paths.

Usage:
    python plot_main_exps.py                       # both figures, pdf + png
    python plot_main_exps.py --dir /tmp/figs
    python plot_main_exps.py --extensions '("pdf",)'
    python plot_main_exps.py ablations             # only the ablations figure
    python plot_main_exps.py trajectories          # only the trajectories figure
"""

from __future__ import annotations

import pathlib
import pickle
import sys
from typing import Sequence

import fire
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils.plotting import style

DATA_DIR = pathlib.Path(__file__).resolve().parent

C_DP = "#1f77b4"
C_DP_ALT = "#ff7f0e"
C_NONDP = "gray"
C_RAND = "#2ca02c"


def main(
    dir: str = pathlib.Path.cwd(),
    extensions: Sequence[str] = ("pdf", "png"),
    usetex: bool = False,
):
    """Plot both main-experiments figures."""
    ablations(dir=dir, extensions=extensions, usetex=usetex)
    trajectories(dir=dir, extensions=extensions, usetex=usetex)


def ablations(
    dir: str = pathlib.Path.cwd(),
    extensions: Sequence[str] = ("pdf", "png"),
    usetex: bool = False,
):
    """Plot the 2x2 hyperparameter-ablation figure (left panel)."""
    rng = np.random.default_rng(0)

    with open(DATA_DIR / "all_losses_dp_gibo_varyingepsilons_10reps_four2.pkl", "rb") as f:
        data_eps_loss = pickle.load(f)
    with open(DATA_DIR / "all_batches_dp_gibo_varyingepsilons_10reps_four2.pkl", "rb") as f:
        data_eps_batches = pickle.load(f)
    with open(DATA_DIR / "all_losses_dp_gibo_varyingmus.pkl", "rb") as f:
        data_mus = pickle.load(f)
    with open(DATA_DIR / "all_losses_003.pkl", "rb") as f:
        data_sigmas = pickle.load(f)

    epsilons = [0.3, 0.5, 3.0, 7.5]
    mus = [0.2, 1.0, 10000.0]
    mus_plot = ["0.2", "1.0", r"$\infty$"]
    mu_colors = [C_DP_ALT, C_DP, C_NONDP]
    mu_labels = [
        r"DP-GIBO ($\mu = 0.2$)",
        r"DP-GIBO ($\mu = 1.0$)",
        r"GIBO ($\mu = \infty$)",
    ]
    sigmas = [0.00, 0.03, 0.25]

    nrows, ncols = 2, 2
    with plt.rc_context(
        style.neurips(
            usetex=usetex,
            rel_width=0.6,
            nrows=nrows * 1.25,
            ncols=ncols,
        )
    ):
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            squeeze=False,
            constrained_layout=True,
        )

        # (0,0): ε → final loss
        ax = axs[0, 0]
        for i, eps in enumerate(epsilons):
            _jittered_scatter(ax, data_eps_loss[eps][:, -1], i, C_DP, rng)
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([f"{e:.1f}" for e in epsilons])
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("Final loss")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # (0,1): ε → function evaluations
        ax = axs[0, 1]
        means, ci_h = [], []
        for eps in epsilons:
            finals = data_eps_batches[eps][:, :].sum(axis=1)
            se = stats.sem(finals)
            h = se * stats.t.ppf(0.975, len(finals) - 1)
            means.append(finals.mean())
            ci_h.append(h)
        ax.bar(
            np.arange(len(epsilons)),
            means,
            yerr=ci_h,
            capsize=2,
            width=0.5,
            alpha=0.5,
            color=C_DP,
            edgecolor="black",
            linewidth=0.6,
            error_kw=dict(elinewidth=1, ecolor="black"),
        )
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([f"{e:.1f}" for e in epsilons])
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("Fn. evaluations")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # (1,0): μ → final loss
        ax = axs[1, 0]
        for i, (mu, c) in enumerate(zip(mus, mu_colors)):
            _jittered_scatter(ax, data_mus[mu][:, -1], i, c, rng)
        ax.set_xticks(range(len(mus)))
        ax.set_xticklabels(mus_plot)
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel("Final loss")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # (1,1): σ → final loss
        ax = axs[1, 1]
        for i, sigma in enumerate(sigmas):
            _jittered_scatter(ax, data_sigmas[sigma][20:30, -1], i, C_DP, rng)
        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([f"{s:.2f}" for s in sigmas])
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel("Final loss")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        legend_handles = [
            Line2D([0], [0], color=col, lw=1.5, label=lab)
            for col, lab in zip(mu_colors, mu_labels)
        ]
        fig.legend(
            handles=legend_handles,
            loc="outside lower center",
            ncol=3,
            frameon=False,
            handlelength=1.5,
            columnspacing=1.2,
            handletextpad=0.4,
        )

        fig.align_labels()

        out_dir = pathlib.Path(dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in extensions:
            fig.savefig(
                out_dir / f"figure_main_ablations.{ext}",
                dpi=400,
                bbox_inches="tight",
                pad_inches=0.05,
            )

        plt.close(fig)


def trajectories(
    dir: str = pathlib.Path.cwd(),
    extensions: Sequence[str] = ("pdf", "png"),
    usetex: bool = False,
):
    """Plot the 2x1 trajectories figure (right panel)."""
    losses_dp = np.load(DATA_DIR / "losses_dp_gibo_15May.npy")
    losses_dp2 = np.load(DATA_DIR / "losses_dp_gibo2_15May.npy")
    losses_rd = np.load(DATA_DIR / "losses_random_15May.npy")

    num_iter = 5
    path_length = 10
    d = 103

    mean_dp = losses_dp.mean(axis=0)
    ci_dp = 1.96 * losses_dp.std(axis=0) / np.sqrt(num_iter)
    mean_dp2 = losses_dp2.mean(axis=0)
    ci_dp2 = 1.96 * losses_dp2.std(axis=0) / np.sqrt(num_iter)
    mean_rd = losses_rd.mean(axis=0)
    ci_rd = 1.96 * losses_rd.std(axis=0) / np.sqrt(num_iter)

    bs = d
    rand_len = path_length * bs
    x_dp = np.arange(0, rand_len + bs, bs)
    x_rd = np.arange(rand_len)

    path_dp1 = np.load(DATA_DIR / "dp-path1.npy")
    path_nondp1 = np.load(DATA_DIR / "nondp-path1.npy")
    path_dp2 = np.load(DATA_DIR / "dp-path2.npy")
    path_nondp2 = np.load(DATA_DIR / "nondp-path2.npy")
    bs_p = d / 3
    rand_len_p = 30 * bs_p
    x_path = np.arange(0, rand_len_p + bs_p, bs_p)

    nrows, ncols = 2, 1
    with plt.rc_context(
        style.neurips(
            usetex=usetex,
            rel_width=0.4,
            nrows=nrows * 1.25,
            ncols=ncols,
        )
    ):
        # match height of the ablations figure when scaled to 0.6 / 0.4 textwidth
        plt.rcParams["figure.figsize"] = (2.2, 2.04)

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            squeeze=False,
            sharex=True,
            constrained_layout=True,
        )

        ax = axs[0, 0]
        ax.plot(x_dp, mean_dp, lw=1, color=C_DP, label="DP-GIBO")
        ax.fill_between(x_dp, mean_dp - ci_dp, mean_dp + ci_dp, alpha=0.2, color=C_DP)
        ax.plot(x_dp, mean_dp2, lw=1, color=C_NONDP, label="GIBO")
        ax.fill_between(x_dp, mean_dp2 - ci_dp2, mean_dp2 + ci_dp2, alpha=0.2, color=C_NONDP)
        ax.plot(x_rd, mean_rd, lw=1, color=C_RAND, label="Random")
        ax.fill_between(x_rd, mean_rd - ci_rd, mean_rd + ci_rd, alpha=0.2, color=C_RAND)
        ax.set_ylabel("Validation loss")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(
            frameon=False,
            loc="upper right",
            handlelength=1.2,
            handletextpad=0.4,
            borderaxespad=0.3,
        )

        ax = axs[1, 0]
        ax.plot(x_path, path_dp1, lw=1, color=C_DP)
        ax.plot(x_path, path_nondp1, lw=1, color=C_NONDP)
        ax.plot(x_path, path_dp2, lw=1, linestyle="--", color=C_DP)
        ax.plot(x_path, path_nondp2, lw=1, linestyle="--", color=C_NONDP)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Hyperparameter")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        style_proxies = [
            Line2D([0], [0], color="black", lw=1, linestyle="-", label="Hyper 1"),
            Line2D([0], [0], color="black", lw=1, linestyle="--", label="Hyper 2"),
        ]
        ax.legend(
            handles=style_proxies,
            frameon=False,
            loc="upper right",
            handlelength=1.2,
            handletextpad=0.4,
            borderaxespad=0.3,
        )

        fig.align_labels()

        out_dir = pathlib.Path(dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in extensions:
            fig.savefig(
                out_dir / f"figure_main_trajectories.{ext}",
                dpi=400,
                bbox_inches="tight",
                pad_inches=0.05,
            )

        plt.close(fig)


def _jittered_scatter(ax, finals, i, color, rng):
    x_base = np.full_like(finals, i, dtype=float)
    jitter = (rng.random(len(finals)) - 0.5) * 0.2
    ax.scatter(x_base + jitter, finals, color=color, alpha=0.4, s=12, edgecolors="none")
    ax.hlines(finals.mean(), i - 0.2, i + 0.2, color=color, lw=1.2)


if __name__ == "__main__":
    fire.Fire(
        {
            "main": main,
            "ablations": ablations,
            "trajectories": trajectories,
        }
    )
