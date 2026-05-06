"""Compare DP-GIBO, Random search, and Global BO across three problem dimensions.

Reads the per-dimension `.npy` files in this folder (`losses_dp_gibo_d{d}.npy`,
`losses_random_d{d}.npy`, `losses_global_BO_d{d}.npy`) and lays out one panel
per dimension showing each method's mean loss curve with a shaded 95% CI band.

Usage:
    python gp_example/plot_three_dimensions.py
    python gp_example/plot_three_dimensions.py --dims '(2,5,10)' --usetex
"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Sequence

import fire
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils.plotting import style

C_DP = "#1f77b4"
C_NONDP = "gray"
C_RAND = "#2ca02c"
C_GLOBAL_BO = "#d62728"


def _load(in_path: pathlib.Path, d: int):
    dp = np.load(in_path / f"losses_dp_gibo_d{d}.npy")
    rd = np.load(in_path / f"losses_random_d{d}.npy")
    glo = np.load(in_path / f"losses_global_BO_d{d}.npy")
    return dp, rd, glo


def _summary(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = curves.shape[0]
    mean = curves.mean(axis=0)
    ci = 1.96 * curves.std(axis=0) / math.sqrt(max(n, 1))
    return mean, ci


def main(
    dims: Sequence[int] = (2, 5, 10),
    in_dir: str | None = None,
    out_dir: str | None = None,
    out_name: str = "figure_three_dimensions",
    extensions: Sequence[str] = ("pdf", "png"),
    usetex: bool = False,
    sharey: bool = False,
):
    """Plot per-dimension loss-curve comparison.

    Args:
        dims: dimensions to plot, in order.
        in_dir: where to read the .npy curves from (defaults to this folder).
        out_dir: where to write the figure (defaults to this folder).
        out_name: stem for the output figure.
        extensions: figure formats to save.
        usetex: use LaTeX for figure text (requires a TeX install).
        sharey: share the y-axis across panels.
    """
    here = pathlib.Path(__file__).resolve().parent
    in_path = pathlib.Path(in_dir) if in_dir else here
    out_path = pathlib.Path(out_dir) if out_dir else here

    rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for d in dims:
        try:
            dp, rd, glo = _load(in_path, d)
        except FileNotFoundError:
            print(f"  [skip] no .npy files found for d={d} in {in_path}")
            continue
        rows.append((d, dp, rd, glo))
        print(
            f"  d={d:>2}  DP-GIBO {dp[:, -1].mean():.4f}±{dp[:, -1].std():.4f}  "
            f"Random {rd[:, -1].mean():.4f}±{rd[:, -1].std():.4f}  "
            f"Global BO {glo[:, -1].mean():.4f}±{glo[:, -1].std():.4f}  "
            f"(n={dp.shape[0]})"
        )

    if not rows:
        print("Nothing to plot.")
        return

    n_panels = len(rows)
    fmt = FuncFormatter(lambda x, pos: f"{x:.2f}")

    with plt.rc_context(
        style.neurips(usetex=usetex, rel_width=0.66, nrows=2.0, ncols=n_panels)
    ):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_panels,
            squeeze=False,
            sharey=sharey,
            constrained_layout=True,
        )

        for j, (d, dp, rd, glo) in enumerate(rows):
            ax = axes[0, j]

            mean_dp, ci_dp = _summary(dp)
            mean_rd, ci_rd = _summary(rd)
            mean_glo, ci_glo = _summary(glo)

            bs = d + 1
            x_dp = np.arange(0, dp.shape[1]) * bs
            x_rd = np.arange(rd.shape[1])
            x_glo = np.arange(glo.shape[1])

            ax.plot(x_dp, mean_dp, lw=1, color=C_DP, label="DP-GIBO")
            ax.fill_between(
                x_dp, mean_dp - ci_dp, mean_dp + ci_dp, alpha=0.2, color=C_DP
            )
            ax.plot(x_rd, mean_rd, lw=1, color=C_RAND, label="Random search")
            ax.fill_between(
                x_rd, mean_rd - ci_rd, mean_rd + ci_rd, alpha=0.2, color=C_RAND
            )
            ax.plot(x_glo, mean_glo, lw=1, color=C_GLOBAL_BO, label="Global BO")
            ax.fill_between(
                x_glo,
                mean_glo - ci_glo,
                mean_glo + ci_glo,
                alpha=0.2,
                color=C_GLOBAL_BO,
            )

            ax.set_title(rf"$d = {d}$")
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
            ax.yaxis.set_major_formatter(fmt)

            if j == 0:
                ax.set_ylabel("Validation loss")
            if j == n_panels // 2:
                ax.set_xlabel("Function evaluations")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=len(labels),
            frameon=False,
            handlelength=1.2,
            handletextpad=0.4,
        )

        fig.align_labels()
        out_path.mkdir(parents=True, exist_ok=True)
        for ext in extensions:
            fig.savefig(
                out_path / f"{out_name}.{ext}",
                dpi=400,
                bbox_inches="tight",
                pad_inches=0.05,
            )
        plt.close(fig)


if __name__ == "__main__":
    fire.Fire(main)
