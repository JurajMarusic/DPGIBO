"""Compare final-iteration validation loss across problem dimensions.

Reads the per-dimension `.npy` files saved by `group_lasso_run.py`
(`group_lasso_{dp_gibo,gibo,random}_losses_d{d}.npy`) and lays out one panel
per dimension showing each method's final loss as a strip plot — per-seed
dots + median line + shaded 2.5%–97.5% quantile band.

Usage:
    python -m lasso_example.plot_final_loss_by_dim
    python -m lasso_example.plot_final_loss_by_dim --dims '(5,10,20,40)'
"""

from __future__ import annotations

import pathlib
import sys
from typing import Sequence

import fire
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils.plotting import style

C_DP = "#1f77b4"
C_NONDP = "gray"
C_RAND = "#2ca02c"


def _load(in_path: pathlib.Path, d: int):
    dp = np.load(in_path / f"group_lasso_dp_gibo_losses_d{d}.npy")[:, -1]
    nondp = np.load(in_path / f"group_lasso_gibo_losses_d{d}.npy")[:, -1]
    rand = np.load(in_path / f"group_lasso_random_losses_d{d}.npy")[:, -1]
    return dp, nondp, rand


def main(
    dims: Sequence[int] = (2, 10, 20),
    in_dir: str | None = None,
    out_dir: str | None = None,
    extensions: Sequence[str] = ("pdf", "png"),
    usetex: bool = False,
    sharey: bool = False,
):
    """Plot per-dimension final-loss comparison.

    Args:
        dims: dimensions to plot, in order.
        in_dir: where to read the .npy curves from (defaults to this folder).
        out_dir: where to write the figure (defaults to in_dir).
        extensions: figure formats to save.
        usetex: use LaTeX for figure text (requires a TeX install).
        sharey: share the y-axis across panels (loss scales often differ
            substantially across dimensions, so default is False).
    """
    in_path = (
        pathlib.Path(in_dir) if in_dir else pathlib.Path(__file__).resolve().parent
    )
    out_path = pathlib.Path(out_dir) if out_dir else in_path

    rows: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for d in dims:
        try:
            dp, nondp, rand = _load(in_path, d)
        except FileNotFoundError:
            print(f"  [skip] no .npy files found for d={d} in {in_path}")
            continue
        rows.append((d, dp, nondp, rand))

        def _summarize(x: np.ndarray) -> str:
            med = np.nanmedian(x)
            lo, hi = np.quantile(x, [0.025, 0.975])
            return f"{med:.4f} [{lo:.4f}, {hi:.4f}]"

        print(
            f"  d={d:>2}  DP-GIBO {_summarize(dp)}  "
            f"GIBO {_summarize(nondp)}  "
            f"Random {_summarize(rand)}  "
            f"(n={len(dp)})"
        )

    if not rows:
        print("Nothing to plot.")
        return

    n_panels = len(rows)
    rng = np.random.default_rng(0)

    with plt.rc_context(
        style.neurips(usetex=usetex, rel_width=1.0, nrows=1.25, ncols=n_panels)
    ):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_panels,
            squeeze=False,
            sharey=sharey,
            constrained_layout=True,
        )

        for j, (d, dp, nondp, rand) in enumerate(rows):
            ax = axes[0, j]
            for i, (finals, color) in enumerate(
                [(dp, C_DP), (nondp, C_NONDP), (rand, C_RAND)]
            ):
                x_base = np.full_like(finals, i, dtype=float)
                jitter = (rng.random(len(finals)) - 0.5) * 0.18
                # ax.scatter(
                #     x_base + jitter,
                #     finals,
                #     color=color,
                #     alpha=0.45,
                #     s=14,
                #     edgecolors="none",
                # )
                med = float(np.nanmedian(finals))
                lo, hi = np.quantile(finals, [0.025, 0.975])
                median_half_w = 0.15
                band_half_w = 0.04
                ax.fill_between(
                    [i - band_half_w, i + band_half_w],
                    float(lo),
                    float(hi),
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                    zorder=1,
                )
                ax.hlines(
                    med,
                    i - median_half_w,
                    i + median_half_w,
                    color=color,
                    linewidth=1.5,
                    zorder=3,
                )
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["DP-GIBO", "GIBO", "Random"])
            ax.set_title(rf"$d = {d}$")
            if j == 0:
                ax.set_ylabel("Final validation loss")
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        fig.align_labels()
        out_path.mkdir(parents=True, exist_ok=True)
        for ext in extensions:
            fig.savefig(
                out_path / f"figure_group_lasso_final_loss_by_dim.{ext}",
                dpi=400,
                bbox_inches="tight",
                pad_inches=0.05,
            )
        plt.close(fig)


if __name__ == "__main__":
    fire.Fire(main)
