"""Final-loss strip plot for the main experiment (DP-GIBO vs GIBO vs Random).

Reads the 15 May `.npy` files in `data/` and renders a single panel showing
each method's final-iteration validation loss as per-seed dots, a median
line, and a shaded 2.5%-97.5% quantile band — mirroring the styling of
`lasso_example/plot_final_loss_by_dim.py`.

Usage:
    python plot_main_loss.py
    python plot_main_loss.py --usetex
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


def main(
    in_dir: str | None = None,
    out_dir: str | None = None,
    out_name: str = "figure_SVM",
    d: int = 103,
    rel_width: float = 1 / 3,
    nrows: float = 1.25,
    extensions: Sequence[str] = ("pdf", "png"),
    usetex: bool = False,
):
    """Plot the main-experiment final-loss strip plot.

    Args:
        in_dir: where to read the .npy files from (defaults to ``data``).
        out_dir: where to write the figure (defaults to repo root).
        out_name: stem for the output figure.
        d: problem dimension (used for the panel title only).
        rel_width: figure width as a fraction of the text column.
        nrows: figure-height multiplier (in NeurIPS rows).
        extensions: figure formats to save.
        usetex: use LaTeX for figure text (requires a TeX install).
    """
    here = pathlib.Path(__file__).resolve().parent
    in_path = pathlib.Path(in_dir) if in_dir else here
    out_path = pathlib.Path(out_dir) if out_dir else here

    dp = np.load(in_path / "losses_dp_gibo_15May.npy")[:, -1]
    gibo = np.load(in_path / "losses_dp_gibo2_15May.npy")[:, -1]
    rand = np.load(in_path / "losses_random_15May.npy")[:, -1]

    def _summarize(x: np.ndarray) -> str:
        med = np.nanmedian(x)
        lo, hi = np.quantile(x, [0.025, 0.975])
        return f"{med:.4f} [{lo:.4f}, {hi:.4f}]"

    print(
        f"  DP-GIBO {_summarize(dp)}  "
        f"GIBO {_summarize(gibo)}  "
        f"Random {_summarize(rand)}  "
        f"(n={len(dp)})"
    )

    rng = np.random.default_rng(0)

    with plt.rc_context(
        style.neurips(usetex=usetex, rel_width=rel_width, nrows=nrows, ncols=1)
    ):
        fig, ax = plt.subplots(constrained_layout=True)

        for i, (finals, color) in enumerate(
            [(dp, C_DP), (gibo, C_NONDP), (rand, C_RAND)]
        ):
            x_base = np.full_like(finals, i, dtype=float)
            jitter = (rng.random(len(finals)) - 0.5) * 0.18
            ax.scatter(
                x_base + jitter,
                finals,
                color=color,
                alpha=0.45,
                s=14,
                edgecolors="none",
            )
            med = float(np.nanmedian(finals))
            lo, hi = np.quantile(finals, [0.025, 0.975])
            median_half_w = 0.15
            band_half_w = 0.02
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
        ax.set_ylabel("Final validation loss")
        ax.set_title(rf"$d = {d}$")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

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
