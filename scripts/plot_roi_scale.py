"""Reproduce Figure 2: controlled ROI-scale robustness."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RATIOS = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
CRG_F1 = np.array([0.5053936195, 0.5119047619, 0.5136899989, 0.5162838533, 0.5154945683, 0.5147293903, 0.5157859002])
PH_F1 = np.array([0.5518296013, 0.5477304887, 0.5339386079, 0.5224640035, 0.5111473627, 0.4919186366, 0.4490445860])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("fig_roi_scale_robustness.pdf"))
    args = p.parse_args()

    crg = (CRG_F1 - CRG_F1[2]) * 100
    ph = (PH_F1 - PH_F1[2]) * 100

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(RATIOS, crg, marker="o", linewidth=1.6, label="CRG")
    ax.plot(RATIOS, ph, marker="s", linestyle="--", linewidth=1.6, label="PH")
    ax.axhline(0, color="0.45", linestyle="--", linewidth=0.8)
    ax.axvline(1, color="0.45", linestyle=":", linewidth=0.8)
    ax.text(1.04, 1.55, "GT ROI", fontsize=8)
    ax.set_xlabel("ROI area ratio")
    ax.set_ylabel(r"$\Delta$ micro-F1 vs. GT ROI (pp)")
    ax.set_xticks(RATIOS)
    ax.set_xticklabels(["0.25", "0.5", "1", "1.5", "2", "3", "5"])
    ax.set_ylim(-9.5, 2.5)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, loc="lower left")
    ax.annotate(f"{crg[-1]:+.2f}", (5, crg[-1]), xytext=(-4, 8), textcoords="offset points", ha="right", fontsize=8)
    ax.annotate(f"{ph[-1]:+.2f}", (5, ph[-1]), xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=8)
    fig.tight_layout(pad=0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    if args.output.suffix.lower() == ".pdf":
        fig.savefig(args.output.with_suffix(".png"), dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
