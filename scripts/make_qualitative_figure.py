"""Create the representative qualitative ROI figure (Figure 3).

Expected term-row fields include image_path, gt_mask_rle/h/w and mask_rle/h/w.
Expected scored files contain question_id, gt_labels and pred_labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from roi_geometry import decode_rle, perturb_area, roi_geometry


def find_row(path: Path, qid: str) -> dict:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if str(r.get("question_id")) == qid:
                    return r
    raise KeyError(f"{qid} not found in {path}")


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    im = im.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(im) > 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qid", default="1_59e1f024cb9e119964b57d942fd72e36")
    p.add_argument("--term-jsonl", type=Path, required=True)
    p.add_argument("--crg-gt", type=Path, required=True)
    p.add_argument("--crg-maira", type=Path, required=True)
    p.add_argument("--crg-area5", type=Path, required=True)
    p.add_argument("--ph-gt", type=Path, required=True)
    p.add_argument("--ph-maira", type=Path, required=True)
    p.add_argument("--ph-area5", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("qualitative_overlocalization.pdf"))
    args = p.parse_args()

    term = find_row(args.term_jsonl, args.qid)
    image = Image.open(term["image_path"]).convert("RGB")
    image_np = np.asarray(image)

    gt = decode_rle(term["gt_mask_rle"], int(term["gt_mask_h"]), int(term["gt_mask_w"]))
    gt = resize_mask(gt, image.size)
    maira = decode_rle(term["mask_rle"], int(term["mask_h"]), int(term["mask_w"]))
    maira = resize_mask(maira, image.size)
    area5 = perturb_area(gt, 5.0)

    scored = {name: find_row(path, args.qid) for name, path in {
        "crg_gt": args.crg_gt, "crg_maira": args.crg_maira, "crg_area5": args.crg_area5,
        "ph_gt": args.ph_gt, "ph_maira": args.ph_maira, "ph_area5": args.ph_area5,
    }.items()}

    panels = [
        ("GT ROI", gt, scored["crg_gt"]["pred_labels"], scored["ph_gt"]["pred_labels"]),
        ("MAIRA ROI", maira, scored["crg_maira"]["pred_labels"], scored["ph_maira"]["pred_labels"]),
        ("Controlled 5x ROI", area5, scored["crg_area5"]["pred_labels"], scored["ph_area5"]["pred_labels"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
    gt_labels = scored["crg_gt"]["gt_labels"]
    for ax, (title, mask, crg, ph) in zip(axes, panels):
        g = roi_geometry(gt, mask)
        ax.imshow(image_np)
        overlay = np.ma.masked_where(~mask, mask.astype(float))
        ax.imshow(overlay, alpha=0.28, cmap="Reds", vmin=0, vmax=1)
        ax.contour(mask.astype(float), levels=[0.5], linewidths=1.1)
        ax.set_title(f"{title}\nArea={g['area_ratio']:.2f}x, IoG={g['iog']:.2f}, IoP={g['iop']:.2f}", fontsize=9)
        ax.axis("off")
        crg_text = "Correct" if set(crg) == set(gt_labels) else (", ".join(crg) or "None")
        ph_text = ", ".join(ph) or "None"
        ax.text(0.0, -0.08, f"CRG: {crg_text}\nPH: {ph_text}", transform=ax.transAxes, va="top", fontsize=7.5)

    fig.suptitle("Effect of ROI Over-localization", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    if args.output.suffix.lower() == ".pdf":
        fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
