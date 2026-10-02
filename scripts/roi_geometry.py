"""ROI geometry metrics and controlled area perturbations used in the KSC paper.

For GT mask G and comparison ROI R:
    area_ratio = |R| / |G|
    IoG        = |R ∩ G| / |G|   (ROI recall)
    IoP        = |R ∩ G| / |R|   (ROI precision)
    IoU        = |R ∩ G| / |R ∪ G|

Controlled perturbation:
- ratio < 1: keep the deepest GT pixels (IoP=1, IoG≈ratio)
- ratio > 1: preserve all GT pixels and add nearest background pixels
             (IoG=1, IoP≈1/ratio)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.ndimage import distance_transform_edt


def decode_rle(rle, height: int, width: int) -> np.ndarray:
    flat = np.zeros(height * width, dtype=np.uint8)
    arr = np.asarray(rle, dtype=np.int64)
    if arr.size % 2 != 0:
        raise ValueError("RLE must contain (start, length) pairs")
    starts = arr[0::2] - 1
    lengths = arr[1::2]
    for start, length in zip(starts, lengths):
        flat[start : start + length] = 1
    return flat.reshape((width, height)).T.astype(bool)


def roi_geometry(gt_mask: np.ndarray, roi_mask: np.ndarray) -> Dict[str, float]:
    gt = np.asarray(gt_mask, dtype=bool)
    roi = np.asarray(roi_mask, dtype=bool)
    if gt.shape != roi.shape:
        raise ValueError(f"Mask shape mismatch: GT={gt.shape}, ROI={roi.shape}")

    g = int(gt.sum())
    r = int(roi.sum())
    inter = int(np.logical_and(gt, roi).sum())
    union = g + r - inter
    if g == 0:
        raise ValueError("GT ROI is empty")

    return {
        "area_ratio": r / g,
        "iog": inter / g,
        "iop": inter / r if r else 0.0,
        "iou": inter / union if union else 0.0,
        "gt_area": g,
        "roi_area": r,
        "intersection": inter,
    }


def perturb_area(gt_mask: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 0:
        raise ValueError("ratio must be > 0")

    gt = np.asarray(gt_mask, dtype=bool)
    flat_gt = gt.ravel()
    gt_idx = np.flatnonzero(flat_gt)
    gt_area = len(gt_idx)
    if gt_area == 0:
        raise ValueError("GT ROI is empty")

    target_area = int(round(gt_area * ratio))
    target_area = max(1, min(target_area, gt.size))
    if target_area == gt_area:
        return gt.copy()

    out = np.zeros(gt.size, dtype=bool)

    if target_area < gt_area:
        dist_inside = distance_transform_edt(gt).ravel()
        scores = dist_inside[gt_idx]
        k = target_area
        chosen = np.argpartition(scores, len(scores) - k)[len(scores) - k :]
        out[gt_idx[chosen]] = True
    else:
        out[gt_idx] = True
        add_n = target_area - gt_area
        bg_idx = np.flatnonzero(~flat_gt)
        add_n = min(add_n, len(bg_idx))
        if add_n:
            dist_out = distance_transform_edt(~gt).ravel()
            scores = dist_out[bg_idx]
            chosen = np.argpartition(scores, add_n - 1)[:add_n]
            out[bg_idx[chosen]] = True

    return out.reshape(gt.shape)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 1, 1.5, 2, 3, 5])
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            gt = decode_rle(row["gt_mask_rle"], int(row["gt_mask_h"]), int(row["gt_mask_w"]))
            for ratio in args.ratios:
                roi = perturb_area(gt, ratio)
                rec = {"question_id": row.get("question_id"), "ratio": ratio, **roi_geometry(gt, roi)}
                fout.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
