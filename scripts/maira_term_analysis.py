"""Summarize MAIRA-2 localization quality after terminology remapping.

Final paper remapping:
  Aorta -> "Thoracic aorta."
  Heart -> "Cardiomediastinal silhouette."

Input JSONL should contain:
  anatomy, maira2_status, maira2_bbox_iou
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

TERM_MAP = {
    "Aorta": "Thoracic aorta.",
    "Heart": "Cardiomediastinal silhouette.",
}


def summarize(path: Path) -> dict:
    stats = defaultdict(lambda: {"n": 0, "detected": 0, "iou_sum_all": 0.0, "iou_sum_detected": 0.0})
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            anatomy = r.get("anatomy")
            if anatomy not in TERM_MAP:
                continue
            s = stats[anatomy]
            s["n"] += 1
            status = r.get("maira2_status")
            iou = float(r.get("maira2_bbox_iou") or 0.0)
            s["iou_sum_all"] += iou
            if status == "detected":
                s["detected"] += 1
                s["iou_sum_detected"] += iou

    out = {}
    for anatomy, s in stats.items():
        n, d = s["n"], s["detected"]
        out[anatomy] = {
            "query": TERM_MAP[anatomy],
            "n": n,
            "detected": d,
            "detection_rate": d / n if n else 0.0,
            "mean_iou_all": s["iou_sum_all"] / n if n else 0.0,
            "mean_iou_detected": s["iou_sum_detected"] / d if d else 0.0,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    args = p.parse_args()
    print(json.dumps(summarize(args.input), indent=2))


if __name__ == "__main__":
    main()
