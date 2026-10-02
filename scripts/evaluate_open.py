"""Evaluate open-ended HEAL-MedVQA predictions after 14-label extraction.

Expected JSONL fields:
  question_id, gt_labels, pred_labels

The experiment used Llama-3.1-8B-Instruct to map generated answers to the
14 finding labels before running this deterministic metric stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

LABELS = [
    "aortic enlargement",
    "atelectasis",
    "calcification",
    "cardiomegaly",
    "consolidation",
    "interstitial lung disease",
    "infiltration",
    "lung opacification",
    "nodule/mass",
    "other lesion",
    "pleural effusion",
    "pleural thickening",
    "pneumothorax",
    "pulmonary fibrosis",
]


def contrib(gt: Iterable[str], pred: Iterable[str]) -> tuple[int, int, int]:
    gt = set(gt)
    pred = set(pred)
    return len(gt & pred), len(pred - gt), len(gt - pred)


def micro_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": p, "recall": r, "micro_f1": f1}


def score_jsonl(path: Path) -> dict:
    tp = fp = fn = exact = n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gt = row["gt_labels"]
            pred = row["pred_labels"]
            a, b, c = contrib(gt, pred)
            tp += a
            fp += b
            fn += c
            exact += int(set(gt) == set(pred))
            n += 1

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **micro_metrics(tp, fp, fn),
        "exact_set": exact / n if n else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    result = score_jsonl(args.input)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
