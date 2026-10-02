"""Select clean qualitative failure cases for Section 4.3.

Selection rule:
- both CRG and PH are correct with GT ROI,
- CRG remains correct after ROI replacement/perturbation,
- PH becomes incorrect.

The paper inspected 5 real-MAIRA and 5 controlled-5x cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["question_id"])] = r
    return out


def err(gt, pred) -> int:
    gt, pred = set(gt), set(pred)
    return len(pred - gt) + len(gt - pred)


def failure_type(gt, pred) -> str:
    gt, pred = set(gt), set(pred)
    fp, fn = pred - gt, gt - pred
    if fp and fn:
        return "mixed"
    if fp:
        return "fp_only"
    if fn:
        return "fn_only"
    return "correct"


def select(crg_gt, crg_new, ph_gt, ph_new) -> list[dict]:
    common = sorted(set(crg_gt) & set(crg_new) & set(ph_gt) & set(ph_new))
    rows = []
    for qid in common:
        gt = crg_gt[qid]["gt_labels"]
        if not (
            err(gt, crg_gt[qid]["pred_labels"]) == 0
            and err(gt, ph_gt[qid]["pred_labels"]) == 0
            and err(gt, crg_new[qid]["pred_labels"]) == 0
            and err(gt, ph_new[qid]["pred_labels"]) > 0
        ):
            continue
        pred = ph_new[qid]["pred_labels"]
        rows.append({
            "question_id": qid,
            "image_id": crg_gt[qid].get("image_id"),
            "question": crg_gt[qid].get("question"),
            "gt_labels": gt,
            "crg_gt": crg_gt[qid]["pred_labels"],
            "crg_new": crg_new[qid]["pred_labels"],
            "ph_gt": ph_gt[qid]["pred_labels"],
            "ph_new": pred,
            "failure_type": failure_type(gt, pred),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--crg-gt", type=Path, required=True)
    p.add_argument("--crg-new", type=Path, required=True)
    p.add_argument("--ph-gt", type=Path, required=True)
    p.add_argument("--ph-new", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows = select(load(args.crg_gt), load(args.crg_new), load(args.ph_gt), load(args.ph_new))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"eligible={len(rows)} saved={args.output}")


if __name__ == "__main__":
    main()
