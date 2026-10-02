"""Paired bootstrap/permutation statistics used in Sections 4.1 and 4.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[str(row["question_id"])] = row
    return out


def contrib(gt, pred) -> np.ndarray:
    gt = set(gt)
    pred = set(pred)
    return np.array([len(gt & pred), len(pred - gt), len(gt - pred)], dtype=np.int64)


def micro_f1(counts: np.ndarray) -> float:
    tp, fp, fn = map(int, counts)
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 0.0


def total_f1(x: np.ndarray) -> float:
    return micro_f1(x.sum(axis=0))


def paired_arrays(base_path: Path, new_path: Path):
    base = read_jsonl(base_path)
    new = read_jsonl(new_path)
    qids = sorted(set(base) & set(new))
    a, b = [], []
    for qid in qids:
        gt = base[qid]["gt_labels"]
        if set(gt) != set(new[qid]["gt_labels"]):
            raise ValueError(f"GT mismatch: {qid}")
        a.append(contrib(gt, base[qid]["pred_labels"]))
        b.append(contrib(gt, new[qid]["pred_labels"]))
    return np.stack(a), np.stack(b), qids


def bootstrap_delta(base: np.ndarray, new: np.ndarray, n_boot: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(base)
    point = total_f1(new) - total_f1(base)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = total_f1(new[idx]) - total_f1(base[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta_f1": point, "delta_f1_pp": point * 100, "ci95_pp": [float(lo * 100), float(hi * 100)]}


def permutation_delta(base: np.ndarray, new: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(total_f1(new) - total_f1(base))
    ge = 0
    n = len(base)
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        p_base = base.copy()
        p_new = new.copy()
        p_base[swap], p_new[swap] = new[swap], base[swap]
        ge += int(abs(total_f1(p_new) - total_f1(p_base)) >= observed)
    return (ge + 1) / (n_perm + 1)


def interaction(a_base, a_new, b_base, b_new, n_boot: int = 20000, seed: int = 0) -> dict:
    if not (len(a_base) == len(a_new) == len(b_base) == len(b_new)):
        raise ValueError("All paired arrays must have equal length")
    da = total_f1(a_new) - total_f1(a_base)
    db = total_f1(b_new) - total_f1(b_base)
    point = da - db

    rng = np.random.default_rng(seed)
    n = len(a_base)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ca = total_f1(a_new[idx]) - total_f1(a_base[idx])
        cb = total_f1(b_new[idx]) - total_f1(b_base[idx])
        boot[i] = ca - cb
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"interaction_pp": point * 100, "ci95_pp": [float(lo * 100), float(hi * 100)]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a-base", type=Path, required=True)
    p.add_argument("--a-new", type=Path, required=True)
    p.add_argument("--b-base", type=Path)
    p.add_argument("--b-new", type=Path)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    a0, a1, qids = paired_arrays(args.a_base, args.a_new)
    result = {
        "n": len(qids),
        "method_a": bootstrap_delta(a0, a1, args.bootstrap, args.seed),
        "method_a_permutation_p": permutation_delta(a0, a1, args.permutations, args.seed),
    }

    if args.b_base and args.b_new:
        b0, b1, bqids = paired_arrays(args.b_base, args.b_new)
        if qids != bqids:
            raise ValueError("Method A/B QID sets differ; filter to the same paired cohort first")
        result["method_b"] = bootstrap_delta(b0, b1, args.bootstrap, args.seed)
        result["method_b_permutation_p"] = permutation_delta(b0, b1, args.permutations, args.seed)
        result["interaction"] = interaction(a0, a1, b0, b1, args.bootstrap, args.seed)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
