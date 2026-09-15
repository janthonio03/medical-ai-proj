import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.analysis.common import (
    get_gt,
    get_prediction,
    norm_answer,
    parse_disease,
    read_jsonl,
)


METHOD_ORDER = ["PH", "VCD", "CRG", "LoBA"]


def grouped_stats(df, group_col):
    rows = []

    for group_value, group in df.groupby(
        group_col,
        dropna=False,
        observed=False,
    ):
        for method in METHOD_ORDER:
            vanilla_errors = int((~group["Vanilla_correct"]).sum())
            vanilla_correct = int(group["Vanilla_correct"].sum())
            rescued = int(group[f"{method}_rescue"].sum())
            harmed = int(group[f"{method}_harm"].sum())

            rows.append(
                {
                    group_col: group_value,
                    "method": method,
                    "n": len(group),
                    "accuracy": float(group[f"{method}_correct"].mean()),
                    "vanilla_accuracy": float(group["Vanilla_correct"].mean()),
                    "vanilla_errors": vanilla_errors,
                    "rescued": rescued,
                    "rescue_rate": rescued / vanilla_errors if vanilla_errors else None,
                    "vanilla_correct": vanilla_correct,
                    "harmed": harmed,
                    "harm_rate": harmed / vanilla_correct if vanilla_correct else None,
                    "changed": int(group[f"{method}_changed"].sum()),
                }
            )

    return pd.DataFrame(rows)


def save_jsonl(df, path):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/processed/test_metadata.jsonl")
    parser.add_argument("--vanilla", default="outputs/vanilla/greedy/closed_predictions.jsonl")
    parser.add_argument("--ph", default="outputs/ph/attn5_cfg2/closed_predictions_greedy_baseline.jsonl")
    parser.add_argument("--vcd", default="outputs/vcd/alpha1_beta0.1_noise500/closed_predictions.jsonl")
    parser.add_argument("--crg", default="outputs/crg/alpha1/closed_predictions.jsonl")
    parser.add_argument("--loba", default="outputs/loba/oracle_roi_alpha0.3_beta2/closed_predictions.jsonl")
    parser.add_argument("--out-tables", default="analysis/tables")
    parser.add_argument("--out-cases", default="analysis/cases")
    args = parser.parse_args()

    out_tables = Path(args.out_tables)
    out_cases = Path(args.out_cases)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_cases.mkdir(parents=True, exist_ok=True)

    method_paths = {
        "Vanilla": Path(args.vanilla),
        "PH": Path(args.ph),
        "VCD": Path(args.vcd),
        "CRG": Path(args.crg),
        "LoBA": Path(args.loba),
    }

    metadata = read_jsonl(args.metadata)
    meta_by_qid = {str(row["question_id"]): row for row in metadata}

    predictions = {}
    gt_map = {}

    for method, path in method_paths.items():
        rows = read_jsonl(path)
        predictions[method] = {}

        for row in rows:
            qid = str(row["question_id"])
            predictions[method][qid] = get_prediction(row, method)
            gt = get_gt(row)
            if gt != "unknown":
                gt_map[qid] = gt

    loba_rows = {
        str(row["question_id"]): row
        for row in read_jsonl(method_paths["LoBA"])
    }

    rows = []

    for qid in predictions["Vanilla"]:
        meta = meta_by_qid[qid]
        gt = gt_map.get(qid, norm_answer(meta.get("answer")))

        row = {
            "question_id": qid,
            "question": meta.get("question", ""),
            "gt": gt,
            "anatomy": meta.get("anatomy", "Unknown"),
            "question_type": meta.get("question_type", "Unknown"),
        }
        row["disease"] = parse_disease(row["question"])

        loba_row = loba_rows.get(qid, {})
        row["roi_area_ratio"] = float(loba_row.get("roi_area_ratio", 0.0))
        row["highlighted_patches"] = int(loba_row.get("highlighted_patches", 0))

        for method in ["Vanilla", *METHOD_ORDER]:
            pred = predictions[method][qid]
            row[method] = pred
            row[f"{method}_correct"] = pred == gt

        for method in METHOD_ORDER:
            row[f"{method}_changed"] = row[method] != row["Vanilla"]
            row[f"{method}_rescue"] = (
                not row["Vanilla_correct"] and row[f"{method}_correct"]
            )
            row[f"{method}_harm"] = (
                row["Vanilla_correct"] and not row[f"{method}_correct"]
            )

        rescue_methods = [m for m in METHOD_ORDER if row[f"{m}_rescue"]]
        harm_methods = [m for m in METHOD_ORDER if row[f"{m}_harm"]]
        row["rescue_methods"] = "+".join(rescue_methods) if rescue_methods else "NONE"
        row["harm_methods"] = "+".join(harm_methods) if harm_methods else "NONE"
        rows.append(row)

    df = pd.DataFrame(rows)
    df["roi_quartile"] = pd.qcut(
        df["roi_area_ratio"].rank(method="first"),
        q=4,
        labels=["Q1_small", "Q2", "Q3", "Q4_large"],
    )

    overall = []
    for method in ["Vanilla", *METHOD_ORDER]:
        entry = {
            "method": method,
            "accuracy": float(df[f"{method}_correct"].mean()),
            "correct": int(df[f"{method}_correct"].sum()),
        }
        if method != "Vanilla":
            rescued = int(df[f"{method}_rescue"].sum())
            harmed = int(df[f"{method}_harm"].sum())
            entry.update(
                {
                    "changed": int(df[f"{method}_changed"].sum()),
                    "rescued": rescued,
                    "harmed": harmed,
                    "net_gain": rescued - harmed,
                }
            )
        overall.append(entry)

    overall_df = pd.DataFrame(overall)
    errors = df[~df["Vanilla_correct"]].copy()

    signatures = (
        errors.groupby("rescue_methods")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    unique_rows = []
    rescue_sets = {}
    for method in METHOD_ORDER:
        rescue_sets[method] = set(
            df.loc[df[f"{method}_rescue"], "question_id"].tolist()
        )
        total = len(rescue_sets[method])
        unique = int((errors["rescue_methods"] == method).sum())
        unique_rows.append(
            {
                "method": method,
                "total_rescues": total,
                "unique_rescues": unique,
                "shared_rescues": total - unique,
            }
        )

    pairwise_rows = []
    for a, b in combinations(METHOD_ORDER, 2):
        inter = rescue_sets[a] & rescue_sets[b]
        union = rescue_sets[a] | rescue_sets[b]
        pairwise_rows.append(
            {
                "method_a": a,
                "method_b": b,
                "rescue_a": len(rescue_sets[a]),
                "rescue_b": len(rescue_sets[b]),
                "intersection": len(inter),
                "union": len(union),
                "jaccard": len(inter) / len(union) if union else 0.0,
            }
        )

    unique_df = pd.DataFrame(unique_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    df.to_csv(out_tables / "method_case_matrix.csv", index=False)
    overall_df.to_csv(out_tables / "method_overall_summary.csv", index=False)
    signatures.to_csv(out_tables / "rescue_overlap_signatures.csv", index=False)
    unique_df.to_csv(out_tables / "unique_rescue_summary.csv", index=False)
    pairwise_df.to_csv(out_tables / "pairwise_rescue_overlap.csv", index=False)
    grouped_stats(df, "roi_quartile").to_csv(
        out_tables / "roi_quartile_method_breakdown.csv", index=False
    )
    grouped_stats(df, "anatomy").to_csv(
        out_tables / "anatomy_method_breakdown.csv", index=False
    )
    grouped_stats(df, "disease").to_csv(
        out_tables / "disease_method_breakdown.csv", index=False
    )
    grouped_stats(df, "gt").to_csv(
        out_tables / "gt_method_breakdown.csv", index=False
    )

    case_columns = [
        "question_id",
        "question",
        "gt",
        "anatomy",
        "disease",
        "roi_area_ratio",
        "highlighted_patches",
        "Vanilla",
        "PH",
        "VCD",
        "CRG",
        "LoBA",
        "rescue_methods",
        "harm_methods",
    ]

    catalog = df[case_columns].copy()
    save_jsonl(catalog[~df["Vanilla_correct"]], out_cases / "all_vanilla_errors.jsonl")
    save_jsonl(catalog[df["rescue_methods"] != "NONE"], out_cases / "all_rescued_cases.jsonl")
    save_jsonl(catalog[df["rescue_methods"].isin(METHOD_ORDER)], out_cases / "unique_rescues.jsonl")
    save_jsonl(catalog[df["harm_methods"] != "NONE"], out_cases / "harm_cases.jsonl")

    summary = {
        "n": len(df),
        "vanilla_correct": int(df["Vanilla_correct"].sum()),
        "vanilla_errors": int((~df["Vanilla_correct"]).sum()),
        "overall": overall,
        "rescue_signatures": signatures.to_dict(orient="records"),
        "unique_rescues": unique_df.to_dict(orient="records"),
        "pairwise_overlap": pairwise_df.to_dict(orient="records"),
    }

    with (out_tables / "case_analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(overall_df.to_string(index=False))
    print("\nRescue signatures")
    print(signatures.to_string(index=False))
    print("\nUnique rescues")
    print(unique_df.to_string(index=False))
    print("\nPairwise overlap")
    print(pairwise_df.to_string(index=False))


if __name__ == "__main__":
    main()
