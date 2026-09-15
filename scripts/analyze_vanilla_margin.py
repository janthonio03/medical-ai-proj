import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.common import read_jsonl


METHODS = ["PH", "VCD", "CRG", "LoBA"]


def summary_stats(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
        }
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-matrix",
        default="analysis/tables/method_case_matrix.csv",
    )
    parser.add_argument(
        "--loba-results",
        default="outputs/loba/oracle_roi_alpha0.3_beta2/closed_predictions.jsonl",
    )
    parser.add_argument("--out", default="analysis/tables")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.case_matrix)
    loba_rows = read_jsonl(args.loba_results)

    margin_rows = []
    for row in loba_rows:
        first_token = row.get("first_token", {})
        margin_rows.append(
            {
                "question_id": str(row["question_id"]),
                "baseline_yes_logp": first_token.get("baseline_yes_logp"),
                "baseline_no_logp": first_token.get("baseline_no_logp"),
            }
        )

    margins = pd.DataFrame(margin_rows)
    df["question_id"] = df["question_id"].astype(str)
    margins["question_id"] = margins["question_id"].astype(str)

    df = df.merge(
        margins,
        on="question_id",
        how="left",
        validate="one_to_one",
    )

    missing = int(df["baseline_yes_logp"].isna().sum())
    if missing:
        raise RuntimeError(f"missing baseline margins: {missing}")

    df["yes_minus_no"] = (
        df["baseline_yes_logp"] - df["baseline_no_logp"]
    )
    df["confidence_margin"] = df["yes_minus_no"].abs()
    df["gt_margin"] = np.where(
        df["gt"].str.lower() == "yes",
        df["yes_minus_no"],
        -df["yes_minus_no"],
    )
    df["wrong_margin"] = np.where(
        ~df["Vanilla_correct"],
        -df["gt_margin"],
        np.nan,
    )
    df["correct_margin"] = np.where(
        df["Vanilla_correct"],
        df["gt_margin"],
        np.nan,
    )

    df["yesno_margin_prediction"] = np.where(
        df["yes_minus_no"] >= 0,
        "yes",
        "no",
    )
    mismatch = int((df["yesno_margin_prediction"] != df["Vanilla"]).sum())

    errors = df[~df["Vanilla_correct"]].copy()
    errors["error_conf_quartile"] = pd.qcut(
        errors["wrong_margin"].rank(method="first"),
        q=4,
        labels=[
            "Q1_low_wrong_conf",
            "Q2",
            "Q3",
            "Q4_high_wrong_conf",
        ],
    )

    rescue_rows = []
    direct_error_rows = []

    for method in METHODS:
        rescued_mask = errors[f"{method}_rescue"]

        for label, mask in [
            ("rescue", rescued_mask),
            ("resistant_error", ~rescued_mask),
        ]:
            stats = summary_stats(errors.loc[mask, "wrong_margin"])
            direct_error_rows.append(
                {
                    "method": method,
                    "group": label,
                    "n": stats["n"],
                    "wrong_margin_mean": stats["mean"],
                    "wrong_margin_median": stats["median"],
                    "wrong_margin_q1": stats["q1"],
                    "wrong_margin_q3": stats["q3"],
                }
            )

    for quartile, group in errors.groupby(
        "error_conf_quartile",
        observed=False,
    ):
        for method in METHODS:
            rescued = int(group[f"{method}_rescue"].sum())
            rescue_rows.append(
                {
                    "quartile": str(quartile),
                    "method": method,
                    "n_errors": len(group),
                    "rescued": rescued,
                    "rescue_rate": rescued / len(group),
                    "median_wrong_margin": float(group["wrong_margin"].median()),
                }
            )

    correct = df[df["Vanilla_correct"]].copy()
    correct["correct_conf_quartile"] = pd.qcut(
        correct["correct_margin"].rank(method="first"),
        q=4,
        labels=[
            "Q1_low_correct_conf",
            "Q2",
            "Q3",
            "Q4_high_correct_conf",
        ],
    )

    harm_rows = []
    direct_correct_rows = []

    for method in METHODS:
        harm_mask = correct[f"{method}_harm"]
        for label, mask in [
            ("harm", harm_mask),
            ("stable_correct", ~harm_mask),
        ]:
            stats = summary_stats(correct.loc[mask, "correct_margin"])
            direct_correct_rows.append(
                {
                    "method": method,
                    "group": label,
                    "n": stats["n"],
                    "correct_margin_mean": stats["mean"],
                    "correct_margin_median": stats["median"],
                    "correct_margin_q1": stats["q1"],
                    "correct_margin_q3": stats["q3"],
                }
            )

    for quartile, group in correct.groupby(
        "correct_conf_quartile",
        observed=False,
    ):
        for method in METHODS:
            harmed = int(group[f"{method}_harm"].sum())
            harm_rows.append(
                {
                    "quartile": str(quartile),
                    "method": method,
                    "n_correct": len(group),
                    "harmed": harmed,
                    "harm_rate": harmed / len(group),
                    "median_correct_margin": float(group["correct_margin"].median()),
                }
            )

    error_group_df = pd.DataFrame(direct_error_rows)
    rescue_df = pd.DataFrame(rescue_rows)
    correct_group_df = pd.DataFrame(direct_correct_rows)
    harm_df = pd.DataFrame(harm_rows)

    df.to_csv(out_dir / "vanilla_margin_case_matrix.csv", index=False)
    error_group_df.to_csv(out_dir / "rescue_vs_resistant_margin.csv", index=False)
    rescue_df.to_csv(out_dir / "error_confidence_quartile_rescue.csv", index=False)
    correct_group_df.to_csv(out_dir / "harm_vs_stable_margin.csv", index=False)
    harm_df.to_csv(out_dir / "correct_confidence_quartile_harm.csv", index=False)

    print(f"N: {len(df)}")
    print(f"Yes/No margin vs Vanilla mismatch: {mismatch}")
    print("\nRESCUE VS RESISTANT ERROR")
    print(error_group_df.to_string(index=False))
    print("\nERROR CONFIDENCE QUARTILES")
    print(rescue_df.to_string(index=False))
    print("\nHARM VS STABLE CORRECT")
    print(correct_group_df.to_string(index=False))
    print("\nCORRECT CONFIDENCE QUARTILES")
    print(harm_df.to_string(index=False))


if __name__ == "__main__":
    main()
