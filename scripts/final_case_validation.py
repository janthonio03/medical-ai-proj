import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["PH", "VCD", "CRG", "LoBA"]


def exact_mcnemar(rescue, harm):
    n = rescue + harm
    if n == 0:
        return 1.0

    k = min(rescue, harm)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def holm_adjust(pvalues):
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted = {}
    previous = 0.0
    m = len(ordered)

    for rank, (name, pvalue) in enumerate(ordered):
        value = min(1.0, (m - rank) * pvalue)
        value = max(value, previous)
        adjusted[name] = value
        previous = value

    return adjusted


def paired_bootstrap_diff(
    n,
    rescue,
    harm,
    n_boot=50000,
    seed=42,
):
    unchanged = n - rescue - harm
    probs = np.asarray([rescue, harm, unchanged], dtype=float)
    probs /= probs.sum()

    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, probs, size=n_boot)
    diffs = (draws[:, 0] - draws[:, 1]) / n
    low, high = np.quantile(diffs, [0.025, 0.975])

    return (
        float((rescue - harm) / n),
        float(low),
        float(high),
    )


def pick_near_median(df, mask, margin_col, phenotype, k=3):
    subset = df[mask].copy()
    if len(subset) == 0:
        return pd.DataFrame()

    median = subset[margin_col].median()
    subset["_distance"] = (subset[margin_col] - median).abs()
    subset = subset.sort_values(["_distance", "question_id"]).head(k)
    subset["phenotype"] = phenotype
    subset["selection_note"] = f"closest to median {margin_col}"
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="analysis/tables/vanilla_margin_case_matrix.csv",
    )
    parser.add_argument("--out", default="analysis/tables")
    parser.add_argument("--bootstrap", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    results = []
    raw_p = {}

    for method in METHODS:
        rescue = int(df[f"{method}_rescue"].sum())
        harm = int(df[f"{method}_harm"].sum())
        correct = int(df[f"{method}_correct"].sum())

        pvalue = exact_mcnemar(rescue, harm)
        raw_p[method] = pvalue

        delta, ci_low, ci_high = paired_bootstrap_diff(
            n=len(df),
            rescue=rescue,
            harm=harm,
            n_boot=args.bootstrap,
            seed=args.seed,
        )

        results.append(
            {
                "method": method,
                "n": len(df),
                "vanilla_accuracy": float(df["Vanilla_correct"].mean()),
                "method_accuracy": correct / len(df),
                "delta_accuracy": delta,
                "delta_pp": 100.0 * delta,
                "rescue": rescue,
                "harm": harm,
                "discordant": rescue + harm,
                "mcnemar_exact_p": pvalue,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "bootstrap_ci95_low_pp": 100.0 * ci_low,
                "bootstrap_ci95_high_pp": 100.0 * ci_high,
            }
        )

    adjusted = holm_adjust(raw_p)
    for row in results:
        row["holm_adjusted_p"] = adjusted[row["method"]]
        row["significant_raw_0.05"] = row["mcnemar_exact_p"] < 0.05
        row["significant_holm_0.05"] = row["holm_adjusted_p"] < 0.05

    stats_df = pd.DataFrame(results)
    stats_df.to_csv(out_dir / "final_statistical_validation.csv", index=False)

    candidates = []

    definitions = [
        (
            "VCD_only_rescue",
            df["VCD_rescue"]
            & ~df["PH_rescue"]
            & ~df["CRG_rescue"]
            & ~df["LoBA_rescue"],
            "wrong_margin",
        ),
        (
            "CRG_only_rescue",
            df["CRG_rescue"]
            & ~df["PH_rescue"]
            & ~df["VCD_rescue"]
            & ~df["LoBA_rescue"],
            "wrong_margin",
        ),
        (
            "PH_or_LoBA_rescue",
            df["PH_rescue"] | df["LoBA_rescue"],
            "wrong_margin",
        ),
        ("VCD_harm", df["VCD_harm"], "correct_margin"),
        ("CRG_harm", df["CRG_harm"], "correct_margin"),
    ]

    for phenotype, mask, margin_col in definitions:
        selected = pick_near_median(
            df,
            mask,
            margin_col,
            phenotype,
            k=3,
        )
        if len(selected):
            candidates.append(selected)

    all_failed = (
        ~df["Vanilla_correct"]
        & ~df["PH_rescue"]
        & ~df["VCD_rescue"]
        & ~df["CRG_rescue"]
        & ~df["LoBA_rescue"]
    )

    resistant = (
        df[all_failed]
        .sort_values("wrong_margin", ascending=False)
        .head(5)
        .copy()
    )
    resistant["phenotype"] = "high_confidence_resistant"
    resistant["selection_note"] = "highest Vanilla wrong margin"
    candidates.append(resistant)

    candidates_df = pd.concat(candidates, ignore_index=True)
    keep_cols = [
        "phenotype",
        "selection_note",
        "question_id",
        "question",
        "gt",
        "anatomy",
        "disease",
        "roi_area_ratio",
        "Vanilla",
        "PH",
        "VCD",
        "CRG",
        "LoBA",
        "yes_minus_no",
        "wrong_margin",
        "correct_margin",
    ]
    candidates_df = candidates_df[
        [col for col in keep_cols if col in candidates_df.columns]
    ]
    candidates_df.to_csv(out_dir / "qualitative_case_candidates.csv", index=False)

    summary = {
        "n": len(df),
        "statistics": stats_df.to_dict(orient="records"),
        "qualitative_candidate_n": len(candidates_df),
    }
    with (out_dir / "final_validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("MCNEMAR + BOOTSTRAP")
    print(
        stats_df[
            [
                "method",
                "method_accuracy",
                "delta_pp",
                "rescue",
                "harm",
                "mcnemar_exact_p",
                "holm_adjusted_p",
                "bootstrap_ci95_low_pp",
                "bootstrap_ci95_high_pp",
            ]
        ].to_string(index=False)
    )
    print("\nQUALITATIVE CASE CANDIDATES")
    print(candidates_df.to_string(index=False))


if __name__ == "__main__":
    main()
