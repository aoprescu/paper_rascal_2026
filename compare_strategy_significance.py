"""
Statistical comparison of training strategies.

Use this script after the training pipeline has produced cv_aggregated.csv.
The unit of analysis is hospital, not CV fold:

    one hospital x one strategy = one paired observation

This avoids treating repeated CV folds as independent observations.

Default test:
  - Friedman test across all strategies.
  - Post-hoc paired Wilcoxon signed-rank tests.
  - Holm correction for multiple comparisons.
  - Default metric: f2_optcal_mean (F2Opt).

Edit the configuration block below and run:
  python compare_strategy_significance.py

Requires:
  pip install pandas numpy scipy
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV = "cv_aggregated.csv"
OUTDIR = "stats_strategy_comparisons"

STRATEGIES = ["Local", "External", "Hybrid", "BigData", "BigDataHybrid"]
METRICS = ["f2_optcal_mean"]  # F2Opt
ALPHA = 0.05
HIGHER_IS_BETTER = True

# Options: "holm" or "bonferroni"
P_CORRECTION = "holm"

# If cv_aggregated.csv has volume_group, also repeat the same analysis by group.
RUN_BY_VOLUME_GROUP = True


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, returned in original order."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(n, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (n - rank) * p_values[idx]
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(n, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def bonferroni_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    return np.minimum(p_values * len(p_values), 1.0)


def paired_effect_summary(a, b):
    diff = a - b
    return {
        "strategy_a_mean": float(np.mean(a)),
        "strategy_b_mean": float(np.mean(b)),
        "mean_diff_a_minus_b": float(np.mean(diff)),
        "median_diff_a_minus_b": float(np.median(diff)),
        "n_a_better": int(np.sum(diff > 0)),
        "n_b_better": int(np.sum(diff < 0)),
        "n_equal": int(np.sum(diff == 0)),
    }


def analyze_one_table(df, metrics, strategies, label, group_value, correction):
    friedman_rows = []
    posthoc_rows = []
    descriptive_rows = []
    rank_rows = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        pivot = df.pivot_table(
            index="hospital",
            columns="strategy",
            values=metric,
            aggfunc="mean",
        )
        present_strategies = [s for s in strategies if s in pivot.columns]
        pivot = pivot[present_strategies].dropna()

        for strategy in present_strategies:
            values = pivot[strategy].to_numpy()
            descriptive_rows.append({
                "analysis": label,
                "group": group_value,
                "metric": metric,
                "strategy": strategy,
                "n_hospitals": len(values),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "iqr": (
                    float(np.percentile(values, 75) - np.percentile(values, 25))
                    if len(values)
                    else np.nan
                ),
            })

        if len(pivot) < 3 or len(present_strategies) < 3:
            friedman_rows.append({
                "analysis": label,
                "group": group_value,
                "metric": metric,
                "n_hospitals": len(pivot),
                "n_strategies": len(present_strategies),
                "test": "friedman",
                "statistic": np.nan,
                "p_value": np.nan,
                "kendall_w": np.nan,
                "note": "Not enough complete paired observations",
            })
            continue

        rank_ascending = not HIGHER_IS_BETTER
        ranks = pivot.rank(axis=1, ascending=rank_ascending, method="average")
        for strategy in present_strategies:
            rank_rows.append({
                "analysis": label,
                "group": group_value,
                "metric": metric,
                "strategy": strategy,
                "mean_rank": float(ranks[strategy].mean()),
                "median_rank": float(ranks[strategy].median()),
            })

        samples = [pivot[s].to_numpy() for s in present_strategies]
        statistic, p_value = stats.friedmanchisquare(*samples)
        n = len(pivot)
        k = len(present_strategies)
        kendall_w = statistic / (n * (k - 1))

        friedman_rows.append({
            "analysis": label,
            "group": group_value,
            "metric": metric,
            "n_hospitals": n,
            "n_strategies": k,
            "test": "friedman",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "kendall_w": float(kendall_w),
            "note": "",
        })

        pair_rows = []
        raw_p_values = []
        for strategy_a, strategy_b in itertools.combinations(present_strategies, 2):
            a = pivot[strategy_a].to_numpy()
            b = pivot[strategy_b].to_numpy()
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                    a,
                    b,
                    zero_method="wilcox",
                    alternative="two-sided",
                )
            except ValueError:
                wilcoxon_stat, wilcoxon_p = np.nan, np.nan

            row = {
                "analysis": label,
                "group": group_value,
                "metric": metric,
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "n_hospitals": n,
                "test": "wilcoxon_signed_rank",
                "statistic": float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else np.nan,
                "p_value": float(wilcoxon_p) if not np.isnan(wilcoxon_p) else np.nan,
                **paired_effect_summary(a, b),
            }
            mean_diff = row["mean_diff_a_minus_b"]
            if mean_diff > 0:
                row["winner_by_mean"] = strategy_a if HIGHER_IS_BETTER else strategy_b
            elif mean_diff < 0:
                row["winner_by_mean"] = strategy_b if HIGHER_IS_BETTER else strategy_a
            else:
                row["winner_by_mean"] = "tie"
            pair_rows.append(row)
            raw_p_values.append(wilcoxon_p)

        valid_mask = ~pd.isna(raw_p_values)
        adjusted = np.full(len(raw_p_values), np.nan, dtype=float)
        if np.any(valid_mask):
            valid_p = np.asarray(raw_p_values, dtype=float)[valid_mask]
            if correction == "bonferroni":
                adjusted[valid_mask] = bonferroni_adjust(valid_p)
            else:
                adjusted[valid_mask] = holm_adjust(valid_p)

        for row, p_adjusted in zip(pair_rows, adjusted):
            row["p_adjusted"] = float(p_adjusted) if not np.isnan(p_adjusted) else np.nan
            row["p_correction"] = correction
            row["alpha"] = ALPHA
            row["significant"] = bool(p_adjusted < ALPHA) if not np.isnan(p_adjusted) else False
            posthoc_rows.append(row)

    return friedman_rows, posthoc_rows, descriptive_rows, rank_rows


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    required = {"hospital", "strategy"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    friedman_rows = []
    posthoc_rows = []
    descriptive_rows = []
    rank_rows = []

    rows_f, rows_p, rows_d, rows_r = analyze_one_table(
        df=df,
        metrics=METRICS,
        strategies=STRATEGIES,
        label="global",
        group_value="all",
        correction=P_CORRECTION,
    )
    friedman_rows.extend(rows_f)
    posthoc_rows.extend(rows_p)
    descriptive_rows.extend(rows_d)
    rank_rows.extend(rows_r)

    if RUN_BY_VOLUME_GROUP and "volume_group" in df.columns:
        for group_value, df_group in df.groupby("volume_group"):
            rows_f, rows_p, rows_d, rows_r = analyze_one_table(
                df=df_group,
                metrics=METRICS,
                strategies=STRATEGIES,
                label="volume_group",
                group_value=group_value,
                correction=P_CORRECTION,
            )
            friedman_rows.extend(rows_f)
            posthoc_rows.extend(rows_p)
            descriptive_rows.extend(rows_d)
            rank_rows.extend(rows_r)

    friedman_df = pd.DataFrame(friedman_rows)
    posthoc_df = pd.DataFrame(posthoc_rows)
    descriptive_df = pd.DataFrame(descriptive_rows)
    rank_df = pd.DataFrame(rank_rows)

    friedman_path = os.path.join(OUTDIR, "friedman_results.csv")
    posthoc_path = os.path.join(OUTDIR, "posthoc_wilcoxon_results.csv")
    descriptive_path = os.path.join(OUTDIR, "strategy_descriptives.csv")
    rank_path = os.path.join(OUTDIR, "friedman_mean_ranks.csv")
    significant_path = os.path.join(OUTDIR, "significant_pairwise_results.csv")

    friedman_df.to_csv(friedman_path, index=False)
    posthoc_df.to_csv(posthoc_path, index=False)
    descriptive_df.to_csv(descriptive_path, index=False)
    rank_df.to_csv(rank_path, index=False)
    if "significant" in posthoc_df.columns:
        posthoc_df[posthoc_df["significant"]].to_csv(significant_path, index=False)
    else:
        pd.DataFrame().to_csv(significant_path, index=False)

    print(f"Saved: {friedman_path}")
    print(f"Saved: {posthoc_path}")
    print(f"Saved: {descriptive_path}")
    print(f"Saved: {rank_path}")
    print(f"Saved: {significant_path}")

    if len(friedman_df):
        print("\nFriedman summary:")
        cols = ["analysis", "group", "metric", "n_hospitals", "p_value", "kendall_w"]
        print(friedman_df[cols].to_string(index=False))

    if len(posthoc_df):
        print("\nPairwise significant comparisons:")
        sig = posthoc_df[posthoc_df["significant"]]
        if len(sig):
            cols = [
                "analysis",
                "group",
                "metric",
                "strategy_a",
                "strategy_b",
                "winner_by_mean",
                "mean_diff_a_minus_b",
                "p_adjusted",
            ]
            print(sig[cols].to_string(index=False))
        else:
            print("No significant pairwise comparisons after correction.")


if __name__ == "__main__":
    main()
