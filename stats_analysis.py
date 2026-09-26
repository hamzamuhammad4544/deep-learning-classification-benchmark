"""
stats_analysis.py
------------------
Part 3 of the assignment: statistical testing across the 100 paired values
per algorithm, per dataset.

Method chosen: Wilcoxon signed-rank test (paired, non-parametric), with a
Friedman test as an overall omnibus check per dataset.

Why non-parametric / paired tests:
  * The 100 values per algorithm come from the SAME 100 train/test splits
    across all 6 algorithms (see evaluate.py) -- i.e. related / paired
    samples, not independent ones. A paired test (Wilcoxon signed-rank,
    or a paired t-test) is required, not an independent two-sample test.
  * Accuracy / F1 / AUC are bounded in [0, 1] and, especially on the
    easier datasets, cluster near the ceiling -- this violates the
    normality assumption that a paired t-test relies on. The Wilcoxon
    signed-rank test makes no distributional assumption, only that the
    paired differences are symmetric, which is a much safer bet here.
  * The Friedman test is the natural omnibus counterpart: it ranks all 6
    algorithms within each of the 100 runs and tests whether the average
    ranks differ, without assuming normality either.

Win / Tie / Loss definition (per dataset, per pair of algorithms):
  * Win  : row algorithm's paired Wilcoxon test says it is significantly
           BETTER than the column algorithm (p < ALPHA) and its mean
           metric is higher.
  * Loss : row algorithm is significantly WORSE (p < ALPHA, mean lower).
  * Tie  : p >= ALPHA (no significant difference detected).

These per-dataset outcomes are then summed across the 5 datasets into a
single "wins-ties-losses" string per cell, e.g. "3-1-1" = row algorithm
won on 3 datasets, tied on 1, lost on 1 (out of 5).
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

ALPHA = 0.05


def friedman_per_dataset(results, metric="accuracy"):
    """Run one Friedman test per dataset across all 6 algorithms.
    Returns a DataFrame: dataset, statistic, p_value, significant."""
    rows = []
    for dataset_name, algo_results in results.items():
        algo_names = list(algo_results.keys())
        samples = [algo_results[a][metric] for a in algo_names]
        stat, p = friedmanchisquare(*samples)
        rows.append({
            "dataset": dataset_name,
            "statistic": stat,
            "p_value": p,
            "significant (p<0.05)": p < ALPHA,
        })
    return pd.DataFrame(rows)


def _pairwise_outcome(a_values, b_values):
    """Wilcoxon signed-rank test between two paired arrays.
    Returns 'win', 'tie', or 'loss' from a_values's perspective."""
    diff = a_values - b_values
    if np.allclose(diff, 0):
        return "tie"
    try:
        stat, p = wilcoxon(a_values, b_values)
    except ValueError:
        # all differences identical/zero-variance edge case
        return "tie"
    if p >= ALPHA:
        return "tie"
    return "win" if a_values.mean() > b_values.mean() else "loss"


def win_tie_loss_table(results, metric="accuracy"):
    """Build the Win-Tie-Loss table (aggregated across all datasets),
    in the exact row/column layout the assignment asks for."""
    algo_names = list(next(iter(results.values())).keys())
    counts = {a: {b: {"win": 0, "tie": 0, "loss": 0} for b in algo_names}
              for a in algo_names}

    for dataset_name, algo_results in results.items():
        for a in algo_names:
            for b in algo_names:
                if a == b:
                    continue
                outcome = _pairwise_outcome(algo_results[a][metric],
                                             algo_results[b][metric])
                counts[a][b][outcome] += 1

    table = pd.DataFrame(index=algo_names, columns=algo_names, dtype=object)
    for a in algo_names:
        for b in algo_names:
            if a == b:
                table.loc[a, b] = "-"
            else:
                c = counts[a][b]
                table.loc[a, b] = f"{c['win']}-{c['tie']}-{c['loss']}"
    return table


def summary_table(results):
    """Table 1: mean accuracy / mean F1 / mean AUC per dataset x algorithm."""
    rows = []
    for dataset_name, algo_results in results.items():
        for algo_name, metrics in algo_results.items():
            rows.append({
                "Dataset": dataset_name,
                "Algorithm": algo_name,
                "Mean Accuracy": metrics["accuracy"].mean(),
                "Mean F1": metrics["f1"].mean(),
                "Mean AUC": metrics["auc"].mean(),
            })
    return pd.DataFrame(rows)
