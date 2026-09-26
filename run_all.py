"""
run_all.py
----------
Runs the entire assignment pipeline end to end:

  1. For each of the 5 datasets, run the 100x 70/30 hold-out evaluation
     for all 6 algorithms (evaluate.py).
  2. Save the raw per-run metric arrays to results/raw_results.pkl (so you
     never have to re-run the expensive part just to re-plot or re-test).
  3. Build and save Table 1 (Mean Accuracy / Mean F1 / Mean AUC) as
     results/table1_summary.csv.
  4. Plot the 6 mean ROC curves per dataset -> results/roc_<dataset>.png.
  5. Run the Friedman test per dataset -> results/friedman_results.csv.
  6. Build the Win-Tie-Loss table -> results/table2_win_tie_loss.csv.

Usage:
    python run_all.py                  # full 100 runs per dataset (slow)
    python run_all.py --n_runs 5        # quick smoke test

Expect the full run to take a while: 5 datasets x 6 algorithms x 100 runs
= 3000 fits, of which 1500 are Keras models. On a laptop CPU this is
realistically on the order of 1-3 hours; Spambase (4601 rows) is the
slowest dataset. Consider running it in the background / overnight, or
lowering FNN_EPOCHS in models.py if you need faster turnaround.
"""

import argparse
import os
import pickle
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import DATASETS
from evaluate import run_holdout_experiment, mean_roc_curve
from stats_analysis import summary_table, friedman_per_dataset, win_tie_loss_table

RESULTS_DIR = "results"


def plot_roc(dataset_name, algo_results):
    n_runs_used = len(next(iter(algo_results.values()))["accuracy"])
    plt.figure(figsize=(6, 6))
    for algo_name, metrics in algo_results.items():
        mean_fpr, mean_tpr = mean_roc_curve(metrics["roc"])
        mean_auc = metrics["auc"].mean()
        plt.plot(mean_fpr, mean_tpr, label=f"{algo_name} (AUC={mean_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC over {n_runs_used} runs - {dataset_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/roc_{dataset_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main(n_runs):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for dataset_name, loader in DATASETS.items():
        print(f"\n=== {dataset_name} ===")
        X, y, preprocessor = loader()
        results = run_holdout_experiment(X, y, preprocessor, n_runs=n_runs)
        all_results[dataset_name] = results
        plot_path = plot_roc(dataset_name, results)
        print(f"  saved ROC plot -> {plot_path}")

    with open(f"{RESULTS_DIR}/raw_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved raw per-run results -> {RESULTS_DIR}/raw_results.pkl")

    table1 = summary_table(all_results)
    table1.to_csv(f"{RESULTS_DIR}/table1_summary.csv", index=False)
    print(f"Saved Table 1 -> {RESULTS_DIR}/table1_summary.csv")
    print(table1.pivot(index="Algorithm", columns="Dataset",
                        values="Mean Accuracy").round(3))

    friedman_df = friedman_per_dataset(all_results, metric="accuracy")
    friedman_df.to_csv(f"{RESULTS_DIR}/friedman_results.csv", index=False)
    print(f"\nSaved Friedman test results -> {RESULTS_DIR}/friedman_results.csv")
    print(friedman_df)

    wtl = win_tie_loss_table(all_results, metric="accuracy")
    wtl.to_csv(f"{RESULTS_DIR}/table2_win_tie_loss.csv")
    print(f"\nSaved Table 2 (Win-Tie-Loss, accuracy) -> "
          f"{RESULTS_DIR}/table2_win_tie_loss.csv")
    print(wtl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=100,
                         help="Number of hold-out repetitions (default: 100)")
    args = parser.parse_args()
    main(args.n_runs)
