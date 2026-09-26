# Binary Classification Benchmark: Logistic Regression vs. Neural Networks

A benchmarking pipeline that compares 6 algorithms — plain and regularized
Logistic Regression, and plain and regularized 3-layer Neural Networks —
across 5 binary classification datasets, with statistical significance
testing on the results.

## Contents

```
data/                 the 5 UCI CSV files (no header, label = last column)
data_loader.py         loads + preprocesses each dataset
models.py               the 6 algorithms (LR / LR-L2 / LR-L1 / FNN / FNN-L2 / FNN-L1)
evaluate.py              100x 70/30 hold-out evaluation harness shared by all 6 algorithms
stats_analysis.py       Friedman test + pairwise Wilcoxon -> Win-Tie-Loss table
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Each script is meant to be run and explored individually rather than as one
end-to-end pipeline:

```bash
# Sanity-check the datasets load and preprocess correctly:
python data_loader.py

# Run the 100x hold-out evaluation for a dataset (edit evaluate.py to pick
# which dataset / algorithms to run, or import it interactively):
python evaluate.py

# Run the statistical tests (Friedman + Wilcoxon) over evaluation results:
python stats_analysis.py
```

Running things this way lets you inspect, tweak, and re-run individual
stages (data loading, model definitions, evaluation, statistics) without
committing to the full 5-dataset x 6-algorithm x 100-run sweep every time.

Each stage produces (or can be adapted to save) outputs into a `results/`
folder, such as:

- `raw_results.pkl` — every individual accuracy/F1/AUC value from every one
  of the 100 runs, for every algorithm, for every dataset (so you never
  have to re-run the expensive part just to re-plot or re-analyze).
- `table1_summary.csv` — Mean Accuracy / Mean F1 / Mean AUC per
  dataset x algorithm.
- `roc_<dataset>.png` — one ROC plot per dataset, with all 6 algorithms'
  mean ROC curve (averaged over the 100 runs) overlaid.
- `friedman_results.csv` — an omnibus Friedman test per dataset (are the
  6 algorithms significantly different at all, on that dataset?).
- `table2_win_tie_loss.csv` — the pairwise Win-Tie-Loss table, aggregated
  across all 5 datasets, using paired Wilcoxon signed-rank tests on the
  100 paired accuracy values per pair of algorithms.

## Design choices

- **"3-layer FNN"** is implemented as Input -> 1 Hidden layer (ReLU) ->
  Output (sigmoid), trained with binary cross-entropy. To use a different
  depth, add a second `layers.Dense(...)` in `models.py::_build_fnn_keras`.
- **Paired design**: every hold-out iteration uses the *same* train/test
  split for all 6 algorithms (same `random_state=i`), making the 100
  values per algorithm a paired sample — required for the Wilcoxon
  signed-rank test.
- **Preprocessing is refit every iteration**, on the training split only
  (scaling, imputation, one-hot encoding), to avoid leaking test-set
  information into those statistics.
- **Statistical testing**: Wilcoxon signed-rank (paired, non-parametric)
  was used instead of a paired t-test because accuracy/F1/AUC are bounded
  in [0, 1] and several datasets have runs clustering near ceiling
  performance — this violates the normality assumption a t-test relies
  on. A Friedman test (also non-parametric) is run per dataset as an
  omnibus check across all 6 algorithms at once.

## Timing

5 datasets x 6 algorithms x 100 runs = 3000 model fits, 1500 of them
Keras networks. On a typical laptop CPU this realistically takes
1–3 hours, with Spambase (4,601 rows) the slowest dataset by far. Options
if you need it faster:
- Lower `FNN_EPOCHS` in `models.py` (currently 50).
- Run one dataset / one algorithm at a time instead of the full sweep.
