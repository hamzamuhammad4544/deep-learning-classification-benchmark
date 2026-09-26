"""
evaluate.py
-----------
Shared 100x hold-out (70/30) evaluation harness.

For a given dataset (raw X, y, preprocessor) this runs, for i = 1..N_RUNS:
  1. A stratified 70/30 train/test split using random_state = i.
     The SAME split is reused for all 6 algorithms in that iteration, so
     the resulting 100 values per algorithm are paired across algorithms
     -- this is what makes the significance testing in stats_analysis.py
     valid (paired tests, not independent-samples tests).
  2. Fit the (cloned) preprocessor on the train split only, transform both
     splits (prevents test-set leakage into scaling/encoding statistics).
  3. Train every one of the 6 algorithms from scratch on the transformed
     training data.
  4. Record accuracy, F1, and AUC on the transformed test data, plus the
     raw (fpr, tpr) ROC points for that run.

Returns a nested dict:
    results[dataset_name][algorithm_name] = {
        "accuracy": np.array of length N_RUNS,
        "f1":       np.array of length N_RUNS,
        "auc":      np.array of length N_RUNS,
        "roc":      list of N_RUNS (fpr, tpr) tuples,
    }
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from models import ALGORITHMS, IS_KERAS

N_RUNS = 100
TEST_SIZE = 0.30


def _fit_predict(algo_name, builder, X_train, y_train, X_test):
    model = builder()
    model.fit(X_train, y_train)
    if IS_KERAS[algo_name]:
        proba = model.predict_proba_pos(X_test)
        pred = (proba >= 0.5).astype(int)
    else:
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
    return pred, proba


def run_holdout_experiment(X, y, preprocessor, n_runs=N_RUNS, verbose=True):
    results = {name: {"accuracy": [], "f1": [], "auc": [], "roc": []}
               for name in ALGORITHMS}

    for i in range(n_runs):
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=i, stratify=y
        )

        pre = clone(preprocessor)
        X_train = pre.fit_transform(X_train_raw)
        X_test = pre.transform(X_test_raw)
        # Keras needs dense float32 arrays; sklearn is fine with either.
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
            X_test = X_test.toarray()
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)

        for algo_name, builder in ALGORITHMS.items():
            pred, proba = _fit_predict(algo_name, builder, X_train, y_train, X_test)
            results[algo_name]["accuracy"].append(accuracy_score(y_test, pred))
            results[algo_name]["f1"].append(f1_score(y_test, pred))
            results[algo_name]["auc"].append(roc_auc_score(y_test, proba))
            fpr, tpr, _ = roc_curve(y_test, proba)
            results[algo_name]["roc"].append((fpr, tpr))

        if verbose and (i + 1) % 10 == 0:
            print(f"    run {i + 1}/{n_runs} done")

    for algo_name in ALGORITHMS:
        results[algo_name]["accuracy"] = np.array(results[algo_name]["accuracy"])
        results[algo_name]["f1"] = np.array(results[algo_name]["f1"])
        results[algo_name]["auc"] = np.array(results[algo_name]["auc"])

    return results


def mean_roc_curve(roc_list, n_points=100):
    """Interpolate each run's ROC curve onto a common FPR grid and average
    the TPRs, so multiple runs collapse into a single mean ROC curve."""
    mean_fpr = np.linspace(0, 1, n_points)
    tprs = []
    for fpr, tpr in roc_list:
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr
