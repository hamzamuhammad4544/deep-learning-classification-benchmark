"""
data_loader.py
--------------
Loads the 5 UCI classification datasets and returns, for each one:
  - X : raw feature DataFrame (mixed types, missing values as NaN)
  - y : binary target as a 0/1 numpy array
  - preprocessor : an *unfit* sklearn ColumnTransformer that imputes,
                    scales numeric columns and one-hot encodes categorical
                    columns.

IMPORTANT: the preprocessor is intentionally left UNFIT here. It must be
fit on the training split only, inside each hold-out iteration, and then
used to transform the test split. Fitting it on the whole dataset up
front would leak test-set information into the scaler / encoder / imputer
statistics.

All 5 CSV files are expected in ./data/ with NO header row and the class
label as the LAST column (this matches how they were downloaded from the
UCI mirrors).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

DATA_DIR = "data"

# Any column with at most this many distinct values (after dropping NaN)
# is treated as categorical -> one-hot encoded. Columns with more unique
# values are treated as continuous -> imputed (median) + scaled.
CATEGORICAL_MAX_UNIQUE = 10


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Auto-detect categorical vs. continuous columns and build a
    ColumnTransformer that imputes + scales / one-hot encodes accordingly."""
    categorical_cols, numeric_cols = [], []
    for col in X.columns:
        series = X[col]
        is_object = series.dtype == object
        n_unique = series.nunique(dropna=True)
        if is_object or n_unique <= CATEGORICAL_MAX_UNIQUE:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def _binary_encode(series: pd.Series) -> np.ndarray:
    """Map a 2-valued label column to 0/1 (sorted lexicographic order)."""
    uniques = sorted(series.astype(str).unique())
    if len(uniques) != 2:
        raise ValueError(f"Expected a binary target, got values: {uniques}")
    mapping = {uniques[0]: 0, uniques[1]: 1}
    return series.astype(str).map(mapping).values.astype(int)


def load_ionosphere():
    df = pd.read_csv(f"{DATA_DIR}/ionosphere.csv", header=None)
    X = df.iloc[:, :-1].apply(pd.to_numeric)
    y = _binary_encode(df.iloc[:, -1])
    return X, y, _build_preprocessor(X)


def load_sonar():
    df = pd.read_csv(f"{DATA_DIR}/sonar.csv", header=None)
    X = df.iloc[:, :-1].apply(pd.to_numeric)
    y = _binary_encode(df.iloc[:, -1])
    return X, y, _build_preprocessor(X)


def load_spambase():
    df = pd.read_csv(f"{DATA_DIR}/spambase.csv", header=None)
    X = df.iloc[:, :-1].apply(pd.to_numeric)
    y = df.iloc[:, -1].astype(int).values
    return X, y, _build_preprocessor(X)


def load_german():
    # Statlog German Credit (categorical version): 20 attributes, mix of
    # string codes (e.g. 'A11', 'A34') and numeric values. Target is the
    # last column: 1 = good credit risk, 2 = bad credit risk.
    df = pd.read_csv(f"{DATA_DIR}/german.csv", header=None, sep=r"\s+")
    # (german.csv from the UCI mirror is comma-separated; sep regex above
    #  also tolerates a whitespace-separated variant if you swap files.)
    if df.shape[1] == 1:  # fallback in case it wasn't whitespace-separated
        df = pd.read_csv(f"{DATA_DIR}/german.csv", header=None)
    X = df.iloc[:, :-1]
    y = _binary_encode(df.iloc[:, -1])
    return X, y, _build_preprocessor(X)


def load_horse_colic():
    # Horse Colic: 28 documented attributes. Column 2 (0-indexed) is the
    # Hospital Number, a record ID with no predictive value, so it is
    # dropped. Column 23 (0-indexed) -- "surgical lesion?" (1=yes, 2=no)
    # -- is used as the binary target, matching the two classes present
    # in this file. Missing values are marked '?' in the raw file.
    df = pd.read_csv(f"{DATA_DIR}/horse-colic.csv", header=None, na_values="?")
    target_col = 23
    id_col = 2
    y = _binary_encode(df[target_col].astype(int).astype(str))
    feature_cols = [c for c in df.columns if c not in (target_col, id_col)]
    X = df[feature_cols].reset_index(drop=True)
    X.columns = range(X.shape[1])  # renumber to a contiguous 0..N-1 range;
    # ColumnTransformer treats integer column labels as POSITIONAL indices,
    # so the gaps left by dropping columns 2 and 23 must be closed here.
    return X, y, _build_preprocessor(X)


DATASETS = {
    "Ionosphere": load_ionosphere,
    "Sonar": load_sonar,
    "German Credit": load_german,
    "Spambase": load_spambase,
    "Horse Colic": load_horse_colic,
}


if __name__ == "__main__":
    # Quick sanity check when run directly: python data_loader.py
    for name, loader in DATASETS.items():
        X, y, pre = loader()
        print(f"{name:15s} X={X.shape}  y={y.shape}  classes={np.unique(y)}  "
              f"class_balance={np.bincount(y)}")
