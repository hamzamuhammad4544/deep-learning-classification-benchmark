"""
models.py
---------
The 6 algorithms:

  1. Logistic Regression                      (LR)
  2. Logistic Regression + L2                 (LR-L2)
  3. Logistic Regression + L1                 (LR-L1)
  4. 3-layer Fully Connected Neural Network    (FNN)
  5. 3-layer FNN + L2                          (FNN-L2)
  6. 3-layer FNN + L1                          (FNN-L1)

"""

from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

HIDDEN_UNITS = 16  # kept modest: several of the datasets have <500 rows
FNN_EPOCHS = 50
FNN_BATCH_SIZE = 32
FNN_VERBOSE = 0


# ---------------------------------------------------------------------
# Logistic Regression variants (sklearn, default parameters otherwise)
# ---------------------------------------------------------------------
def build_lr():
    return LogisticRegression(penalty=None, max_iter=1000)


def build_lr_l2():
    return LogisticRegression(penalty="l2", max_iter=1000)  # sklearn default


def build_lr_l1():
    return LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)


# ---------------------------------------------------------------------
# 3-layer Fully Connected Neural Network variants (Keras)
# ---------------------------------------------------------------------
def _build_fnn_keras(input_dim, regularizer=None):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(HIDDEN_UNITS, activation="relu",
                     kernel_regularizer=regularizer),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",              # Keras default
        loss="binary_crossentropy",    # cross-entropy, as required
        metrics=["accuracy"],
    )
    return model


class KerasFNNWrapper:
    """Thin wrapper so the FNN exposes the same .fit / .predict / .predict_proba
    interface as the sklearn models, keeping evaluate.py identical for all 6
    algorithms."""

    def __init__(self, regularizer=None, input_dim=None):
        self.regularizer = regularizer
        self.input_dim = input_dim
        self.model = None

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        self.model = _build_fnn_keras(self.input_dim, self.regularizer)
        self.model.fit(
            X, y,
            epochs=FNN_EPOCHS,
            batch_size=FNN_BATCH_SIZE,
            verbose=FNN_VERBOSE,
        )
        return self

    def predict_proba_pos(self, X):
        """Return P(class=1) for each row, shape (n_samples,)."""
        return self.model.predict(X, verbose=0).ravel()

    def predict(self, X):
        return (self.predict_proba_pos(X) >= 0.5).astype(int)


def build_fnn():
    return KerasFNNWrapper(regularizer=None)


def build_fnn_l2():
    return KerasFNNWrapper(regularizer=regularizers.l2(1e-3))


def build_fnn_l1():
    return KerasFNNWrapper(regularizer=regularizers.l1(1e-3))


# ---------------------------------------------------------------------
# Registry used by evaluate.py / run_all.py
# ---------------------------------------------------------------------
ALGORITHMS = {
    "LR": build_lr,
    "LR-L2": build_lr_l2,
    "LR-L1": build_lr_l1,
    "FNN": build_fnn,
    "FNN-L2": build_fnn_l2,
    "FNN-L1": build_fnn_l1,
}

IS_KERAS = {
    "LR": False, "LR-L2": False, "LR-L1": False,
    "FNN": True, "FNN-L2": True, "FNN-L1": True,
}
