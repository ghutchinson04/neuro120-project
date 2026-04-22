"""Nonlinear baselines (supplementary negative result).

Three model families are compared on identical grouped-CV splits:

1. Linear logistic regression (matches the main-analysis baseline).
2. RBF-SVM on PCA-reduced features.
3. Autoencoder-latent features followed by logistic regression.

All three run on the *same* stimulus-grouped CV splits used elsewhere in
the project, so any performance differences are attributable to model
family rather than to the split. With only 33 electrodes in
Norman-Haignere, nonlinear models do not materially outperform the linear
baseline -- we report this as a negative result rather than suppress it.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config import N_SPLITS, RANDOM_STATE
from decoding import make_grouped_splits, make_logreg


# autoencoder latent feature extractor used by the nonlinear comparison
def _apply_activation(z: np.ndarray, kind: str) -> np.ndarray:
    """Apply an sklearn-style activation name to a raw linear output.

    Used to match the activation of the fitted MLP's first hidden layer
    without relying on private sklearn internals.
    """
    if kind == "identity":
        return z
    if kind == "logistic":
        return 1.0 / (1.0 + np.exp(-z))
    if kind == "tanh":
        return np.tanh(z)
    if kind == "relu":
        return np.maximum(0.0, z)
    raise ValueError(f"Unsupported activation: {kind}")


def _fit_transform_autoencoder(
    train_X: np.ndarray,
    test_X: np.ndarray,
    latent_dim: int = 8,
    pca_dim: int = 32,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale -> PCA (variance denoising) -> 1-hidden-layer autoencoder.

    The encoder activations (``Xtr_pca @ W + b`` passed through the
    MLP's activation function) are returned as the latent features.
    Fit on training folds only; test features are projected without
    any refit, so there is no leakage across folds.

    Parameters
    ----------
    train_X, test_X : np.ndarray
        Feature matrices of shape ``(n_train, n_features)`` and
        ``(n_test, n_features)``.
    latent_dim : int, default=8
        Target latent dimensionality.
    pca_dim : int, default=32
        Dimensionality reduction applied before the autoencoder fit.
    random_state : int, default=:data:`config.RANDOM_STATE`
        Seed for PCA and the MLP.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(Ztr, Zte)`` latent features of shape ``(n_train, latent_dim)``
        and ``(n_test, latent_dim)``.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_X)
    Xte = scaler.transform(test_X)

    n_pca = max(2, min(pca_dim, Xtr.shape[0] - 1, Xtr.shape[1]))
    pca = PCA(n_components=n_pca, random_state=random_state)
    Xtr_p = pca.fit_transform(Xtr)
    Xte_p = pca.transform(Xte)

    n_latent = max(2, min(latent_dim, Xtr_p.shape[1] - 1))
    ae = MLPRegressor(
        hidden_layer_sizes=(n_latent,),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=2000,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ae.fit(Xtr_p, Xtr_p)

    W = ae.coefs_[0]
    b = ae.intercepts_[0]
    Ztr = _apply_activation(Xtr_p @ W + b, ae.activation)
    Zte = _apply_activation(Xte_p @ W + b, ae.activation)
    return Ztr, Zte


# model evaluators that share the same grouped cv splits
def _evaluate_model_on_splits(
    fit_predict_fn,
    X: np.ndarray,
    y_enc: np.ndarray,
    splits,
) -> Tuple[np.ndarray, List[float]]:
    """Run ``fit_predict_fn(X_tr, y_tr, X_te)`` on every CV fold.

    Returns the concatenated held-out predictions and the list of
    per-fold balanced accuracies. The callable signature lets us reuse
    one evaluator across linear logreg, RBF-SVM, and the autoencoder
    pipeline below.
    """
    y_pred = np.empty_like(y_enc)
    fold_bacc: List[float] = []
    for tr, te in splits:
        preds = fit_predict_fn(X[tr], y_enc[tr], X[te])
        y_pred[te] = preds
        fold_bacc.append(balanced_accuracy_score(y_enc[te], preds))
    return y_pred, fold_bacc


def run_nonlinear_comparison(
    tasks: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_STATE,
    latent_dim: int = 8,
) -> pd.DataFrame:
    """Compare linear logreg, RBF-SVM, and autoencoder-latent + logreg.

    Parameters
    ----------
    tasks : dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
        Mapping ``task_name -> (X, y, stimulus_id)``. ``X`` has shape
        ``(n_samples, n_features)``; ``y`` and ``stimulus_id`` are
        ``(n_samples,)``. Each task gets its own seeded grouped-CV
        split so tasks are reproducible independently.
    n_splits : int, default=:data:`config.N_SPLITS`
        Grouped-CV folds.
    seed : int, default=:data:`config.RANDOM_STATE`
        Seed for CV and every model.
    latent_dim : int, default=8
        Autoencoder latent size.

    Returns
    -------
    pandas.DataFrame
        One row per (task, model) with columns
        ``task``, ``model``, ``bacc``, ``macro_f1``,
        ``fold_bacc_std``, and ``recall_<class>`` for each observed class.
    """
    rows: List[Dict[str, object]] = []

    for task_name, (X, y, stimulus_id) in tasks.items():
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        stimulus_id = np.asarray(stimulus_id)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        class_names = list(le.classes_)
        splits = make_grouped_splits(stimulus_id, y_enc, n_splits=n_splits, seed=seed)

        def fit_predict_linear_logreg(Xtr, ytr, Xte):
            """Balanced logistic regression with in-fold scaling."""
            model = make_logreg(class_weight="balanced", random_state=seed)
            model.fit(Xtr, ytr)
            return model.predict(Xte)

        def fit_predict_rbf_svm(Xtr, ytr, Xte):
            """Scale -> PCA (95% var) -> RBF-SVM, all fit on training fold only."""
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=0.95, random_state=seed)),
                    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
                ]
            )
            model.fit(Xtr, ytr)
            return model.predict(Xte)

        def fit_predict_autoencoder_logreg(Xtr, ytr, Xte):
            """Scale -> PCA -> shallow autoencoder latent -> balanced logreg."""
            Ztr, Zte = _fit_transform_autoencoder(
                Xtr, Xte, latent_dim=latent_dim, random_state=seed
            )
            clf = LogisticRegression(
                max_iter=10000, class_weight="balanced", random_state=seed
            )
            clf.fit(Ztr, ytr)
            return clf.predict(Zte)

        for model_name, fn in [
            ("linear_logreg", fit_predict_linear_logreg),
            ("rbf_svm", fit_predict_rbf_svm),
            ("autoencoder_latent_logreg", fit_predict_autoencoder_logreg),
        ]:
            y_pred, fold_bacc = _evaluate_model_on_splits(fn, X, y_enc, splits)
            bacc = float(balanced_accuracy_score(y_enc, y_pred))
            f1 = float(f1_score(y_enc, y_pred, average="macro"))
            row: Dict[str, object] = {
                "task": task_name,
                "model": model_name,
                "bacc": bacc,
                "macro_f1": f1,
                "fold_bacc_std": float(np.std(fold_bacc)),
            }
            # per class recall helps diagnose whether a model is biased towards
            # the larger class, a common failure mode with small imbalanced data
            for c_i, c in enumerate(class_names):
                mask = y_enc == c_i
                if mask.any():
                    row[f"recall_{c}"] = float(
                        balanced_accuracy_score(
                            (y_enc[mask] == c_i).astype(int),
                            (y_pred[mask] == c_i).astype(int),
                        )
                    )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["task", "model"]).reset_index(drop=True)
