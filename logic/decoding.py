"""Grouped cross-validation and time-resolved decoders.

All CV splits are stimulus-grouped via :class:`StratifiedGroupKFold` so
that (a) no stimulus ID ever appears in both train and test folds and
(b) class proportions stay approximately balanced across folds. This
matters here because multiple electrodes observe the *same* stimulus:
without grouping, features from the same stimulus would appear in both
train and test and inflate decoding accuracy.

Feature scaling is always fit inside each training fold via an sklearn
:class:`Pipeline`; this prevents test-set leakage into the scaler
statistics.

This module is pure computation: no file I/O, no plotting, no global
state. Every random choice is seeded by the caller.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import N_SPLITS, RANDOM_STATE
from data_utils import sliding_window_iter


# model factories used across the project
def make_logreg(
    class_weight: Optional[str] = "balanced",
    C: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Return the standard logistic-regression pipeline used across the project.

    We use a balanced logistic regression because the song class is
    roughly half the size of the speech/music classes in Norman-Haignere;
    class weighting keeps the decoder from collapsing to the majority
    prediction.

    Parameters
    ----------
    class_weight : str or None, default="balanced"
        Passed through to :class:`sklearn.linear_model.LogisticRegression`.
    C : float, default=1.0
        Inverse L2 regularisation strength.
    random_state : int, default=:data:`config.RANDOM_STATE`
        Seed for solver tie-breaking.

    Returns
    -------
    sklearn.pipeline.Pipeline
        ``StandardScaler -> LogisticRegression``. ``StandardScaler`` is
        fit inside each training fold automatically when the pipeline
        is used within a CV loop.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=10000,
                    class_weight=class_weight,
                    C=C,
                    random_state=random_state,
                ),
            ),
        ]
    )


# grouped cv splits that prevent stimulus level leakage
def make_grouped_splits(
    stimulus_id: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_STATE,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return stimulus-grouped, class-stratified CV splits.

    We split by stimulus identity rather than by individual rows to
    prevent leakage: multiple feature views of the same stimulus would
    otherwise appear in both train and test folds and inflate decoding
    accuracy.

    Parameters
    ----------
    stimulus_id : np.ndarray
        Group key per row, shape ``(n_samples,)``.
    y : np.ndarray
        Class labels per row, shape ``(n_samples,)``. Used for
        stratification so each fold has a similar class distribution.
    n_splits : int, default=:data:`config.N_SPLITS`
        Number of CV folds.
    seed : int, default=:data:`config.RANDOM_STATE`
        Seed for the shuffled fold assignment.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of ``(train_idx, test_idx)`` pairs.

    Raises
    ------
    ValueError
        If ``stimulus_id`` and ``y`` have different lengths.
    """
    if len(stimulus_id) != len(y):
        raise ValueError(
            f"stimulus_id and y must have the same length; "
            f"got {len(stimulus_id)} and {len(y)}"
        )
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups=stimulus_id))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute the standard classification-metrics bundle used across the project.

    We report balanced accuracy as the primary metric because the song
    class is smaller than the other two in Norman-Haignere; plain
    accuracy would reward majority-class prediction.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels, shape ``(n_samples,)``.
    y_pred : np.ndarray
        Predicted labels, shape ``(n_samples,)``.
    class_names : Sequence[str] or None, default=None
        Display names in class-index order. Defaults to the sorted set
        of observed labels.
    y_proba : np.ndarray or None, default=None
        Predicted positive-class probability for binary tasks, shape
        ``(n_samples,)``. Enables ROC AUC reporting when present.

    Returns
    -------
    dict
        * ``"balanced_accuracy"``, ``"macro_f1"`` : scalar scores.
        * ``"confusion_matrix"`` : raw integer counts as a nested list.
        * ``"confusion_matrix_normalized"`` : row-normalised version.
        * ``"per_class_recall"`` : mapping class name -> recall.
        * ``"class_names"`` : list of class names in row/column order.
        * ``"roc_auc"`` (binary only) : scalar AUC or ``NaN`` if not defined.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    if class_names is None:
        class_names = [str(c) for c in classes]

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    per_class_recall = {
        str(c): float(cm_norm[i, i]) for i, c in enumerate(class_names)
    }

    out: Dict[str, object] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "per_class_recall": per_class_recall,
        "class_names": list(class_names),
    }
    if y_proba is not None and len(classes) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def run_grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    stimulus_id: np.ndarray,
    model_fn: Callable[[], Pipeline] = None,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_STATE,
    return_proba: bool = False,
) -> Dict[str, object]:
    """Run one grouped-CV pass and return held-out predictions plus metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    y : np.ndarray
        Labels, shape ``(n_samples,)``. Any dtype; encoded internally.
    stimulus_id : np.ndarray
        Group key for CV, shape ``(n_samples,)``. Identical IDs are
        guaranteed to stay in the same fold.
    model_fn : Callable[[], Pipeline] or None, default=None
        Factory returning a fresh sklearn estimator per fold. Defaults
        to :func:`make_logreg` with balanced class weighting.
    n_splits : int, default=:data:`config.N_SPLITS`
        Number of CV folds.
    seed : int, default=:data:`config.RANDOM_STATE`
        Seed for the CV splitter and the model.
    return_proba : bool, default=False
        If True, also return an ``(n_samples, n_classes)`` array of
        predicted probabilities (populated fold-by-fold).

    Returns
    -------
    dict
        * ``"y_true"``, ``"y_pred"`` : string-valued arrays, shape ``(n_samples,)``.
        * ``"fold_bacc"`` : per-fold balanced accuracy, shape ``(n_splits,)``.
        * ``"metrics"`` : output of :func:`compute_metrics`.
        * ``"class_names"`` : encoded class order.
        * ``"seed"``, ``"n_splits"`` : echo of the inputs for bookkeeping.
        * ``"proba"`` (when ``return_proba``): ``(n_samples, n_classes)``.

    Raises
    ------
    AssertionError
        If ``X``, ``y``, and ``stimulus_id`` have mismatched row counts.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    stimulus_id = np.asarray(stimulus_id)
    assert X.shape[0] == y.shape[0] == stimulus_id.shape[0], (
        f"row mismatch: X={X.shape[0]}, y={y.shape[0]}, stimulus_id={stimulus_id.shape[0]}"
    )

    if model_fn is None:
        model_fn = lambda: make_logreg(class_weight="balanced", random_state=seed)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    y_pred = np.empty_like(y_enc)
    proba = None
    if return_proba:
        proba = np.zeros((len(y_enc), len(class_names)))

    fold_bacc: List[float] = []
    splits = make_grouped_splits(stimulus_id, y_enc, n_splits=n_splits, seed=seed)
    for tr, te in splits:
        model = model_fn()
        model.fit(X[tr], y_enc[tr])
        y_pred[te] = model.predict(X[te])
        if return_proba and hasattr(model, "predict_proba"):
            proba[te] = model.predict_proba(X[te])
        fold_bacc.append(balanced_accuracy_score(y_enc[te], y_pred[te]))

    # metrics on concatenated held out predictions
    y_true_str = le.inverse_transform(y_enc)
    y_pred_str = le.inverse_transform(y_pred)
    # binary proba for the positive class if it exists and there are two classes
    proba_pos = None
    if return_proba and proba is not None and proba.shape[1] == 2:
        proba_pos = proba[:, 1]
    metrics = compute_metrics(y_true_str, y_pred_str, class_names=class_names, y_proba=proba_pos)

    out: Dict[str, object] = {
        "y_true": y_true_str,
        "y_pred": y_pred_str,
        "fold_bacc": np.array(fold_bacc),
        "metrics": metrics,
        "class_names": class_names,
        "seed": seed,
        "n_splits": n_splits,
    }
    if return_proba:
        out["proba"] = proba
    return out


# time resolved decoders
def _feature_window(tensor: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mean-pool a 3-D tensor over time samples ``[start:end)``.

    Parameters
    ----------
    tensor : np.ndarray
        Shape ``(n_electrodes, n_stimuli, n_time)``.
    start, end : int
        Half-open time-sample range ``[start, end)``.

    Returns
    -------
    np.ndarray
        ``(n_stimuli, n_electrodes)`` feature matrix, ready for sklearn.
    """
    return tensor[:, :, start:end].mean(axis=2).T


def run_time_resolved_binary_decoder(
    X_tensor: np.ndarray,
    y: np.ndarray,
    stimulus_id: np.ndarray,
    t: np.ndarray,
    class_a: str = "song",
    class_b: str = "music",
    subset_idx: Optional[np.ndarray] = None,
    window_sec: float = 0.15,
    step_sec: float = 0.05,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_STATE,
    return_proba: bool = True,
) -> Dict[str, np.ndarray]:
    """Sliding-window binary decoder for ``class_a`` vs ``class_b``.

    At each sliding window, mean-pool the 3-D tensor over time to get a
    ``(n_stim, n_elec)`` feature matrix, then run one grouped-CV pass
    via :func:`run_grouped_cv`. Stimuli whose coarse label is not in
    ``{class_a, class_b}`` are dropped before decoding.

    Parameters
    ----------
    X_tensor : np.ndarray
        ``(n_electrodes, n_stimuli, n_time)`` ECoG tensor.
    y : np.ndarray
        ``(n_stimuli,)`` coarse labels.
    stimulus_id : np.ndarray
        ``(n_stimuli,)`` unique IDs for stimulus-grouped CV.
    t : np.ndarray
        ``(n_time,)`` time axis in seconds.
    class_a, class_b : str
        Labels to keep; all others are dropped.
    subset_idx : np.ndarray or None, default=None
        Electrode indices to keep along axis 0 of ``X_tensor``. None
        means use all electrodes.
    window_sec, step_sec : float, default=(0.15, 0.05)
        Sliding-window parameters in seconds. 50 ms steps preserve
        temporal resolution without making adjacent estimates too
        correlated.
    n_splits : int, default=:data:`config.N_SPLITS`
        Grouped CV folds.
    seed : int, default=:data:`config.RANDOM_STATE`
        Seed for CV and the model.
    return_proba : bool, default=True
        Forwarded to :func:`run_grouped_cv`.

    Returns
    -------
    dict
        * ``"time"`` : ``(n_windows,)`` window-centre times (seconds).
        * ``"bacc"`` : ``(n_windows,)`` held-out balanced accuracy.
        * ``"auc"`` : ``(n_windows,)`` ROC AUC per window (``NaN`` if
          unavailable, e.g. one class missing in a fold).
        * ``"per_fold_bacc"`` : ``(n_windows, n_splits)`` per-fold bacc.
        * ``"y_pred_per_window"`` : dict mapping window index to
          ``(y_true, y_pred)`` pairs (used for window-wise bootstraps).
        * ``"class_a"``, ``"class_b"`` : echo of the inputs.
    """
    X_tensor = np.asarray(X_tensor)
    y = np.asarray(y)
    stimulus_id = np.asarray(stimulus_id)

    keep = np.isin(y, [class_a, class_b])
    X_tensor = X_tensor[:, keep, :]
    y = y[keep]
    stimulus_id = stimulus_id[keep]

    if subset_idx is not None:
        subset_idx = np.asarray(subset_idx)
        X_tensor = X_tensor[subset_idx]

    times: List[float] = []
    bacc: List[float] = []
    auc: List[float] = []
    per_fold: List[np.ndarray] = []
    per_window_pred: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for w_i, (start, end, center) in enumerate(sliding_window_iter(t, window_sec, step_sec)):
        Xw = _feature_window(X_tensor, start, end)
        res = run_grouped_cv(
            Xw,
            y,
            stimulus_id,
            n_splits=n_splits,
            seed=seed,
            return_proba=return_proba,
        )
        times.append(center)
        bacc.append(res["metrics"]["balanced_accuracy"])
        auc.append(float(res["metrics"].get("roc_auc", float("nan"))))
        per_fold.append(res["fold_bacc"])
        per_window_pred[w_i] = (res["y_true"], res["y_pred"])

    return {
        "time": np.array(times),
        "bacc": np.array(bacc),
        "auc": np.array(auc),
        "per_fold_bacc": np.array(per_fold),
        "y_pred_per_window": per_window_pred,
        "class_a": class_a,
        "class_b": class_b,
    }


def cross_temporal_generalization(
    X_tensor: np.ndarray,
    y: np.ndarray,
    stimulus_id: np.ndarray,
    t: np.ndarray,
    subset_idx: Optional[np.ndarray] = None,
    class_a: str = "song",
    class_b: str = "music",
    window_sec: float = 0.15,
    step_sec: float = 0.05,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_STATE,
) -> Dict[str, np.ndarray]:
    """Cross-temporal generalisation: train at time ``i``, test at time ``j``.

    The returned ``(n_t, n_t)`` balanced-accuracy matrix summarises how
    stable the decoding geometry is across time. Entry ``[i, j]`` is
    the held-out balanced accuracy when training features come from
    window ``i`` and test features from window ``j``.

    Parameters
    ----------
    X_tensor : np.ndarray
        ``(n_electrodes, n_stimuli, n_time)`` tensor.
    y : np.ndarray
        ``(n_stimuli,)`` coarse labels.
    stimulus_id : np.ndarray
        ``(n_stimuli,)`` IDs for grouped CV.
    t : np.ndarray
        ``(n_time,)`` time axis in seconds.
    subset_idx : np.ndarray or None, default=None
        Electrode indices to keep.
    class_a, class_b : str
        Binary contrast to decode.
    window_sec, step_sec : float
        Sliding-window parameters.
    n_splits, seed : int
        Grouped-CV controls.

    Returns
    -------
    dict
        * ``"times"`` : ``(n_windows,)`` centre-time array.
        * ``"matrix"`` : ``(n_windows, n_windows)`` generalisation matrix.
        * ``"n_splits"``, ``"seed"`` : echo of the inputs.

    Notes
    -----
    We reuse the *same* grouped CV splits for train and test times so a
    stimulus is never in the training set at one window and the test
    set at another. Without this, cross-temporal scores would be
    inflated by within-stimulus similarity across time.
    """
    X_tensor = np.asarray(X_tensor)
    y = np.asarray(y)
    stimulus_id = np.asarray(stimulus_id)

    keep = np.isin(y, [class_a, class_b])
    X_tensor = X_tensor[:, keep, :]
    y = y[keep]
    stimulus_id = stimulus_id[keep]

    if subset_idx is not None:
        subset_idx = np.asarray(subset_idx)
        X_tensor = X_tensor[subset_idx]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # precompute features for every window
    windows = list(sliding_window_iter(t, window_sec, step_sec))
    n_win = len(windows)
    feats: List[np.ndarray] = [_feature_window(X_tensor, s, e) for s, e, _ in windows]
    times = np.array([c for _, _, c in windows])

    # shared grouped splits so train and test never overlap at the stimulus level
    splits = make_grouped_splits(stimulus_id, y_enc, n_splits=n_splits, seed=seed)

    matrix = np.zeros((n_win, n_win), dtype=float)
    # aggregate test predictions per (i, j) over folds then compute balanced
    # accuracy once so the whole cross temporal matrix uses the same averaging
    for i in range(n_win):
        # accumulators keyed by train window
        y_true_all = {j: [] for j in range(n_win)}
        y_pred_all = {j: [] for j in range(n_win)}
        for tr, te in splits:
            model = make_logreg(class_weight="balanced", random_state=seed)
            model.fit(feats[i][tr], y_enc[tr])
            for j in range(n_win):
                y_pred_all[j].append(model.predict(feats[j][te]))
                y_true_all[j].append(y_enc[te])
        for j in range(n_win):
            yt = np.concatenate(y_true_all[j])
            yp = np.concatenate(y_pred_all[j])
            matrix[i, j] = balanced_accuracy_score(yt, yp)

    return {
        "times": times,
        "matrix": matrix,
        "n_splits": n_splits,
        "seed": seed,
    }
