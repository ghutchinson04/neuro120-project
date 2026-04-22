"""Representational dissimilarity matrices (RDMs) and divergence statistics.

We quantify how strongly the neural geometry separates two stimulus
classes using a size-invariant *divergence* statistic::

    D(a, b) = mean_between(a, b) - 0.5 * (mean_within(a) + mean_within(b))

where ``mean_between`` averages the off-block RDM cells connecting
class ``a`` rows to class ``b`` rows, and ``mean_within`` averages the
strictly upper-triangular cells inside a class block. Subtracting the
within-class baseline makes the statistic robust to differences in
class-internal noise or compression.

This module is pure computation: every function is deterministic given
its inputs (plus ``seed`` for the resampling tests) and has no file I/O
or plotting.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from data_utils import sliding_window_iter


# acoustic regressor helpers used by the partialled divergence path

def partial_out_features(
    X: np.ndarray,
    A: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    """Residualise each column of ``X`` against the regressor matrix ``A``.

    Both matrices are mean-centred across stimuli so the intercept is
    absorbed implicitly. Returns ``X_centred - A_centred @ beta``, where
    ``beta`` solves the (optionally ridge-penalised) least-squares
    problem ``A_centred @ beta ≈ X_centred``.

    Parameters
    ----------
    X : np.ndarray
        Neural features at one time-window, shape ``(n_stimuli, n_features)``.
    A : np.ndarray
        Stimulus-level regressors (e.g. acoustic features), shape
        ``(n_stimuli, n_regressors)``.
    ridge : float, default=0.0
        Ridge penalty on ``beta``. ``0`` uses the Moore-Penrose
        pseudo-inverse, which is numerically stable even when ``A`` is
        rank-deficient but silently ignores colinear directions. A
        small positive ridge (e.g. ``1.0``) stabilises the fit when
        acoustic regressors are highly correlated.

    Returns
    -------
    np.ndarray
        Residualised features of shape ``(n_stimuli, n_features)``.

    Raises
    ------
    ValueError
        If the row counts of ``X`` and ``A`` disagree.
    """
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=float)
    if X.shape[0] != A.shape[0]:
        raise ValueError(
            f"row mismatch: X has {X.shape[0]} stim, A has {A.shape[0]}"
        )
    A_c = A - A.mean(axis=0, keepdims=True)
    X_c = X - X.mean(axis=0, keepdims=True)
    if ridge > 0:
        k = A_c.shape[1]
        gram = A_c.T @ A_c + ridge * np.eye(k)
        beta = np.linalg.solve(gram, A_c.T @ X_c)
    else:
        beta = np.linalg.pinv(A_c) @ X_c
    return X_c - A_c @ beta


# rdm and divergence computations

def compute_rdm(
    X: np.ndarray,
    metric: str = "correlation",
    standardize: bool = True,
) -> np.ndarray:
    """Return the symmetric representational dissimilarity matrix of ``X``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)`` where each
        row is one stimulus.
    metric : str, default="correlation"
        Any distance accepted by :func:`scipy.spatial.distance.pdist`
        (``"correlation"``, ``"cosine"``, ``"euclidean"``, ...).
    standardize : bool, default=True
        If True, z-score each feature across stimuli before computing
        distances. Prevents a single high-variance feature from
        dominating correlation/cosine/euclidean metrics.

    Returns
    -------
    np.ndarray
        Symmetric RDM of shape ``(n_samples, n_samples)`` with zero
        diagonal.

    Raises
    ------
    AssertionError
        If ``X`` is not 2-D.
    """
    X = np.asarray(X, dtype=float)
    assert X.ndim == 2, f"X must be 2-D (n_samples, n_features); got shape {X.shape}"
    if standardize:
        X = StandardScaler().fit_transform(X)
    return squareform(pdist(X, metric=metric))


def compute_divergence(
    rdm: np.ndarray,
    labels: Sequence[str],
    pair: Tuple[str, str] = ("song", "music"),
) -> Dict[str, float]:
    """Compute between-minus-within representational divergence for a class pair.

    Parameters
    ----------
    rdm : np.ndarray
        Square representational dissimilarity matrix of shape
        ``(n_stimuli, n_stimuli)``. Lower values indicate more similar
        representations.
    labels : Sequence[str]
        Array of class labels of shape ``(n_stimuli,)``.
    pair : tuple[str, str], default=("song", "music")
        The two classes whose between-category dissimilarity is compared
        against their within-category dissimilarity.

    Returns
    -------
    dict[str, float | tuple[str, str]]
        * ``"between"``  -- mean dissimilarity between the two classes.
        * ``"within_a"`` -- mean within-class dissimilarity for ``pair[0]``.
        * ``"within_b"`` -- mean within-class dissimilarity for ``pair[1]``.
        * ``"divergence"`` -- ``between - 0.5 * (within_a + within_b)``.
        * ``"pair"`` -- echo of the input pair for downstream bookkeeping.

    Raises
    ------
    ValueError
        If either class has fewer than two stimuli (a within-class
        block needs at least one off-diagonal cell to be well defined).

    Notes
    -----
    This metric is used throughout the project to quantify how strongly
    the neural geometry separates song from music beyond within-class
    variability. Subtracting the within-class baseline makes the
    statistic insensitive to class-internal scale: a subset whose
    song/music blocks are both compact but well-separated will score
    high, even if its absolute distances are small.
    """
    labels_arr = np.asarray(labels)
    a, b = pair
    ia = np.where(labels_arr == a)[0]
    ib = np.where(labels_arr == b)[0]
    if ia.size < 2 or ib.size < 2:
        raise ValueError(
            f"Need at least 2 samples per class; got {a}={ia.size}, {b}={ib.size}"
        )

    between = float(rdm[np.ix_(ia, ib)].mean())
    wa_block = rdm[np.ix_(ia, ia)]
    wb_block = rdm[np.ix_(ib, ib)]
    within_a = float(wa_block[np.triu_indices_from(wa_block, k=1)].mean())
    within_b = float(wb_block[np.triu_indices_from(wb_block, k=1)].mean())
    divergence = between - 0.5 * (within_a + within_b)
    return {
        "between": between,
        "within_a": within_a,
        "within_b": within_b,
        "divergence": divergence,
        "pair": (a, b),
    }


# sliding window wrappers that sweep the rdm across time

def _divergence_from_window(
    X_tensor: np.ndarray,
    labels: np.ndarray,
    start: int,
    end: int,
    pair: Tuple[str, str],
    metric: str,
    A: Optional[np.ndarray] = None,
    ridge: float = 0.0,
) -> Dict[str, float]:
    """Score divergence for one sliding-window slice.

    Mean-pools ``X_tensor[:, :, start:end]`` to a ``(n_stim, n_elec)``
    feature matrix, optionally residualises against ``A``, and returns
    the full :func:`compute_divergence` result dict.
    """
    Xw = X_tensor[:, :, start:end].mean(axis=2).T
    if A is not None:
        Xw = partial_out_features(Xw, A, ridge=ridge)
    rdm = compute_rdm(Xw, metric=metric, standardize=True)
    return compute_divergence(rdm, labels, pair=pair)


def time_resolved_divergence(
    X_tensor: np.ndarray,
    labels: Sequence[str],
    t: np.ndarray,
    pair: Tuple[str, str] = ("song", "music"),
    window_sec: float = 0.15,
    step_sec: float = 0.05,
    metric: str = "correlation",
    A: Optional[np.ndarray] = None,
    ridge: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Sliding-window divergence curve for a single class pair.

    At each sliding window we mean-pool the 3-D tensor over time,
    optionally residualise against ``A``, z-score across stimuli, build
    an RDM, and score divergence with :func:`compute_divergence`.

    Parameters
    ----------
    X_tensor : np.ndarray
        ``(n_electrodes, n_stimuli, n_time)`` tensor.
    labels : Sequence[str]
        ``(n_stimuli,)`` class labels.
    t : np.ndarray
        ``(n_time,)`` time axis in seconds.
    pair : tuple[str, str], default=("song", "music")
        Class pair for the divergence statistic.
    window_sec, step_sec : float
        Sliding-window parameters in seconds.
    metric : str, default="correlation"
        Distance metric passed to :func:`compute_rdm`.
    A : np.ndarray or None, default=None
        Stimulus-level regressors of shape ``(n_stim, n_regressors)``.
        When provided, the window-averaged feature matrix is
        residualised against ``A`` before the RDM is built
        (acoustic-partialled divergence).
    ridge : float, default=0.0
        Ridge penalty forwarded to :func:`partial_out_features`.

    Returns
    -------
    dict
        * ``"time"`` : ``(n_windows,)`` window-centre times in seconds.
        * ``"divergence"`` : ``(n_windows,)`` between-minus-within scores.
        * ``"between"`` : ``(n_windows,)`` mean between-class dissimilarity.
        * ``"within"`` : ``(n_windows,)`` averaged within-class dissimilarity.
    """
    X_tensor = np.asarray(X_tensor)
    labels = np.asarray(labels)
    if A is not None:
        A = np.asarray(A, dtype=float)

    times: List[float] = []
    divs: List[float] = []
    betw: List[float] = []
    with_: List[float] = []
    for start, end, center in sliding_window_iter(t, window_sec, step_sec):
        stats = _divergence_from_window(
            X_tensor, labels, start, end, pair, metric, A=A, ridge=ridge
        )
        times.append(center)
        divs.append(stats["divergence"])
        betw.append(stats["between"])
        with_.append(0.5 * (stats["within_a"] + stats["within_b"]))
    return {
        "time": np.array(times),
        "divergence": np.array(divs),
        "between": np.array(betw),
        "within": np.array(with_),
    }


# inference, bootstrap confidence intervals and label permutations

def bootstrap_divergence_curve(
    X_tensor: np.ndarray,
    labels: Sequence[str],
    t: np.ndarray,
    pair: Tuple[str, str] = ("song", "music"),
    window_sec: float = 0.15,
    step_sec: float = 0.05,
    metric: str = "correlation",
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> Dict[str, np.ndarray]:
    """Resample stimuli (stratified, without-replacement within class)
    to get a bootstrap distribution for every window.

    We resample *stimulus indices* inside each class, keeping class sizes
    constant. Using without-replacement within-class bootstraps here would
    just return the original set every time, so we use with-replacement
    bootstrap but deduplicate before computing RDM to avoid zero-distance
    rows polluting within-class means.
    """
    X_tensor = np.asarray(X_tensor)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    windows = list(sliding_window_iter(t, window_sec, step_sec))
    times = np.array([c for _, _, c in windows])

    a, b = pair
    ia = np.where(labels == a)[0]
    ib = np.where(labels == b)[0]

    boot = np.full((n_boot, len(windows)), np.nan)
    for r in range(n_boot):
        sa = rng.choice(ia, size=ia.size, replace=True)
        sb = rng.choice(ib, size=ib.size, replace=True)
        idx = np.concatenate([sa, sb])
        # drop duplicates to avoid zero within class distances
        idx = np.unique(idx)
        sub_labels = labels[idx]
        if np.sum(sub_labels == a) < 2 or np.sum(sub_labels == b) < 2:
            continue
        sub = X_tensor[:, idx, :]
        for w_i, (start, end, _) in enumerate(windows):
            boot[r, w_i] = _divergence_from_window(
                sub, sub_labels, start, end, pair, metric
            )["divergence"]

    lo = np.nanquantile(boot, 0.025, axis=0)
    hi = np.nanquantile(boot, 0.975, axis=0)
    med = np.nanmedian(boot, axis=0)
    return {
        "time": times,
        "boot": boot,
        "ci_lo": lo,
        "ci_hi": hi,
        "median": med,
    }


def permutation_test_divergence_curve(
    X_tensor: np.ndarray,
    labels: Sequence[str],
    t: np.ndarray,
    pair: Tuple[str, str] = ("song", "music"),
    window_sec: float = 0.15,
    step_sec: float = 0.05,
    metric: str = "correlation",
    A: Optional[np.ndarray] = None,
    ridge: float = 0.0,
    n_perm: int = 1000,
    seed: int = RANDOM_STATE,
) -> Dict[str, np.ndarray]:
    """Shuffle class labels (preserving class sizes within the pair) and
    recompute the divergence curve to build a null distribution.

    When ``A`` is provided, the window-averaged feature matrix is
    residualised against ``A`` *inside every permutation*, so the null
    reflects divergence after acoustic features have been controlled
    for (residualising before shuffling would leak class information
    through ``A``).

    Returns observed curve, null samples, per-window empirical p-value
    with the conservative ``(+1)/(N+1)`` correction, and 95th-percentile
    null envelope.
    """
    X_tensor = np.asarray(X_tensor)
    labels = np.asarray(labels)
    if A is not None:
        A = np.asarray(A, dtype=float)
    rng = np.random.default_rng(seed)

    windows = list(sliding_window_iter(t, window_sec, step_sec))
    times = np.array([c for _, _, c in windows])

    def _curve(lbls: np.ndarray) -> np.ndarray:
        return np.array([
            _divergence_from_window(
                X_tensor, lbls, s, e, pair, metric, A=A, ridge=ridge
            )["divergence"]
            for s, e, _ in windows
        ])

    observed = _curve(labels)

    # shuffle labels within the pair only, class sizes are preserved
    a, b = pair
    in_pair = np.where(np.isin(labels, [a, b]))[0]
    pair_labels = labels[in_pair]

    null = np.full((n_perm, len(windows)), np.nan)
    for r in range(n_perm):
        shuffled = pair_labels.copy()
        rng.shuffle(shuffled)
        full_labels = labels.copy()
        full_labels[in_pair] = shuffled
        null[r] = _curve(full_labels)

    p_per_window = (np.sum(null >= observed[None, :], axis=0) + 1) / (n_perm + 1)
    env_95 = np.nanquantile(null, 0.95, axis=0)
    obs_peak = float(np.nanmax(observed))
    null_peaks = np.nanmax(null, axis=1)
    p_peak = float((np.sum(null_peaks >= obs_peak) + 1) / (n_perm + 1))

    return {
        "time": times,
        "observed": observed,
        "null": null,
        "env_95": env_95,
        "p_per_window": p_per_window,
        "obs_peak": obs_peak,
        "null_peaks": null_peaks,
        "p_peak": p_peak,
    }
