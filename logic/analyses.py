"""High-level analysis drivers.

Each function here is a thin composition of lower-level modules
(:mod:`data_utils`, :mod:`decoding`, :mod:`rdm`, :mod:`subsets`,
:mod:`stats`). Every driver returns a plain dict of numpy arrays and
Python scalars so callers (the notebook, :mod:`pipeline`, or an ad-hoc
script) can save, plot, or diff the outputs without touching analysis
internals.

Design rationale
----------------
* Computation is strictly separated from plotting and file I/O: this
  module never writes a CSV or draws a figure.
* Every sampling step takes an explicit ``seed`` so results reproduce
  exactly across runs.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from config import (
    BOOTSTRAP_N,
    EARLY_WINDOW,
    N_SPLITS,
    RANDOM_STATE,
    RANDOM_SUBSETS_N,
    STEP_SEC,
    SUBSET_SIZE,
    WINDOW_SEC,
)
from data_utils import window_features
from decoding import (
    run_grouped_cv,
    run_time_resolved_binary_decoder,
)
from rdm import (
    compute_divergence,
    compute_rdm,
)
from stats import bootstrap_diff_ci, empirical_p_value
from subsets import sample_random_subsets


# score a single electrode subset on one task with one metric
def evaluate_subset(
    subset_indices: np.ndarray,
    X_tensor: np.ndarray,
    y: np.ndarray,
    stimulus_id: np.ndarray,
    t: np.ndarray,
    task: str = "song_vs_music",
    metric: str = "bacc",
    time_window: Tuple[float, float] = EARLY_WINDOW,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_STATE,
    rdm_metric: str = "correlation",
) -> float:
    """Score one electrode subset on one task with one metric.

    Used by the matched random-subset control (Figure 4) and the
    leave-one-out analysis (Figure 7). Both applications need a
    *single scalar* per subset; this function enforces that contract.

    Parameters
    ----------
    subset_indices : np.ndarray
        Electrode indices into axis 0 of ``X_tensor``, shape ``(k,)``.
    X_tensor : np.ndarray
        ``(n_electrodes, n_stimuli, n_time)`` ECoG tensor.
    y : np.ndarray
        ``(n_stimuli,)`` coarse labels.
    stimulus_id : np.ndarray
        ``(n_stimuli,)`` grouping keys for CV.
    t : np.ndarray
        ``(n_time,)`` time axis in seconds.
    task : {"song_vs_music", "three_class"}, default="song_vs_music"
        Which decoding problem to score.
    metric : {"bacc", "divergence"}, default="bacc"
        ``"bacc"`` is held-out balanced accuracy under grouped CV;
        ``"divergence"`` is the RDM divergence of the mean-pooled
        window features for the song/music contrast.
    time_window : tuple[float, float], default=:data:`config.EARLY_WINDOW`
        Inclusive window in seconds used to mean-pool features.
    n_splits : int, default=:data:`config.N_SPLITS`
        Grouped-CV folds (only used when ``metric=="bacc"``).
    seed : int, default=:data:`config.RANDOM_STATE`
        Seed forwarded to downstream samplers.
    rdm_metric : str, default="correlation"
        Distance metric for the RDM (only used when
        ``metric=="divergence"``).

    Returns
    -------
    float
        The requested scalar score for the subset.

    Raises
    ------
    ValueError
        If ``task`` or ``metric`` is not one of the accepted values.
    """
    X_tensor = np.asarray(X_tensor)
    y = np.asarray(y)
    stimulus_id = np.asarray(stimulus_id)
    subset_indices = np.asarray(subset_indices)

    if task == "song_vs_music":
        keep = np.isin(y, ["song", "music"])
    elif task == "three_class":
        keep = np.ones_like(y, dtype=bool)
    else:
        raise ValueError(f"Unknown task: {task}")

    X_sub = X_tensor[subset_indices][:, keep, :]
    y_sub = y[keep]
    sid_sub = stimulus_id[keep]
    X_feat = window_features(X_sub, t, time_window)

    if metric == "bacc":
        res = run_grouped_cv(
            X_feat, y_sub, sid_sub, n_splits=n_splits, seed=seed, return_proba=False
        )
        return float(res["metrics"]["balanced_accuracy"])
    if metric == "divergence":
        rdm = compute_rdm(X_feat, metric=rdm_metric, standardize=True)
        stats = compute_divergence(rdm, y_sub, pair=("song", "music"))
        return float(stats["divergence"])
    raise ValueError(f"Unknown metric: {metric}")


# compare the song only subset against a matched random subset null
def compare_true_vs_random_subsets(
    X_tensor: np.ndarray,
    y: np.ndarray,
    stimulus_id: np.ndarray,
    electrode_group: np.ndarray,
    t: np.ndarray,
    *,
    metric: str = "bacc",
    task: str = "song_vs_music",
    subset_size: int = SUBSET_SIZE,
    n_subsets: int = RANDOM_SUBSETS_N,
    time_window: Tuple[float, float] = EARLY_WINDOW,
    extras: Optional[Dict[str, np.ndarray]] = None,
    seed: int = RANDOM_STATE,
    bootstrap_diff_n: int = BOOTSTRAP_N,
) -> Dict[str, object]:
    """Compare the song-only subset against a matched random-subset null.

    This is the headline analysis (Figure 4). We ask: does the small
    song-selective electrode pool separate song from music *better*
    than any equally-sized random subset drawn from the non-song pool?
    Random matched subsets are the right null because they control for
    subset size and electrode count simultaneously.

    Parameters
    ----------
    X_tensor : np.ndarray
        ``(n_electrodes, n_stimuli, n_time)`` ECoG tensor.
    y : np.ndarray
        ``(n_stimuli,)`` coarse labels.
    stimulus_id : np.ndarray
        ``(n_stimuli,)`` grouping keys for CV.
    electrode_group : np.ndarray
        ``(n_electrodes,)`` in ``{"song", "speech", "music"}``; used to
        identify the true song-only subset and the non-song null pool.
    t : np.ndarray
        ``(n_time,)`` time axis in seconds.
    metric : {"bacc", "divergence"}, keyword-only, default="bacc"
        Score passed to :func:`evaluate_subset`.
    task : {"song_vs_music", "three_class"}, keyword-only, default="song_vs_music"
        Decoding task.
    subset_size : int, default=:data:`config.SUBSET_SIZE`
        Number of electrodes per random subset. Non-positive values
        default to the number of song electrodes.
    n_subsets : int, default=:data:`config.RANDOM_SUBSETS_N`
        Number of random null subsets to draw.
    time_window : tuple[float, float], default=:data:`config.EARLY_WINDOW`
        Feature pooling window.
    extras : dict[str, np.ndarray] or None, default=None
        Extra named subsets (e.g. ``{"all": ..., "no_song": ...}``)
        scored once for reference lines in the plot.
    seed : int, default=:data:`config.RANDOM_STATE`
        Seed for subset sampling, CV, and the bootstrap.
    bootstrap_diff_n : int, default=:data:`config.BOOTSTRAP_N`
        Bootstrap replicates for the ``true - mean(null)`` CI.

    Returns
    -------
    dict
        * ``"true_score"`` : scalar song-only score.
        * ``"null_scores"`` : ``(n_subsets,)`` array of random-subset scores.
        * ``"reference_scores"`` : mapping extra-subset name to its score.
        * ``"empirical_p_greater"`` : one-sided p-value for
          ``true_score > null``.
        * ``"diff_ci"`` : bootstrap percentile CI on the difference, dict
          with ``"lo"``/``"median"``/``"hi"``.
        * ``"diff_bootstrap_samples"`` : ``(bootstrap_diff_n,)`` raw samples.
        * ``"subset_indices"`` : list of the random subsets used, for audit.
        * ``"metric"``, ``"task"``, ``"subset_size"``, ``"n_subsets"``,
          ``"time_window"``, ``"seed"`` : echo of the inputs.
    """
    electrode_group = np.asarray(electrode_group)
    song_idx = np.where(electrode_group == "song")[0]
    if subset_size <= 0:
        subset_size = song_idx.size

    # true score on the song only subset
    true_score = evaluate_subset(
        song_idx,
        X_tensor,
        y,
        stimulus_id,
        t,
        task=task,
        metric=metric,
        time_window=time_window,
        seed=seed,
    )

    # reference scores for other fixed subsets
    ref_scores: Dict[str, float] = {}
    if extras:
        for name, idx in extras.items():
            ref_scores[name] = evaluate_subset(
                idx,
                X_tensor,
                y,
                stimulus_id,
                t,
                task=task,
                metric=metric,
                time_window=time_window,
                seed=seed,
            )

    # random subset null distribution
    subset_list = sample_random_subsets(
        electrode_group,
        n_subsets=n_subsets,
        size=subset_size,
        exclude_song=True,
        seed=seed,
    )
    null_scores = np.array(
        [
            evaluate_subset(
                idx,
                X_tensor,
                y,
                stimulus_id,
                t,
                task=task,
                metric=metric,
                time_window=time_window,
                seed=seed,
            )
            for idx in subset_list
        ]
    )

    p_greater = empirical_p_value(true_score, null_scores, alternative="greater")
    diff_ci = bootstrap_diff_ci(
        sample_true=[true_score],
        sample_null=null_scores,
        n_boot=bootstrap_diff_n,
        seed=seed,
    )

    return {
        "metric": metric,
        "task": task,
        "subset_size": int(subset_size),
        "n_subsets": int(n_subsets),
        "time_window": tuple(time_window),
        "true_score": float(true_score),
        "null_scores": null_scores,
        "reference_scores": ref_scores,
        "empirical_p_greater": float(p_greater),
        "diff_ci": {"lo": diff_ci["lo"], "median": diff_ci["median"], "hi": diff_ci["hi"]},
        "diff_bootstrap_samples": diff_ci["diffs"],
        "subset_indices": [idx.tolist() for idx in subset_list],
        "seed": int(seed),
    }


# time resolved song vs music decoder with a stimulus level bootstrap ci
def time_resolved_songmusic_with_ci(
    X_tensor: np.ndarray,
    y: np.ndarray,
    stimulus_id: np.ndarray,
    t: np.ndarray,
    subset_idx: np.ndarray,
    window_sec: float = WINDOW_SEC,
    step_sec: float = STEP_SEC,
    n_splits: int = N_SPLITS,
    n_boot: int = 200,
    seed: int = RANDOM_STATE,
) -> Dict[str, np.ndarray]:
    """Time-resolved song-vs-music decoder with a stimulus-level bootstrap CI.

    Runs :func:`run_time_resolved_binary_decoder` once to produce the
    point estimate, then bootstraps *the held-out predictions* per
    window to build a confidence band. We bootstrap held-out
    predictions (rather than training data) to avoid leakage: if you
    bootstrap inputs first and then run CV, the same stimulus shows
    up in both training and test folds and the CI shrinks artificially.

    Parameters
    ----------
    X_tensor : np.ndarray
        ``(n_electrodes, n_stimuli, n_time)`` ECoG tensor.
    y : np.ndarray
        ``(n_stimuli,)`` coarse labels.
    stimulus_id : np.ndarray
        ``(n_stimuli,)`` grouping keys for CV.
    t : np.ndarray
        ``(n_time,)`` time axis in seconds.
    subset_idx : np.ndarray
        Electrode indices to decode on, shape ``(k,)``.
    window_sec, step_sec : float
        Sliding-window parameters.
    n_splits : int
        Grouped-CV folds.
    n_boot : int, default=200
        Bootstrap replicates per window.
    seed : int
        Seed for the decoder and the bootstrap.

    Returns
    -------
    dict
        * ``"time"`` : ``(n_windows,)`` window-centre times.
        * ``"bacc"``, ``"auc"`` : ``(n_windows,)`` point estimates.
        * ``"bacc_ci_lo"``, ``"bacc_ci_hi"`` : 95% CI band, shape
          ``(n_windows,)``.
        * ``"per_fold_bacc"`` : ``(n_windows, n_splits)`` per-fold scores.
        * ``"class_a"``, ``"class_b"`` : echo of the contrast.
    """
    main = run_time_resolved_binary_decoder(
        X_tensor,
        y,
        stimulus_id,
        t,
        class_a="song",
        class_b="music",
        subset_idx=subset_idx,
        window_sec=window_sec,
        step_sec=step_sec,
        n_splits=n_splits,
        seed=seed,
        return_proba=True,
    )

    rng = np.random.default_rng(seed)
    n_t = main["time"].size
    boot = np.zeros((n_boot, n_t), dtype=float)

    for w_i in range(n_t):
        y_true, y_pred = main["y_pred_per_window"][w_i]
        # stratified bootstrap so we do not drop either class entirely
        y_true_arr = np.asarray(y_true)
        classes = np.unique(y_true_arr)
        class_indices = {c: np.where(y_true_arr == c)[0] for c in classes}
        for b in range(n_boot):
            idx = np.concatenate(
                [rng.choice(ci_, size=ci_.size, replace=True) for ci_ in class_indices.values()]
            )
            boot[b, w_i] = balanced_accuracy_score(y_true_arr[idx], np.asarray(y_pred)[idx])

    ci_lo = np.quantile(boot, 0.025, axis=0)
    ci_hi = np.quantile(boot, 0.975, axis=0)
    return {
        "time": main["time"],
        "bacc": main["bacc"],
        "auc": main["auc"],
        "bacc_ci_lo": ci_lo,
        "bacc_ci_hi": ci_hi,
        "per_fold_bacc": main["per_fold_bacc"],
        "class_a": main["class_a"],
        "class_b": main["class_b"],
    }
