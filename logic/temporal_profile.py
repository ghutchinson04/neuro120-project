"""Event-locked temporal profile extraction and cross-dataset feature table.

Two use cases:

* Bellier: given a continuous HFA matrix and a binary vocal mask, build
  event-locked averages around vocal onsets and (by symmetry) instrumental
  onsets, for hemisphere-specific STG groups.

* Norman-Haignere: given the trial-locked dataset tensor, collapse across
  stimuli in each coarse class (song / music / speech) per electrode group
  (song / music / speech electrodes) to get a comparable profile.

For each profile we extract three simple shape features:

    peak_latency_s           argmax within [0, peak_win_s] post-onset
    onset_sustained_ratio    mean[0, 0.2]s divided by mean[0.5, 1.5]s
    auc                      trapezoid integral of HFA in [0, post_s]

The intent is a *qualitative* cross-dataset overlay; we do not run any
between-dataset statistical test.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from bellier_data import detect_onsets
from config import BELLIER_FS, PROFILE_POST_SEC, PROFILE_PRE_SEC


# numpy 2 removed np,trapz, np,trapezoid is the replacement when available
_trapz = getattr(np, "trapezoid", None) or np.trapz


# event locked average trace across windows around each event
def event_locked_profile(
    hfa: np.ndarray,
    event_samples: np.ndarray,
    fs: float,
    pre_sec: float = PROFILE_PRE_SEC,
    post_sec: float = PROFILE_POST_SEC,
    group_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Average HFA across a group of electrodes, event-locked.

    The function builds an ``(n_events, n_win)`` buffer by slicing the
    electrode-averaged HFA trace around each event, drops events whose
    window falls outside the recording, then reports the across-event
    mean and SEM.

    Parameters
    ----------
    hfa : np.ndarray
        Continuous HFA matrix of shape ``(T, n_elec)``.
    event_samples : np.ndarray
        Integer sample indices of event onsets, shape ``(n_events,)``.
    fs : float
        Sampling rate in Hz (matches ``hfa``).
    pre_sec : float, default=:data:`config.PROFILE_PRE_SEC`
        Seconds *before* each event included in the window.
    post_sec : float, default=:data:`config.PROFILE_POST_SEC`
        Seconds *after* each event included in the window.
    group_indices : np.ndarray, optional
        Electrode column indices to include. ``None`` means use all
        electrodes.

    Returns
    -------
    dict
        Keys ``time_s`` (``(n_win,)``; ``0`` = event onset),
        ``mean`` (``(n_win,)`` across-event mean of the
        electrode-averaged trace), ``sem`` (``(n_win,)`` across-event
        SEM), ``n_events`` (events that fully fit inside the recording),
        and ``n_elec`` (number of electrodes used).

    Raises
    ------
    ValueError
        If ``hfa`` is not 2-D or ``group_indices`` is empty.
    """
    if hfa.ndim != 2:
        raise ValueError("hfa must be (T, n_elec)")
    T, E = hfa.shape
    if group_indices is None:
        group_indices = np.arange(E)
    group_indices = np.asarray(group_indices, dtype=int)
    if group_indices.size == 0:
        raise ValueError("group_indices is empty")

    pre = int(round(pre_sec * fs))
    post = int(round(post_sec * fs))
    win_len = pre + post
    time_s = (np.arange(-pre, post)) / fs

    # electrode averaged trace, then slice around each event
    trace = hfa[:, group_indices].mean(axis=1)  # shape (t,)
    n_ev = len(event_samples)
    buffer = np.full((n_ev, win_len), np.nan, dtype=float)
    for i, s in enumerate(event_samples):
        lo = int(s) - pre
        hi = int(s) + post
        if lo < 0 or hi > T:
            continue
        buffer[i] = trace[lo:hi]

    valid = ~np.all(np.isnan(buffer), axis=1)
    buffer = buffer[valid]
    if buffer.size == 0:
        return {
            "time_s": time_s,
            "mean": np.full(win_len, np.nan),
            "sem": np.full(win_len, np.nan),
            "n_events": 0,
            "n_elec": int(group_indices.size),
        }

    mean = np.nanmean(buffer, axis=0)
    sem = np.nanstd(buffer, axis=0, ddof=1) / np.sqrt(max(1, buffer.shape[0]))
    return {
        "time_s": time_s,
        "mean": mean,
        "sem": sem,
        "n_events": int(buffer.shape[0]),
        "n_elec": int(group_indices.size),
    }


# shape features extracted from a single event locked profile
def extract_features(
    time_s: np.ndarray,
    profile: np.ndarray,
    peak_win_s: tuple = (0.0, 1.0),
    onset_win_s: tuple = (0.0, 0.2),
    sustained_win_s: tuple = (0.5, 1.5),
    auc_win_s: tuple = (0.0, 1.5),
) -> Dict[str, float]:
    """Summarise an event-locked profile with three shape statistics.

    Parameters
    ----------
    time_s : np.ndarray
        Time axis in seconds, shape ``(n_win,)``. ``0`` marks event onset.
    profile : np.ndarray
        Mean HFA trace of shape ``(n_win,)`` aligned to ``time_s``.
    peak_win_s : tuple[float, float]
        Window in which to locate ``peak_latency_s`` via argmax.
    onset_win_s, sustained_win_s : tuple[float, float]
        Windows whose mean HFA defines the
        ``onset_sustained_ratio = mean(onset) / mean(sustained)``.
        NaN is returned if the denominator is zero or non-finite.
    auc_win_s : tuple[float, float]
        Window over which ``auc`` (trapezoidal integral of HFA) is
        computed.

    Returns
    -------
    dict[str, float]
        Keys ``peak_latency_s``, ``onset_sustained_ratio``, ``auc``.
        Any feature whose window is empty returns NaN so that callers
        can see missing values rather than silently getting a zero.
    """
    t = np.asarray(time_s)
    y = np.asarray(profile)

    def _mean(t0: float, t1: float) -> float:
        m = (t >= t0) & (t <= t1)
        return float(np.nanmean(y[m])) if m.any() else float("nan")

    peak_mask = (t >= peak_win_s[0]) & (t <= peak_win_s[1])
    if peak_mask.any():
        local = y[peak_mask]
        local_t = t[peak_mask]
        peak_latency = float(local_t[np.nanargmax(local)])
    else:
        peak_latency = float("nan")

    onset = _mean(*onset_win_s)
    sustained = _mean(*sustained_win_s)
    if sustained == 0 or not np.isfinite(sustained):
        ratio = float("nan")
    else:
        ratio = float(onset / sustained)

    auc_mask = (t >= auc_win_s[0]) & (t <= auc_win_s[1])
    if auc_mask.any():
        auc = float(_trapz(y[auc_mask], t[auc_mask]))
    else:
        auc = float("nan")

    return {
        "peak_latency_s": peak_latency,
        "onset_sustained_ratio": ratio,
        "auc": auc,
    }


# dataset specific wrappers that build profiles for bellier and norman
def bellier_profiles(
    supergrid: Dict[str, object],
    subsets_idx: Dict[str, np.ndarray],
    vocal_mask: np.ndarray,
    groups: Iterable[str] = ("right_STG", "left_STG"),
) -> Dict[str, object]:
    """Vocal / instrumental event-locked profiles per STG hemisphere.

    Vocal and instrumental onset samples are derived by
    :func:`bellier_data.detect_onsets` from the aligned vocal mask;
    instrumental onsets come from the complement mask so both event
    types share the same time base. For each STG group, an
    electrode-averaged HFA trace is produced and summarised by
    :func:`extract_features`.

    Parameters
    ----------
    supergrid : dict
        Output of :func:`bellier_data.build_supergrid`. Must contain the
        ``hfa`` key holding a ``(T, n_elec)`` array.
    subsets_idx : dict[str, np.ndarray]
        Mapping from group name (e.g. ``"right_STG"``) to electrode
        column indices inside ``supergrid["hfa"]``.
    vocal_mask : np.ndarray
        Binary mask of shape ``(T,)`` aligned to ``hfa`` time axis.
    groups : iterable[str]
        Group names to include; groups with zero electrodes are skipped.

    Returns
    -------
    dict
        ``{"profiles": {group__event: profile_dict, ...},
        "features": pandas.DataFrame}``. Each ``profile_dict`` is the
        output of :func:`event_locked_profile`; the dataframe has one
        row per (group, event) with the three shape features.
    """
    hfa = np.asarray(supergrid["hfa"])
    vocal_onsets = detect_onsets(vocal_mask.astype(int))
    instrumental_onsets = detect_onsets((~vocal_mask.astype(bool)).astype(int))

    profiles: Dict[str, object] = {}
    feature_rows: List[Dict[str, object]] = []

    for group in groups:
        idx = np.asarray(subsets_idx[group])
        if idx.size == 0:
            continue
        for event_name, events in (
            ("vocal", vocal_onsets),
            ("instrumental", instrumental_onsets),
        ):
            prof = event_locked_profile(
                hfa, events, fs=BELLIER_FS, group_indices=idx
            )
            feats = extract_features(prof["time_s"], prof["mean"])
            key = f"{group}__{event_name}"
            profiles[key] = prof
            feature_rows.append(
                {
                    "dataset": "bellier",
                    "group": group,
                    "event": event_name,
                    "n_elec": prof["n_elec"],
                    "n_events": prof["n_events"],
                    **feats,
                }
            )

    return {"profiles": profiles, "features": pd.DataFrame(feature_rows)}


def norman_profiles(
    ds: Dict[str, object],
    electrode_groups: Iterable[str] = ("song", "music", "speech"),
    stimulus_classes: Iterable[str] = ("song", "music", "speech"),
) -> Dict[str, object]:
    """Trial-locked profile for each (electrode_group, stimulus_class) pair.

    Each stimulus already corresponds to a discrete trial, so "events"
    here are the trials themselves. Per-trial electrode means are
    averaged across trials to get the same ``(time_s, mean, sem)``
    triplet produced by :func:`bellier_profiles`, which keeps
    downstream plotting and feature extraction dataset-agnostic.

    Parameters
    ----------
    ds : dict
        Output of :func:`data_utils.build_dataset`. Must contain
        ``X_tensor`` with shape ``(n_elec, n_stim, n_time)``, ``t``
        with shape ``(n_time,)``, ``electrode_group`` with shape
        ``(n_elec,)``, and ``y_coarse`` with shape ``(n_stim,)``.
    electrode_groups, stimulus_classes : iterable[str]
        Groups and coarse classes to include. Missing groups are
        silently skipped.

    Returns
    -------
    dict
        Same structure as :func:`bellier_profiles`:
        ``{"profiles": ..., "features": pandas.DataFrame}``.
    """
    X = np.asarray(ds["X_tensor"])  # shape (n_elec, n_stim, n_time)
    t = np.asarray(ds["t"])
    elec_group = np.asarray(ds["electrode_group"])
    y = np.asarray(ds["y_coarse"])

    profiles: Dict[str, object] = {}
    feature_rows: List[Dict[str, object]] = []

    for egroup in electrode_groups:
        e_idx = np.where(elec_group == egroup)[0]
        if e_idx.size == 0:
            continue
        for cls in stimulus_classes:
            s_idx = np.where(y == cls)[0]
            if s_idx.size == 0:
                continue
            block = X[np.ix_(e_idx, s_idx)]  # shape (n_e_group, n_s_class, n_t)
            # collapse electrodes per trial, then average across trials
            per_trial = block.mean(axis=0)  # shape (n_s_class, n_t)
            mean = per_trial.mean(axis=0)
            sem = per_trial.std(axis=0, ddof=1) / np.sqrt(max(1, per_trial.shape[0]))

            prof = {
                "time_s": t,
                "mean": mean,
                "sem": sem,
                "n_events": int(per_trial.shape[0]),
                "n_elec": int(e_idx.size),
            }
            feats = extract_features(t, mean)
            key = f"{egroup}__{cls}"
            profiles[key] = prof
            feature_rows.append(
                {
                    "dataset": "norman",
                    "group": egroup,
                    "event": cls,
                    "n_elec": prof["n_elec"],
                    "n_events": prof["n_events"],
                    **feats,
                }
            )

    return {"profiles": profiles, "features": pd.DataFrame(feature_rows)}
