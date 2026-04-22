"""Data loading and dataset assembly for the Norman-Haignere ECoG files.

Public entry points
-------------------
* :func:`build_dataset` -- load the 33 individual-electrode ``.mat`` files,
  align them on a shared stimulus order, and return a single dict with
  shape-checked tensors plus all metadata needed for stimulus-grouped
  cross-validation (``stimulus_id``, ``electrode_group``, ``electrode_id``).
* :func:`window_features` -- mean-pool the 3-D ECoG tensor over a time
  window and return a 2-D ``(n_stimuli, n_electrodes)`` sklearn-ready
  feature matrix.
* :func:`load_acoustic_features` -- load the cochleogram + spectrotemporal
  modulation regressors from the Norman-Haignere MATLAB files and
  align them to the same stimulus order as the dataset.
* :func:`sliding_window_iter` -- iterate ``(start, end, center)`` triples
  for a sliding-window decoder.

Everything in this module is a pure function: no globals, no plotting,
no side effects other than reading the ``.mat`` files.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import scipy.io as sio

from config import (
    DATA_DIR,
    MAJOR_CLASSES,
    MUSIC_FINE,
    NORMAN_ACOUSTIC_MAT,
    NORMAN_COMPONENT_RESP_MAT,
    SONG_FINE,
    SPEECH_FINE,
    TIME_MAX,
    TIME_MIN,
)


# low level helpers for reading norman haignere files
def _mat_cell_to_list(x) -> List[str]:
    """Flatten a MATLAB cell/array of strings into a list of cleaned strings.

    Strips the ``.wav`` suffix and surrounding whitespace that the
    Norman-Haignere files use for stimulus names.
    """
    if isinstance(x, np.ndarray):
        return [str(v).replace(".wav", "").strip() for v in np.ravel(x)]
    return [str(x).replace(".wav", "").strip()]


def _coarse_from_fine(fine_label: str) -> str:
    """Map a fine-grained stimulus category label to its coarse class.

    The Norman-Haignere dataset distinguishes English vs foreign speech
    at the fine level; we collapse them to a single ``"speech"`` class
    because no downstream analysis depends on the language contrast.
    Unknown labels fall through to ``"other"`` so the caller can filter
    them out cleanly.
    """
    if fine_label in SONG_FINE:
        return "song"
    if fine_label in SPEECH_FINE:
        return "speech"
    if fine_label in MUSIC_FINE:
        return "music"
    return "other"


def _load_electrode_file(path: Path) -> Dict[str, np.ndarray]:
    """Load one electrode ``.mat`` file and normalize its fields.

    Returns a dict with ``R`` (``(n_stim, n_time)`` HFA), ``t`` (time
    axis in seconds), ``stim_names``, and both ``fine_categories`` and
    ``coarse_categories`` per stimulus. Converts the MATLAB 1-based
    category index to 0-based if necessary.
    """
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    R = np.asarray(d["R"], dtype=float)
    t = np.asarray(d["t"], dtype=float).squeeze()
    stim_names = _mat_cell_to_list(d["stim_names"])

    C = d["C"]
    category_labels = [str(v) for v in np.ravel(C.category_labels)]
    category_assignments = np.asarray(C.category_assignments).astype(int).squeeze()
    if category_assignments.min() == 1:
        category_assignments = category_assignments - 1
    fine_categories = np.array([category_labels[i] for i in category_assignments])
    coarse_categories = np.array([_coarse_from_fine(l) for l in fine_categories])

    return {
        "R": R,
        "t": t,
        "stim_names": stim_names,
        "fine_categories": fine_categories,
        "coarse_categories": coarse_categories,
    }


def _load_group(
    file_paths: Sequence[Path],
    reference_names: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Load many electrode ``.mat`` files into one stimulus-aligned tensor.

    Each file is reordered so that row ``i`` of its response matrix
    corresponds to ``reference_names[i]`` (or to the first file's order
    if ``reference_names`` is not supplied). We then check that every
    file shares the same time axis and coarse category assignments,
    raising a clear error if any file disagrees.

    Parameters
    ----------
    file_paths : Sequence[Path]
        Paths to electrode ``.mat`` files.
    reference_names : Sequence[str] or None, default=None
        Canonical stimulus ordering. If ``None``, the first file's
        ordering becomes the reference for the rest.

    Returns
    -------
    dict
        ``responses``        : ``(n_electrodes, n_stim, n_time)`` float array.
        ``stim_names``       : ``(n_stim,)`` string array (reference order).
        ``fine_categories``  : ``(n_stim,)`` string array.
        ``coarse_categories``: ``(n_stim,)`` string array in {"song", "speech", "music", "other"}.
        ``t``                : ``(n_time,)`` time axis in seconds.
        ``paths``            : list of :class:`Path` objects, sorted alphabetically.

    Raises
    ------
    ValueError
        If any file's time axis or coarse category assignments disagree
        with the reference after stimulus alignment.
    """
    tensors = []
    ref_names = None
    ref_fine = None
    ref_coarse = None
    ref_t = None

    for p in sorted(file_paths):
        item = _load_electrode_file(p)

        if reference_names is None and ref_names is None:
            ref_names = item["stim_names"]
            ref_fine = item["fine_categories"]
            ref_coarse = item["coarse_categories"]
            ref_t = item["t"]
        else:
            target = reference_names if reference_names is not None else ref_names
            idx = [item["stim_names"].index(n) for n in target]
            item["R"] = item["R"][idx]
            item["fine_categories"] = item["fine_categories"][idx]
            item["coarse_categories"] = item["coarse_categories"][idx]

            if ref_fine is None:
                ref_fine = item["fine_categories"]
                ref_coarse = item["coarse_categories"]
                ref_t = item["t"]

            if not np.all(item["coarse_categories"] == ref_coarse):
                raise ValueError(f"Category mismatch in {Path(p).name}")
            if not np.allclose(item["t"], ref_t):
                raise ValueError(f"Time axis mismatch in {Path(p).name}")

        tensors.append(item["R"])

    if reference_names is None:
        reference_names = ref_names

    return {
        "responses": np.stack(tensors, axis=0),
        "stim_names": np.array(reference_names),
        "fine_categories": np.array(ref_fine),
        "coarse_categories": np.array(ref_coarse),
        "t": ref_t,
        "paths": [Path(p) for p in sorted(file_paths)],
    }


# public dataset builder, aligns electrodes and slices the time axis
def build_dataset(
    data_dir: Path | str = DATA_DIR,
    time_window_s: tuple[float, float] = (TIME_MIN, TIME_MAX),
    keep_classes: Iterable[str] = MAJOR_CLASSES,
) -> Dict[str, object]:
    """Build the aligned Norman-Haignere dataset once for the whole analysis.

    Loads the song-, speech-, and music-selective electrode ``.mat``
    files, aligns them on a common stimulus ordering, concatenates them
    along the electrode axis in the stable order ``song -> speech ->
    music``, and slices the shared time axis to ``time_window_s``. All
    shapes are asserted before returning.

    Parameters
    ----------
    data_dir : Path or str, default=:data:`config.DATA_DIR`
        Directory containing ``song-elec*-response.mat``,
        ``speech-elec*-response.mat``, and ``music-elec*-response.mat``.
    time_window_s : tuple[float, float], default=(:data:`config.TIME_MIN`, :data:`config.TIME_MAX`)
        Inclusive ``(tmin, tmax)`` bounds in seconds. Samples outside
        this window are dropped from the returned tensor.
    keep_classes : Iterable[str], default=:data:`config.MAJOR_CLASSES`
        Coarse classes to retain. Stimuli whose coarse label is not in
        this set are dropped entirely (rows removed from every tensor
        and label array).

    Returns
    -------
    dict
        * ``X_tensor`` : ``(n_electrodes, n_stimuli, n_time)`` float tensor.
        * ``y_coarse`` : ``(n_stimuli,)`` string array of class labels.
        * ``stimulus_id`` : ``(n_stimuli,)`` string array, guaranteed unique.
        * ``electrode_group`` : ``(n_electrodes,)`` in {"song", "speech", "music"}.
        * ``electrode_id`` : ``(n_electrodes,)`` human-readable names.
        * ``t`` : ``(n_time,)`` float array, time axis in seconds.
        * ``time_mask`` : bool mask applied to the original time axis.
        * ``meta`` : dict with counts, time-window, and classes.

    Raises
    ------
    FileNotFoundError
        If ``data_dir`` does not exist or any of the three electrode
        groups is missing.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    song_files = sorted(glob.glob(str(data_dir / "song-elec*-response.mat")))
    speech_files = sorted(glob.glob(str(data_dir / "speech-elec*-response.mat")))
    music_files = sorted(glob.glob(str(data_dir / "music-elec*-response.mat")))
    if not (song_files and speech_files and music_files):
        raise FileNotFoundError("Missing song/speech/music electrode files in DATA_DIR.")

    song_group = _load_group(song_files)
    ref_names = song_group["stim_names"].tolist()
    speech_group = _load_group(speech_files, reference_names=ref_names)
    music_group = _load_group(music_files, reference_names=ref_names)

    # stimulus level labels come from any aligned group, they are equal after align
    stim_names = song_group["stim_names"]
    coarse = song_group["coarse_categories"]
    keep_classes = list(keep_classes)
    keep_stim = np.isin(coarse, keep_classes)

    # concatenate electrodes along axis 0 in a stable order, song then speech then music
    tensor = np.concatenate(
        [
            song_group["responses"][:, keep_stim, :],
            speech_group["responses"][:, keep_stim, :],
            music_group["responses"][:, keep_stim, :],
        ],
        axis=0,
    )
    electrode_group = np.array(
        ["song"] * song_group["responses"].shape[0]
        + ["speech"] * speech_group["responses"].shape[0]
        + ["music"] * music_group["responses"].shape[0]
    )
    electrode_id = np.array(
        [f"song-elec{i+1}" for i in range(song_group["responses"].shape[0])]
        + [f"speech-elec{i+1}" for i in range(speech_group["responses"].shape[0])]
        + [f"music-elec{i+1}" for i in range(music_group["responses"].shape[0])]
    )

    # keep only samples that fall inside the requested time window
    t_full = song_group["t"]
    tmin, tmax = time_window_s
    time_mask = (t_full >= tmin) & (t_full <= tmax)
    t = t_full[time_mask]
    tensor = tensor[:, :, time_mask]

    y_coarse = coarse[keep_stim]
    stimulus_id = stim_names[keep_stim]

    # sanity check that every array has the expected shape
    n_e, n_s, n_t = tensor.shape
    assert y_coarse.shape == (n_s,)
    assert stimulus_id.shape == (n_s,)
    assert electrode_group.shape == (n_e,)
    assert electrode_id.shape == (n_e,)
    assert len(np.unique(stimulus_id)) == n_s, "stimulus_id must be unique"
    assert t.shape == (n_t,)

    meta = {
        "n_electrodes": int(n_e),
        "n_stimuli": int(n_s),
        "n_time": int(n_t),
        "dt_s": float(np.median(np.diff(t))) if n_t > 1 else float("nan"),
        "time_window_s": (float(t[0]), float(t[-1])),
        "n_per_group": {g: int(np.sum(electrode_group == g)) for g in ("song", "speech", "music")},
        "n_per_class": {c: int(np.sum(y_coarse == c)) for c in keep_classes},
        "keep_classes": keep_classes,
    }

    return {
        "X_tensor": tensor,
        "y_coarse": y_coarse,
        "stimulus_id": stimulus_id,
        "electrode_group": electrode_group,
        "electrode_id": electrode_id,
        "t": t,
        "time_mask": time_mask,
        "meta": meta,
    }


def window_features(
    X_tensor: np.ndarray,
    t: np.ndarray,
    window_s: tuple[float, float],
) -> np.ndarray:
    """Mean-pool a 3-D electrode tensor over a time window.

    This is the standard featurisation for single-window decoders: we
    collapse the time axis inside ``window_s`` and transpose so each
    row is a stimulus, each column an electrode.

    Parameters
    ----------
    X_tensor : np.ndarray
        Shape ``(n_electrodes, n_stimuli, n_time)``.
    t : np.ndarray
        Time axis of shape ``(n_time,)`` in seconds.
    window_s : tuple[float, float]
        Inclusive ``(t0, t1)`` bounds in seconds.

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(n_stimuli, n_electrodes)`` ready for
        :mod:`sklearn` estimators.

    Raises
    ------
    AssertionError
        If ``X_tensor`` is not 3-D.
    ValueError
        If no time samples fall inside ``window_s``.
    """
    assert X_tensor.ndim == 3, f"X_tensor must be 3-D; got shape {X_tensor.shape}"
    t0, t1 = window_s
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        raise ValueError(f"No time samples inside window {window_s}")
    return X_tensor[:, :, mask].mean(axis=2).T


def load_acoustic_features(
    stim_names_target: Sequence[str] | None = None,
    acoustic_mat: Path | str = NORMAN_ACOUSTIC_MAT,
    response_mat: Path | str = NORMAN_COMPONENT_RESP_MAT,
) -> Dict[str, np.ndarray]:
    """Load Norman-Haignere cochleogram and spectrotemporal-modulation features.

    ``acoustic_features.mat`` ships with two arrays:

    * ``F_coch``  shape ``(165, 6)``   - cochlear energy per frequency band.
    * ``F_mod``   shape ``(165, 7, 9)`` - spectrotemporal modulation energy.

    Both are ordered to match ``stim_names`` from
    ``ecog_component_responses.mat``. We load that canonical ordering,
    optionally align rows to ``stim_names_target`` (so the features line
    up with our `build_dataset` output after dropping ``other``/``env``
    stimuli), and partial out ``F_coch`` from ``F_mod`` -- the same
    orthogonalisation Norman-Haignere et al. use before combining the
    two feature families (see ``reference_code/ecog_acoustic_corr.m``).

    Parameters
    ----------
    stim_names_target
        If provided, rows are reindexed so that output row ``i``
        corresponds to ``stim_names_target[i]``. ``.wav`` suffixes are
        stripped on both sides before matching.
    acoustic_mat, response_mat
        File paths.

    Returns
    -------
    dict with keys:
        ``F_coch``     (n_stim, 6)  cochleogram features
        ``F_mod``      (n_stim, 63) flattened modulation features
        ``F_mod_resid`` (n_stim, 63) F_mod with F_coch regressed out
        ``A_full``     (n_stim, 69) concatenation of F_coch + F_mod_resid
        ``stim_names`` (n_stim,)    aligned stimulus names
        ``coch_freqs``, ``spec_mod_scales``, ``temp_mod_rates`` metadata
    """
    acoustic_mat = Path(acoustic_mat)
    response_mat = Path(response_mat)
    if not acoustic_mat.exists():
        raise FileNotFoundError(f"acoustic_features.mat not found: {acoustic_mat}")
    if not response_mat.exists():
        raise FileNotFoundError(
            f"ecog_component_responses.mat not found: {response_mat} "
            f"(needed for the canonical stimulus ordering)."
        )

    a = sio.loadmat(acoustic_mat, squeeze_me=True)
    r = sio.loadmat(response_mat, squeeze_me=True, struct_as_record=False)

    F_coch = np.asarray(a["F_coch"], dtype=float)
    F_mod = np.asarray(a["F_mod"], dtype=float)
    canonical_names = [
        str(x).replace(".wav", "").strip() for x in np.ravel(r["stim_names"])
    ]
    if F_coch.shape[0] != len(canonical_names) or F_mod.shape[0] != len(canonical_names):
        raise ValueError(
            "acoustic_features.mat row count does not match canonical stim_names."
        )

    if stim_names_target is not None:
        target = [str(s).replace(".wav", "").strip() for s in stim_names_target]
        name_to_idx = {n: i for i, n in enumerate(canonical_names)}
        missing = [n for n in target if n not in name_to_idx]
        if missing:
            raise KeyError(
                f"{len(missing)} target stimuli not in acoustic_features.mat: {missing[:5]}..."
            )
        idx = np.array([name_to_idx[n] for n in target], dtype=int)
        F_coch = F_coch[idx]
        F_mod = F_mod[idx]
        out_names = np.array(target)
    else:
        out_names = np.array(canonical_names)

    n_stim = F_coch.shape[0]
    F_mod_flat = F_mod.reshape(n_stim, -1)

    # orthogonalise modulation features against the cochleogram, as in
    # the matlab reference code ecog_acoustic_corr,m, this avoids double
    # counting the audio energy that already shows up in f_coch
    F_mod_resid = F_mod_flat - F_coch @ (np.linalg.pinv(F_coch) @ F_mod_flat)

    A_full = np.concatenate([F_coch, F_mod_resid], axis=1)

    return {
        "F_coch": F_coch,
        "F_mod": F_mod_flat,
        "F_mod_resid": F_mod_resid,
        "A_full": A_full,
        "stim_names": out_names,
        "coch_freqs": np.asarray(a["coch_freqs"]),
        "spec_mod_scales": np.asarray(a["spec_mod_scales"]),
        "temp_mod_rates": np.asarray(a["temp_mod_rates"]),
    }


def sliding_window_iter(
    t: np.ndarray,
    window_sec: float,
    step_sec: float,
):
    """Yield ``(start_idx, end_idx, center_time_s)`` triples for a sliding window.

    The generator walks a window of ``window_sec`` seconds across the
    time axis in steps of ``step_sec`` seconds. ``end_idx`` is exclusive
    so ``t[start:end]`` selects exactly the samples in the window.

    Parameters
    ----------
    t : np.ndarray
        Time axis of shape ``(n_time,)`` in seconds, assumed roughly
        uniformly spaced.
    window_sec : float
        Window length in seconds.
    step_sec : float
        Step between successive windows, in seconds.

    Yields
    ------
    tuple[int, int, float]
        ``(start_idx, end_idx, center_time_s)`` for each window.

    Raises
    ------
    ValueError
        If the time axis has fewer than two samples.
    """
    if t.size < 2:
        raise ValueError("time axis too short for a sliding window")
    dt = float(np.median(np.diff(t)))
    window_pts = max(1, int(round(window_sec / dt)))
    step_pts = max(1, int(round(step_sec / dt)))
    max_end = t.size
    for start in range(0, max_end - window_pts + 1, step_pts):
        end = start + window_pts
        yield start, end, float(t[start:end].mean())
