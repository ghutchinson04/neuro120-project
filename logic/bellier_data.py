"""Data loading and supergrid assembly for the Bellier 2023 ECoG dataset.

Each ``P*_HFA_data.mat`` stores ``ecog`` (``(T, n_elec)``) at 100 Hz
aligned to the 190.72 s Pink Floyd stimulus, plus an ``artifacts`` mask
and a ``dataInfo`` struct listing reference / noisy / epileptic
channels to drop. The companion ``P*_MNI_electrode_coordinates.mat``
stores MNI positions and FreeSurfer anatomical labels.

Public entry points
-------------------
* :func:`load_patient` -- load one patient with cleaned HFA + metadata.
* :func:`build_supergrid` -- pool all 29 patients into a single
  ``(T, n_electrodes)`` HFA matrix plus per-electrode provenance.
* :func:`load_vocal_segments` -- parse the vocal-segment CSV into a
  100 Hz binary mask aligned to the song.
* :func:`electrode_subsets` -- index subsets for
  ``{all, right_STG, left_STG, non_STG}`` used by downstream decoders.
* :func:`detect_onsets` -- 0->1 transitions of a binary mask.
* :func:`assign_anatomy` -- map a FreeSurfer label to our coarse group.

This module is pure data handling: it reads MATLAB files and writes a
cache ``.npz``, but performs no decoding, plotting, or statistics.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.io as sio

from config import (
    BELLIER_FS,
    BELLIER_HFA_DIR,
    BELLIER_T,
    BELLIER_VOCAL_CSV,
    CACHE_DIR,
    ensure_dirs,
)


# anatomy lookup tables mapping freesurfer regions to coarse groups
_STG_KEYS = {"superiortemporal"}
_SMC_KEYS = {"precentral", "postcentral"}
_IFG_KEYS = {"parstriangularis", "parsopercularis", "parsorbitalis"}


def assign_anatomy(label: str) -> Dict[str, str]:
    """Map a raw FreeSurfer anatomical label to a coarse group and hemisphere.

    We collapse many labels to a few task-relevant groups:

    * ``STG``   -- superior temporal gyrus (auditory cortex).
    * ``SMC``   -- pre/postcentral sensorimotor cortex.
    * ``IFG``   -- inferior frontal gyrus (pars triangularis / opercularis /
      orbitalis).
    * ``other`` -- anything else.

    Hemisphere is parsed from the label: ``ctx-lh-...`` / ``ctx-rh-...``
    for cortical labels, or the ``Left-`` / ``Right-`` prefix for
    non-cortical structures.

    Parameters
    ----------
    label : str
        Raw FreeSurfer label, e.g. ``"ctx-lh-superiortemporal"``.

    Returns
    -------
    dict[str, str]
        ``{"group": str, "hemi": str}`` with ``hemi`` in
        ``{"lh", "rh", "unknown"}``.
    """
    s = str(label)
    hemi = "unknown"
    m = re.match(r"ctx-(lh|rh)-(.+)$", s)
    if m:
        hemi = m.group(1)
        region = m.group(2)
    else:
        # non cortical labels start with right or left followed by a region
        if s.startswith("Right-"):
            hemi = "rh"
        elif s.startswith("Left-"):
            hemi = "lh"
        region = s.lower()

    if region in _STG_KEYS:
        group = "STG"
    elif region in _SMC_KEYS:
        group = "SMC"
    elif region in _IFG_KEYS:
        group = "IFG"
    else:
        group = "other"
    return {"group": group, "hemi": hemi}


# per patient loading helpers
def _patient_paths(pid: str) -> tuple[Path, Path]:
    """Return ``(hfa_file, mni_file)`` for one patient, or raise if missing.

    Used by :func:`load_patient` to fail loudly on a missing file
    rather than silently propagating a MATLAB load error.
    """
    hfa_f = Path(BELLIER_HFA_DIR) / f"{pid}_HFA_data.mat"
    mni_f = Path(BELLIER_HFA_DIR) / f"{pid}_MNI_electrode_coordinates.mat"
    if not hfa_f.exists() or not mni_f.exists():
        raise FileNotFoundError(f"Missing Bellier files for {pid}: {hfa_f} / {mni_f}")
    return hfa_f, mni_f


def _indices_to_drop(info) -> np.ndarray:
    """Union of reference / noisy / epileptic channel indices (0-based).

    The Bellier ``dataInfo`` struct lists these with MATLAB 1-based
    indexing. We subtract 1 and take the union so ``load_patient`` can
    drop them with a single ``np.setdiff1d`` call.
    """
    parts: List[np.ndarray] = []
    for fld in ("idxRefElec", "idxNoisyElecs", "idxEpilepticElecs"):
        v = getattr(info, fld)
        a = np.atleast_1d(np.asarray(v)).ravel()
        if a.size == 0:
            continue
        # matlab uses 1 based indexing so shift down by one
        parts.append(a.astype(int) - 1)
    if not parts:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(parts))


def load_patient(pid: str) -> Dict[str, object]:
    """Load one patient's HFA + artifact mask + anatomy with bad channels dropped.

    Reference, noisy, and epileptic channels (as flagged in the
    Bellier ``dataInfo`` struct) are removed before any downstream
    analysis, so the caller never has to re-filter. We also hard-check
    that every per-electrode array has the same length and that the
    recording length and sampling rate match the published values.

    Parameters
    ----------
    pid : str
        Patient identifier, e.g. ``"P01"``.

    Returns
    -------
    dict
        * ``"patient_id"`` : echo of ``pid``.
        * ``"laterality"`` : implantation side from ``dataInfo``.
        * ``"hfa"`` : ``(T, n_keep)`` cleaned HFA matrix.
        * ``"artifacts"`` : ``(T, n_keep)`` bool mask, same shape as
          ``hfa``.
        * ``"mni"`` : ``(n_keep, 3)`` MNI coordinates.
        * ``"anatomy_raw"`` : ``(n_keep,)`` raw FreeSurfer labels.
        * ``"group"``, ``"hemi"`` : coarse anatomy (see
          :func:`assign_anatomy`).
        * ``"channel_label"`` : per-patient electrode names.
        * ``"n_dropped"``, ``"n_kept"`` : integer counts for reporting.

    Raises
    ------
    FileNotFoundError
        If either the HFA or MNI file for ``pid`` is missing.
    ValueError
        If the per-electrode arrays disagree in length, or the
        recording length/sampling rate differ from the expected
        :data:`config.BELLIER_T` / :data:`config.BELLIER_FS`.
    """
    hfa_f, mni_f = _patient_paths(pid)

    m = sio.loadmat(hfa_f, squeeze_me=True, struct_as_record=False)
    ecog = np.asarray(m["ecog"], dtype=float)  # shape (t, e)
    artifacts = np.asarray(m["artifacts"]).astype(bool)  # shape (t, e)
    info = m["dataInfo"]
    fs = float(info.fs)
    laterality = str(info.implantationLaterality)

    mni = sio.loadmat(mni_f, squeeze_me=True, struct_as_record=False)["elec_mni_frvr"]
    elecpos = np.asarray(mni.elecpos, dtype=float)  # shape (e, 3)
    anat_strs = np.asarray(mni.anatLabels, dtype=object).ravel().astype(str)
    labels = np.asarray(mni.label, dtype=object).ravel().astype(str)

    E = ecog.shape[1]
    if not (E == artifacts.shape[1] == elecpos.shape[0] == anat_strs.size == labels.size):
        raise ValueError(
            f"{pid}: inconsistent electrode counts "
            f"(ecog={E}, artifacts={artifacts.shape[1]}, elecpos={elecpos.shape[0]}, "
            f"anat={anat_strs.size}, label={labels.size})"
        )
    if ecog.shape[0] != BELLIER_T:
        raise ValueError(f"{pid}: expected T={BELLIER_T}, got {ecog.shape[0]}")
    if abs(fs - BELLIER_FS) > 1e-6:
        raise ValueError(f"{pid}: expected fs={BELLIER_FS}, got {fs}")

    drop = _indices_to_drop(info)
    keep = np.setdiff1d(np.arange(E), drop)

    ecog = ecog[:, keep]
    artifacts = artifacts[:, keep]
    elecpos = elecpos[keep]
    anat_strs = anat_strs[keep]
    labels = labels[keep]

    anat_parsed = [assign_anatomy(s) for s in anat_strs]
    group = np.array([a["group"] for a in anat_parsed])
    hemi = np.array([a["hemi"] for a in anat_parsed])

    return {
        "patient_id": pid,
        "laterality": laterality,
        "hfa": ecog,                    # shape (t, n_keep)
        "artifacts": artifacts,         # shape (t, n_keep)
        "mni": elecpos,                 # shape (n_keep, 3)
        "anatomy_raw": anat_strs,       # shape (n_keep,)
        "group": group,                 # shape (n_keep,), values stg smc ifg other
        "hemi": hemi,                   # shape (n_keep,), values lh rh unknown
        "channel_label": labels,        # shape (n_keep,), per patient names
        "n_dropped": int(drop.size),
        "n_kept": int(keep.size),
    }


# supergrid, pools every patient into one shared matrix
def _all_patient_ids() -> List[str]:
    """Return every ``P*`` patient identifier found in ``BELLIER_HFA_DIR``.

    Sorted numerically by the integer after the leading ``P`` so
    downstream reporting is in ``P1, P2, ..., P29`` order rather than
    lexicographic.
    """
    files = sorted(Path(BELLIER_HFA_DIR).glob("P*_HFA_data.mat"))
    pids = [f.stem.split("_")[0] for f in files]
    pids.sort(key=lambda s: int(s[1:]))
    return pids


def build_supergrid(cache: bool = True, verbose: bool = True) -> Dict[str, object]:
    """Pool all 29 patients into a single ``(T, n_electrodes)`` HFA matrix.

    Every patient's recording is 19072 samples at 100 Hz aligned to the
    same song, so the time axis is shared and we just concatenate along
    the electrode axis. Per-electrode metadata arrays track provenance
    (patient ID, channel label, MNI position, anatomy).

    Parameters
    ----------
    cache : bool, default=True
        If True, reuse ``results/cache/bellier_supergrid.npz`` when
        present and write it otherwise. Loading the cache cuts startup
        from minutes to <1 s.
    verbose : bool, default=True
        Print a per-patient ``kept/dropped`` summary while building.

    Returns
    -------
    dict
        * ``"hfa"`` : ``(T, n_electrodes)`` float HFA matrix.
        * ``"artifacts"`` : ``(T, n_electrodes)`` bool mask.
        * ``"patient_id"`` : ``(n_electrodes,)`` patient each column came from.
        * ``"channel_label"`` : ``(n_electrodes,)`` per-patient names.
        * ``"mni"`` : ``(n_electrodes, 3)`` MNI coordinates.
        * ``"group"``, ``"hemi"``, ``"anatomy_raw"`` : anatomical labels.
        * ``"fs"`` : float sampling rate (100).
        * ``"T"`` : int number of samples (19072).
    """
    ensure_dirs()
    cache_path = CACHE_DIR / "bellier_supergrid.npz"
    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        return {
            "hfa": z["hfa"],
            "artifacts": z["artifacts"],
            "patient_id": z["patient_id"],
            "channel_label": z["channel_label"],
            "mni": z["mni"],
            "group": z["group"],
            "hemi": z["hemi"],
            "anatomy_raw": z["anatomy_raw"],
            "fs": float(z["fs"].item()),
            "T": int(z["T"].item()),
        }

    pids = _all_patient_ids()
    hfa_cols: List[np.ndarray] = []
    art_cols: List[np.ndarray] = []
    pid_arr: List[str] = []
    label_arr: List[str] = []
    mni_rows: List[np.ndarray] = []
    group_arr: List[str] = []
    hemi_arr: List[str] = []
    anat_raw_arr: List[str] = []

    for pid in pids:
        rec = load_patient(pid)
        hfa_cols.append(rec["hfa"])
        art_cols.append(rec["artifacts"])
        mni_rows.append(rec["mni"])
        n = rec["hfa"].shape[1]
        pid_arr.extend([pid] * n)
        label_arr.extend(rec["channel_label"].tolist())
        group_arr.extend(rec["group"].tolist())
        hemi_arr.extend(rec["hemi"].tolist())
        anat_raw_arr.extend(rec["anatomy_raw"].tolist())
        if verbose:
            print(f"  {pid}: kept {n} / dropped {rec['n_dropped']}")

    hfa = np.concatenate(hfa_cols, axis=1)
    artifacts = np.concatenate(art_cols, axis=1)
    mni = np.concatenate(mni_rows, axis=0)
    patient_id = np.array(pid_arr)
    channel_label = np.array(label_arr)
    group = np.array(group_arr)
    hemi = np.array(hemi_arr)
    anatomy_raw = np.array(anat_raw_arr)

    assert hfa.shape == artifacts.shape
    assert hfa.shape[0] == BELLIER_T
    assert hfa.shape[1] == patient_id.size == mni.shape[0]

    if cache:
        np.savez_compressed(
            cache_path,
            hfa=hfa.astype(np.float32),
            artifacts=artifacts,
            patient_id=patient_id,
            channel_label=channel_label,
            mni=mni.astype(np.float32),
            group=group,
            hemi=hemi,
            anatomy_raw=anatomy_raw,
            fs=np.asarray(BELLIER_FS),
            T=np.asarray(BELLIER_T),
        )

    return {
        "hfa": hfa,
        "artifacts": artifacts,
        "patient_id": patient_id,
        "channel_label": channel_label,
        "mni": mni,
        "group": group,
        "hemi": hemi,
        "anatomy_raw": anatomy_raw,
        "fs": BELLIER_FS,
        "T": BELLIER_T,
    }


# vocal labels parsed from a csv into a sample aligned mask
def load_vocal_segments(
    csv_path: Path | str = BELLIER_VOCAL_CSV,
    n_samples: int = BELLIER_T,
    fs: int = BELLIER_FS,
) -> np.ndarray:
    """Parse ``vocal_segments.csv`` into a sample-aligned boolean vocal mask.

    Parameters
    ----------
    csv_path : Path or str, default=:data:`config.BELLIER_VOCAL_CSV`
        Path to a CSV with at least ``start_s`` and ``end_s`` columns
        (seconds, relative to song start).
    n_samples : int, default=:data:`config.BELLIER_T`
        Length of the output mask, in samples.
    fs : int, default=:data:`config.BELLIER_FS`
        Sampling rate in Hz used to convert seconds to sample indices.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n_samples,)``; ``mask[i]`` is True iff
        sample ``i`` falls within any annotated interval
        ``[start_s, end_s)``.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If the CSV is missing the required columns, or if any interval
        has ``end_s < start_s``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"vocal_segments.csv not found at {csv_path}. "
            f"Please supply a CSV with columns start_s,end_s."
        )
    df = pd.read_csv(csv_path)
    required = {"start_s", "end_s"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must have columns {required}; got {list(df.columns)}")

    mask = np.zeros(n_samples, dtype=bool)
    duration = n_samples / fs
    for _, row in df.iterrows():
        t0 = float(row["start_s"])
        t1 = float(row["end_s"])
        if t1 < t0:
            raise ValueError(f"invalid vocal segment: start_s={t0} > end_s={t1}")
        t0 = max(0.0, t0)
        t1 = min(duration, t1)
        i0 = int(round(t0 * fs))
        i1 = int(round(t1 * fs))
        mask[i0:i1] = True
    return mask


# electrode subsets grouped by anatomy and hemisphere
def electrode_subsets(supergrid: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Return the four anatomical electrode-index subsets used downstream.

    Parameters
    ----------
    supergrid : dict
        Output of :func:`build_supergrid`.

    Returns
    -------
    dict[str, np.ndarray]
        Integer index arrays into ``supergrid["hfa"]``'s electrode axis
        for keys ``{"all", "right_STG", "left_STG", "non_STG"}``.
    """
    group = np.asarray(supergrid["group"])
    hemi = np.asarray(supergrid["hemi"])
    n = group.size
    is_stg = group == "STG"
    return {
        "all": np.arange(n),
        "right_STG": np.where(is_stg & (hemi == "rh"))[0],
        "left_STG": np.where(is_stg & (hemi == "lh"))[0],
        "non_STG": np.where(~is_stg)[0],
    }


def detect_onsets(mask: np.ndarray) -> np.ndarray:
    """Return sample indices where ``mask`` transitions from 0 to 1.

    Used to build event-locked profiles around vocal-onset (and by
    symmetry instrumental-onset) times in the Bellier recording.

    Parameters
    ----------
    mask : np.ndarray
        Boolean or 0/1 array of shape ``(n_samples,)``.

    Returns
    -------
    np.ndarray
        Integer array of the sample indices where ``mask[i-1] == False``
        and ``mask[i] == True``.
    """
    m = np.asarray(mask).astype(bool).astype(np.int8)
    return np.flatnonzero(np.diff(np.concatenate([[0], m])) == 1)
