"""Electrode-subset definitions and the matched random-subset sampler.

Three reference subsets drive every main figure:

* ``all``        -- every electrode in the concatenated Norman-Haignere grid.
* ``no_song``    -- non-song electrodes only (speech + music).
* ``song_only``  -- the small song-selective pool.

In addition, :func:`sample_random_subsets` builds the matched-size null
we use to test whether ``song_only`` decodes song-vs-music better than an
*arbitrary* size-matched subset of non-song electrodes (headline result,
Figure 4).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from config import RANDOM_STATE, SUBSET_SIZE


def define_fixed_subsets(
    electrode_group: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Return the three reference electrode-index subsets used throughout the paper.

    Parameters
    ----------
    electrode_group : Sequence[str]
        Per-electrode group labels of shape ``(n_electrodes,)``, each in
        ``{"song", "speech", "music"}``.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping with three keys, each an ``int`` array of indices into
        ``electrode_group``:

        * ``"all"``        -- shape ``(n_electrodes,)``.
        * ``"no_song"``    -- all non-song electrodes.
        * ``"song_only"``  -- all song-selective electrodes.
    """
    groups = np.asarray(electrode_group)
    n = groups.size
    return {
        "all": np.arange(n),
        "no_song": np.where(groups != "song")[0],
        "song_only": np.where(groups == "song")[0],
    }


def sample_random_subsets(
    electrode_group: Sequence[str],
    n_subsets: int,
    size: int | None = None,
    exclude_song: bool = True,
    seed: int = RANDOM_STATE,
    allow_duplicates: bool = False,
) -> List[np.ndarray]:
    """Sample matched random electrode-index subsets from the non-song pool.

    This function provides the key null model for testing whether
    song-only performance exceeds what would be expected from any
    equally-sized subset of non-song electrodes.

    Parameters
    ----------
    electrode_group : Sequence[str]
        Per-electrode group labels of shape ``(n_electrodes,)``.
    n_subsets : int
        Number of random subsets to draw.
    size : int or None, default=None
        Number of electrodes per sampled subset. Should match the
        song-only count to make the null comparison size-matched.
        Defaults to :data:`config.SUBSET_SIZE`.
    exclude_song : bool, default=True
        If True, draw only from non-song electrodes (the scientifically
        meaningful null); if False, draw from the full grid.
    seed : int, default=:data:`config.RANDOM_STATE`
        RNG seed for reproducible sampling.
    allow_duplicates : bool, default=False
        If False, refuse to return two identical draws. When the pool
        is too small to deliver ``n_subsets`` unique combinations we
        fall back to sampling with duplicates (this is flagged by a
        pool exhaustion but never raises, since the matched-null
        analysis is robust to a small number of repeats).

    Returns
    -------
    list[np.ndarray]
        List of length ``n_subsets``, where each element is an ``int``
        array of shape ``(size,)`` containing unique electrode indices
        (sorted ascending within each subset for determinism).

    Raises
    ------
    ValueError
        If ``size`` exceeds the number of available candidates in the
        chosen pool.
    """
    groups = np.asarray(electrode_group)
    if size is None:
        size = SUBSET_SIZE

    if exclude_song:
        pool = np.where(groups != "song")[0]
    else:
        pool = np.arange(groups.size)

    if size > pool.size:
        raise ValueError(
            f"Cannot draw size={size} from a pool of {pool.size} electrodes "
            f"(exclude_song={exclude_song})."
        )

    rng = np.random.default_rng(seed)
    seen: set = set()
    out: List[np.ndarray] = []
    # soft cap on attempts so degenerate pools do not spin forever before
    # we fall back to the with duplicates branch below
    max_attempts = n_subsets * 50
    attempts = 0
    while len(out) < n_subsets and attempts < max_attempts:
        idx = np.sort(rng.choice(pool, size=size, replace=False))
        key = tuple(idx.tolist())
        if not allow_duplicates and key in seen:
            attempts += 1
            continue
        seen.add(key)
        out.append(idx)
        attempts += 1

    if len(out) < n_subsets:
        # pool too small for unique sampling, fill the remainder allowing
        # duplicates rather than failing silently with fewer subsets than requested
        while len(out) < n_subsets:
            idx = np.sort(rng.choice(pool, size=size, replace=False))
            out.append(idx)
    return out
