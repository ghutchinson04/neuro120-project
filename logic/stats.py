"""Bootstrap, empirical p-values, and other resampling utilities.

This module contains pure analysis functions only: no file I/O, no
plotting, no global state. Every random operation takes an explicit
``seed`` so results are fully reproducible.

Conventions
-----------
* ``n_boot`` / ``n_perm`` default to 1000 so that the (+1)/(N+1)
  correction keeps p-values bounded below 1e-3 when the observed is
  never matched by the null.
* ``alpha`` is the two-sided coverage level (so the default
  ``alpha=0.05`` returns a 95% percentile CI).
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def percentile_ci(
    x: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return a percentile confidence interval for a 1-D sample.

    Parameters
    ----------
    x : Sequence[float]
        Sample values of shape ``(n,)``. ``NaN`` entries are ignored.
    alpha : float, default=0.05
        Two-sided miscoverage level; ``alpha=0.05`` yields a 95% CI.

    Returns
    -------
    tuple of (lo, median, hi) floats
        Lower quantile (``alpha/2``), median, and upper quantile
        (``1 - alpha/2``) of ``x``.
    """
    arr = np.asarray(x, dtype=float)
    return (
        float(np.nanquantile(arr, alpha / 2.0)),
        float(np.nanmedian(arr)),
        float(np.nanquantile(arr, 1.0 - alpha / 2.0)),
    )


def empirical_p_value(
    observed: float,
    null_samples: Sequence[float],
    alternative: str = "greater",
) -> float:
    """Compute an empirical p-value with the conservative (+1)/(N+1) correction.

    The correction prevents a p-value of exactly 0 when none of the
    ``n`` null draws matches or exceeds the observed statistic; this is
    standard practice for permutation tests (see Phipson & Smyth 2010).

    Parameters
    ----------
    observed : float
        Observed test statistic.
    null_samples : Sequence[float]
        Null-distribution draws of shape ``(n,)``.
    alternative : {"greater", "less", "two-sided"}, default="greater"
        * ``"greater"``: ``P(null >= observed)``
        * ``"less"``:    ``P(null <= observed)``
        * ``"two-sided"``: tail probability relative to the null mean,
          i.e. ``P(|null - mean(null)| >= |observed - mean(null)|)``.

    Returns
    -------
    float
        Empirical p-value in ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``alternative`` is not one of the three accepted strings.
    """
    null = np.asarray(null_samples, dtype=float)
    n = null.size
    if alternative == "greater":
        k = int(np.sum(null >= observed))
    elif alternative == "less":
        k = int(np.sum(null <= observed))
    elif alternative == "two-sided":
        mu = np.nanmean(null)
        k = int(np.sum(np.abs(null - mu) >= abs(observed - mu)))
    else:
        raise ValueError(
            f"alternative must be 'greater', 'less', or 'two-sided'; got {alternative!r}"
        )
    return float((k + 1) / (n + 1))


def bootstrap_diff_ci(
    sample_true: Sequence[float],
    sample_null: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for ``mean(sample_true) - mean(sample_null)``.

    Both samples are resampled with replacement independently. When
    ``sample_true`` has length 1 (the common case where ``sample_true``
    is a single point estimate), the true side is kept fixed and the
    returned CI is ``true - bootstrap(mean(sample_null))``.

    Parameters
    ----------
    sample_true : Sequence[float]
        Observed scores for the "true" group, shape ``(n_true,)``.
        Typically length 1 (a single song-only score).
    sample_null : Sequence[float]
        Observed scores for the null/reference group, shape ``(n_null,)``.
    n_boot : int, default=1000
        Number of bootstrap replicates.
    alpha : float, default=0.05
        Two-sided miscoverage level for the returned CI.
    seed : int, default=42
        RNG seed.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``"lo"``, ``"median"``, ``"hi"`` : percentile CI bounds.
        * ``"diffs"`` : ``(n_boot,)`` array of bootstrap differences.
    """
    rng = np.random.default_rng(seed)
    st = np.asarray(sample_true, dtype=float)
    sn = np.asarray(sample_null, dtype=float)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        bt = rng.choice(st, size=st.size, replace=True).mean() if st.size > 1 else st[0]
        bn = rng.choice(sn, size=sn.size, replace=True).mean()
        diffs[b] = bt - bn
    lo, med, hi = percentile_ci(diffs, alpha=alpha)
    return {"lo": lo, "median": med, "hi": hi, "diffs": diffs}
