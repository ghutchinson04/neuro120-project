"""Blocked-CV vocal-vs-instrumental decoder for the Bellier supergrid.

Pipeline
--------
1. Slice the continuous HFA into fixed-length windows
   (:func:`make_windows`) and label each window by majority vote of the
   vocal mask.
2. Assign windows to interleaved blocked time folds
   (:func:`interleaved_blocked_folds`) so no contiguous chunk of audio
   leaks across train/test and every fold still sees both classes.
3. Fit logistic regression under blocked CV -- always. This is the
   headline result (Figure 8).
4. Fit a tiny temporal CNN (:func:`run_cnn_subset`) *only* for subsets
   where logreg already clears chance by at least
   :data:`config.CNN_MIN_BACC_OVER_CHANCE`. The CNN is strictly
   supplementary; it never rescues a null logreg result.

Public entry point: :func:`run_vocal_instrumental_decoder`.

Blocked CV is used here because neighbouring time windows are heavily
correlated in HFA: a plain shuffled k-fold would leak nearby windows
from train into test and inflate every score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    BELLIER_BOOT_N,
    BELLIER_CV_FOLDS,
    BELLIER_FS,
    BELLIER_WIN_SEC,
    BELLIER_WIN_STEP_S,
    CNN_BATCH,
    CNN_EPOCHS,
    CNN_LR,
    CNN_MIN_BACC_OVER_CHANCE,
    RANDOM_STATE,
)


CHANCE_BACC = 0.5
# cap cnn input width, logreg always uses the full subset
CNN_MAX_N_ELEC = 256


# windowing and folding

@dataclass
class WindowedData:
    # per window 3d tensor used by the cnn
    X_tensor: np.ndarray   # shape (n_wins, n_elec, win_samples)
    # time averaged per window features used by logreg
    X_mean: np.ndarray     # shape (n_wins, n_elec)
    # binary vocal label per window
    y: np.ndarray          # shape (n_wins,)
    # sample index of each window centre used to build folds
    center_idx: np.ndarray # shape (n_wins,)
    # fold assignment in [0, k) where negative values mark buffer windows
    fold: np.ndarray       # shape (n_wins,)
    # window length in samples
    win_samples: int


def make_windows(
    hfa: np.ndarray,
    mask: np.ndarray,
    fs: int = BELLIER_FS,
    win_sec: float = BELLIER_WIN_SEC,
    step_sec: float = BELLIER_WIN_STEP_S,
) -> WindowedData:
    """Slice HFA into fixed-length windows with majority-vote vocal labels.

    Each window contributes one row to the decoder. The label is 1 iff
    the vocal mask is True on more than half of the window's samples;
    using a majority vote keeps labels robust to annotation jitter.

    Parameters
    ----------
    hfa : np.ndarray
        HFA recording of shape ``(T, n_elec)``.
    mask : np.ndarray
        Sample-level vocal-present mask of shape ``(T,)``.
    fs : int, default=:data:`config.BELLIER_FS`
        Sampling rate (Hz). Used to convert ``win_sec``/``step_sec`` to
        integer sample counts.
    win_sec, step_sec : float
        Window length and stride, in seconds.

    Returns
    -------
    WindowedData
        Dataclass bundling the per-window tensor (for the CNN), the
        time-averaged matrix (for logistic regression), binary labels,
        window-centre sample indices, and a placeholder fold array
        filled in later by :func:`interleaved_blocked_folds`.

    Raises
    ------
    ValueError
        If ``hfa`` and ``mask`` have mismatched time axes, or if the
        window/step parameters collapse to zero samples.
    """
    hfa = np.ascontiguousarray(hfa)
    mask = np.asarray(mask, dtype=bool)
    T, E = hfa.shape
    if mask.size != T:
        raise ValueError(f"hfa T={T} != mask T={mask.size}")

    # convert seconds to integer sample counts
    win = int(round(win_sec * fs))
    step = int(round(step_sec * fs))
    if win <= 0 or step <= 0:
        raise ValueError("win_sec and step_sec must produce >= 1 sample")

    # start index of every window along the time axis
    starts = np.arange(0, T - win + 1, step, dtype=int)

    # build the per window tensor and its time averaged companion
    X_tensor = np.stack([hfa[s : s + win].T for s in starts]).astype(np.float32)
    X_mean = X_tensor.mean(axis=2)

    # label each window by majority vote over its vocal mask samples
    y_frac = np.array([mask[s : s + win].mean() for s in starts])
    y = (y_frac > 0.5).astype(int)

    # window centre sample indices feed the fold assignment below
    center_idx = starts + win // 2

    return WindowedData(
        X_tensor=X_tensor,
        X_mean=X_mean.astype(np.float32),
        y=y,
        center_idx=center_idx,
        fold=np.zeros(starts.size, dtype=int),
        win_samples=win,
    )


def interleaved_blocked_folds(
    center_idx: np.ndarray,
    T: int,
    n_folds: int = BELLIER_CV_FOLDS,
    n_blocks_per_fold: int = 3,
    buffer_samples: int = 0,
) -> np.ndarray:
    """Assign windows to folds by round-robin over contiguous time blocks.

    Time is split into ``n_folds * n_blocks_per_fold`` equal contiguous
    blocks; block ``b`` is assigned to fold ``b % n_folds``. Each fold
    therefore consists of ``n_blocks_per_fold`` non-adjacent time
    segments, one from each region of the timeline. This preserves the
    temporal-block structure (no half-window leakage across blocks) while
    guaranteeing that sparse class labels (e.g. vocals concentrated at
    the start of a song) are present in every fold.

    ``buffer_samples`` drops windows whose centre lies within that many
    samples of any internal block boundary.
    """
    # split the full timeline into equally sized contiguous blocks
    n_blocks = int(n_folds * n_blocks_per_fold)
    boundaries = np.linspace(0, T, n_blocks + 1)

    # assign each window to a block via its centre sample
    block = np.digitize(center_idx, boundaries[1:-1])
    block = np.clip(block, 0, n_blocks - 1)

    # round robin blocks onto folds so every fold spans the whole timeline
    fold = (block % n_folds).astype(int)

    # optionally drop windows whose centre is near any internal boundary
    if buffer_samples > 0:
        for b in boundaries[1:-1]:
            fold[np.abs(center_idx - b) < buffer_samples] = -1
    return fold


# scoring helpers

def _bacc_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile 95% CI for balanced accuracy via window-level bootstrap.

    Resamples ``(y_true, y_pred)`` pairs with replacement ``n_boot``
    times and returns the 2.5/97.5 percentile of the resulting
    balanced-accuracy samples. Bootstraps that degenerate to a single
    class are skipped.
    """
    scores = []
    n = len(y_true)
    # degenerate case, balanced accuracy is not defined with a single class
    if n < 2 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    # resample (y_true, y_pred) pairs with replacement and rescore
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        # skip resamples that collapsed to a single class
        if len(np.unique(yt)) < 2:
            continue
        scores.append(balanced_accuracy_score(yt, yp))

    if not scores:
        return float("nan"), float("nan")
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def _run_blocked_cv(
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    fit_predict_fold,
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    """Blocked-CV scaffolding shared by logreg and CNN.

    Filters buffer rows (``fold == -1``), iterates folds, delegates fit
    + test-set prediction to ``fit_predict_fold``, aggregates per-fold
    metrics, and returns a bootstrap CI on balanced accuracy.

    ``fit_predict_fold(X_train, y_train, X_test, k, rng)`` must return
    ``(y_pred_test, y_score_test)``. The same ``rng`` is passed through
    the fold loop and then reused by ``_bacc_bootstrap`` so the
    seed-to-output mapping is deterministic across both stages.
    """
    # drop buffer rows that were not assigned to any fold
    keep = fold >= 0
    X, y, fold = X[keep], y[keep], fold[keep]

    # one rng is shared across training and the later bootstrap
    rng = np.random.default_rng(seed)

    # sentinel values for rows that never land in any test split
    folds = sorted(np.unique(fold).tolist())
    y_pred = np.full(y.shape, -1, dtype=int)
    y_score = np.full(y.shape, np.nan, dtype=float)
    fold_rows: List[Dict[str, object]] = []

    for k in folds:
        test = fold == k
        train = ~test
        n_tr, n_te = int(train.sum()), int(test.sum())

        # skip folds where either split has only one class
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            fold_rows.append({"fold": int(k), "bacc": float("nan"), "auc": float("nan"),
                              "n_train": n_tr, "n_test": n_te})
            continue

        # delegate the actual fit and test set prediction to the caller
        preds, scores = fit_predict_fold(X[train], y[train], X[test], int(k), rng)
        y_pred[test] = preds
        y_score[test] = scores
        fold_rows.append({
            "fold": int(k),
            "bacc": float(balanced_accuracy_score(y[test], preds)),
            "auc": float(roc_auc_score(y[test], scores)),
            "n_train": n_tr,
            "n_test": n_te,
        })

    # aggregate across valid folds
    valid = y_pred >= 0
    mean_bacc = (
        float(np.nanmean([r["bacc"] for r in fold_rows])) if fold_rows else float("nan")
    )
    mean_auc = (
        float(np.nanmean([r["auc"] for r in fold_rows])) if fold_rows else float("nan")
    )

    # window level bootstrap ci on balanced accuracy
    bacc_lo, bacc_hi = _bacc_bootstrap(y[valid], y_pred[valid], n_boot=n_boot, rng=rng)
    return {
        "fold_scores": fold_rows,
        "mean_bacc": mean_bacc,
        "mean_auc": mean_auc,
        "bacc_ci95": (bacc_lo, bacc_hi),
        "y_pred": y_pred,
        "y_score": y_score,
        "y_true": y,
    }


# logistic regression

def run_logreg_subset(
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    n_boot: int = BELLIER_BOOT_N,
    seed: int = RANDOM_STATE,
) -> Dict[str, object]:
    """Blocked-CV logistic regression on time-averaged windows.

    Uses :class:`LogisticRegressionCV` so ``C`` is tuned via an inner
    3-fold CV on the training folds, with balanced accuracy as the
    selection criterion. Scaling is fit inside each outer fold.

    Parameters
    ----------
    X : np.ndarray
        Time-averaged features of shape ``(n_windows, n_elec)``.
    y : np.ndarray
        Binary labels of shape ``(n_windows,)``.
    fold : np.ndarray
        Fold assignment of shape ``(n_windows,)``. Entries equal to
        ``-1`` are dropped (boundary buffer).
    n_boot : int, default=:data:`config.BELLIER_BOOT_N`
        Bootstrap replicates for the balanced-accuracy CI.
    seed : int, default=:data:`config.RANDOM_STATE`
        Seed forwarded to the RNG used for the bootstrap.

    Returns
    -------
    dict
        * ``"fold_scores"`` : per-fold dicts with ``bacc``, ``auc``,
          ``n_train``, ``n_test``.
        * ``"mean_bacc"``, ``"mean_auc"`` : mean across folds.
        * ``"bacc_ci95"`` : ``(lo, hi)`` 95% percentile CI for
          balanced accuracy.
        * ``"y_pred"``, ``"y_score"``, ``"y_true"`` : held-out arrays
          concatenated across folds (``-1`` / ``NaN`` for dropped rows).
    """
    # fit and predict callback for a single outer fold
    def _fit(Xtr, ytr, Xte, _k, _rng):
        # scaler is fit on train only, logreg tunes c via an inner 3 fold split
        # on balanced accuracy
        pipe = Pipeline(steps=[
            ("sc", StandardScaler()),
            ("lr", LogisticRegressionCV(
                Cs=np.logspace(-3, 3, 7), cv=3, max_iter=2000,
                scoring="balanced_accuracy", n_jobs=1,
            )),
        ])
        pipe.fit(Xtr, ytr)
        # return hard labels and positive class probabilities
        return pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]

    return _run_blocked_cv(X, y, fold, _fit, n_boot=n_boot, seed=seed)


# tiny temporal cnn

def _build_cnn(n_elec: int, win_samples: int):
    """Construct the TinyTemporalCNN used for the Bellier supplementary result.

    Architecture: two 1-D convolutional blocks with GELU and dropout,
    followed by global average pooling and a single linear head. The
    output is a binary logit (one number per window).
    """

    class TinyTemporalCNN(nn.Module):
        def __init__(self, n_elec: int, win_samples: int, hidden: int = 32):
            super().__init__()
            self.conv1 = nn.Conv1d(n_elec, hidden, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(hidden * 2, 1)

        def forward(self, x):  # (b, n_elec, win_samples)
            h = self.act(self.conv1(x))
            h = self.drop(h)
            h = self.act(self.conv2(h))
            h = self.pool(h).squeeze(-1)
            return self.fc(h).squeeze(-1)

    return TinyTemporalCNN(n_elec, win_samples)


def _cnn_subsample_channels(n_elec: int, cap: int, seed: int) -> np.ndarray:
    """Return a deterministic subset of channel indices when ``n_elec > cap``.

    The CNN's first conv has ``n_elec`` input channels; capping keeps
    runtime and parameter count bounded when the ``all`` subset runs
    into the thousands of electrodes.
    """
    # no subsampling needed when the subset already fits under the cap
    if n_elec <= cap:
        return np.arange(n_elec)
    # seeded choice so every fold sees the same channels
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_elec, cap, replace=False))


def run_cnn_subset(
    X_tensor: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    n_boot: int = BELLIER_BOOT_N,
    seed: int = RANDOM_STATE,
    epochs: int = CNN_EPOCHS,
    batch_size: int = CNN_BATCH,
    lr: float = CNN_LR,
    max_n_elec: int = CNN_MAX_N_ELEC,
) -> Dict[str, object]:
    """Blocked-CV TinyTemporalCNN with per-fold electrode standardisation.

    Parameters
    ----------
    X_tensor : np.ndarray
        Per-window tensor of shape ``(n_windows, n_elec, win_samples)``.
    y : np.ndarray
        Binary labels of shape ``(n_windows,)``.
    fold : np.ndarray
        Fold assignment of shape ``(n_windows,)``; ``-1`` entries are
        dropped.
    n_boot : int
        Bootstrap replicates for the balanced-accuracy CI.
    seed : int
        Seed for the RNG, for ``torch.manual_seed`` per fold, and for
        the deterministic channel subsample.
    epochs, batch_size, lr : int, int, float
        Training hyperparameters.
    max_n_elec : int
        Hard cap on the number of input electrodes (see
        :func:`_cnn_subsample_channels`).

    Returns
    -------
    dict
        Same structure as :func:`run_logreg_subset` plus two extra
        diagnostic fields:

        * ``"cnn_n_elec"`` : number of electrodes actually fed to the CNN.
        * ``"cnn_subsampled"`` : True iff subsampling was applied.
    """
    # cap the input channels once, before any fold split
    chan_idx = _cnn_subsample_channels(X_tensor.shape[1], max_n_elec, seed)
    if chan_idx.size < X_tensor.shape[1]:
        X_tensor = X_tensor[:, chan_idx, :]

    # fit and predict callback for a single outer fold
    def _fit(Xtr, ytr, Xte, k, rng):
        # per fold electrode standardisation computed on train only
        mean = Xtr.mean(axis=(0, 2), keepdims=True)
        std = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
        Xtr_n = ((Xtr - mean) / std).astype(np.float32)
        Xte_n = ((Xte - mean) / std).astype(np.float32)
        ytr_f = ytr.astype(np.float32)

        # deterministic model weights per fold
        torch.manual_seed(seed + k)
        net = _build_cnn(Xtr_n.shape[1], Xtr_n.shape[2])
        opt = torch.optim.Adam(net.parameters(), lr=lr)

        # reweight the positive class when the split is imbalanced
        pos = max(1e-3, min(1 - 1e-3, float(ytr_f.mean())))
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(1 - pos) / pos], dtype=torch.float32)
        )

        # mini batch training, the shared rng makes the shuffle sequence reproducible
        Xtr_t = torch.from_numpy(Xtr_n)
        ytr_t = torch.from_numpy(ytr_f)
        net.train()
        for _ in range(epochs):
            idx = rng.permutation(len(Xtr_t))
            for i in range(0, len(idx), batch_size):
                b = idx[i : i + batch_size]
                opt.zero_grad()
                loss = crit(net(Xtr_t[b]), ytr_t[b])
                loss.backward()
                opt.step()

        # score the test fold then threshold at a half to get hard labels
        net.eval()
        with torch.no_grad():
            scores = torch.sigmoid(net(torch.from_numpy(Xte_n))).numpy()
        return (scores > 0.5).astype(int), scores

    # reuse the shared cv scaffolding and tack on cnn specific metadata
    res = _run_blocked_cv(X_tensor, y, fold, _fit, n_boot=n_boot, seed=seed)
    res["cnn_n_elec"] = int(X_tensor.shape[1])
    res["cnn_subsampled"] = bool(chan_idx.size == max_n_elec)
    return res


# top level entry point

DEFAULT_SUBSETS: Sequence[str] = ("all", "right_STG", "left_STG", "non_STG")


def _summary_row(
    subset: str,
    model: str,
    n_elec: int,
    n_windows: int,
    result: Dict[str, object],
) -> Dict[str, object]:
    """Flatten a per-subset result dict into one row for the summary table."""
    lo, hi = result["bacc_ci95"]
    return {
        "subset": subset,
        "model": model,
        "n_elec": int(n_elec),
        "n_windows": int(n_windows),
        "mean_bacc": float(result["mean_bacc"]),
        "mean_auc": float(result["mean_auc"]),
        "bacc_ci95_low": float(lo),
        "bacc_ci95_high": float(hi),
    }


def run_vocal_instrumental_decoder(
    supergrid: Dict[str, object],
    subsets_idx: Dict[str, np.ndarray],
    vocal_mask: np.ndarray,
    subsets: Sequence[str] = DEFAULT_SUBSETS,
    n_boot: int = BELLIER_BOOT_N,
    seed: int = RANDOM_STATE,
) -> Dict[str, object]:
    """Blocked-CV vocal-vs-instrumental decoder across anatomical subsets.

    For each named subset we build windowed features, assign
    interleaved blocked folds, always fit logistic regression, and
    additionally fit the TinyTemporalCNN iff logreg clears chance by
    :data:`config.CNN_MIN_BACC_OVER_CHANCE`.

    Parameters
    ----------
    supergrid : dict
        Output of :func:`bellier_data.build_supergrid`.
    subsets_idx : dict[str, np.ndarray]
        Output of :func:`bellier_data.electrode_subsets`.
    vocal_mask : np.ndarray
        Sample-aligned vocal-present mask of shape ``(T,)``.
    subsets : Sequence[str], default=:data:`DEFAULT_SUBSETS`
        Which keys of ``subsets_idx`` to run.
    n_boot : int, default=:data:`config.BELLIER_BOOT_N`
        Bootstrap replicates for the balanced-accuracy CIs.
    seed : int, default=:data:`config.RANDOM_STATE`
        Seed forwarded to CV/bootstrap/CNN.

    Returns
    -------
    dict
        * ``"summary"`` : :class:`pandas.DataFrame` with one row per
          ``(subset, model)`` and columns ``mean_bacc``, ``mean_auc``,
          ``bacc_ci95_low``, ``bacc_ci95_high``, ``n_elec``, ``n_windows``.
        * ``"fold_scores"`` : nested ``dict[subset][model] -> list[fold dicts]``.
        * ``"cnn_ran"`` : ``dict[subset] -> bool`` flag.
        * ``"raw"`` : ``dict[(subset, model)] -> full result dict`` for
          deeper inspection (per-window predictions, probabilities).
    """
    hfa = np.asarray(supergrid["hfa"])

    # per subset accumulators for the summary table and raw results
    rows: List[Dict[str, object]] = []
    fold_scores: Dict[str, Dict[str, List[dict]]] = {}
    cnn_ran: Dict[str, bool] = {}
    raw: Dict[tuple, dict] = {}

    for name in subsets:
        # slice the supergrid down to the requested electrodes
        idx = np.asarray(subsets_idx[name])
        if idx.size == 0:
            print(f"  [decoder] subset '{name}' empty, skipping")
            continue
        hfa_sub = hfa[:, idx]

        # build windows then round robin them onto interleaved blocked folds,
        # every fold spans both the vocal dense first half and the vocal sparse
        # second half so sparse classes still land in each test split
        wd = make_windows(hfa_sub, vocal_mask)
        wd.fold[:] = interleaved_blocked_folds(
            wd.center_idx, T=hfa.shape[0], n_folds=BELLIER_CV_FOLDS,
            n_blocks_per_fold=3,
        )
        n_wins = wd.y.size

        print(
            f"  [decoder] subset={name:10s} n_elec={idx.size:5d} "
            f"n_wins={n_wins:5d} pos_rate={wd.y.mean():.3f}"
        )

        # logreg runs unconditionally as the headline result
        lr_res = run_logreg_subset(wd.X_mean, wd.y, wd.fold, n_boot=n_boot, seed=seed)
        rows.append(_summary_row(name, "logreg", idx.size, n_wins, lr_res))
        fold_scores.setdefault(name, {})["logreg"] = lr_res["fold_scores"]
        raw[(name, "logreg")] = lr_res
        print(
            f"    logreg  bacc={lr_res['mean_bacc']:.3f} "
            f"CI=[{lr_res['bacc_ci95'][0]:.3f}, {lr_res['bacc_ci95'][1]:.3f}]"
        )

        # the cnn runs only when logreg already beats chance by the configured margin
        if lr_res["mean_bacc"] >= CHANCE_BACC + CNN_MIN_BACC_OVER_CHANCE:
            cnn_res = run_cnn_subset(
                wd.X_tensor, wd.y, wd.fold, n_boot=n_boot, seed=seed
            )
            rows.append(_summary_row(name, "cnn", cnn_res["cnn_n_elec"], n_wins, cnn_res))
            fold_scores[name]["cnn"] = cnn_res["fold_scores"]
            raw[(name, "cnn")] = cnn_res
            cnn_ran[name] = True
            print(
                f"    cnn     bacc={cnn_res['mean_bacc']:.3f} "
                f"CI=[{cnn_res['bacc_ci95'][0]:.3f}, {cnn_res['bacc_ci95'][1]:.3f}] "
                f"(n_elec_in={cnn_res['cnn_n_elec']})"
            )
        else:
            cnn_ran[name] = False
            print(
                f"    cnn     skipped "
                f"(logreg bacc {lr_res['mean_bacc']:.3f} < "
                f"chance+{CNN_MIN_BACC_OVER_CHANCE} = "
                f"{CHANCE_BACC + CNN_MIN_BACC_OVER_CHANCE:.3f})"
            )

    summary = pd.DataFrame(rows)
    return {
        "summary": summary,
        "fold_scores": fold_scores,
        "cnn_ran": cnn_ran,
        "raw": raw,
    }
