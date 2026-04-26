"""End-to-end analysis pipeline for the song-vs-music ECoG project.

This module is the single entry point that reproduces every figure,
table, and cache artifact used in the write-up. Each ``run_*`` function
corresponds to one section of the report and is safe to run in
isolation; :func:`run_all` chains them and writes a consolidated
``results/metrics.json``.

Outputs
-------
* ``results/figures/`` -- ``.png`` and ``.pdf`` versions of every
  paper figure.
* ``results/tables/``  -- CSVs with numerical summaries (one CSV per
  analysis block).
* ``results/cache/``   -- compressed ``.npz`` files with the raw
  arrays (null distributions, per-window scores, bootstrap samples,
  etc.) so notebooks can replot without re-running CV.
* ``results/metrics.json`` -- a flat summary of every headline number,
  written at the end of :func:`run_all`.

Design notes
------------
* All heavy computation lives in other modules (``decoding``,
  ``rdm``, ``analyses``, ``bellier_*``). This file only composes them
  and serialises outputs; this keeps the pipeline easy to audit.
* Running ``python pipeline.py`` from a fresh checkout and a populated
  ``data/`` directory re-creates every numeric claim in the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    BOOTSTRAP_N,
    CACHE_DIR,
    EARLY_WINDOW,
    N_SPLITS,
    PERM_N,
    RANDOM_STATE,
    RANDOM_SUBSETS_N,
    RESULTS_DIR,
    STEP_SEC,
    SUBSET_SIZE,
    TAB_DIR,
    WINDOW_SEC,
    ensure_dirs,
)
from data_utils import build_dataset, load_acoustic_features, window_features
from decoding import (
    cross_temporal_generalization,
    run_grouped_cv,
    run_time_resolved_binary_decoder,
)
from nonlinear import run_nonlinear_comparison
from rdm import (
    bootstrap_divergence_curve,
    permutation_test_divergence_curve,
    time_resolved_divergence,
)
from stats import empirical_p_value
from subsets import define_fixed_subsets
from analyses import (
    compare_true_vs_random_subsets,
    evaluate_subset,
    time_resolved_songmusic_with_ci,
)
from bellier_data import build_supergrid, electrode_subsets, load_vocal_segments
from bellier_decoder import run_vocal_instrumental_decoder
from temporal_profile import bellier_profiles, norman_profiles
from plots import (
    plot_bellier_decoder_subset_bars,
    plot_bellier_event_profiles,
    plot_confusion_matrix,
    plot_cross_temporal_heatmaps,
    plot_divergence_partial_comparison,
    plot_divergence_with_stats,
    plot_loo_contributions,
    plot_nonlinear_comparison,
    plot_random_subset_null,
    plot_temporal_profile_overlay,
    plot_time_resolved_curves,
)



# small io helpers for tables and cached arrays
def _save_table(df: pd.DataFrame, stem: str) -> Path:
    """Write ``df`` to ``results/tables/{stem}.csv`` and return the path."""
    path = TAB_DIR / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def _save_cache(stem: str, **arrays) -> Path:
    """Compressed ``np.savez`` of ``**arrays`` into ``results/cache/{stem}.npz``."""
    path = CACHE_DIR / f"{stem}.npz"
    np.savez_compressed(path, **arrays)
    return path


# top level sections driven by the notebook, one per figure or table
def run_baseline_3class(ds: dict, time_window=EARLY_WINDOW, seed=RANDOM_STATE):
    """Step 1 -- clean grouped-CV 3-class baseline (Figure 2a).

    Fits a balanced logistic regression on a single early-window
    feature vector and reports balanced accuracy plus a normalised
    confusion matrix. This is the simplest sanity check: the model
    should be above chance on song/music/speech even before any of
    the subset analyses.
    """
    X = window_features(ds["X_tensor"], ds["t"], time_window)
    res = run_grouped_cv(X, ds["y_coarse"], ds["stimulus_id"], n_splits=N_SPLITS, seed=seed)
    m = res["metrics"]
    cm_norm = np.asarray(m["confusion_matrix_normalized"])
    fig_paths = plot_confusion_matrix(
        cm_norm,
        m["class_names"],
        title=f"3-class decoder (grouped CV, window {time_window})",
        stem="fig2_baseline_confusion",
    )
    summary = pd.DataFrame(
        [
            {
                "task": "three_class",
                "window_s": f"{time_window[0]}-{time_window[1]}",
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                **{f"recall_{k}": v for k, v in m["per_class_recall"].items()},
            }
        ]
    )
    _save_table(summary, "baseline_3class_summary")
    return {"summary": summary, "metrics": m, "figure": fig_paths}


def run_random_subset_control(ds: dict, *, n_subsets=RANDOM_SUBSETS_N, seed=RANDOM_STATE):
    """Step 2 (headline) -- matched random-subset null for song-vs-music.

    The comparison that drives the paper's main claim: the song-only
    electrode subset is pitted against :data:`RANDOM_SUBSETS_N`
    random subsets of the *same size* drawn without replacement from
    the pool of non-song electrodes. A one-sided empirical p-value
    and a bootstrap CI on the (song-only - random) difference are
    reported so both the null and its effect size are visible.

    The same protocol is run for two complementary metrics:

    * ``bacc``        -- primary, a decoding score that rewards
      classifier-accessible signal.
    * ``divergence``  -- secondary, a geometry score that does not
      depend on any classifier.

    Saves Figure 4 (two panels) plus three CSV tables.
    """
    subs = define_fixed_subsets(ds["electrode_group"])
    extras = {"all": subs["all"], "no_song": subs["no_song"]}

    out_by_metric: Dict[str, dict] = {}
    figs: Dict[str, dict] = {}
    rows: List[dict] = []

    for metric, xlabel, title in [
        ("bacc", "song-vs-music balanced accuracy", "Matched random-subset control (balanced accuracy)"),
        ("divergence", "song-vs-music divergence", "Matched random-subset control (RDM divergence)"),
    ]:
        res = compare_true_vs_random_subsets(
            ds["X_tensor"],
            ds["y_coarse"],
            ds["stimulus_id"],
            ds["electrode_group"],
            ds["t"],
            metric=metric,
            task="song_vs_music",
            subset_size=SUBSET_SIZE,
            n_subsets=n_subsets,
            time_window=EARLY_WINDOW,
            extras=extras,
            seed=seed,
            bootstrap_diff_n=BOOTSTRAP_N,
        )

        figs[metric] = plot_random_subset_null(
            null_scores=res["null_scores"],
            true_score=res["true_score"],
            p_value=res["empirical_p_greater"],
            diff_ci=res["diff_ci"],
            title=title,
            xlabel=xlabel,
            stem=f"fig4_random_subset_null_{metric}",
            extra_lines={"all": res["reference_scores"]["all"], "no_song": res["reference_scores"]["no_song"]},
        )

        _save_cache(
            f"random_subset_null_{metric}",
            null_scores=res["null_scores"],
            true_score=np.array([res["true_score"]]),
            diff_samples=res["diff_bootstrap_samples"],
            subset_indices=np.array(res["subset_indices"]),
        )

        rows.append(
            {
                "metric": metric,
                "task": "song_vs_music",
                "subset_size": res["subset_size"],
                "n_random_subsets": res["n_subsets"],
                "true_song_only": res["true_score"],
                "null_mean": float(np.mean(res["null_scores"])),
                "null_std": float(np.std(res["null_scores"])),
                "null_median": float(np.median(res["null_scores"])),
                "ref_all": res["reference_scores"]["all"],
                "ref_no_song": res["reference_scores"]["no_song"],
                "p_greater_vs_random": res["empirical_p_greater"],
                "diff_ci_lo": res["diff_ci"]["lo"],
                "diff_ci_median": res["diff_ci"]["median"],
                "diff_ci_hi": res["diff_ci"]["hi"],
            }
        )
        out_by_metric[metric] = res

    summary = pd.DataFrame(rows)
    _save_table(summary, "random_subset_control_summary")
    # raw null scores saved in long form for easier downstream plotting
    long_rows = []
    for metric, res in out_by_metric.items():
        for v in res["null_scores"]:
            long_rows.append({"metric": metric, "null_score": float(v)})
    _save_table(pd.DataFrame(long_rows), "random_subset_control_null_long")

    return {"summary": summary, "by_metric": out_by_metric, "figures": figs}


def run_time_resolved_songmusic(ds: dict, *, n_boot=200, seed=RANDOM_STATE):
    """Step 3 -- time-resolved song-vs-music decoding per subset (Figure 5).

    Slides a short window across the trial and fits a fresh logistic
    regression per window. For each of the three anatomical subsets
    (``all``, ``no_song``, ``song_only``) we report balanced accuracy
    and AUC together with a per-window bootstrap CI computed on
    held-out predictions (so the CI does not leak across CV folds).
    """
    subs = define_fixed_subsets(ds["electrode_group"])
    curves = {}
    rows = []
    for name in ("all", "no_song", "song_only"):
        out = time_resolved_songmusic_with_ci(
            ds["X_tensor"],
            ds["y_coarse"],
            ds["stimulus_id"],
            ds["t"],
            subset_idx=subs[name],
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            n_splits=N_SPLITS,
            n_boot=n_boot,
            seed=seed,
        )
        curves[name] = out
        _save_cache(f"time_songmusic_{name}", **{k: np.asarray(v) for k, v in out.items() if k != "y_pred_per_window"})

        early_mask = (out["time"] >= EARLY_WINDOW[0]) & (out["time"] <= EARLY_WINDOW[1])
        full_mean = float(np.mean(out["bacc"]))
        early_mean = float(np.mean(out["bacc"][early_mask])) if early_mask.any() else np.nan
        peak_i = int(np.argmax(out["bacc"]))
        rows.append(
            {
                "subset": name,
                "peak_bacc": float(out["bacc"][peak_i]),
                "peak_time_s": float(out["time"][peak_i]),
                "early_window_mean_bacc": early_mean,
                "full_window_mean_bacc": full_mean,
                "mean_auc": float(np.nanmean(out["auc"])),
            }
        )

    summary = pd.DataFrame(rows)
    _save_table(summary, "time_resolved_songmusic_summary")

    fig_paths = plot_time_resolved_curves(
        curves={k: {"time": v["time"], "bacc": v["bacc"], "bacc_ci_lo": v["bacc_ci_lo"], "bacc_ci_hi": v["bacc_ci_hi"]} for k, v in curves.items()},
        metric_key="bacc",
        chance=0.5,
        title="Song vs music time-resolved decoder",
        ylabel="balanced accuracy",
        stem="fig5_songmusic_time_curves",
        ci_key="bacc_ci",
    )
    return {"curves": curves, "summary": summary, "figure": fig_paths}


def run_formalized_divergence(
    ds: dict, *, n_boot=BOOTSTRAP_N, n_perm=PERM_N, seed=RANDOM_STATE
):
    """Step 4 -- time-resolved divergence with bootstrap CI + permutation null.

    Complements Step 3 with a representation-space (classifier-free)
    readout. For each anatomical subset we compute:

    1. The observed divergence curve.
    2. A stratified-by-stimulus bootstrap CI on the curve.
    3. A class-label permutation null envelope plus a per-window and
       peak p-value.

    Produces one combined Figure 3 and one joint summary table.
    """
    subs = define_fixed_subsets(ds["electrode_group"])
    rows = []
    figs: Dict[str, dict] = {}
    curves_per_subset = {}

    subset_names = [name for name in ("all", "no_song", "song_only") if name in subs]

    for name in subset_names:
        sub = ds["X_tensor"][subs[name]]

        observed = time_resolved_divergence(
            sub,
            ds["y_coarse"],
            ds["t"],
            pair=("song", "music"),
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
        )
        boot = bootstrap_divergence_curve(
            sub,
            ds["y_coarse"],
            ds["t"],
            pair=("song", "music"),
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            n_boot=n_boot,
            seed=seed,
        )
        perm = permutation_test_divergence_curve(
            sub,
            ds["y_coarse"],
            ds["t"],
            pair=("song", "music"),
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            n_perm=n_perm,
            seed=seed,
        )

        curves_per_subset[name] = {"observed": observed, "boot": boot, "perm": perm}

        _save_cache(
            f"divergence_{name}",
            time=observed["time"],
            observed=observed["divergence"],
            boot_ci_lo=boot["ci_lo"],
            boot_ci_hi=boot["ci_hi"],
            perm_env95=perm["env_95"],
            perm_p_per_window=perm["p_per_window"],
        )

        early_mask = (observed["time"] >= EARLY_WINDOW[0]) & (observed["time"] <= EARLY_WINDOW[1])
        early_mean_obs = float(np.mean(observed["divergence"][early_mask])) if early_mask.any() else np.nan
        peak_i = int(np.nanargmax(observed["divergence"]))
        rows.append(
            {
                "subset": name,
                "peak_divergence": float(observed["divergence"][peak_i]),
                "peak_time_s": float(observed["time"][peak_i]),
                "early_window_mean": early_mean_obs,
                "boot_ci_lo_at_peak": float(boot["ci_lo"][peak_i]),
                "boot_ci_hi_at_peak": float(boot["ci_hi"][peak_i]),
                "p_peak_permutation": float(perm["p_peak"]),
            }
        )

    combined_curves = {}
    for name in subset_names:
        observed = curves_per_subset[name]["observed"]
        boot = curves_per_subset[name]["boot"]
        combined_curves[name] = {
            "time": observed["time"],
            "divergence": observed["divergence"],
            "divergence_ci_lo": boot["ci_lo"],
            "divergence_ci_hi": boot["ci_hi"],
        }

    figs["combined"] = plot_time_resolved_curves(
        curves=combined_curves,
        metric_key="divergence",
        chance=None,
        title="Song vs music divergence",
        ylabel="divergence (between - within)",
        stem="fig3_divergence_combined",
        ci_key="divergence_ci",
    )

    summary = pd.DataFrame(rows)
    _save_table(summary, "divergence_stats")
    return {"curves": curves_per_subset, "summary": summary, "figures": figs}


def run_acoustic_partition(
    ds: dict,
    *,
    n_perm: int = PERM_N,
    ridge: float = 1.0,
    feature_set: str = "A_full",
    seed: int = RANDOM_STATE,
):
    """Acoustic-partialled song-vs-music divergence (mechanistic add-on).

    For each anatomical subset (``all``, ``no_song``, ``song_only``) we
    compute two time-resolved RDM divergence curves:

    1. *Raw* -- the ordinary divergence curve (already produced in
       :func:`run_formalized_divergence`, reproduced here so everything
       lives in one figure/table).
    2. *Acoustic-partialled* -- at every sliding window, the
       ``(n_stim, n_elec)`` feature matrix is residualised against the
       Norman-Haignere cochleogram + spectrotemporal modulation features
       (see :func:`data_utils.load_acoustic_features`) before the RDM is
       computed. The residual curve measures song-vs-music geometry that
       cannot be explained by low-level audio statistics.

    Permutation tests on both curves share identical class-shuffles; the
    partialled test residualises *inside* every shuffle to prevent leaks.
    Saves ``acoustic_partition_divergence.csv`` and
    ``fig_acoustic_partition_{subset}.{png,pdf}``.

    Parameters
    ----------
    feature_set : {"A_full", "F_coch", "F_mod_resid"}
        Which acoustic regressors to partial out. ``A_full`` uses
        cochleogram + modulation (recommended).
    ridge : float
        Ridge penalty used in the stimulus-level regression. A small
        value (~1) stabilises the fit when some acoustic features are
        nearly colinear.
    """
    subs = define_fixed_subsets(ds["electrode_group"])
    acoustic = load_acoustic_features(stim_names_target=ds["stimulus_id"])
    if feature_set not in acoustic:
        raise ValueError(f"feature_set must be one of F_coch, F_mod_resid, A_full; got {feature_set}")
    A = acoustic[feature_set]
    print(
        f"  [acoustic partition] regressor '{feature_set}' "
        f"shape={A.shape} ridge={ridge}"
    )

    rows = []
    figs: Dict[str, dict] = {}
    curves_per_subset: Dict[str, dict] = {}

    for name in ("all", "no_song", "song_only"):
        sub = ds["X_tensor"][subs[name]]

        raw = time_resolved_divergence(
            sub, ds["y_coarse"], ds["t"], pair=("song", "music"),
            window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        )
        raw_perm = permutation_test_divergence_curve(
            sub, ds["y_coarse"], ds["t"], pair=("song", "music"),
            window_sec=WINDOW_SEC, step_sec=STEP_SEC,
            n_perm=n_perm, seed=seed,
        )
        partial = time_resolved_divergence(
            sub, ds["y_coarse"], ds["t"], A=A,
            pair=("song", "music"), window_sec=WINDOW_SEC, step_sec=STEP_SEC,
            ridge=ridge,
        )
        partial_perm = permutation_test_divergence_curve(
            sub, ds["y_coarse"], ds["t"], A=A,
            pair=("song", "music"), window_sec=WINDOW_SEC, step_sec=STEP_SEC,
            n_perm=n_perm, ridge=ridge, seed=seed,
        )

        curves_per_subset[name] = {
            "raw": raw, "raw_perm": raw_perm,
            "partial": partial, "partial_perm": partial_perm,
        }
        figs[name] = plot_divergence_partial_comparison(
            raw, partial, raw_perm, partial_perm,
            title=f"Acoustic-partialled divergence - {name}",
            stem=f"fig_acoustic_partition_{name}",
        )
        _save_cache(
            f"acoustic_partition_{name}",
            time=raw["time"],
            raw_divergence=raw["divergence"],
            partial_divergence=partial["divergence"],
            raw_env95=raw_perm["env_95"],
            partial_env95=partial_perm["env_95"],
            raw_p_per_window=raw_perm["p_per_window"],
            partial_p_per_window=partial_perm["p_per_window"],
        )

        early_mask = (raw["time"] >= EARLY_WINDOW[0]) & (raw["time"] <= EARLY_WINDOW[1])
        peak_i_raw = int(np.nanargmax(raw["divergence"]))
        peak_i_par = int(np.nanargmax(partial["divergence"]))
        rows.append(
            {
                "subset": name,
                "peak_divergence_raw": float(raw["divergence"][peak_i_raw]),
                "peak_divergence_partial": float(partial["divergence"][peak_i_par]),
                "peak_time_raw_s": float(raw["time"][peak_i_raw]),
                "peak_time_partial_s": float(partial["time"][peak_i_par]),
                "early_mean_raw": float(np.mean(raw["divergence"][early_mask]))
                if early_mask.any() else float("nan"),
                "early_mean_partial": float(np.mean(partial["divergence"][early_mask]))
                if early_mask.any() else float("nan"),
                "fraction_surviving_at_peak": (
                    float(partial["divergence"][peak_i_raw] / raw["divergence"][peak_i_raw])
                    if raw["divergence"][peak_i_raw] > 0 else float("nan")
                ),
                "p_peak_raw": float(raw_perm["p_peak"]),
                "p_peak_partial": float(partial_perm["p_peak"]),
                "n_electrodes": int(sub.shape[0]),
            }
        )

    summary = pd.DataFrame(rows)
    _save_table(summary, "acoustic_partition_divergence")
    return {
        "summary": summary,
        "curves": curves_per_subset,
        "figures": figs,
        "feature_set": feature_set,
        "ridge": ridge,
    }


def run_cross_temporal(ds: dict, *, seed=RANDOM_STATE):
    """Step 5 -- cross-temporal generalisation heatmaps (Figure 6).

    Trains a decoder at window ``t`` and tests at every window ``t'``
    using the same CV splits, yielding a ``(n_win, n_win)`` balanced
    accuracy matrix per subset. Strong off-diagonal values indicate a
    stable representation; diagonal-only structure indicates a
    rapidly-drifting code.
    """
    subs = define_fixed_subsets(ds["electrode_group"])
    results = {}
    for name in ("all", "no_song", "song_only"):
        r = cross_temporal_generalization(
            ds["X_tensor"],
            ds["y_coarse"],
            ds["stimulus_id"],
            ds["t"],
            subset_idx=subs[name],
            class_a="song",
            class_b="music",
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            n_splits=N_SPLITS,
            seed=seed,
        )
        results[name] = r
        _save_cache(f"cross_temporal_{name}", times=r["times"], matrix=r["matrix"])

    # shared color scale between heatmaps using the 5th and 95th percentiles
    vmin = float(min(np.nanpercentile(r["matrix"], 5) for r in results.values()))
    vmax = float(max(np.nanpercentile(r["matrix"], 95) for r in results.values()))
    fig_paths = plot_cross_temporal_heatmaps(
        {name: r["matrix"] for name, r in results.items()},
        times=next(iter(results.values()))["times"],
        vmin=vmin,
        vmax=vmax,
        stem="fig6_cross_temporal",
    )

    # scalar summaries, diagonal mean and off diagonal mean where |i j| >= 5
    rows = []
    for name, r in results.items():
        M = r["matrix"]
        n = M.shape[0]
        diag = np.diag(M)
        off_mask = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= 5
        rows.append(
            {
                "subset": name,
                "diag_mean_bacc": float(np.mean(diag)),
                "offdiag_mean_bacc": float(np.mean(M[off_mask])),
                "max_bacc": float(np.max(M)),
            }
        )
    summary = pd.DataFrame(rows)
    _save_table(summary, "cross_temporal_summary")

    return {"results": results, "summary": summary, "figure": fig_paths}


def run_loo_clean(ds: dict, *, n_perm=1000, seed=RANDOM_STATE):
    """Step 6 -- leave-one-electrode-out contributions (Figure 7).

    For every electrode we drop it, refit, and record
    ``contribution = baseline - dropped`` for both the time-resolved
    mean balanced accuracy and the early-window divergence. Positive
    contribution means the electrode was informative. We use the
    time-resolved mean for bacc (instead of the single-window score)
    because that score saturates near 1.0 with the full 33-electrode
    set and hides single-electrode effects.

    The permutation test shuffles electrode-group labels ``n_perm``
    times and compares mean song vs. mean non-song contributions.
    This tests the *specificity* claim directly: if song electrodes
    really matter more than other electrodes for song-vs-music, the
    observed difference should exceed the shuffled null.
    """
    eg = np.asarray(ds["electrode_group"])
    n_e = eg.size
    X_all = ds["X_tensor"]

    def _mean_bacc(subset_idx):
        r = run_time_resolved_binary_decoder(
            X_all,
            ds["y_coarse"],
            ds["stimulus_id"],
            ds["t"],
            subset_idx=subset_idx,
            window_sec=WINDOW_SEC,
            step_sec=STEP_SEC,
            n_splits=N_SPLITS,
            seed=seed,
            return_proba=False,
        )
        return float(np.mean(r["bacc"]))

    baseline_bacc = _mean_bacc(np.arange(n_e))
    baseline_div = evaluate_subset(
        np.arange(n_e),
        X_all,
        ds["y_coarse"],
        ds["stimulus_id"],
        ds["t"],
        metric="divergence",
        time_window=EARLY_WINDOW,
        seed=seed,
    )

    rows = []
    for i in range(n_e):
        keep = np.delete(np.arange(n_e), i)
        b = _mean_bacc(keep)
        d = evaluate_subset(
            keep,
            X_all,
            ds["y_coarse"],
            ds["stimulus_id"],
            ds["t"],
            metric="divergence",
            time_window=EARLY_WINDOW,
            seed=seed,
        )
        rows.append(
            {
                "electrode_index": int(i),
                "electrode_id": ds["electrode_id"][i],
                "electrode_group": eg[i],
                "delta_bacc": float(baseline_bacc - b),
                "delta_divergence": float(baseline_div - d),
            }
        )
    df = pd.DataFrame(rows)

    # permutation test comparing mean delta_bacc for song electrodes versus others
    rng = np.random.default_rng(seed)
    song_mask = (eg == "song")
    observed_song = df.loc[song_mask, "delta_bacc"].mean()
    observed_other = df.loc[~song_mask, "delta_bacc"].mean()
    observed_stat = observed_song - observed_other

    null_stats = np.empty(n_perm)
    for r in range(n_perm):
        shuffled = eg.copy()
        rng.shuffle(shuffled)
        sm = (shuffled == "song")
        null_stats[r] = df.loc[sm, "delta_bacc"].mean() - df.loc[~sm, "delta_bacc"].mean()
    p_value = empirical_p_value(observed_stat, null_stats, alternative="greater")

    _save_cache("loo_contributions", **{col: df[col].to_numpy() for col in df.columns if df[col].dtype != object})
    _save_cache("loo_perm_null", null_stats=null_stats, observed_stat=np.array([observed_stat]))
    _save_table(df, "loo_contributions")

    group_summary = (
        df.groupby("electrode_group", as_index=False)
        .agg(
            mean_delta_bacc=("delta_bacc", "mean"),
            std_delta_bacc=("delta_bacc", "std"),
            mean_delta_divergence=("delta_divergence", "mean"),
            n=("delta_bacc", "size"),
        )
    )
    group_summary["perm_p_song_vs_rest_bacc"] = p_value
    _save_table(group_summary, "loo_group_summary")

    p_text = (
        f"mean song delta = {observed_song:.3f}\n"
        f"mean non-song delta = {observed_other:.3f}\n"
        f"permutation p (song > rest) = {p_value:.4f}"
    )
    fig_paths = plot_loo_contributions(
        df,
        stem="fig7_loo_contributions_bacc",
        metric_col="delta_bacc",
        ylabel="drop in balanced accuracy when electrode removed",
        title="Leave-one-electrode-out (song vs music, bacc)",
        p_value_text=p_text,
    )
    _ = plot_loo_contributions(
        df,
        stem="fig7_loo_contributions_divergence",
        metric_col="delta_divergence",
        ylabel="drop in divergence when electrode removed",
        title="Leave-one-electrode-out (song vs music, divergence)",
    )
    return {
        "df": df,
        "group_summary": group_summary,
        "observed_stat": observed_stat,
        "null_stats": null_stats,
        "p_value": p_value,
        "figure": fig_paths,
    }


def run_nonlinear_supplement(ds: dict, *, seed=RANDOM_STATE):
    """Supplementary nonlinear comparison (negative result).

    Runs linear logreg, RBF-SVM, and autoencoder-latent+logreg on two tasks:

    1. ``three_class`` - song/speech/music on an early-window feature.
    2. ``song_vs_music`` - binary subset of the same features.

    Saves ``fig_nonlinear_comparison.{png,pdf}`` and
    ``nonlinear_comparison.csv``. This is kept as a supplement: it does
    not drive any main claim, but documents that richer model families
    do not improve over a clean linear baseline.
    """
    X = window_features(ds["X_tensor"], ds["t"], EARLY_WINDOW)
    y = np.asarray(ds["y_coarse"])
    sid = np.asarray(ds["stimulus_id"])

    keep_sm = np.isin(y, ["song", "music"])
    tasks = {
        "three_class_all_electrodes": (X, y, sid),
        "song_vs_music_all_electrodes": (X[keep_sm], y[keep_sm], sid[keep_sm]),
    }

    df = run_nonlinear_comparison(tasks, n_splits=N_SPLITS, seed=seed)
    _save_table(df, "nonlinear_comparison")
    fig = plot_nonlinear_comparison(df, stem="fig_nonlinear_comparison")

    # delta relative to the linear baseline, positive means the nonlinear model helped
    rows = []
    for task, sub in df.groupby("task"):
        lin = float(sub[sub["model"] == "linear_logreg"]["bacc"].iloc[0])
        for _, row in sub.iterrows():
            if row["model"] == "linear_logreg":
                continue
            rows.append({
                "task": task,
                "model": row["model"],
                "delta_bacc_vs_linear": float(row["bacc"]) - lin,
            })
    delta = pd.DataFrame(rows)
    _save_table(delta, "nonlinear_delta_vs_linear")

    return {"summary": df, "delta_vs_linear": delta, "figure": fig}


# bellier 2023 extension sections
def run_bellier_decoder(*, seed=RANDOM_STATE):
    """Bellier vocal-vs-instrumental decoder on the 29-patient supergrid.

    Runs logistic regression on ``{all, right_STG, left_STG, non_STG}``
    under blocked-time 5-fold CV. A TinyTemporalCNN is run only on
    subsets where logreg clears chance by ``CNN_MIN_BACC_OVER_CHANCE``.
    Saves ``bellier_decoder_summary.csv`` and ``fig8_bellier_decoder_subsets``.
    """
    sg = build_supergrid(cache=True, verbose=True)
    subs = electrode_subsets(sg)
    vocal_mask = load_vocal_segments()  # raises filenotfounderror if the csv is missing

    res = run_vocal_instrumental_decoder(sg, subs, vocal_mask, seed=seed)
    _save_table(res["summary"], "bellier_decoder_summary")
    fig = plot_bellier_decoder_subset_bars(res["summary"])

    # save raw per fold scores for the record
    fold_rows = []
    for subset, by_model in res["fold_scores"].items():
        for model, folds in by_model.items():
            for row in folds:
                fold_rows.append({"subset": subset, "model": model, **row})
    _save_table(pd.DataFrame(fold_rows), "bellier_decoder_fold_scores")

    return {
        "summary": res["summary"],
        "fold_scores": fold_rows,
        "cnn_ran": res["cnn_ran"],
        "figure": fig,
        "supergrid_n_elec": int(sg["hfa"].shape[1]),
        "n_windows": int(res["raw"][list(res["raw"].keys())[0]]["y_true"].size)
        if res["raw"] else 0,
    }


def run_bellier_profiles(*, ds_norman: Optional[dict] = None, seed=RANDOM_STATE):
    """Event-locked temporal profiles for Bellier and a cross-dataset overlay.

    Bellier profiles are computed for ``right_STG`` and ``left_STG`` at
    vocal and instrumental onsets. Norman trial-locked profiles are
    computed for the song / music / speech electrode groups crossed with
    song / music / speech stimulus classes. A qualitative overlay (Bellier
    STG vocal vs Norman song-electrode song-trial) is saved as fig10.
    """
    sg = build_supergrid(cache=True, verbose=False)
    subs = electrode_subsets(sg)
    vocal_mask = load_vocal_segments()

    b = bellier_profiles(sg, subs, vocal_mask)
    if ds_norman is None:
        ds_norman = build_dataset()
    n = norman_profiles(ds_norman)

    feats = pd.concat([b["features"], n["features"]], ignore_index=True)
    _save_table(feats, "temporal_profile_features")

    fig9 = plot_bellier_event_profiles(b["profiles"])
    fig10 = plot_temporal_profile_overlay(b["profiles"], n["profiles"])

    # cache raw profile arrays so the notebook can reload without recomputing
    cache_entries = {}
    for src, profs in (("bellier", b["profiles"]), ("norman", n["profiles"])):
        for key, p in profs.items():
            cache_entries[f"{src}__{key}__time"] = np.asarray(p["time_s"])
            cache_entries[f"{src}__{key}__mean"] = np.asarray(p["mean"])
            cache_entries[f"{src}__{key}__sem"] = np.asarray(p["sem"])
    _save_cache("temporal_profiles", **cache_entries)

    return {
        "bellier_features": b["features"],
        "norman_features": n["features"],
        "combined_features": feats,
        "bellier_profiles": b["profiles"],
        "norman_profiles": n["profiles"],
        "figures": {"event_profiles": fig9, "overlay": fig10},
    }


def bellier_build_component_tensor(supergrid,
                                   vocal_present,
                                   window_size,
                                   step_size=None,
                                   label_threshold=0.5,
                                   min_valid_frac=0.8):
    """Build Bellier tensor D for component modeling.

    Args:
        supergrid: dict with at least:
            'hfa'        -> [time x electrode]
            'artifacts'  -> [time x electrode] boolean
        vocal_present: [time] binary array
        window_size: number of samples per window
        step_size: hop between windows, defaults to window_size
        label_threshold: fraction of vocal samples needed to label a window vocal
        min_valid_frac: minimum artifact-free fraction required per electrode/window

    Returns:
        Z dict with:
            D               [window x time x electrode]
            y               [window] binary labels (1=vocal, 0=instrumental)
            keep_windows    [window] original start indices
            valid_mask      [window x electrode] whether electrode/window is valid
    """

    if step_size is None:
        step_size = window_size

    X = np.asarray(supergrid['hfa'], dtype=np.float32)
    artifacts = np.asarray(supergrid['artifacts'], dtype=bool)
    vocal_present = np.asarray(vocal_present).astype(np.float32)

    n_time, n_elec = X.shape
    starts = np.arange(0, n_time - window_size + 1, step_size)

    D = []
    y = []
    valid_mask = []
    keep_windows = []

    for s in starts:
        e = s + window_size

        x_win = X[s:e, :].copy()
        a_win = artifacts[s:e, :]

        # fraction of valid samples for each electrode in this window
        valid_frac = 1.0 - np.mean(a_win, axis=0)
        vmask = valid_frac >= min_valid_frac

        # replace artifact samples with 0; model is simple and expects dense tensor
        x_win[a_win] = 0.0

        # zero out bad electrode/windows entirely
        x_win[:, ~vmask] = 0.0

        frac_vocal = np.mean(vocal_present[s:e])
        y_win = np.int32(frac_vocal >= label_threshold)

        D.append(x_win[np.newaxis, :, :])
        y.append(y_win)
        valid_mask.append(vmask[np.newaxis, :])
        keep_windows.append(s)

    D = np.concatenate(D, axis=0)                  # [window x time x electrode]
    y = np.asarray(y, dtype=np.int32)              # [window]
    valid_mask = np.concatenate(valid_mask, axis=0)  # [window x electrode]
    keep_windows = np.asarray(keep_windows, dtype=np.int32)

    Z = {
        'D': D,
        'y': y,
        'valid_mask': valid_mask,
        'keep_windows': keep_windows,
        'window_size': window_size,
        'step_size': step_size,
    }

    return Z


def bellier_fit_vocal_components(D,
                                 K,
                                 activation_penalty,
                                 activation_scale=0.001,
                                 n_iter=10000,
                                 n_iter_per_eval=20,
                                 seed=0,
                                 kernel_size=None,
                                 step_size=[0.01, 0.0032, 0.001, 0.0003, 0.0001]):
    """Fit the simple Norman-Haignere component model to Bellier windows.

    Args:
        D: [window x time x electrode]
        K: number of components
        activation_penalty: L1 penalty on activations
        other args are passed directly to train_simple

    Returns:
        output dict from train_simple
    """
    try:
        from HaignereModel import train_simple
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Bellier component model requires TensorFlow. "
            "Use the TensorFlow-enabled kernel/environment for this function."
        ) from exc
    Z = train_simple(
        D=D,
        K=K,
        activation_penalty=activation_penalty,
        activation_scale=activation_scale,
        n_iter=n_iter,
        n_iter_per_eval=n_iter_per_eval,
        seed=seed,
        kernel_size=kernel_size,
        step_size=step_size,
    )

    return Z


def bellier_component_vocal_selectivity(model_fit,
                                        y,
                                        n_perm=1000,
                                        seed=0):
    """Measure vocal preference of fitted components.

    Args:
        model_fit: output of train_simple or train_complex
        y: [window] binary labels (1=vocal, 0=instrumental)

    Returns:
        dict with component selectivity statistics
    """

    rng = np.random.RandomState(seed)

    R = model_fit['R']   # [window x time x component]
    K = R.shape[2]

    comp_mean = np.mean(R, axis=1)   # [window x component]

    mean_vocal = np.zeros(K)
    mean_instr = np.zeros(K)
    diff = np.zeros(K)
    p = np.zeros(K)

    vocal_ix = (y == 1)
    instr_ix = (y == 0)

    for k in range(K):
        x = comp_mean[:, k]

        mean_vocal[k] = np.mean(x[vocal_ix])
        mean_instr[k] = np.mean(x[instr_ix])
        diff[k] = mean_vocal[k] - mean_instr[k]

        null = np.zeros(n_perm)
        for i in range(n_perm):
            yperm = rng.permutation(y)
            null[i] = np.mean(x[yperm == 1]) - np.mean(x[yperm == 0])

        p[k] = (1 + np.sum(null >= diff[k])) / (n_perm + 1)

    order = np.argsort(-diff)

    Z = {
        'mean_vocal': mean_vocal,
        'mean_instr': mean_instr,
        'diff': diff,
        'p': p,
        'order': order,
        'best_component': order[0],
        'component_mean_timecourse': np.mean(R, axis=0),  # [time x component]
    }

    return Z


def bellier_electrode_weights_for_component(model_fit,
                                            component_index,
                                            supergrid=None):
    """Return electrode weights for one selected component.

    Args:
        model_fit: output of train_simple/train_complex
        component_index: integer
        supergrid: optional Bellier metadata dict

    Returns:
        dict with electrode weights and optional metadata
    """

    W = model_fit['W']   # [component x electrode]
    w = W[component_index, :]

    order = np.argsort(-w)

    Z = {
        'weights': w,
        'order': order,
        'component_index': component_index,
    }

    if supergrid is not None:
        for key in ['patient_id', 'channel_label', 'group', 'hemi', 'anatomy_raw', 'mni']:
            if key in supergrid:
                Z[key] = supergrid[key]

    return Z


def bellier_top_vocal_electrodes(model_fit,
                                 component_stats,
                                 supergrid=None,
                                 top_n=25):
    """Get the top electrodes for the most vocal-preferring component."""

    k = component_stats['best_component']
    ew = bellier_electrode_weights_for_component(model_fit, k, supergrid=supergrid)

    order = ew['order'][:top_n]

    Z = {
        'component_index': k,
        'component_diff': component_stats['diff'][k],
        'component_p': component_stats['p'][k],
        'electrode_index': order,
        'weights': ew['weights'][order],
    }

    if supergrid is not None:
        for key in ['patient_id', 'channel_label', 'group', 'hemi', 'anatomy_raw']:
            if key in ew:
                Z[key] = np.asarray(ew[key])[order]

    return Z


def run_bellier_vocal_component_model(supergrid,
                                      vocal_present,
                                      K=10,
                                      activation_penalty=0.01,
                                      window_size=50,
                                      step_size=50,
                                      label_threshold=0.5,
                                      min_valid_frac=0.8,
                                      n_perm=1000,
                                      seed=0,
                                      activation_scale=0.001,
                                      n_iter=10000,
                                      n_iter_per_eval=20,
                                      kernel_size=None,
                                      train_step_size=[0.01, 0.0032, 0.001, 0.0003, 0.0001]):
    """Full Bellier vocal-component pipeline, in the style of the pasted repo."""

    
    data = bellier_build_component_tensor(
        supergrid=supergrid,
        vocal_present=vocal_present,
        window_size=window_size,
        step_size=step_size,
        label_threshold=label_threshold,
        min_valid_frac=min_valid_frac,
    )

    model_fit = bellier_fit_vocal_components(
        D=data['D'],
        K=K,
        activation_penalty=activation_penalty,
        activation_scale=activation_scale,
        n_iter=n_iter,
        n_iter_per_eval=n_iter_per_eval,
        seed=seed,
        kernel_size=kernel_size,
        step_size=train_step_size,
    )

    component_stats = bellier_component_vocal_selectivity(
        model_fit=model_fit,
        y=data['y'],
        n_perm=n_perm,
        seed=seed,
    )

    top_electrodes = bellier_top_vocal_electrodes(
        model_fit=model_fit,
        component_stats=component_stats,
        supergrid=supergrid,
        top_n=25,
    )

    Z = {
        'data': data,
        'model_fit': model_fit,
        'component_stats': component_stats,
        'top_electrodes': top_electrodes,
    }

    return Z    


# top level orchestration that runs every section end to end
def run_all(*, n_subsets=RANDOM_SUBSETS_N, n_boot=BOOTSTRAP_N, n_perm=PERM_N, seed=RANDOM_STATE, skip: Optional[list] = None, include_bellier: bool = True):
    """Run every analysis end-to-end and write ``results/metrics.json``.

    Parameters
    ----------
    n_subsets, n_boot, n_perm : int
        Resampling budgets for the random-subset null, bootstrap CIs,
        and label-permutation tests respectively. Defaults come from
        :mod:`config`.
    seed : int
        Global seed for every module that uses a numpy/sklearn RNG.
    skip : list[str] | None
        Names of sections to skip (e.g. ``["cross_temporal"]``). All
        other sections still run.
    include_bellier : bool, default True
        If ``False``, the Bellier 2023 extension is skipped even when
        not listed in ``skip`` (useful when the Bellier data is not
        on disk).

    Returns
    -------
    dict
        The same nested dictionary that is written to
        ``results/metrics.json``.
    """
    ensure_dirs()
    skip = set(skip or [])
    ds = build_dataset()

    metrics: Dict[str, object] = {"meta": ds["meta"], "seed": seed}

    if "baseline" not in skip:
        b = run_baseline_3class(ds, seed=seed)
        metrics["baseline_3class"] = b["metrics"]
    if "random_subset" not in skip:
        r = run_random_subset_control(ds, n_subsets=n_subsets, seed=seed)
        metrics["random_subset_control"] = r["summary"].to_dict(orient="records")
    if "time_resolved" not in skip:
        t = run_time_resolved_songmusic(ds, n_boot=200, seed=seed)
        metrics["time_resolved_songmusic"] = t["summary"].to_dict(orient="records")
    if "divergence" not in skip:
        d = run_formalized_divergence(ds, n_boot=n_boot, n_perm=n_perm, seed=seed)
        metrics["divergence_stats"] = d["summary"].to_dict(orient="records")
    if "acoustic_partition" not in skip:
        try:
            ap = run_acoustic_partition(ds, n_perm=n_perm, seed=seed)
            metrics["acoustic_partition_summary"] = ap["summary"].to_dict(orient="records")
        except FileNotFoundError as exc:
            metrics["acoustic_partition_error"] = str(exc)
            print(f"[acoustic partition] skipped: {exc}")
    if "cross_temporal" not in skip:
        c = run_cross_temporal(ds, seed=seed)
        metrics["cross_temporal_summary"] = c["summary"].to_dict(orient="records")
    if "loo" not in skip:
        l = run_loo_clean(ds, n_perm=n_perm, seed=seed)
        metrics["loo_group_summary"] = l["group_summary"].to_dict(orient="records")
        metrics["loo_permutation_p"] = float(l["p_value"])
    if "nonlinear" not in skip:
        nl = run_nonlinear_supplement(ds, seed=seed)
        metrics["nonlinear_supplement"] = nl["summary"].to_dict(orient="records")
        metrics["nonlinear_delta_vs_linear"] = nl["delta_vs_linear"].to_dict(orient="records")

    if include_bellier and "bellier" not in skip:
        bellier_section: Dict[str, object] = {}
        try:
            bd = run_bellier_decoder(seed=seed)
            bellier_section["decoder_summary"] = bd["summary"].to_dict(orient="records")
            bellier_section["decoder_fold_scores"] = bd["fold_scores"]
            bellier_section["cnn_ran"] = bd["cnn_ran"]
            bellier_section["supergrid_n_elec"] = bd["supergrid_n_elec"]
            bp = run_bellier_profiles(ds_norman=ds, seed=seed)
            bellier_section["profile_features"] = bp["combined_features"].to_dict(orient="records")
        except FileNotFoundError as exc:
            bellier_section["error"] = f"Bellier extension skipped: {exc}"
            print(f"[bellier] skipped: {exc}")
        metrics["bellier"] = bellier_section

    path = RESULTS_DIR / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    return metrics


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Cannot serialize {type(o)}")


if __name__ == "__main__":  # pragma: no cover
    p = argparse.ArgumentParser()
    p.add_argument("--n-subsets", type=int, default=RANDOM_SUBSETS_N)
    p.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    p.add_argument("--n-perm", type=int, default=PERM_N)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--no-bellier", action="store_true", help="skip Bellier extension")
    args = p.parse_args()
    run_all(
        n_subsets=args.n_subsets,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        seed=args.seed,
        skip=args.skip,
        include_bellier=not args.no_bellier,
    )

def run_bellier_matched_random_subset_control(
    bellier_component_out: dict,
    *,
    n_subsets: int = 1000,
    subset_size: int = 7,
    seed: int = RANDOM_STATE,
    save: bool = True,
):
    """Bellier matched random-subset control for vocal-vs-instrumental decoding.

    Compares the top component-loading electrodes against a null
    distribution of matched random subsets of the same size.

    Parameters
    ----------
    bellier_component_out : dict
        Output of run_bellier_vocal_component_model(...).
    n_subsets : int
        Number of random matched subsets.
    subset_size : int
        Number of electrodes in the true subset and each random subset.
    seed : int
        RNG seed.
    save : bool
        Whether to save summary/null tables.

    Returns
    -------
    dict
        {
            "summary": pd.DataFrame,
            "null_scores": np.ndarray,
            "true_score": float,
            "empirical_p_greater": float,
            "subset_indices": list[np.ndarray],
            "true_subset": np.ndarray,
        }
    """
    sg = build_supergrid(cache=True, verbose=False)
    vocal_mask = load_vocal_segments()
    rng = np.random.default_rng(seed)

    true_subset = np.asarray(
        bellier_component_out["top_electrodes"]["electrode_index"][:subset_size],
        dtype=int,
    )

    all_idx = np.arange(sg["hfa"].shape[1], dtype=int)
    pool_idx = np.setdiff1d(all_idx, true_subset)

    # --- evaluate the true subset once ---
    true_res = run_vocal_instrumental_decoder(
        sg,
        {"true_subset": true_subset},
        vocal_mask,
        subsets=["true_subset"],
        seed=seed,
    )
    true_score = float(
        true_res["summary"]
        .query("model == 'logreg'")["mean_bacc"]
        .iloc[0]
    )

    # --- build null distribution over matched random subsets ---
    null_scores = np.zeros(n_subsets, dtype=float)
    subset_indices = []

    for i in range(n_subsets):
        rand_idx = np.asarray(
            rng.choice(pool_idx, size=subset_size, replace=False),
            dtype=int,
        )
        subset_indices.append(rand_idx)

        res = run_vocal_instrumental_decoder(
            sg,
            {"rand_subset": rand_idx},
            vocal_mask,
            subsets=["rand_subset"],
            seed=seed,
        )
        null_scores[i] = float(
            res["summary"]
            .query("model == 'logreg'")["mean_bacc"]
            .iloc[0]
        )

    empirical_p_greater = (1 + np.sum(null_scores >= true_score)) / (n_subsets + 1)

    summary = pd.DataFrame(
        [{
            "task": "bellier_vocal_vs_instrumental",
            "subset_size": subset_size,
            "n_random_subsets": n_subsets,
            "true_score": true_score,
            "null_mean": float(np.mean(null_scores)),
            "null_std": float(np.std(null_scores)),
            "null_median": float(np.median(null_scores)),
            "empirical_p_greater": float(empirical_p_greater),
        }]
    )

    if save:
        _save_table(summary, "bellier_matched_random_subset_summary")
        _save_cache(
            "bellier_matched_random_subset_null",
            null_scores=null_scores,
            true_score=np.array([true_score]),
            true_subset=true_subset,
            subset_indices=np.array(subset_indices, dtype=int),
        )

    return {
        "summary": summary,
        "null_scores": null_scores,
        "true_score": true_score,
        "empirical_p_greater": float(empirical_p_greater),
        "subset_indices": subset_indices,
        "true_subset": true_subset,
    }