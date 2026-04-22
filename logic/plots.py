"""Plotting helpers for the song-vs-music project.

Every function in this module takes pre-computed arrays and renders one
figure; there is no hidden state and no inline computation beyond
trivial shaping. Saving to disk is centralised in :func:`_save`, which
writes both ``.png`` (220 DPI) and ``.pdf`` to
:data:`config.FIG_DIR` and closes the figure to keep memory bounded.

Color decisions (``SUBSET_COLORS``, ``EVENT_COLORS``) are declared up
front so every figure uses the same palette and nothing depends on a
matplotlib default that could drift between versions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from config import FIG_DIR


# consistent subset colors across figures
SUBSET_COLORS = {
    "all": "#555555",
    "no_song": "#2C7FB8",
    "song_only": "#D62728",
}


def _save(fig: plt.Figure, stem: str, dpi: int = 220) -> Dict[str, Path]:
    """Save ``fig`` as PNG and PDF under :data:`config.FIG_DIR`.

    Writes ``{stem}.png`` at ``dpi`` and ``{stem}.pdf`` (vector) and
    returns a mapping ``{"png": path, "pdf": path}``. The caller is
    responsible for closing the figure.
    """
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return {"png": png, "pdf": pdf}


def plot_random_subset_null(
    null_scores: np.ndarray,
    true_score: float,
    p_value: float,
    diff_ci: Dict[str, float],
    title: str,
    xlabel: str,
    stem: str,
    extra_lines: Optional[Dict[str, float]] = None,
) -> Dict[str, Path]:
    """Random-subset null histogram with the true-subset score overlaid.

    Parameters
    ----------
    null_scores : np.ndarray
        Shape ``(n_random_subsets,)``. One metric value per random
        matched subset draw.
    true_score : float
        Metric value for the reference (e.g. song-only) subset. Plotted
        as a thick coloured vertical line.
    p_value : float
        Empirical one-sided p-value ``P(null >= true)``; rendered as
        annotation text.
    diff_ci : dict
        Bootstrap CI for ``true - null``. Keys ``lo``, ``hi``
        (and optionally ``median``).
    title, xlabel, stem : str
        Plot title, x-axis label, and filename stem for :func:`_save`.
    extra_lines : dict[str, float], optional
        Additional reference scores to mark (e.g. ``{"all": 0.82}``);
        colored via :data:`SUBSET_COLORS` when the key matches.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
    ax.hist(null_scores, bins=40, alpha=0.7, color="#AEB6BF", edgecolor="white")
    ax.axvline(
        true_score,
        color=SUBSET_COLORS["song_only"],
        linewidth=2.5,
        label=f"song-only = {true_score:.3f}",
    )
    null_mean = float(np.mean(null_scores))
    ax.axvline(null_mean, color="black", linewidth=1.2, linestyle="--", label=f"null mean = {null_mean:.3f}")
    if extra_lines:
        for name, val in extra_lines.items():
            color = SUBSET_COLORS.get(name, "#333333")
            ax.axvline(val, color=color, linewidth=1.5, linestyle=":", label=f"{name} = {val:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count (random subsets)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.text(
        0.02,
        0.98,
        (
            f"empirical p = {p_value:.4f}\n"
            f"true - null mean 95% CI = [{diff_ci['lo']:.3f}, {diff_ci['hi']:.3f}]"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
    )
    ax.grid(alpha=0.2)
    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def plot_time_resolved_curves(
    curves: Dict[str, Dict[str, np.ndarray]],
    metric_key: str,
    chance: Optional[float],
    title: str,
    ylabel: str,
    stem: str,
    ci_key: Optional[str] = None,
) -> Dict[str, Path]:
    """Overlay one time-resolved curve per subset on the same axes.

    Parameters
    ----------
    curves : dict[str, dict[str, np.ndarray]]
        Mapping ``subset -> {"time": (n_win,), metric_key: (n_win,),
        "{ci_key}_lo": (n_win,), "{ci_key}_hi": (n_win,)}``.
        ``ci_key``-suffixed arrays are optional.
    metric_key : str
        Key into each ``curves[subset]`` dict whose value is plotted as
        the central line.
    chance : float, optional
        Draw a dashed horizontal chance line at this value. ``None``
        disables the line (e.g. for divergence).
    title, ylabel, stem : str
        Plot title, y-axis label, and filename stem.
    ci_key : str, optional
        If given, ``curves[subset][f"{ci_key}_lo"]`` and ``_hi`` are
        filled as a semi-transparent CI band.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for name, curve in curves.items():
        color = SUBSET_COLORS.get(name, None)
        ax.plot(curve["time"], curve[metric_key], label=name, color=color, linewidth=2)
        lo = curve.get(f"{ci_key}_lo") if ci_key else None
        hi = curve.get(f"{ci_key}_hi") if ci_key else None
        if lo is not None and hi is not None:
            ax.fill_between(curve["time"], lo, hi, color=color, alpha=0.18, linewidth=0)
    if chance is not None:
        ax.axhline(chance, color="black", linestyle="--", linewidth=1, label=f"chance ({chance:.2f})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=False, fontsize=9)
    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def plot_divergence_with_stats(
    observed: Dict[str, np.ndarray],
    boot_ci: Dict[str, np.ndarray],
    perm_envelope: Dict[str, np.ndarray],
    title: str,
    stem: str,
) -> Dict[str, Path]:
    """Divergence curve with bootstrap CI band and permutation envelope.

    Parameters
    ----------
    observed : dict
        Output of :func:`rdm.time_resolved_divergence`; must contain
        ``time`` and ``divergence`` arrays of shape ``(n_win,)``.
    boot_ci : dict
        Output of :func:`rdm.bootstrap_divergence_curve`; expects
        ``ci_lo``, ``ci_hi`` arrays of shape ``(n_win,)``.
    perm_envelope : dict
        Output of :func:`rdm.permutation_test_divergence_curve`;
        expects ``env_95`` (``(n_win,)``) for the null 95th-percentile
        envelope.
    title, stem : str
        Plot title and filename stem.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    t = observed["time"]
    ax.plot(t, observed["divergence"], color=SUBSET_COLORS["song_only"], linewidth=2, label="observed")
    ax.fill_between(
        t,
        boot_ci["ci_lo"],
        boot_ci["ci_hi"],
        color=SUBSET_COLORS["song_only"],
        alpha=0.2,
        linewidth=0,
        label="bootstrap 95% CI",
    )
    ax.plot(
        t,
        perm_envelope["env_95"],
        color="black",
        linewidth=1,
        linestyle=":",
        label="null 95th pct (shuffled labels)",
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("divergence (between - within)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=False, fontsize=9)
    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def plot_divergence_partial_comparison(
    raw: Dict[str, np.ndarray],
    partial: Dict[str, np.ndarray],
    raw_perm: Dict[str, np.ndarray],
    partial_perm: Dict[str, np.ndarray],
    title: str,
    stem: str,
) -> Dict[str, Path]:
    """Raw vs. acoustic-partialled divergence on the same axes.

    Both curves share the same time axis. For each curve we draw the
    observed divergence and the label-shuffle 95th-percentile envelope;
    the partialled curve isolates song-vs-music geometry that survives
    after cochleogram + spectrotemporal-modulation features have been
    regressed out stimulus-by-stimulus.

    Parameters
    ----------
    raw, partial : dict
        Output of :func:`rdm.time_resolved_divergence` with and without
        acoustic partialling (``A=None`` vs ``A=<regressors>``). Both
        must contain ``time`` and ``divergence`` of shape ``(n_win,)``.
    raw_perm, partial_perm : dict
        Matching outputs of the permutation tests (``env_95`` arrays
        of shape ``(n_win,)``).
    title, stem : str
        Plot title and filename stem.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    t = raw["time"]
    ax.plot(t, raw["divergence"], color=SUBSET_COLORS["song_only"],
            linewidth=2.2, label="raw divergence")
    ax.plot(t, raw_perm["env_95"], color=SUBSET_COLORS["song_only"],
            linewidth=1, linestyle=":", alpha=0.7, label="raw null 95th")
    ax.plot(t, partial["divergence"], color="#1B7837",
            linewidth=2.2, label="acoustic-partialled")
    ax.plot(t, partial_perm["env_95"], color="#1B7837",
            linewidth=1, linestyle=":", alpha=0.7, label="partialled null 95th")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("song vs music divergence")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=False, fontsize=9)
    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def plot_cross_temporal_heatmaps(
    matrices: Dict[str, np.ndarray],
    times: np.ndarray,
    vmin: float,
    vmax: float,
    stem: str,
) -> Dict[str, Path]:
    """Side-by-side cross-temporal generalisation heatmaps.

    Parameters
    ----------
    matrices : dict[str, np.ndarray]
        Mapping ``subset_name -> (n_win, n_win)`` balanced-accuracy
        matrix. Rows index training time, columns test time.
    times : np.ndarray
        Shape ``(n_win,)`` time axis in seconds; used for both axes.
    vmin, vmax : float
        Shared color scale applied to every subplot so amplitude is
        comparable across subsets.
    stem : str
        Filename stem for :func:`_save`.
    """
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.5 * len(matrices), 4.5), sharey=True)
    if len(matrices) == 1:
        axes = [axes]
    for ax, (name, M) in zip(axes, matrices.items()):
        im = ax.imshow(
            M,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], times[0], times[-1]],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(name)
        ax.set_xlabel("test time (s)")
    axes[0].set_ylabel("train time (s)")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("balanced accuracy")
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def plot_loo_contributions(
    df,
    stem: str,
    metric_col: str = "delta_bacc",
    ylabel: str = "drop in balanced accuracy when electrode removed",
    title: str = "Leave-one-electrode-out contribution",
    group_order: Iterable[str] = ("song", "speech", "music"),
    p_value_text: Optional[str] = None,
) -> Dict[str, Path]:
    """Box + jittered strip plot of per-electrode LOO contributions.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``electrode_group`` and ``metric_col``.
        One row per electrode.
    stem : str
        Filename stem for :func:`_save`.
    metric_col : str, default ``"delta_bacc"``
        Column to plot on the y-axis (e.g. ``delta_bacc`` or
        ``delta_divergence``).
    ylabel, title : str
        Axis label and plot title.
    group_order : iterable[str]
        X-axis order of electrode groups. Missing groups produce
        empty positions.
    p_value_text : str, optional
        If given, rendered in the top-left as an annotation box
        (typically the song-vs-rest permutation p-value).
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    group_order = list(group_order)
    data = [df.loc[df["electrode_group"] == g, metric_col].to_numpy() for g in group_order]
    ax.boxplot(data, labels=group_order, widths=0.5, showfliers=False)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        jitter = rng.normal(0, 0.04, size=vals.size)
        ax.scatter(
            np.full_like(vals, i + 1, dtype=float) + jitter,
            vals,
            color=SUBSET_COLORS["song_only"] if group_order[i] == "song" else "#333333",
            alpha=0.75,
            s=28,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("electrode group")
    ax.set_title(title)
    if p_value_text:
        ax.text(
            0.02,
            0.98,
            p_value_text,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
        )
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def plot_nonlinear_comparison(
    df,
    stem: str = "fig_nonlinear_comparison",
) -> Dict[str, Path]:
    """Grouped bars of model balanced accuracy per task.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`nonlinear.run_nonlinear_comparison`. Must
        contain ``task``, ``model``, ``bacc``, ``fold_bacc_std``.
    stem : str
        Filename stem for :func:`_save`.
    """
    tasks = list(df["task"].unique())
    models = list(df["model"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    colors = {"linear_logreg": "#555555", "rbf_svm": "#2C7FB8", "autoencoder_latent_logreg": "#D62728"}
    for ax, task in zip(axes, tasks):
        sub = df[df["task"] == task]
        x = np.arange(len(models))
        vals = [float(sub[sub["model"] == m]["bacc"].iloc[0]) for m in models]
        err = [float(sub[sub["model"] == m]["fold_bacc_std"].iloc[0]) for m in models]
        ax.bar(x, vals, yerr=err, capsize=4, color=[colors.get(m, "#888") for m in models])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(task)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, linestyle="--", color="black", linewidth=1)
        ax.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("balanced accuracy (grouped CV)")
    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def plot_dataset_overview(
    meta: Dict[str, object],
    electrode_group: Sequence[str],
    y_coarse: Sequence[str],
    stem: str = "fig1_dataset_overview",
) -> Dict[str, Path]:
    """Three-panel dataset summary (Figure 1).

    Panel 1: stimuli per coarse class. Panel 2: electrodes per group.
    Panel 3: monospace annotation explaining the four subsets used
    throughout the paper (``all``, ``no_song``, ``song_only``,
    random matched controls) and the key time/sampling parameters.

    Parameters
    ----------
    meta : dict
        ``ds["meta"]`` from :func:`data_utils.build_dataset`. Keys
        ``time_window_s`` and ``dt_s`` are read if present.
    electrode_group : sequence[str]
        One anatomical group label per electrode, shape ``(n_elec,)``.
    y_coarse : sequence[str]
        One coarse class label per stimulus, shape ``(n_stim,)``.
    stem : str
        Filename stem for :func:`_save`.
    """
    # two clean bar panels with generous label sizes, subset size bookkeeping
    # is reported in the figure caption rather than a crowded third panel,
    # this keeps the figure legible at the standard paper size
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    classes, counts = np.unique(np.asarray(y_coarse), return_counts=True)
    axes[0].bar(classes, counts, color=["#D62728", "#4472C4", "#2CA02C"],
                edgecolor="black", linewidth=0.6)
    axes[0].set_title("stimuli per class", fontsize=12)
    axes[0].set_ylabel("n stimuli", fontsize=11)
    axes[0].tick_params(axis="both", labelsize=11)
    for i, v in enumerate(counts):
        axes[0].text(i, v + 0.35, str(int(v)), ha="center", fontsize=12)
    axes[0].set_ylim(0, max(counts) * 1.18)
    axes[0].grid(alpha=0.2, axis="y")

    groups, g_counts = np.unique(np.asarray(electrode_group), return_counts=True)
    axes[1].bar(groups, g_counts, color=["#D62728", "#4472C4", "#2CA02C"],
                edgecolor="black", linewidth=0.6)
    axes[1].set_title("electrodes per selectivity group", fontsize=12)
    axes[1].set_ylabel("n electrodes", fontsize=11)
    axes[1].tick_params(axis="both", labelsize=11)
    for i, v in enumerate(g_counts):
        axes[1].text(i, v + 0.25, str(int(v)), ha="center", fontsize=12)
    axes[1].set_ylim(0, max(g_counts) * 1.18)
    axes[1].grid(alpha=0.2, axis="y")

    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def plot_confusion_matrix(
    cm_norm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    stem: str,
) -> Dict[str, Path]:
    """Render a row-normalised confusion matrix (rows = true class).

    Parameters
    ----------
    cm_norm : np.ndarray
        Shape ``(n_classes, n_classes)``. Each row should sum to 1.
    class_names : sequence[str]
        Tick labels; length must equal ``n_classes``.
    title, stem : str
        Plot title and filename stem.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=10,
            )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


# bellier extension figures
BELLIER_SUBSET_COLORS = {
    "all": "#555555",
    "right_STG": "#E45756",
    "left_STG": "#4C78A8",
    "non_STG": "#AEB6BF",
}

EVENT_COLORS = {
    "vocal": "#E45756",
    "instrumental": "#4C78A8",
    # norman haignere stimulus classes mapped onto the same qualitative axis
    "song": "#E45756",
    "music": "#F58518",
    "speech": "#4C78A8",
}


def plot_bellier_decoder_subset_bars(
    summary,
    stem: str = "fig8_bellier_decoder_subsets",
    title: str = "Bellier vocal vs instrumental decoder (blocked 5-fold CV)",
) -> Dict[str, Path]:
    """Grouped bars of mean balanced accuracy per subset and model.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`bellier_decoder.run_vocal_instrumental_decoder`.
        Must contain ``subset``, ``model``, ``mean_bacc``,
        ``bacc_ci95_low``, ``bacc_ci95_high``, ``n_elec``.
    stem, title : str
        Filename stem and plot title.
    """
    df = summary.copy()
    subsets = [s for s in ("all", "right_STG", "left_STG", "non_STG")
               if s in df["subset"].unique()]
    models = [m for m in ("logreg", "cnn") if m in df["model"].unique()]

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.6))
    x = np.arange(len(subsets))
    width = 0.8 / max(1, len(models))

    for mi, model in enumerate(models):
        means, lo_err, hi_err = [], [], []
        for sub in subsets:
            row = df[(df["subset"] == sub) & (df["model"] == model)]
            if row.empty:
                means.append(np.nan)
                lo_err.append(0)
                hi_err.append(0)
                continue
            m = float(row["mean_bacc"].iloc[0])
            lo = float(row["bacc_ci95_low"].iloc[0])
            hi = float(row["bacc_ci95_high"].iloc[0])
            means.append(m)
            lo_err.append(max(0.0, m - lo))
            hi_err.append(max(0.0, hi - m))
        offset = (mi - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width * 0.9,
            yerr=[lo_err, hi_err],
            capsize=3,
            label=model,
            color="#4C78A8" if model == "logreg" else "#E45756",
            edgecolor="black",
            linewidth=0.5,
        )
        for xi, m in zip(x + offset, means):
            if np.isfinite(m):
                ax.text(xi, m + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylabel("Balanced accuracy (95% CI)")
    ax.set_title(title)
    ax.set_ylim(0.35, min(1.0, float(np.nanmax(df["bacc_ci95_high"])) + 0.08))
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def plot_bellier_event_profiles(
    profiles: Dict[str, dict],
    groups: Sequence[str] = ("right_STG", "left_STG"),
    stem: str = "fig9_bellier_event_profiles",
    title: str = "Bellier: event-locked HFA for vocal vs instrumental onsets",
) -> Dict[str, Path]:
    """One subplot per STG group with vocal and instrumental traces.

    Parameters
    ----------
    profiles : dict[str, dict]
        Output of :func:`temporal_profile.bellier_profiles` under the
        ``profiles`` key. Expected keys are ``{group}__vocal`` and
        ``{group}__instrumental``; each value carries ``time_s``,
        ``mean``, ``sem`` arrays of shape ``(n_win,)``.
    groups : sequence[str]
        Groups to render, one subplot each.
    stem, title : str
        Filename stem and figure title.
    """
    fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 3.8), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, group in zip(axes, groups):
        for event in ("vocal", "instrumental"):
            key = f"{group}__{event}"
            if key not in profiles:
                continue
            p = profiles[key]
            t, m, s = p["time_s"], p["mean"], p["sem"]
            color = EVENT_COLORS[event]
            ax.plot(t, m, label=f"{event} (n={p['n_events']})", color=color, linewidth=1.6)
            ax.fill_between(t, m - s, m + s, color=color, alpha=0.18, linewidth=0)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{group}  (n_elec={profiles[f'{group}__vocal']['n_elec']})")
        ax.set_xlabel("Time re: onset (s)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("HFA (z)")
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths


def plot_temporal_profile_overlay(
    bellier_profiles: Dict[str, dict],
    norman_profiles: Dict[str, dict],
    bellier_groups: Sequence[str] = ("right_STG", "left_STG"),
    norman_pair: tuple = ("song", "song"),  # (electrode_group, stimulus_class)
    stem: str = "fig10_cross_dataset_profiles",
    title: str = "Cross-dataset temporal profiles: Bellier STG vs Norman song electrodes",
) -> Dict[str, Path]:
    """Cross-dataset overlay: Bellier STG vocal onsets vs Norman song trials.

    Both curves are peak-normalised within the post-onset window
    ``[0, 1.5]`` s so only *shape* (rise time, decay) is compared;
    absolute amplitude is not meaningful because the two datasets use
    different HFA normalisations.

    Parameters
    ----------
    bellier_profiles, norman_profiles : dict[str, dict]
        Outputs of :func:`temporal_profile.bellier_profiles` and
        :func:`temporal_profile.norman_profiles` (the ``profiles``
        sub-dicts, not the full returns).
    bellier_groups : sequence[str]
        Bellier groups to overlay (``right_STG``, ``left_STG``).
    norman_pair : tuple[str, str]
        ``(electrode_group, stimulus_class)`` pair for the Norman
        overlay trace.
    stem, title : str
        Filename stem and figure title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.2))

    def _norm(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = (t >= 0) & (t <= 1.5)
        if not m.any():
            return y
        peak = float(np.nanmax(y[m]))
        peak = peak if peak > 1e-9 else 1.0
        return y / peak

    for group in bellier_groups:
        key = f"{group}__vocal"
        if key not in bellier_profiles:
            continue
        p = bellier_profiles[key]
        t, y = p["time_s"], p["mean"]
        ax.plot(
            t,
            _norm(t, y),
            label=f"Bellier {group} (vocal onset)",
            color=BELLIER_SUBSET_COLORS[group],
            linewidth=1.8,
        )

    eg, cls = norman_pair
    key = f"{eg}__{cls}"
    if key in norman_profiles:
        p = norman_profiles[key]
        t, y = p["time_s"], p["mean"]
        ax.plot(
            t,
            _norm(t, y),
            label=f"Norman {eg} electrodes ({cls} trials)",
            color="#E45756",
            linestyle="--",
            linewidth=1.8,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Time re: onset (s)")
    ax.set_ylabel("HFA (normalized to peak)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    plt.tight_layout()
    paths = _save(fig, stem)
    plt.close(fig)
    return paths
