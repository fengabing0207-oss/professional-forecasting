"""
Figures from a real backtest — calibration and skill-over-time.

Two plots, both driven straight off the rolling backtest's prediction frame
(`backtest_results/predictions_all_models.csv`), so they reproduce with one
command and always reflect the leakage-free evaluation:

1. reliability_plot   — calibration / reliability diagram. For each model, all
   predicted probabilities (the three one-vs-rest outcomes pooled) binned and
   plotted against observed frequency, with the y=x reference and each model's
   ECE in the legend. On-diagonal = calibrated.
2. rolling_logloss_plot — per-block log loss over backtest time, one line per
   model. Shows where each model's skill holds or degrades across the test span.
3. scoreline_heatmap  — side-by-side P(home i, away j) matrices for one fixture;
   goal models get a heatmap, the 1X2 classifiers a labelled blank panel.

Style is deliberately plain: white background, thin grid, no chartjunk.
"""
from __future__ import annotations
import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")             # headless / reproducible file output
import matplotlib.pyplot as plt   # noqa: E402

from evaluation.metrics import (  # noqa: E402
    OUTCOMES, log_loss, expected_calibration_error,
)

_COLORS = {
    "dixon_coles": "#1f77b4", "negative_binomial": "#ff7f0e",
    "logreg": "#2ca02c", "gbm": "#d62728",
}
# NB tracks Dixon-Coles almost exactly; dash it so the DC line shows through
# rather than being fully overplotted (and to make the overlap legible).
_LINESTYLE = {"negative_binomial": (0, (5, 3))}


def _pooled_calibration(sub: pd.DataFrame, n_bins: int):
    """Pool the three one-vs-rest (prob, hit) pairs across a model's fixtures."""
    probs, hits = [], []
    for k, o in enumerate(OUTCOMES):
        col = ["p_home", "p_draw", "p_away"][k]
        probs.append(sub[col].to_numpy())
        hits.append((sub["actual"] == o).to_numpy().astype(float))
    p = np.concatenate(probs)
    h = np.concatenate(hits)
    bins = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = which == b
        if m.sum() == 0:
            continue
        xs.append(p[m].mean())
        ys.append(h[m].mean())
    # sample-weighted ECE on the pooled events
    ece = expected_calibration_error(p, h, n_bins)
    return np.array(xs), np.array(ys), ece


def reliability_plot(preds: pd.DataFrame, out_path: str,
                     models: Optional[list[str]] = None, n_bins: int = 10) -> str:
    models = models or sorted(preds["model"].unique())
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect")
    for m in models:
        sub = preds[(preds["model"] == m) & preds["predictable"]]
        if sub.empty:
            continue
        xs, ys, ece = _pooled_calibration(sub, n_bins)
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5,
                color=_COLORS.get(m), ls=_LINESTYLE.get(m, "-"),
                label=f"{m} (ECE={ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (reliability) — all outcomes pooled")
    ax.grid(True, lw=0.4, color="0.9")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def rolling_logloss_plot(preds: pd.DataFrame, out_path: str,
                         models: Optional[list[str]] = None,
                         min_block: int = 20) -> str:
    """Per-cutoff-block log loss over time, one line per model.

    Blocks with fewer than ``min_block`` scorable fixtures are dropped (their
    per-block log loss is too noisy to read).
    """
    models = models or sorted(preds["model"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for m in models:
        sub = preds[(preds["model"] == m) & preds["predictable"]]
        xs, ys = [], []
        for cutoff, g in sub.groupby("cutoff"):
            if len(g) < min_block:
                continue
            probs = g[["p_home", "p_draw", "p_away"]].to_numpy()
            xs.append(pd.Timestamp(cutoff))
            ys.append(log_loss(probs, g["actual"].tolist()))
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5,
                color=_COLORS.get(m), ls=_LINESTYLE.get(m, "-"), label=m)
    ax.axhline(np.log(3), color="0.6", lw=1, ls="--", label="uniform (ln 3)")
    ax.set_xlabel("backtest block (refit cutoff)")
    ax.set_ylabel("log loss")
    ax.set_title("Rolling log loss over backtest time (lower = better)")
    ax.grid(True, lw=0.4, color="0.9")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def scoreline_heatmap(fitted: dict, home: str, away: str, out_path: str,
                      neutral: bool = True, max_display: int = 6) -> str:
    """Side-by-side scoreline probability matrices for one illustrative fixture.

    ``fitted`` maps model name -> fitted model. Goal models (those exposing
    ``predict_scoreline``) get a heatmap of P(home goals i, away goals j); the
    1X2 ML classifiers get a blank panel labelled accordingly, because they do
    not produce a scoreline distribution at all. Visually makes the point that
    Dixon-Coles and the NB variant are nearly identical.
    """
    names = list(fitted)
    fig, axes = plt.subplots(1, len(names), figsize=(3.5 * len(names), 3.6),
                             squeeze=False)
    axes = axes[0]
    mats = {}
    for n, m in fitted.items():
        if hasattr(m, "predict_scoreline"):
            M = m.predict_scoreline(home, away, neutral)
            if M is not None:
                mats[n] = M[:max_display + 1, :max_display + 1]
    vmax = max((M.max() for M in mats.values()), default=0.1)

    for ax, n in zip(axes, names):
        if n in mats:
            M = mats[n]
            im = ax.imshow(M, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            ax.set_xlabel(f"{away} goals")
            ax.set_ylabel(f"{home} goals")
            ax.set_xticks(range(max_display + 1))
            ax.set_yticks(range(max_display + 1))
            ax.set_title(n, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, f"{n}\n\n(classifier — no\nscoreline distribution)",
                    ha="center", va="center", fontsize=9, color="0.4",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle(f"Scoreline probability — {home} vs {away} (neutral)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_all_figures(preds: pd.DataFrame, figdir: str) -> list[str]:
    os.makedirs(figdir, exist_ok=True)
    return [
        reliability_plot(preds, os.path.join(figdir, "calibration.png")),
        rolling_logloss_plot(preds, os.path.join(figdir, "rolling_logloss.png")),
    ]


if __name__ == "__main__":
    # standalone: regenerate figures from the saved backtest predictions
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    csv = os.path.join(_ROOT, "backtest_results", "predictions_all_models.csv")
    preds = pd.read_csv(csv, parse_dates=["date", "cutoff"])
    paths = make_all_figures(preds, os.path.join(_ROOT, "figures"))
    print("wrote:", *paths, sep="\n  ")
