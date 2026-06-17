"""
Unified, leakage-free model comparison.

Runs every model through the SAME rolling-origin backtest and scores them with
the SAME metric suite, against the uniform (and, when odds are supplied, market)
baselines. This is the report the whole refactor exists to produce: an honest,
calibrated, apples-to-apples comparison.

Models compared
---------------
- dixon_coles        : parametric Poisson goal model (the incumbent white box)
- negative_binomial  : same mean structure, over-dispersed counts (NB)
- logreg             : multinomial logistic regression on pre-match features
- gbm                : gradient-boosted trees on the same features

Run:  python scripts/backtest_report.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import warnings

import numpy as np
import pandas as pd

# benign LBFGS line-search overflow inside LogisticRegression on early/thin
# folds; results are unaffected, so keep the report output clean.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                        category=RuntimeWarning)

from data import load_matches
from models.dixon_coles import DixonColesForecaster
from models.negative_binomial import NegativeBinomialForecaster
from models.ml_baselines import LogisticForecaster, GBMForecaster
from backtest.rolling_backtest import run_rolling_backtest, BacktestConfig
from evaluation import metrics as M
from evaluation import plots

DATA = os.path.join(_ROOT, "data", "results.csv")
OUTDIR = os.path.join(_ROOT, "backtest_results")
FIGDIR = os.path.join(_ROOT, "figures")


def main():
    # load raw (no global team filter); the backtest re-derives the team
    # universe per cutoff from past-only matches, so it is leak-free.
    matches = load_matches(DATA, since="2021-01-01", min_matches_per_team=0)

    cfg = BacktestConfig(
        test_start="2024-07-01",   # ~2 yrs of out-of-sample evaluation
        step_days=30,
        min_train_matches=500,
        min_matches_per_team=15,
    )
    print(f"Data: {len(matches):,} matches "
          f"{matches['date'].min().date()} -> {matches['date'].max().date()}")
    print(f"Backtest: test from {cfg.test_start}, {cfg.step_days}d blocks, "
          f"expanding window, refit per block (no leakage)\n")

    factories = {
        "dixon_coles": lambda: DixonColesForecaster(half_life_days=540),
        "negative_binomial": lambda: NegativeBinomialForecaster(half_life_days=540),
        "logreg": lambda: LogisticForecaster(),
        "gbm": lambda: GBMForecaster(),
    }

    res = run_rolling_backtest(matches, factories, cfg, verbose=True)
    res.save(os.path.join(OUTDIR, "predictions_all_models.csv"))

    report = M.evaluate_backtest(res.predictions)
    print("\n" + "=" * 78)
    print(f"MODEL COMPARISON  (scored on {report.attrs['n_shared_fixtures']} "
          f"fixtures predictable by all models; lower is better except accuracy)")
    print("=" * 78)
    fmt = {c: "{:.4f}".format for c in
           ["log_loss", "brier", "rps", "exact_nll", "ece_home", "accuracy"]}
    print(report.to_string(index=False, formatters=fmt))
    report.to_csv(os.path.join(OUTDIR, "model_comparison.csv"), index=False)

    # calibration curve for the best model's home-win probability
    best = report.iloc[0]["model"]
    print("\n" + "-" * 78)
    print(f"Calibration (home-win) — best model by log-loss: {best}")
    print("-" * 78)
    sub = res.predictions[(res.predictions["model"] == best)
                          & res.predictions["predictable"]]
    curve = M.calibration_curve(sub["p_home"].to_numpy(),
                                (sub["actual"] == "home").to_numpy(), n_bins=10)
    print(curve.to_string(index=False))
    ece = M.expected_calibration_error(sub["p_home"].to_numpy(),
                                       (sub["actual"] == "home").to_numpy())
    print(f"\nExpected calibration error (home): {ece:.4f}")

    # ---- block-bootstrap uncertainty (respects time structure) -----------
    print("\n" + "=" * 78)
    print("BLOCK BOOTSTRAP  (resample whole refit blocks; 2000 reps, 95% CI)")
    print("=" * 78)
    model_order = ["dixon_coles", "negative_binomial", "logreg", "gbm"]
    metric_keys = ["log_loss", "brier", "rps"]
    ci = {m: {k: M.block_bootstrap_ci(res.predictions, m, k, n_boot=2000)
              for k in metric_keys} for m in model_order}
    header = f"  {'model':18s} " + " ".join(f"{k:>26s}" for k in metric_keys)
    print(header)
    for m in model_order:
        cells = " ".join(
            f"{ci[m][k]['point']:.4f} [{ci[m][k]['lo']:.4f},{ci[m][k]['hi']:.4f}]"
            for k in metric_keys)
        print(f"  {m:18s} {cells}")

    # markdown CI table for the README (exact numbers, copy-paste ready)
    md = ["| model | log loss (95% CI) | Brier (95% CI) | RPS (95% CI) |",
          "|---|---|---|---|"]
    for m in model_order:
        md.append("| {} | {:.3f} [{:.3f}, {:.3f}] | {:.3f} [{:.3f}, {:.3f}] | "
                  "{:.3f} [{:.3f}, {:.3f}] |".format(
                      m,
                      ci[m]["log_loss"]["point"], ci[m]["log_loss"]["lo"], ci[m]["log_loss"]["hi"],
                      ci[m]["brier"]["point"], ci[m]["brier"]["lo"], ci[m]["brier"]["hi"],
                      ci[m]["rps"]["point"], ci[m]["rps"]["lo"], ci[m]["rps"]["hi"]))
    with open(os.path.join(OUTDIR, "ci_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print("\nPaired log-loss differences (A - B; negative => A better):")
    diffs = {}
    for a, b in [("dixon_coles", "logreg"),
                 ("dixon_coles", "negative_binomial"),
                 ("dixon_coles", "gbm")]:
        d = M.block_bootstrap_diff(res.predictions, a, b, "log_loss", n_boot=2000)
        diffs[(a, b)] = d
        verdict = ("significant" if d["excludes_zero"]
                   else "indistinguishable (CI straddles 0)")
        print(f"  {a} - {b:18s}: {d['point_diff']:+.4f}  "
              f"95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]  -> {verdict}")
    with open(os.path.join(OUTDIR, "diff_table.md"), "w") as f:
        f.write("| comparison | Δ log loss | 95% CI | verdict |\n|---|---|---|---|\n")
        for (a, b), d in diffs.items():
            v = "**significant**" if d["excludes_zero"] else "not significant"
            f.write(f"| {a} − {b} | {d['point_diff']:+.4f} | "
                    f"[{d['lo']:+.4f}, {d['hi']:+.4f}] | {v} |\n")

    # ---- figures ---------------------------------------------------------
    figs = plots.make_all_figures(res.predictions, FIGDIR)

    # scoreline heatmap: fit all models once on the full (filtered) history for
    # an illustrative marquee fixture (static illustration, not evaluation).
    full = load_matches(DATA, since="2021-01-01", min_matches_per_team=15)
    as_of = full["date"].max()
    fitted = {name: factory().fit(full, as_of=as_of)
              for name, factory in factories.items()}
    home, away = _pick_fixture(fitted, full)
    figs.append(plots.scoreline_heatmap(
        fitted, home, away, os.path.join(FIGDIR, "scoreline_heatmap.png")))

    print("\nFigures -> " + ", ".join(os.path.relpath(f, _ROOT) for f in figs))
    print("Saved: predictions_all_models.csv, model_comparison.csv, "
          "ci_table.md, diff_table.md -> backtest_results/")
    print("\nNote: exact_nll is defined only for the goal models (full scoreline "
          "distribution); the 1X2 ML models show NaN by design.")


def _pick_fixture(fitted: dict, df) -> tuple:
    """A recognisable, lopsided-ish fixture present in every model's universe."""
    dc = fitted["dixon_coles"]
    universe = getattr(dc, "_model").idx_ if hasattr(dc, "_model") else {}
    preferred = [("Spain", "Cape Verde"), ("Brazil", "Bolivia"),
                 ("France", "Senegal"), ("Argentina", "Peru")]
    for h, a in preferred:
        if h in universe and a in universe:
            return h, a
    teams = list(universe)
    return (teams[0], teams[1]) if len(teams) >= 2 else ("Spain", "Germany")


if __name__ == "__main__":
    main()
