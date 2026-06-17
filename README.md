# goal-model — a transparent, calibration-first football forecaster

A Dixon-Coles bivariate-Poisson model for international football, built to a
single principle: **a model that picks the right winner is not the same as a
model whose probabilities are correct.** Picking winners is easy; being
*calibrated* is the hard, honest part — and the part most hobby predictors skip.

This is a white-box, parametric statistical model (maximum-likelihood estimation
of per-team attack/defense ratings), deliberately chosen over a black-box ML
model so that every number is interpretable and every error is attributable to
a stated assumption rather than hidden weights.

## What it does

1. **Learns** an attack rating and a defense rating for every team, plus a global
   home-advantage term and the Dixon-Coles low-score correlation `rho`, by
   maximum likelihood on ~5,500 international matches since 2021, with
   exponential time-decay weighting (recent form dominates).
2. **Predicts** any fixture: full scoreline distribution, W/D/L probabilities,
   and over/under 2.5 — derived from learned expected goals (lambda), not
   hand-tuned numbers.
3. **Scores itself**: Brier score and reliability bins against `uniform` and
   `market` (de-vigged odds) baselines. Beating the market is the real bar.

## The core idea: fitting lambda

A naive Poisson model needs an expected-goals number (`lambda`) per team per
match. The temptation is to type one in by hand. This project's whole point is
to **estimate** lambda instead:

```
lambda_home = exp( attack[home] - defense[away] + home_adv * (1 - neutral) )
lambda_away = exp( attack[away] - defense[home] )
```

All `attack[]`, `defense[]`, `home_adv`, `rho` are fit by maximizing the
time-weighted log-likelihood of every historical scoreline (`src/dixoncoles.py`).

### Why it matters (a worked example)

During development the fixtures below were first prototyped with **hand-filled**
lambdas. Re-fitting on real data showed how far subjective guesses can drift —
and, instructively, caught an error of judgment:

| Fixture | hand-filled | learned | note |
|---|---|---|---|
| Spain–Cape Verde | 2.00 / 0.45 | 2.66 / 0.43 | hand guess *under*-rated Spain's attack (top-rated in the data) |
| Argentina–Algeria | 1.85 / 0.85 | 1.64 / 0.48 | data attributes Argentina's edge to **defense** (best in dataset), not firepower |
| Norway–Iraq | 2.10 / 0.55 | 2.26 / 0.58 | guess was close; a mid-game urge to *raise* Iraq's rating was the real mistake — 5 yrs of data say Iraq is genuinely this weak; their goal was noise, not signal |

The Norway–Iraq row is the lesson in miniature: adjusting a sound prior to fit a
single half-hour of observation is overfitting to noise. The data-driven prior
resisted it.

## Evaluation: rolling-origin backtest (no leakage)

The original `fit_and_report.py` fit the model on the *entire* dataset and then
"predicted" fixtures inside that same span — every prediction saw matches played
after it (look-ahead leakage), so its skill was optimistic and uninterpretable.

`backtest/rolling_backtest.py` replaces this with a **walk-forward** scheme. The
timeline is cut into blocks; for each block the model is freshly refit on matches
strictly *before* the block's cutoff and then scores every match *in* the block.
Because training is `date < cutoff` and every scored match has `date >= cutoff`,
no match is ever scored using information from its own match day or later. The
backtest asserts this invariant on every run.

Even *universe construction* is leak-free: the "team has ≥ N matches" filter is
re-derived at each cutoff from past matches only (`filter_min_matches` on the
training fold), not once over the whole dataset. Teams below threshold as of a
cutoff are simply absent and their fixtures are recorded un-scorable. (Fixing
this changed which fixtures are scorable but left the headline metrics
essentially unchanged — the original global filter was not materially inflating
the comparison.)

All models implement one small contract (`models/base.py`:
`fit(train, as_of)` + `predict_proba`) so the engine and metrics never need to
know which model they are scoring. Every model is compared on exactly the
fixtures *all* of them could predict.

## Metrics: calibration-first, all proper scoring rules

`evaluation/metrics.py` (migrated and extended from the old `calibration.py`):

- **log loss** — canonical probabilistic loss; punishes confident wrong calls.
- **Brier score** — MSE of the (home, draw, away) vector.
- **RPS** (Ranked Probability Score) — *order-aware*: predicting a draw when the
  result was an away win costs less than predicting a home win. The right metric
  for an ordinal 1X2 outcome.
- **exact-score NLL** — negative log-likelihood of the realised scoreline, for
  models that emit a full score distribution (the goal models).
- **calibration curve + ECE** — per-bin predicted-vs-observed frequency.
- **block bootstrap** — CIs and paired model-vs-model differences that resample
  whole refit blocks, so correlated within-block fixtures don't inflate apparent
  precision.

Baselines: **always-uniform** (1/3, 1/3, 1/3) and a **de-vigged market**
interface (`devig_probs`) — supply decimal odds and the market becomes the bar
(odds data not yet wired in; the interface is ready for closing odds).

## Models compared

| key | model | what it tests |
|---|---|---|
| `dixon_coles` | parametric Poisson goal model | the white-box incumbent |
| `negative_binomial` | same mean structure, over-dispersed counts | does relaxing mean = variance help? |
| `logreg` | multinomial logistic on pre-match form features | linear ML baseline |
| `gbm` | gradient-boosted trees on the same features | does flexibility add value or over-fit? |

## What the backtest found (2024-07 → 2026, refit per 30-day block)

| model | log loss | Brier | RPS | exact-NLL | ECE (home) | acc |
|---|---|---|---|---|---|---|
| dixon_coles | **0.858** | **0.504** | **0.330** | 2.843 | 0.020 | 0.601 |
| negative_binomial | 0.858 | 0.504 | 0.330 | 2.843 | 0.020 | 0.601 |
| logreg | 0.995 | 0.594 | 0.415 | — | 0.031 | 0.524 |
| gbm | 1.023 | 0.609 | 0.425 | — | 0.040 | 0.505 |
| baseline_uniform | 1.099 | 0.667 | 0.477 | — | 0.140 | 0.473 |

Scored on the 1,843 fixtures every model could predict. Differences are read off
the **block bootstrap** (2,000 resamples of whole refit blocks, so the time
structure is respected, not assumed-i.i.d.):

1. **The negative-binomial variant brings no measurable improvement.** Raw goal
   counts are over-dispersed (mean ≈ 1.33, variance ≈ 2.07), which motivated
   trying NB. But the fitted dispersion lands at `r ≈ 1450` (effectively
   Poisson) and NB is statistically indistinguishable from Dixon-Coles — the
   paired log-loss difference is `+0.0000`, 95% CI `[-0.0001, +0.0001]`. The
   reading: the marginal over-dispersion is largely absorbed by between-match
   variation in team strength (the fitted λ already differs match to match),
   leaving little extra dispersion for the count distribution to model once that
   structure is in the mean. The blowout-underestimation noted earlier is
   therefore better pursued through the mean / independence structure than the
   count distribution.
2. **At this sample size and feature set, the structured model beats the
   flexible one.** `gbm` is the weakest non-trivial model and loses to plain
   logistic regression; Dixon-Coles beats both decisively (DC − logreg =
   `-0.136`, 95% CI `[-0.169, -0.098]`; DC − gbm = `-0.162`, CI
   `[-0.196, -0.124]` — both exclude 0). With only a few thousand matches and a
   handful of hand-crafted form features, flexible ML under-performs the
   structured model: when features and samples are limited, model structure can
   matter more than flexibility. Whether richer features change that is the open
   question, not a settled verdict on ML.
3. **Dixon-Coles is well calibrated** out-of-sample: ECE 0.020, the reliability
   curve tracks the diagonal across all ten probability bins
   (`figures/calibration.png`).

Figures (regenerated by the report, or `python evaluation/plots.py`):

| | |
|---|---|
| `figures/calibration.png` | reliability diagram, all models, ECE in the legend |
| `figures/rolling_logloss.png` | per-block log loss over backtest time |

## Layout

```
src/data.py                    load + filter + time-decay weights
src/dixoncoles.py              the goal model: MLE fit + scoreline/W-D-L   <- core
src/calibration.py             original Brier/reliability (superseded by evaluation/)
models/base.py                 shared ForecastModel contract
models/dixon_coles.py          adapter: Dixon-Coles -> contract
models/negative_binomial.py    over-dispersed (NB) goal model
models/ml_baselines.py         logistic + gradient-boosted 1X2 baselines
backtest/rolling_backtest.py   walk-forward, no-leakage evaluation harness
evaluation/metrics.py          log loss, Brier, RPS, exact-NLL, calibration, block bootstrap
evaluation/plots.py            calibration + rolling-log-loss figures
scripts/backtest_report.py     the real report: all models, one backtest, metrics + figures
scripts/fit_and_report.py      original single-fit demo (in-sample; illustrative only)
data/results.csv               martj42/international_results (public, 1872–present)
figures/                       generated PNGs (calibration, rolling log loss)
```

## Run

```bash
pip install -r requirements.txt
python scripts/backtest_report.py     # leakage-free model comparison (the real report)
python scripts/fit_and_report.py      # original in-sample ratings demo
```

## Roadmap

- [ ] de-vig live odds → model-vs-market edge report (analysis only; no betting hooks)
- [x] rolling-origin backtest — leakage-free walk-forward evaluation
- [x] unified metric suite — log loss, Brier, RPS, exact-NLL, calibration + baselines
- [x] negative-binomial variant vs gradient-boosted / logistic baseline
- [x] leak-free team-universe construction (per-cutoff filter)
- [x] block-bootstrap CIs + calibration / rolling-log-loss figures
- [ ] richer per-match ML features (date-exact form, Elo, competition tier)

## Data & method credits

- Dataset: martj42/international_results (CC0).
- Method: Dixon & Coles (1997), "Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market."

*This is a forecasting and market-efficiency research project. It contains no
betting integration and recommends no wagers.*
