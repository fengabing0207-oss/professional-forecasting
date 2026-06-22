# Probability Cup prediction engine

This Phase 1 engine turns manually entered Probability Cup questions plus market
odds or manual probabilities into calibrated binary prediction rows. After the
match resolves, it scores those predictions with Brier score and Relative Brier
Points (RBP).

It is deliberately conservative. The existing Dixon-Coles / score-matrix model
is useful for goal-derived questions. It is not a corners, cards, shots,
offsides, penalties, or player-prop model. Unsupported props are market/manual
only until a real model exists.

## What it does

- reads a structured question bank CSV
- reads manual odds or direct probabilities
- removes vig from 2-way and 3-way markets
- optionally accepts a `--model-probs` CSV from an existing model workflow
- blends market/model/manual probabilities into `p_final`
- clamps final probabilities away from impossible 0/1 values
- scores resolved predictions with Brier score and RBP

## What it does not do yet

- scrape websites
- ingest paid APIs
- store API keys
- train prop models for corners, cards, shots, offsides, or players
- claim unsupported prop predictions are model-derived
- replace the existing research/backtest pipeline

## Question bank template

Create a CSV using `data/cup/question_bank_template.csv`:

```text
question_id,match_id,match_date,home_team,away_team,raw_question,event_type,selection,threshold,player,p_manual,manual_weight,notes
```

Important fields:

- `question_id`: stable unique ID for this binary question
- `match_id`: your match identifier
- `home_team`, `away_team`: team names for auditability
- `raw_question`: the original Probability Cup wording
- `event_type`: one of the supported or market/manual-only event types below
- `selection`: selected team, outcome key, or exact score such as `2-1`
- `threshold`: numeric line for over/under and margin questions
- `p_manual`: optional fallback probability in `[0, 1]`
- `notes`: assumptions or mapping details

## Manual odds template

Create a CSV using `data/cup/manual_odds_template.csv`:

```text
question_id,market_id,outcome_key,odds_format,odds_value,direct_probability,bookmaker,retrieved_at,notes
```

Use `odds_format` values:

- `decimal`: put decimal odds in `odds_value`, for example `2.00`
- `american`: put American odds in `odds_value`, for example `+130` or `-150`
- `probability` or `direct`: put a probability in `direct_probability`

Rows with the same `market_id` are treated as one market. Two- and three-way
markets are no-vig normalized proportionally. Single-row inputs keep their raw
implied or direct probability because there is no companion outcome to normalize
against.

## Prediction export

The Phase 1 CLI is useful immediately with market/manual probabilities:

```bash
python -m cup.predict_cup \
  --questions data/cup/question_bank_template.csv \
  --odds data/cup/manual_odds_template.csv \
  --output outputs/cup/predictions.csv
```

If you have model probabilities from the existing Dixon-Coles pipeline, pass an
intermediate file:

```bash
python -m cup.predict_cup \
  --questions data/cup/question_bank.csv \
  --odds data/cup/manual_odds.csv \
  --model-probs outputs/cup/model_probs.csv \
  --output outputs/cup/predictions.csv
```

`model_probs` columns:

```text
question_id,p_model,model_family,notes
```

Output columns:

```text
question_id,match_id,raw_question,event_type,selection,p_market,p_model,p_manual,p_final,status,model_family,notes
```

Default blending:

- market + model: `0.70 * p_market + 0.30 * p_model`
- market only: `p_market`
- model only: `p_model`
- manual only: `p_manual`
- no probability: blank `p_final`, status `missing_probability`

Override defaults with `--market-weight`, `--model-weight`, `--manual-weight`,
`--min-prob`, and `--max-prob`.

## Scoring resolved questions

Fill `data/cup/results_template.csv`:

```text
question_id,actual_result,crowd_brier,notes
```

`actual_result` must be `1` if the question resolved yes/true and `0` otherwise.
`crowd_brier` is optional.

Run:

```bash
python -m cup.scoring \
  --predictions outputs/cup/predictions.csv \
  --results data/cup/results.csv \
  --output outputs/cup/scored_predictions.csv
```

Scoring formulas:

- `user_brier = (p_final - actual_result)^2`
- `rbp = (crowd_brier - user_brier) * 100`

Positive RBP means the prediction beat the crowd benchmark.

## Norway vs Senegal smoke-test example

The repository includes a realistic-shaped sample under `data/cup/examples/`.
It is for testing only. The odds, model probabilities, and results are dummy
sample values, not official prices, official Probability Cup data, or real match
results.

Run prediction export:

```bash
python -m cup.predict_cup \
  --questions data/cup/examples/norway_senegal_questions.csv \
  --odds data/cup/examples/norway_senegal_manual_odds.csv \
  --model-probs data/cup/examples/norway_senegal_model_probs.csv \
  --output outputs/cup/norway_senegal_predictions.csv
```

Run scoring:

```bash
python -m cup.scoring \
  --predictions outputs/cup/norway_senegal_predictions.csv \
  --results data/cup/examples/norway_senegal_results.csv \
  --output outputs/cup/norway_senegal_scored_predictions.csv
```

The example covers:

- `model_and_market`: Norway win (`team_win`) with dummy market and model inputs
- `unsupported_market_only`: prop questions using dummy market odds
- `manual_only`: second-half/fouls/shots/halftime examples using dummy manual inputs
- player prop market-only behavior for Sadio Mane anytime scorer

Generated files live under `outputs/`, which is gitignored.

## Development tests

Install dev-only test dependencies, then run pytest:

```bash
pip install -r requirements-dev.txt
pytest -q
```

If pytest is not installed in the active environment, the tests can still be
smoke-run manually without installing packages:

```bash
python -c "import importlib, inspect; mods=['tests.test_market_odds','tests.test_anchor_and_scoring','tests.test_question_mapper','tests.test_predict_cup']; funcs=[]; [funcs.extend([obj for name,obj in inspect.getmembers(importlib.import_module(m), inspect.isfunction) if name.startswith('test_')]) for m in mods]; [f() for f in funcs]; print('manual test runner passed:', len(funcs), 'tests')"
```

## Model-supported event types

These can be mapped to a goal score matrix when model probabilities are supplied:

- `home_win`
- `away_win`
- `draw`
- `team_win`
- `team_not_lose`
- `total_goals_over`
- `total_goals_under`
- `team_goals_over`
- `team_goals_under`
- `both_teams_score_yes`
- `both_teams_score_no`
- `clean_sheet`
- `exact_score`
- `win_by_margin`

The score-matrix helpers follow the existing project convention: the
Dixon-Coles adapter normalizes the finite matrix after max-goal truncation, and
the Probability Cup helpers compute probabilities on that normalized matrix.

## Market/manual-only event types

These are not model-supported in Phase 1:

- `corners_threshold`
- `team_corners_threshold`
- `shots_on_target_threshold`
- `team_shots_on_target_threshold`
- `offsides_threshold`
- `fouls_more_than_opponent`
- `cards_threshold`
- `red_card`
- `penalty_or_red_card`
- `player_anytime_scorer`
- `first_goal`
- `second_half_result`
- `halftime_result`
- `halftime_draw`
- `halftime_home_win`
- `halftime_away_win`
- `unsupported_market_only`

For these event types, the engine ignores `p_model` and uses market probability
if available, otherwise manual probability. If neither exists, it outputs
`missing_probability`.

## Why market anchoring

Markets aggregate information that this repo does not yet model: lineups,
injuries, tactical changes, weather, discipline profiles, and public news. For
goal-derived questions, the model can add structured signal. For unsupported
props, market/manual inputs are more honest than fake model precision.

## Phase 2 roadmap

Do not treat these as implemented yet:

- real odds ingestion
- corners negative-binomial model
- shots-on-target model
- cards/fouls/offsides base-rate models
- player anytime scorer model
- formation/style feature table
- post-match Bayesian team/player state updater
- event-type calibration layer
- model-vs-market residual tracker
- crowd bias / public bias tracker
