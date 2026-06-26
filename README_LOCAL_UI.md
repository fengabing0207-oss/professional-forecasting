# Local Probability Cup UI

This Phase 1.3 UI is a local browser wrapper around the existing Probability
Cup importer, prediction exporter, scoring workflow, and CSV history. It is a
small Flask app with SQLite storage. It is not deployed and does not interact
with SportsPredict automatically.

## Install

```bash
pip install -r requirements-ui.txt
```

The app uses `pandas` through the existing engine code, which is already in the
main project requirements.

## Run

```bash
python -m webapp.app
```

Open:

```text
http://127.0.0.1:5050
```

## Storage

History is stored locally in:

```text
.local/probability_cup_history.sqlite3
```

Override with:

```bash
PROB_CUP_DB_PATH=/path/to/history.sqlite3 python -m webapp.app
```

`.local/` is ignored and should not be committed.

## Workflow

1. Create a session with home team, away team, match ID, and optional date.
2. Paste copied questions into Import Questions.
3. Review the generated question CSV and save a reviewed snapshot.
4. Paste or upload manual odds and optional model probability CSVs.
5. Generate predictions and inspect status/risk flags.
6. Manually submit probabilities to SportsPredict.
7. After the match, paste or upload results CSV and score the predictions.
8. Use the homepage latest CSV links or History to download previous snapshots.

There is a Norway/Senegal dummy sample loader for smoke testing. It is not
official odds, probabilities, or real performance validation.

The session list shows latest CSV downloads for questions, predictions, and
scoring after those snapshots exist. Missing snapshot types are shown as muted
labels instead of links.

## Product Direction

The long-term UI is a card-based live prediction mode. CSV views remain
available for debugging and audit history, but live use should happen through a
10-question card workflow. See `docs/PROBABILITY_CUP_PRODUCT_SPEC.md` before
making Probability Cup UI changes.

## Live Prediction Workflow

1. Import questions.
2. Review question parsing.
3. Open Live Prediction Mode from the session page.
4. Review assistant ranges, reasoning, risk flags, and exposure warnings.
5. Enter final probabilities as percents.
6. Generate predictions and inspect the manual submission sheet.
7. Manually submit final probabilities to SportsPredict.

The Manual Probability Workbench uses percent mode. Enter `51` for 51%, which
is written to the generated manual odds CSV as `0.51`. Blank inputs stay blank.
Ambiguous decimal percent entries such as `0.51` are rejected; enter `51`
instead. The workbench does not invent probabilities and does not use
`parser_confidence` as a probability.

Live Prediction Mode is available at:

```text
/sessions/<session_id>/live
```

The assistant is deterministic and heuristic. It does not scrape, call external
APIs, train a model, or submit anything automatically.

## Settled History Calibration

The Calibration page accepts manually copied settled-history CSVs and runs local
diagnostics without saving the uploaded data. Use it after a completed match to
compare user probability, crowd probability, actual outcome, and platform RBP.

Expected columns:

```text
session_id
match_id
match_date
home_team
away_team
question_id
raw_question
event_type
selection
user_prob
crowd_prob
actual_result
platform_rbp
notes
```

`user_prob` and `crowd_prob` should be decimals between 0 and 1.
`actual_result` should be 0 or 1. `platform_rbp` is preserved as copied from
SportsPredict, with numeric summaries computed in separate analysis columns.

The page shows:

- overall RBP, Brier, crowd-edge, and directional-correctness metrics
- event-type performance
- probability-bucket performance
- largest RBP wins and losses
- largest user/crowd deviations
- guardrail suggestions for recurring loss patterns
- a normalized CSV download

Do not commit live settled-history data unless that is explicitly requested.

## Boundaries

- No scraping.
- No API keys.
- No automatic SportsPredict submission.
- No ML prop models.
- No Monte Carlo simulation.
- No LLM agent.
- Unsupported props remain market/manual-only.
- Full-time goal-model outputs must not be used for halftime, corners, shots,
  fouls/cards, offsides, or player scoring.

## Validation

```bash
PYTHONPYCACHEPREFIX=/private/tmp/worldcupper-pycache .venv/bin/python -m compileall src backtest evaluation cup market tests webapp
.venv/bin/python -m pytest -q
git ls-files outputs
git ls-files .local
git status
```
