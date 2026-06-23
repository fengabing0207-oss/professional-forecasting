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
8. Use History to download previous snapshots.

There is a Norway/Senegal dummy sample loader for smoke testing. It is not
official odds, probabilities, or real performance validation.

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

