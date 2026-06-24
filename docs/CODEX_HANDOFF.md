# Codex Handoff

Before making Probability Cup UI changes, read:

1. README_LOCAL_UI.md
2. docs/PROBABILITY_CUP_PRODUCT_SPEC.md
3. webapp/engine_bridge.py
4. webapp/calibration.py
5. webapp/templates/predictions.html

Current product direction:
Move from CSV-heavy workflow toward a card-based live prediction mode.

## Current State

Implemented:

- Flask local app
- SQLite local history
- session snapshots
- importer/prediction/scoring bridge
- calibration page
- latest CSV downloads
- Manual Probability Workbench

Known UX issue:
Manual Probability Workbench is better than raw CSV, but still not the final
form. The desired next UX is card-based Live Prediction Mode.

## Guardrails

- Do not scrape SportsPredict.
- Do not automate submission.
- Do not add API keys.
- Do not train ML yet.
- Do not build Monte Carlo yet.
- Do not build a crowd_prior_model yet.
- Do not commit `.local/`, `outputs/`, `data/cup/live/`, or live settled data.
- Do not copy SportsPredict branding, logos, exact styling, proprietary assets, or page source.

## Recommended Next PR

Open PR #3 for Live Prediction Mode after PR #2 is preserved and merged. PR #3
should add a card-based route at `/sessions/<session_id>/live`, plus the
deterministic `webapp/prediction_assistant.py` helper described in the product
spec.
