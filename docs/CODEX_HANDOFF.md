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
- Live Prediction Mode with local context, assistant, and market snapshots
- Manual live market anchors for recommendation grounding

Current direction:
Live Prediction Mode is the primary real-match workflow. PR #5 adds manual live
market anchors so heuristic suggestions can be blended with user-entered market
probabilities or odds. Market anchors are manually entered only. Percent input
uses percent mode, so enter `51` for 51%, not `0.51`.

Market odds input supports American odds such as `-160` and `+220`, and decimal
odds such as `1.62` and `2.85`. Odds are converted to rough implied
probability only. No devig or bookmaker-margin adjustment is applied in PR #5.

Market anchors influence recommendations and are saved locally as audit
snapshots, but they never auto-submit and never become final probabilities by
themselves. Final probability remains explicit user-confirmed input. The manual
odds CSV includes final probabilities only, not market anchors.

## Guardrails

- Do not scrape SportsPredict.
- Do not login to SportsPredict.
- Do not automate submission.
- Do not add API keys.
- Do not add external live feeds.
- Do not train ML yet.
- Do not build Monte Carlo yet.
- Do not build a crowd_prior_model yet.
- Do not commit `.local/`, `outputs/`, `data/cup/live/`, or live settled data.
- Do not copy SportsPredict branding, logos, exact styling, proprietary assets, or page source.

## Current PR

PR #5 adds a local, manual market-anchor layer. Keep it transparent and
bounded: no scraping, no login, no API, no external feed, no auto-submit, no ML
training, no Monte Carlo, and no crowd model.
