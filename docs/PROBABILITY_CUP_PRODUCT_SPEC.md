# Probability Cup Product Spec

Probability Cup Local should become a local prediction assistant, not a CSV editor.

## Product Vision

The UI should optimize for a 10-question live match workflow. The user should
not edit raw CSV during live use unless debugging.

Target workflow:

1. Create match session.
2. Paste up to 10 raw Probability Cup questions manually.
3. Parse questions into event_type / selection / threshold / player / period.
4. Show a card-based live prediction mode.
5. For each question, show suggested probability, suggested range, confidence, reasoning, and risk flags.
6. User edits final probability with slider/input.
7. App generates manual odds CSV internally.
8. App runs existing prediction engine.
9. App saves prediction snapshot.
10. User manually submits final probabilities to SportsPredict.
11. After match, user pastes settled results into calibration.
12. Calibration updates guardrails and review notes.

## Non-Goals

- No scraping.
- No auto-submit.
- No API keys.
- No model training until enough settled data exists.
- No hidden 0.5 defaults.
- No treating parser_confidence as probability.
- No copying SportsPredict branding or assets.

## Live Prediction Mode

Route:

```text
GET/POST /sessions/<session_id>/live
```

Purpose:
A SportsPredict-style local workflow for reviewing and finalizing up to 10
questions. This should be original local UI, not a copy of SportsPredict
branding, logos, styling, proprietary assets, or page source.

Layout:

Match header:

- home_team vs away_team
- match_date
- match_id
- progress: Question 1 / 10
- completion count: 7 / 10 final probabilities entered

Question card:

- raw_question
- event_type badge
- selection
- threshold
- player
- period
- parser status
- model support status:
  - goal-model supported
  - market/manual-only
  - needs manual
  - unsupported / needs review

Assistant panel:

- suggested_probability_percent
- optional manual market anchor probability
- anchored recommendation when market anchor exists
- suggested_range
- confidence: low / medium / high
- reasoning
- risk_flags
- exposure warning if relevant

Final probability input:

- slider from 0 to 100
- numeric percent input
- save button
- next / previous buttons

Footer:

- question list mini-nav
- missing probability count
- Generate Predictions button
- Export submission sheet button

Interaction:

- User can move prev/next between cards.
- User can enter final probability as percent.
- Final probability is saved locally.
- Blank stays blank.
- Invalid values below 0 or above 100 are rejected.
- `0.51` in percent mode is rejected or clearly normalized only if explicitly supported.
- No silent 50% default.
- Parser confidence never becomes probability.
- Market anchors never become final probability automatically.
- Each card should show whether it still needs user action.

Submission sheet:
After final probabilities are entered, show a simple table:

```text
Q1 raw question final_probability_percent
Q2 raw question final_probability_percent
...
```

This is for manual SportsPredict entry only. No auto-submit.

## Manual Market Anchors

Live Prediction Mode may accept optional, manually entered market anchors. These
are local notes from the user, not scraped data and not an external feed.

Supported inputs:

- market anchor percent, e.g. `47`
- American odds, e.g. `-160` or `+220`
- decimal odds, e.g. `1.62` or `2.85`
- market source / notes

If both percent and odds are entered, percent takes priority. Odds convert to
rough implied probability only. No devig or bookmaker-margin adjustment is
performed in PR #5.

When both heuristic and market anchor exist, the recommended probability is a
transparent blend:

```text
0.60 * market_anchor + 0.40 * heuristic
```

Market anchors and recommendations remain guidance. Final probability must
still be explicitly confirmed by the user, and the manual odds CSV is generated
from final probabilities only.

## Prediction Assistant v0 Heuristics

This should be deterministic and transparent. Do not call external APIs. Do not
train ML.

Future file target:

```text
webapp/prediction_assistant.py
```

Planned functions:

```text
suggest_probability_for_question(row, match_context=None, calibration_context=None) -> dict
suggest_probabilities_for_question_csv(question_csv_text, match_context=None, calibration_context=None) -> DataFrame
assistant_rows_to_manual_odds_csv(rows, match_id=None) -> str
normalize_final_probability_percent(value) -> float | None
detect_match_script_exposure(rows) -> list[str]
```

Initial heuristic examples:

penalty_or_red_card:

- suggested 26-32%
- risk if final > 40%

halftime_draw:

- suggested 38-45%

compound BTTS + 3+ goals:

- suggested 40-48%
- risk: compound_condition
- warning: favorite can win 2-0 / 3-0 while this loses

player_second_half_shot_on_target:

- suggested 40-55%
- risk if final > 55%
- warning: second-half-only player SOT is high variance

player_shot_on_target:

- suggested 40-60%
- risk if final > 60% unless star context is explicitly provided

fouls_more_than_opponent:

- suggested 52-59%
- risk if final > 60%
- calibration note: prior losses from high-confidence fouls

corners_threshold:

- total 9+ corners: 45-55%
- underdog/team 5+ corners: 30-40%
- risk if underdog 5+ corners > 45%

shots_on_target_threshold:

- second-half total 4+ SOT: 52-60%
- both teams 2H 1+ SOT: 50-57%
- underdog 4+ SOT: 30-42%

offsides_threshold:

- suggested 45-58%
- confidence low-medium

team_win:

- if no market/team-strength context, avoid aggressive auto-suggestion
- user review required

Exposure warnings:
Detect when the card set implies the same fragile match script too many times:

```text
favorite win high + underdog SOT high
favorite win high + BTTS high
favorite win high + underdog 5+ corners high
multiple second-half-only props above 55%
multiple player props above 60%
multiple underdog-output overs
```

Warning text:

```text
You may be overexposed to the same match script.
```

## Implementation Order

1. Preserve current PR #2.
2. Merge PR #2 after manual browser test.
3. Open PR #3 for Live Prediction Mode.
4. PR #3 should add prediction_assistant.py and /sessions/<id>/live.
5. Keep existing CSV workbench as fallback/debug mode.
6. Do not remove existing functionality.
