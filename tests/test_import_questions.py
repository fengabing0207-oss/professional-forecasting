import os
import tempfile

import pandas as pd

from cup.import_questions import import_raw_text, parse_raw_question


def _write_tmp(text):
    fh = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    try:
        fh.write(text)
        return fh.name
    finally:
        fh.close()


def test_raw_text_importer_maps_obvious_team_win_question():
    path = _write_tmp("Q1: Will Norway win the match?\n")
    try:
        out = import_raw_text(path, "NOR_SEN", "", "Norway", "Senegal")
    finally:
        os.unlink(path)
    row = out.iloc[0]
    assert row["question_id"] == "q1"
    assert row["event_type"] == "team_win"
    assert row["selection"] == "Norway"
    assert row["status"] == "parsed"
    assert pd.isna(row["p_manual"]) or row["p_manual"] == ""


def test_raw_text_importer_maps_halftime_draw_as_market_manual_only():
    parsed = parse_raw_question("Will the match be tied at halftime?", "Norway", "Senegal")
    assert parsed.event_type == "halftime_draw"
    assert parsed.selection == "draw_halftime"
    assert parsed.status == "parsed"
    assert "market/manual-only" in parsed.notes


def test_raw_text_importer_maps_corners_threshold_as_market_only_prop():
    parsed = parse_raw_question("Will there be 9+ total corners?", "Norway", "Senegal")
    assert parsed.event_type == "corners_threshold"
    assert parsed.selection == "total"
    assert parsed.threshold == 9.0
    assert parsed.status == "parsed"
    assert "market/manual-only" in parsed.notes


def test_ambiguous_question_becomes_needs_review():
    parsed = parse_raw_question("Will the match get weird?", "Norway", "Senegal")
    assert parsed.event_type == "unsupported_market_only"
    assert parsed.status == "needs_review"


def test_importer_does_not_create_probabilities():
    path = _write_tmp("Will Norway win the match?\nWill there be 9+ total corners?\n")
    try:
        out = import_raw_text(path, "NOR_SEN", "", "Norway", "Senegal")
    finally:
        os.unlink(path)
    assert out["p_manual"].isna().all() or (out["p_manual"] == "").all()
    assert "p_market" not in out.columns
    assert "p_final" not in out.columns
