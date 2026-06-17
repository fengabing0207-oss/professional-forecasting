"""
Data layer.

Loads the public `international_results` dataset (martj42/international_results,
results from 1872 onward) and prepares a clean, recent, weighted training frame.

Output frame columns: home, away, hg (home goals), ag (away goals),
date, neutral (bool), weight (time-decay weight, filled by the model layer).
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def filter_min_matches(df: pd.DataFrame, min_matches_per_team: int) -> pd.DataFrame:
    """Restrict to teams with enough games to estimate reliably.

    Iteratively drops low-sample teams (dropping one team's matches can push
    another below the threshold). A pure function of the frame it is handed, so
    the rolling backtest can call it on a *cutoff-restricted* fold to build the
    team universe without ever seeing future matches (no selection-time leak).
    """
    if min_matches_per_team <= 0:
        return df.reset_index(drop=True)
    df = df.copy()
    while True:
        counts = pd.concat([df["home"], df["away"]]).value_counts()
        weak = set(counts[counts < min_matches_per_team].index)
        if not weak:
            break
        df = df[~df["home"].isin(weak) & ~df["away"].isin(weak)].copy()
    return df.reset_index(drop=True)


def load_matches(
    path: str,
    since: str = "2021-01-01",
    drop_friendlies: bool = False,
    min_matches_per_team: int = 15,
) -> pd.DataFrame:
    """Load, clean, and filter the raw results CSV into a model-ready frame.

    Parameters
    ----------
    since : keep only matches on/after this date (recency: old form is noise).
    drop_friendlies : if True, exclude friendlies (competitive form only).
    min_matches_per_team : drop teams with too few games to estimate reliably,
        applied over the *whole* loaded span. The rolling backtest passes 0 here
        and re-derives the team universe per cutoff (leak-free) via
        ``filter_min_matches`` instead.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])           # unplayed -> NA
    df = df[df["date"] >= pd.Timestamp(since)].copy()

    if drop_friendlies:
        df = df[df["tournament"] != "Friendly"].copy()

    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df = df.rename(columns={
        "home_team": "home", "away_team": "away",
        "home_score": "hg", "away_score": "ag",
    })
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)

    df = filter_min_matches(df, min_matches_per_team)
    return df[["home", "away", "hg", "ag", "date", "neutral", "tournament"]]


def add_time_weights(df: pd.DataFrame, half_life_days: float = 540.0,
                     as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Dixon-Coles style exponential time decay.

    A match `half_life_days` old counts half as much as a match today.
    Default ~1.5 years: recent international form dominates, ancient games fade.
    """
    as_of = as_of or df["date"].max()
    age_days = (as_of - df["date"]).dt.days.clip(lower=0)
    xi = np.log(2) / half_life_days
    df = df.copy()
    df["weight"] = np.exp(-xi * age_days)
    return df
