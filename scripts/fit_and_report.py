"""
Fit the Dixon-Coles model on real international results and report.

What this demonstrates
----------------------
1. Learned attack/defense ratings for every team (data, not hand-tuned).
2. For the fixtures discussed during development, the DATA-LEARNED lambda vs
   the HAND-FILLED lambda used in the quick prototype -- the whole point of
   "fitting lambda" is to replace subjective guesses with estimates.
3. A worked W/D/L + scoreline prediction from the fitted model.

Run:  python scripts/fit_and_report.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from data import load_matches, add_time_weights
from dixoncoles import DixonColes

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "results.csv")

# fixtures from tonight, with the hand-filled lambdas used in the prototype
# and actual results where known.
HAND = {
    ("Norway", "Iraq"):      {"lh": 2.10, "la": 0.55, "actual": (3, 1)},
    ("Argentina", "Algeria"):{"lh": 1.85, "la": 0.85, "actual": None},
    ("Spain", "Cape Verde"): {"lh": 2.00, "la": 0.45, "actual": (0, 0)},
    ("Belgium", "Egypt"):    {"lh": 1.60, "la": 0.85, "actual": (1, 1)},
}


def main():
    df = load_matches(DATA, since="2021-01-01", min_matches_per_team=15)
    df = add_time_weights(df, half_life_days=540)
    print(f"Training on {len(df):,} matches, {df['home'].nunique()} home / "
          f"{pd.concat([df['home'], df['away']]).nunique()} total teams, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    model = DixonColes().fit(df)
    print(f"Converged: {model.fit_result_.success} | "
          f"home_adv={model.home_adv_:.3f} | rho={model.rho_:.3f}\n")

    tbl = model.ratings_table()
    print("=== Top 12 attack ratings (learned) ===")
    print(tbl.head(12).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\n=== Best 12 defenses (lowest concede; higher defense = better) ===")
    print(tbl.sort_values("defense", ascending=False).head(12)
          .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print("\n" + "=" * 70)
    print("LEARNED lambda  vs  HAND-FILLED lambda   (neutral venue, World Cup)")
    print("=" * 70)
    for (h, a), info in HAND.items():
        if h not in model.idx_ or a not in model.idx_:
            print(f"  {h} vs {a}: team not in filtered training set, skipped")
            continue
        pred = model.predict(h, a, neutral=True)
        print(f"\n{h} vs {a}")
        print(f"  hand-filled : lam_{h[:3]}={info['lh']:.2f}  lam_{a[:3]}={info['la']:.2f}")
        print(f"  LEARNED     : lam_{h[:3]}={pred['lambda_home']:.2f}  "
              f"lam_{a[:3]}={pred['lambda_away']:.2f}")
        print(f"  fitted W/D/L: {h} {pred['p_home']*100:.1f}% | "
              f"Draw {pred['p_draw']*100:.1f}% | {a} {pred['p_away']*100:.1f}%  "
              f"| over2.5 {pred['over_2_5']*100:.1f}%")
        ts = ", ".join(f"{i}-{j} {p*100:.0f}%" for (i, j), p in pred["top_scores"][:4])
        print(f"  top scores  : {ts}")
        if info["actual"]:
            print(f"  ACTUAL      : {info['actual'][0]}-{info['actual'][1]}")


if __name__ == "__main__":
    main()
