"""Produce the Predictions of a Matchday from its Snapshot and earlier history.

Usage: uv run python -m clpred.predict <matchday number>

Writes matchday-NN/predictions.csv (one row per match) and
matchday-NN/score_distributions.csv (probability of every scoreline).
Only matches played before the Matchday's first kick-off are used. A match
with a missing input is still predicted and the gap is named in `degraded`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from clpred.model import EUROPEAN, Settings, fit, market_distribution, outcome_probabilities, pick, score_matrix

# Share of the model in the blend with the market, tuned on the replay.
MODEL_SHARE = 0.4
SHOWN_GOALS = 6


def main() -> None:
    matchday = int(sys.argv[1])
    folder = Path(f"matchday-{matchday:02d}")
    matches = pd.read_csv("data/matches.csv", low_memory=False)
    fixtures = pd.read_csv("data/fixtures.csv")
    fixtures = fixtures[fixtures.matchday == matchday].sort_values(["kickoff_utc", "home"])
    odds = pd.read_csv(folder / "odds.csv").set_index(["home", "away"]) if (folder / "odds.csv").exists() else None

    settings = Settings()
    as_of = fixtures.date.min()
    history = matches[matches.date < as_of]
    model = fit(history, as_of, settings)
    domestic_clubs = set(history[~history.competition.isin(EUROPEAN)].home)

    rows, distributions = [], []
    for fixture in fixtures.itertuples():
        degraded = []
        for club in (fixture.home, fixture.away):
            if club not in domestic_clubs:
                degraded.append(f"no domestic history for {club}")
        home_rate, away_rate = model.expected_goals(fixture.home, fixture.away)
        model_matrix = score_matrix(home_rate, away_rate, settings.low_score_rho)

        market_matrix = None
        key = (fixture.home, fixture.away)
        if odds is not None and key in odds.index and pd.notna(odds.loc[key].get("odds_home")):
            price = odds.loc[key]
            over, under = price.get("odds_over_2_5"), price.get("odds_under_2_5")
            if pd.isna(over) or pd.isna(under):
                over = under = None
                degraded.append("no over/under odds")
            market_matrix = market_distribution(price.odds_home, price.odds_draw, price.odds_away, over, under,
                                                rho=settings.low_score_rho)
            blend = MODEL_SHARE * model_matrix + (1 - MODEL_SHARE) * market_matrix
        else:
            degraded.append("no odds: model only, no Baseline")
            blend = model_matrix

        picked = pick(blend)
        likeliest = np.unravel_index(np.argmax(blend), blend.shape)
        home_win, draw, away_win = outcome_probabilities(blend)
        outcome_confidence = home_win if picked[0] > picked[1] else draw if picked[0] == picked[1] else away_win
        top = sorted(((blend[i, j], i, j) for i in range(SHOWN_GOALS + 1) for j in range(SHOWN_GOALS + 1)), reverse=True)[:3]
        goals = np.arange(blend.shape[0])
        row = {
            "matchday": matchday, "date": fixture.date, "kickoff_utc": fixture.kickoff_utc,
            "home": fixture.home, "away": fixture.away,
            "pick": f"{picked[0]}-{picked[1]}",
            "confidence_pick": round(float(blend[picked]), 4),
            "confidence_outcome": round(outcome_confidence, 4),
            "prob_home": round(home_win, 4), "prob_draw": round(draw, 4), "prob_away": round(away_win, 4),
            "most_likely_score": f"{likeliest[0]}-{likeliest[1]}",
            "top_scores": "; ".join(f"{i}-{j} {p:.1%}" for p, i, j in top),
            "expected_home_goals": round(float(blend.sum(axis=1) @ goals), 2),
            "expected_away_goals": round(float(blend.sum(axis=0) @ goals), 2),
            "model_home_goals": round(home_rate, 2), "model_away_goals": round(away_rate, 2),
            "model_pick": "{}-{}".format(*pick(model_matrix)),
            "model_prob_home": round(outcome_probabilities(model_matrix)[0], 4),
            "model_prob_draw": round(outcome_probabilities(model_matrix)[1], 4),
            "model_prob_away": round(outcome_probabilities(model_matrix)[2], 4),
            "home_venue_effect": round(float(model.venue[model.clubs[fixture.home]]), 3),
            "degraded": "; ".join(degraded),
        }
        if market_matrix is not None:
            market = outcome_probabilities(market_matrix)
            row.update(baseline_pick="{}-{}".format(*pick(market_matrix)),
                       baseline_prob_home=round(market[0], 4), baseline_prob_draw=round(market[1], 4),
                       baseline_prob_away=round(market[2], 4),
                       odds_captured_at_utc=odds.loc[key].captured_at_utc)
        rows.append(row)
        for i in range(blend.shape[0]):
            for j in range(blend.shape[1]):
                distributions.append({"home": fixture.home, "away": fixture.away, "home_goals": i, "away_goals": j,
                                      "probability": round(float(blend[i, j]), 6)})

    pd.DataFrame(rows).to_csv(folder / "predictions.csv", index=False)
    pd.DataFrame(distributions).to_csv(folder / "score_distributions.csv", index=False)
    print(f"{len(rows)} Predictions for Matchday {matchday}, history up to {history.date.max()}, "
          f"{sum(bool(r['degraded']) for r in rows)} degraded")


if __name__ == "__main__":
    main()
