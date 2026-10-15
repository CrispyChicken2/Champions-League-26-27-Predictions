import pandas as pd
import pytest

from clpred.player_ratings import Ratings, fit, stretches


def player(pid, minutes, sub=False):
    return {"player_id": pid, "player": pid, "position": "Sub" if sub else "FW", "time": str(minutes)}


def shot(minute, xg, side):
    return {"minute": str(minute), "xG": str(xg), "h_a": side}


MATCH = {
    "rosters": {"h": [player("h1", 90), player("h2", 60), player("h3", 30, sub=True)],
                "a": [player("a1", 90)]},
    "shots": [shot(10, 0.3, "h"), shot(70, 0.5, "h"), shot(93, 0.1, "a")],
}


def test_a_substitution_splits_the_match_and_shots_follow_their_minute():
    parts = stretches(MATCH)
    assert [p["minutes"] for p in parts] == [60, 30]
    assert parts[0]["home_players"] == ["h1", "h2"] and parts[1]["home_players"] == ["h1", "h3"]
    assert parts[0]["home_xg"] == pytest.approx(0.3) and parts[1]["home_xg"] == pytest.approx(0.5)
    assert parts[1]["away_xg"] == pytest.approx(0.1)  # stoppage time counts in the last stretch


def test_the_player_on_the_pitch_when_chances_come_gets_the_credit():
    rows = []
    for k in range(40):
        # "star" plays the first half of each match, "bench" the second; chances only come in the first half.
        match = {"rosters": {"h": [player("star", 45), player("bench", 45, sub=True)], "a": [player("keeper", 90)]},
                 "shots": [shot(20, 1.0, "h"), shot(80, 0.1, "h")]}
        rows.append({"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=k), "stretches": stretches(match)})
    ratings = fit(pd.DataFrame(rows), pd.Timestamp("2024-06-01"))
    assert ratings.lineup(["star"])[0] > ratings.lineup(["bench"])[0]


def test_unknown_players_count_as_average():
    import numpy as np
    ratings = Ratings({"a": 0}, np.array([0.2]), np.array([0.1]), 0.3, 0.2)
    assert ratings.lineup(["a", "stranger"]) == (pytest.approx(0.2), pytest.approx(0.1))
