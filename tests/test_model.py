import numpy as np
import pandas as pd
import pytest

from clpred.model import Settings, fit, market_distribution, outcome_probabilities, pick, points, score_matrix


def test_score_distribution_is_a_probability_distribution():
    matrix = score_matrix(1.6, 1.1, rho=-0.05)
    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()
    assert sum(outcome_probabilities(matrix)) == pytest.approx(1.0)


def test_stronger_home_side_is_favoured():
    home, draw, away = outcome_probabilities(score_matrix(2.4, 0.7, rho=0.0))
    assert home > draw > away


def test_pick_follows_expected_points_not_the_likeliest_score():
    # 1-1 is the likeliest single score, but a home win is the likelier outcome.
    matrix = np.zeros((11, 11))
    matrix[1, 1], matrix[1, 0], matrix[2, 1], matrix[2, 0], matrix[0, 1] = 0.30, 0.25, 0.20, 0.15, 0.10
    assert np.unravel_index(matrix.argmax(), matrix.shape) == (1, 1)
    assert pick(matrix) == (1, 0)


def test_points_rule():
    assert points((2, 1), (2, 1)) == 3
    assert points((2, 1), (1, 0)) == 1
    assert points((1, 1), (0, 0)) == 1
    assert points((2, 1), (1, 1)) == 0
    assert points((0, 2), (1, 0)) == 0


def test_baseline_reproduces_the_odds_it_was_built_from():
    truth = score_matrix(1.9, 1.0, rho=0.0)
    home, draw, away = outcome_probabilities(truth)
    totals = np.add.outer(np.arange(11), np.arange(11))
    over = truth[totals > 2.5].sum()
    margin = 1.06  # a bookmaker's margin must not change the result
    rebuilt = market_distribution(1 / (home * margin), 1 / (draw * margin), 1 / (away * margin),
                                  1 / (over * margin), 1 / ((1 - over) * margin), rho=0.0)
    assert outcome_probabilities(rebuilt) == pytest.approx((home, draw, away), abs=0.005)
    assert rebuilt[totals > 2.5].sum() == pytest.approx(over, abs=0.005)


def league(seed: int = 0, rounds: int = 12) -> pd.DataFrame:
    """A made-up league in which 'Strong' scores a lot and 'Weak' concedes a lot."""
    rng = np.random.default_rng(seed)
    rate = {"Strong": 2.6, "Middle": 1.4, "Other": 1.3, "Weak": 0.6}
    rows, day = [], pd.Timestamp("2025-08-01")
    for _ in range(rounds):
        for home in rate:
            for away in rate:
                if home != away:
                    day += pd.Timedelta(days=1)
                    rows.append({"date": day.date().isoformat(), "competition": "E0", "home": home, "away": away,
                                 "home_goals": rng.poisson(rate[home] * 1.2 * 1.4 / rate[away] ** 0.5),
                                 "away_goals": rng.poisson(rate[away] * 1.4 / rate[home] ** 0.5),
                                 "home_shots_on_target": np.nan, "away_shots_on_target": np.nan})
    return pd.DataFrame(rows)


def test_fit_ranks_clubs_by_strength_and_finds_home_advantage():
    model = fit(league(), "2026-06-01", Settings())
    strength = {club: model.attack[k] + model.defence[k] for club, k in model.clubs.items()}
    assert strength["Strong"] > strength["Middle"] > strength["Weak"]
    assert model.home_advantage > 0
    home_goals, away_goals = model.expected_goals("Strong", "Weak", european=False)
    assert home_goals > away_goals


def test_fit_ignores_matches_on_or_after_the_cut_off():
    data = league()
    cut_off = "2025-12-01"
    future = data[data.date >= cut_off].assign(home_goals=9, away_goals=0)
    tampered = pd.concat([data[data.date < cut_off], future])
    honest = fit(data, cut_off, Settings())
    leaked = fit(tampered, cut_off, Settings())
    assert honest.expected_goals("Middle", "Other", european=False) == pytest.approx(
        leaked.expected_goals("Middle", "Other", european=False))


def test_champions_league_matches_weigh_more():
    # Upsets both home and away, so only the clubs' own strengths can explain them.
    blank = {"date": "2026-05-01", "competition": "ucl", "home_shots_on_target": np.nan, "away_shots_on_target": np.nan}
    upset = pd.DataFrame([{**blank, "home": "Weak", "away": "Strong", "home_goals": 5, "away_goals": 0}] * 3
                         + [{**blank, "home": "Strong", "away": "Weak", "home_goals": 0, "away_goals": 5}] * 3)
    both = pd.concat([league(rounds=4), upset])
    light = fit(both, "2026-06-01", Settings(ucl_weight=1.0)).expected_goals("Weak", "Strong")
    heavy = fit(both, "2026-06-01", Settings(ucl_weight=4.0)).expected_goals("Weak", "Strong")
    assert heavy[0] > light[0]
