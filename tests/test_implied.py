import numpy as np
import pandas as pd
import pytest

from clpred.implied import implied_goals
from clpred.model import outcome_probabilities, score_matrix
from clpred.squad_value import best_eleven_value, parse_squad

DETAILS = "\t01/01/1995 (31)\t\t1,90m\tright\t01/07/2022\t\t30/06/2030\t"


def test_implied_goals_recover_the_rates_behind_the_odds():
    home, draw, away = outcome_probabilities(score_matrix(1.8, 0.9, rho=0.0))
    margin = 1.05
    odds = [pd.Series([1 / (p * margin)]) for p in (home, draw, away)]
    implied_home, implied_away = implied_goals(*odds)
    assert implied_home[0] == pytest.approx(1.8, abs=0.05)
    assert implied_away[0] == pytest.approx(0.9, abs=0.05)


def test_missing_odds_give_no_implied_goals():
    implied_home, implied_away = implied_goals(pd.Series([np.nan]), pd.Series([3.4]), pd.Series([2.9]))
    assert np.isnan(implied_home[0]) and np.isnan(implied_away[0])


def test_best_eleven_takes_one_keeper_and_ten_outfield_players():
    text = "".join(f"1\t\n\tKeeper {k}\nGoalkeeper\n{DETAILS}€{value}.00m\n" for k, value in enumerate((20, 5)))
    text += "".join(f"2\t\n\tPlayer {k}\nCentre-Back\n{DETAILS}€{value}.00m\n" for k, value in enumerate(range(1, 13)))
    squad = parse_squad(text)
    assert len(squad) == 14
    assert best_eleven_value(squad) == 20 + sum(range(3, 13))
