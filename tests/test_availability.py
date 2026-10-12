from datetime import date

import numpy as np
import pandas as pd
import pytest

from clpred.availability import MAX_ADJUSTMENT, absent_players, adjustment, best_eleven_value, plain, value_loss


def squad() -> pd.DataFrame:
    players = [("Keeper One", "Goalkeeper", 20.0), ("Keeper Two", "Goalkeeper", 2.0)]
    players += [(f"Starter {k}", "Centre-Back", 10.0) for k in range(9)]
    players += [("Star Forward", "Centre-Forward", 100.0), ("Reserve", "Left Winger", 4.0), ("Youth", "Right-Back", 1.0)]
    return pd.DataFrame(players, columns=["player", "position", "value_m"]).assign(club="Club")


def absences(*rows) -> pd.DataFrame:
    columns = ["club", "player", "position", "kind", "reason", "since", "expected_return"]
    return pd.DataFrame([("Club", player, "", kind, reason, "", back) for player, kind, reason, back in rows], columns=columns)


def test_best_eleven_is_one_keeper_and_ten_outfield_players():
    assert best_eleven_value(squad()) == pytest.approx(20 + 9 * 10 + 100)


def test_losing_a_fringe_player_costs_nothing():
    assert value_loss(squad(), {plain("Youth")}) == 0.0


def test_losing_the_star_costs_the_gap_to_his_replacement():
    full, without = 210.0, 20 + 9 * 10 + 4
    assert value_loss(squad(), {plain("Star Forward")}) == pytest.approx(np.log(without / full))


def test_losing_the_keeper_brings_in_the_second_keeper():
    full, without = 210.0, 2 + 9 * 10 + 100
    assert value_loss(squad(), {plain("Keeper One")}) == pytest.approx(np.log(without / full))


def test_player_back_before_the_match_is_not_absent():
    table = absences(("Star Forward", "Injuries", "Knock", "10/10/2026"), ("Reserve", "Injuries", "Knock", "20/10/2026"))
    assert absent_players(table, "Club", date(2026, 10, 13)) == {plain("Reserve")}


def test_only_competition_wide_suspensions_count():
    table = absences(("Star Forward", "Suspensions", "Red card suspension", ""), ("Reserve", "Suspensions", "No eligibility", ""))
    assert absent_players(table, "Club", date(2026, 10, 13)) == {plain("Reserve")}


def test_adjustment_is_negative_capped_and_names_the_players():
    table = absences(("Star Forward", "Injuries", "Knee", ""))
    change, counted = adjustment(squad(), table, "Club", date(2026, 10, 13), slope=0.4)
    assert -MAX_ADJUSTMENT <= change < 0 and counted == ["Star Forward"]
    capped, _ = adjustment(squad(), table, "Club", date(2026, 10, 13), slope=50.0)
    assert capped == -MAX_ADJUSTMENT


def test_unknown_club_gets_no_adjustment():
    assert adjustment(squad(), absences(), "Elsewhere", date(2026, 10, 13), slope=0.4) == (0.0, [])
