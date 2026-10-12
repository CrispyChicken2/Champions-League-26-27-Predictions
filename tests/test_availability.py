from datetime import date

import numpy as np
import pandas as pd
import pytest

from clpred.availability import (MAX_ADJUSTMENT, PLAYER_THRESHOLD, absent_players, centred, fit_effects, historical_missing, missing, plain,
                                 rate_changes, same_player, squad_importance)
from clpred.players import importance, parse_page


def season_table() -> pd.DataFrame:
    """A club whose striker produces most of the goals and whose back line always plays."""
    rows = [("Keeper", "Goalkeeper", 40, 40, 0, 0, 3600), ("Reserve Keeper", "Goalkeeper", 40, 0, 0, 0, 0)]
    rows += [(f"Defender {k}", "Centre-Back", 40, 40, 1, 0, 3600) for k in range(4)]
    rows += [("Holder", "Defensive Midfield", 40, 40, 1, 2, 3600)]
    rows += [(f"Midfielder {k}", "Central Midfield", 40, 40, 3, 4, 3600) for k in range(3)]
    rows += [("Winger", "Left Winger", 40, 40, 8, 8, 3600), ("Star", "Centre-Forward", 40, 40, 36, 8, 3600),
             ("Substitute", "Centre-Forward", 40, 20, 2, 0, 400)]
    columns = ["player", "position", "in_squad", "appearances", "goals", "assists", "minutes"]
    return pd.DataFrame(rows, columns=columns).assign(club="Club", season=2025)


def absences(*rows) -> pd.DataFrame:
    columns = ["club", "player", "position", "kind", "reason", "since", "expected_return"]
    return pd.DataFrame([("Club", player, "", kind, reason, "", back) for player, kind, reason, back in rows], columns=columns)


def test_performance_page_rows_are_read_on_one_or_two_lines():
    text = ("1\t\n\tThibaut Courtois\nGoalkeeper\n\t33\t\t45\t45\t-\t2\t1\t-\t-\t-\t1\t2.27\t4.005'\n"
            "10\t\n\tKylian Mbappé\nCentre-Forward\n\t26\t\n\t46\t44\t42\t7\t5\t-\t-\t4\t10\t2.16\t3.621'\n"
            "40\t\n\tUnused Player\nCentre-Back\n\t19\t\tNot used during this season\n")
    players = {p["player"]: p for p in parse_page(text)}
    assert set(players) == {"Thibaut Courtois", "Kylian Mbappé"}
    assert (players["Kylian Mbappé"]["goals"], players["Kylian Mbappé"]["assists"], players["Kylian Mbappé"]["minutes"]) == (42, 7, 3621)
    assert players["Thibaut Courtois"]["appearances"] == 45


def test_the_star_scorer_is_the_most_important_attacker():
    table = importance(season_table()).set_index("player")
    assert table.attack.idxmax() == "Star"
    assert table.attack["Star"] > 3 * table.attack["Midfielder 0"]
    assert table.attack["Keeper"] == 0


def test_defence_importance_goes_to_defensive_regulars_only():
    table = importance(season_table()).set_index("player")
    assert table.defence["Star"] == 0
    assert table.defence["Keeper"] == pytest.approx(table.defence["Defender 0"])
    assert table.defence["Reserve Keeper"] == 0
    assert table.defence.nlargest(6).sum() == pytest.approx(1.0)


def test_missing_sums_the_absent_players():
    table = importance(season_table())
    attack, defence, counted = missing(table, {plain("Star"), plain("Keeper")})
    assert attack == pytest.approx(table.set_index("player").attack["Star"])
    assert defence == pytest.approx(1 / 6)
    assert counted[0] == "Star"
    assert missing(table, set()) == (0.0, 0.0, [])


def test_player_back_before_the_match_is_not_absent():
    table = absences(("Star", "Injuries", "Knock", "10/10/2026"), ("Winger", "Injuries", "Knock", "20/10/2026"))
    assert absent_players(table, "Club", date(2026, 10, 13)) == {plain("Winger")}


def test_only_competition_wide_suspensions_count():
    table = absences(("Star", "Suspensions", "Red card suspension", ""), ("Winger", "Suspensions", "No eligibility", ""))
    assert absent_players(table, "Club", date(2026, 10, 13)) == {plain("Winger")}


def test_missing_attack_lowers_own_goals_and_missing_defence_raises_the_opponents():
    home, away = rate_changes((0.3, 0.0), (0.0, 0.0), attack_effect=0.4, defence_effect=0.4)
    assert home == pytest.approx(-0.12) and away == 0
    home, away = rate_changes((0.0, 0.5), (0.0, 0.0), attack_effect=0.4, defence_effect=0.4)
    assert home == 0 and away == pytest.approx(0.2)


def test_a_club_less_depleted_than_usual_gains():
    home, _ = rate_changes((-0.1, 0.0), (0.0, 0.0), attack_effect=0.4, defence_effect=0.4)
    assert home > 0


def test_rate_changes_are_capped():
    home, away = rate_changes((5.0, 5.0), (0.0, 0.0), attack_effect=0.4, defence_effect=0.4)
    assert home == -MAX_ADJUSTMENT and away == MAX_ADJUSTMENT


def test_player_spellings_are_matched():
    assert same_player("Gleison Bremer", "Bremer") >= 0.75
    assert same_player("Vinicius Junior", "Vinícius Júnior") == 1.0
    assert same_player("Kylian Mbappé", "Ethan Mbappé") < PLAYER_THRESHOLD


def test_squad_importance_only_counts_players_still_at_the_club():
    squad = pd.DataFrame({"club": "Club", "player": [p for p in season_table().player if p != "Star"]})
    table = squad_importance(season_table(), squad, "Club", (2025,))
    assert "Star" not in set(table.player)
    assert squad_importance(season_table(), squad, "Elsewhere", (2025,)).empty


def test_history_marks_a_player_missing_from_the_matchday_squad():
    names = [p for p in season_table().player if p != "Reserve Keeper"]
    rows = []
    for match_id, day, absent in ((1, "2025-09-16", None), (2, "2025-10-01", "Star"), (3, "2025-10-21", None)):
        rows += [{"match_id": match_id, "date": day, "season": 2026, "club": "Club", "player": n, "role": "start"}
                 for n in names if n != absent]
    result = historical_missing(pd.DataFrame(rows), season_table()).set_index("match_id")
    assert result.attack_missing[1] == 0 and result.attack_missing[3] == 0
    assert result.attack_missing[2] > 0.3


def test_fit_recovers_an_effect_planted_in_made_up_matches():
    rng = np.random.default_rng(1)
    n = 6000
    table = pd.DataFrame({"home_rate": 1.6, "away_rate": 1.2,
                          "home_attack_missing": rng.uniform(0, 0.6, n), "away_attack_missing": rng.uniform(0, 0.6, n),
                          "home_defence_missing": rng.uniform(0, 0.6, n), "away_defence_missing": rng.uniform(0, 0.6, n)})
    table, _ = centred(table)
    table["home_goals"] = rng.poisson(1.6 * np.exp(-0.5 * table.home_attack_missing + 0.3 * table.away_defence_missing))
    table["away_goals"] = rng.poisson(1.2 * np.exp(-0.5 * table.away_attack_missing + 0.3 * table.home_defence_missing))
    level, attack, defence = fit_effects(table)
    assert attack == pytest.approx(0.5, abs=0.12) and defence == pytest.approx(0.3, abs=0.12)
    assert abs(level) < 0.05
