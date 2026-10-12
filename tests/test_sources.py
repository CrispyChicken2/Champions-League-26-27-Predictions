"""The page readers, checked against small samples shaped like the saved pages."""

import json

import clpred.recent_results as recent_results
from clpred.snapshot import BOOKMAKER_ROW, CORRECT_SCORE_ROW, HEADING, OVER_UNDER_ROW, extract_absences, extract_squads

CLUBS = {"Roma": ["Roma", "AS Roma"]}
CAPTURED = {"captured_at_utc": "2026-10-12T16:00:00+00:00"}


def test_odds_rows_are_read():
    page = ("|Arsenal - Lille|Arsenal|-|Lille||Tomorrow,||13 Oct 2026,||21:00||1X2|Betclic.fr||CLAIM BONUS|\t|1.20|\t|6.25|\t|11.50|\t|92.6%"
            "||||bwin.fr||CLAIM BONUS|\t|1.20|\t|5.50|\t|9.25|\t|89.0%|")
    assert HEADING.search(page).groups() == ("Arsenal", "Lille")
    assert BOOKMAKER_ROW.findall(page) == [("1.20", "6.25", "11.50"), ("1.20", "5.50", "9.25")]


def test_over_under_and_correct_score_rows_are_read():
    over_under = "Over/Under +2|1|\t||1.23||\t||3.70||\t92.3%||Over/Under +2.5|4|\t||1.52||\t||2.28||\t90.5%||"
    assert OVER_UNDER_ROW.search(over_under).groups() == ("1.52", "2.28")
    correct_score = "|21:00||1X2|Correct Score|1:0|1|\t||6.30||||2:0|1|\t||5.00||||5:0|0|\t||-||||0:0|1|\t||16.00||||"
    assert CORRECT_SCORE_ROW.findall(correct_score) == [("1", "0", "6.30"), ("2", "0", "5.00"), ("0", "0", "16.00")]


def test_squad_page_is_read_including_injured_players(tmp_path):
    # An injured player's details wrap onto two lines.
    text = ("99\t\n\tMile Svilar\xa0\nGoalkeeper\n\t27/08/1999 (27)\t\n\t1,89m\tright\t01/07/2022\t\t30/06/2030\t€35.00m\n"
            "-\t\n\tDevis Vásquez\nGoalkeeper\n\t12/05/1998 (28)\t\t1,95m\tright\t29/07/2025\t\t30/06/2027\t€800k\n")
    (tmp_path / "squad_as-rom.txt").write_text(text, "utf-8")
    manifest = {"squad_as-rom": {"title": "AS Roma - Detailed squad 26/27 | Transfermarkt"}}
    players = extract_squads(tmp_path, manifest, CLUBS)
    assert [(p["player"], p["position"], p["value_m"]) for p in players] == [
        ("Mile Svilar", "Goalkeeper", 35.0), ("Devis Vásquez", "Goalkeeper", 0.8)]


def test_absence_page_is_read_and_stops_at_risk_of_suspension(tmp_path):
    text = ("Player\tAge\tReason\tsince\tExpected return\tMissed matches\nInjuries\n\tMile Svilar\nGoalkeeper\n"
            "\t27\tBroken finger\t09/10/2026\t09/11/2026\t9\nRISK OF SUSPENSION\n\tSomeone Else\nCentre-Back\n\t25\tYellow cards\t\t\t0\n")
    (tmp_path / "absences_as-rom.txt").write_text(text, "utf-8")
    manifest = {"absences_as-rom": {"title": "AS Roma - Suspensions and injuries | Transfermarkt", **CAPTURED}}
    rows = extract_absences(tmp_path, manifest, CLUBS)
    assert len(rows) == 1
    assert (rows[0]["club"], rows[0]["player"], rows[0]["kind"], rows[0]["expected_return"]) == (
        "Roma", "Mile Svilar", "Injuries", "09/11/2026")


def test_results_page_dates_including_yesterday(tmp_path, monkeypatch):
    text = "6. ROUND\n1\nX\n2\n \nLiverpool - Manchester City\n0:3\n3.54\n3.69\n2.02\nYesterday\nArsenal - Leeds\n2:1\n1.44\n4.52\n7.51\n10.10.\n"
    (tmp_path / "E0.txt").write_text(text, "utf-8")
    (tmp_path / "CZE_2024-2025_abc.txt").write_text("Sparta Prague - Plzen\n1:0\n1.80\n3.50\n4.20\n15.02.\n", "utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"E0": CAPTURED, "CZE_2024-2025_abc": CAPTURED}), "utf-8")
    monkeypatch.setattr(recent_results, "SOURCE", tmp_path)
    rows = {(r["home"], r["away"]): r for r in recent_results.read_pages()}
    assert rows[("Liverpool", "Manchester City")]["date"] == "2026-10-11"
    assert rows[("Arsenal", "Leeds")]["date"] == "2026-10-10"
    assert rows[("Sparta Prague", "Plzen")]["date"] == "2025-02-15"
    assert rows[("Liverpool", "Manchester City")]["away_goals"] == 3


def test_recent_rows_skip_matches_already_in_the_history(tmp_path, monkeypatch):
    text = "Arsenal - Leeds\n2:1\n1.44\n4.52\n7.51\n10.10.\nFulham - Leeds\n1:1\n2.0\n3.0\n4.0\n20.09.\n"
    (tmp_path / "E0.txt").write_text(text, "utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"E0": CAPTURED}), "utf-8")
    monkeypatch.setattr(recent_results, "SOURCE", tmp_path)
    history = [{"competition": "E0", "date": "2026-09-20", "home": "Fulham", "away": "Leeds"}]
    name_map = [{"league": "E0", "source_name": n, "uefa_name": ""} for n in ("Arsenal", "Leeds", "Fulham")]
    added, notes = recent_results.recent_rows(history, name_map, {})
    assert [(r["date"], r["home"], r["away"]) for r in added] == [("2026-10-10", "Arsenal", "Leeds")]
    assert notes == []
