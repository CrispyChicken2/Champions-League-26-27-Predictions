"""Build one clean match table from the downloaded history.

Usage: uv run python -m clpred.build_matches

Reads data/history/, writes:
  data/matches.csv      every finished match, domestic and European
  data/fixtures.csv     unplayed 2026/27 Champions League league-phase matches
  data/name_map.csv     how each domestic club name was matched to UEFA's name
"""

import csv
import glob
import io
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from clpred.implied import implied_goals
from clpred.names import best_match
from clpred.recent_results import recent_rows

HISTORY = Path("data/history")
OUT = Path("data")
FIRST_DATE = "2021-07-01"
MATCH_THRESHOLD = 0.82
XG_NAME_THRESHOLD = 0.72
# Domestic clubs that must never take a UEFA name: Paris FC is not Paris Saint-Germain.
NEVER_MATCH = {"Paris FC"}

LEAGUE_COUNTRY = {
    "E0": "ENG", "D1": "GER", "I1": "ITA", "SP1": "ESP", "F1": "FRA", "N1": "NED",
    "B1": "BEL", "P1": "POR", "T1": "TUR", "G1": "GRE", "AUT": "AUT", "NOR": "NOR",
}


def parse_date(text: str) -> str:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unreadable date {text!r}")


def number(row: dict, *keys: str) -> float | None:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            try:
                return float(value)
            except ValueError:
                continue
    return None


def read_european() -> tuple[list[dict], list[dict], dict[str, dict]]:
    played, fixtures, clubs = [], [], {}
    for path in sorted(glob.glob(str(HISTORY / "european_*.json"))):
        competition = Path(path).stem.split("_")[1]
        for match in json.load(io.open(path, encoding="utf-8")):
            home, away = match["homeTeam"], match["awayTeam"]
            if home.get("isPlaceHolder") or away.get("isPlaceHolder"):
                continue
            for team in (home, away):
                names = team.get("translations", {})
                clubs[team["internationalName"]] = {
                    "country": team.get("countryCode"),
                    "spellings": [
                        team["internationalName"],
                        names.get("displayOfficialName", {}).get("EN", team["internationalName"]),
                        names.get("displayName", {}).get("EN", team["internationalName"]),
                    ],
                }
            row = {
                "date": match["kickOffTime"]["date"],
                "competition": competition,
                "stage": match.get("round", {}).get("metaData", {}).get("name", ""),
                "matchday": (match.get("matchday") or {}).get("sequenceNumber", ""),
                "season": match["seasonYear"],
                "home": home["internationalName"],
                "away": away["internationalName"],
                "kickoff_utc": match["kickOffTime"].get("dateTime", ""),
            }
            regular = (match.get("score") or {}).get("regular")
            if match["status"] == "FINISHED" and regular:
                # 90-minute score: extra time and penalties are excluded.
                played.append({**row, "home_goals": regular["home"], "away_goals": regular["away"]})
            elif competition == "ucl" and str(match["seasonYear"]) == "2027" and row["stage"] == "League Phase":
                fixtures.append(row)
    return played, fixtures, clubs


def read_domestic() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(HISTORY / "domestic_*.csv"))):
        league = Path(path).stem.split("_")[1]
        for row in csv.DictReader(io.open(path, encoding="utf-8-sig", errors="replace")):
            home = (row.get("HomeTeam") or row.get("Home") or "").strip()
            away = (row.get("AwayTeam") or row.get("Away") or "").strip()
            home_goals, away_goals = number(row, "FTHG", "HG"), number(row, "FTAG", "AG")
            if not home or not away or home_goals is None or away_goals is None:
                continue
            rows.append({
                "date": parse_date(row["Date"]),
                "competition": league,
                "stage": "",
                "matchday": "",
                "season": "",
                "home": home,
                "away": away,
                "kickoff_utc": "",
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
                "home_shots_on_target": number(row, "HST"),
                "away_shots_on_target": number(row, "AST"),
                "home_xg": number(row, "HxG"),
                "away_xg": number(row, "AxG"),
                "odds_home": number(row, "AvgCH", "AvgH", "B365CH", "B365H"),
                "odds_draw": number(row, "AvgCD", "AvgD", "B365CD", "B365D"),
                "odds_away": number(row, "AvgCA", "AvgA", "B365CA", "B365A"),
            })
    return rows


def attach_xg(domestic: list[dict], name_map: list[dict]) -> None:
    """Fill home_xg / away_xg from Understat for the leagues it covers."""
    by_key = {(r["competition"], r["home"], r["away"], r["date"]): r for r in domestic}
    for path in sorted(glob.glob(str(HISTORY / "xg_*.json"))):
        league = Path(path).stem.split("_")[1]
        candidates: dict[str, list[str]] = {}
        for entry in name_map:
            if entry["league"] == league:
                canonical = entry["uefa_name"] or entry["source_name"]
                candidates.setdefault(canonical, [canonical]).append(entry["source_name"])
        names: dict[str, str | None] = {}
        paired = played = 0
        for match in json.load(io.open(path, encoding="utf-8")):
            if not match["isResult"]:
                continue
            played += 1
            sides = []
            for side in ("h", "a"):
                title = match[side]["title"]
                if title not in names:
                    found, score = best_match(title, candidates)
                    names[title] = found if score >= XG_NAME_THRESHOLD else None
                    if names[title] is None:
                        print(f"xG: unknown club {title!r} in {league} (closest {found!r}, {score:.2f})")
                sides.append(names[title])
            if None in sides:
                continue
            day = datetime.strptime(match["datetime"][:10], "%Y-%m-%d").date()
            # Kick-offs near midnight can fall on the neighbouring day in the other source.
            for shift in (0, -1, 1):
                row = by_key.get((league, sides[0], sides[1], (day + timedelta(days=shift)).isoformat()))
                if row is not None:
                    row["home_xg"], row["away_xg"] = float(match["xG"]["h"]), float(match["xG"]["a"])
                    paired += 1
                    break
        if paired < played:
            print(f"xG: {Path(path).name} paired {paired} of {played} played matches")


def main() -> None:
    european, fixtures, clubs = read_european()
    domestic = [r for r in read_domestic() if r["date"] >= FIRST_DATE]

    # Give each domestic club UEFA's name when it has played in Europe, so its
    # domestic and European matches belong to the same club.
    name_map = []
    renames: dict[tuple[str, str], str] = {}
    for league, country in LEAGUE_COUNTRY.items():
        candidates = {n: c["spellings"] for n, c in clubs.items() if c["country"] == country}
        names = sorted({r[side] for r in domestic if r["competition"] == league for side in ("home", "away")})
        for name in names:
            match, score = best_match(name, candidates)
            accepted = match is not None and score >= MATCH_THRESHOLD and name not in NEVER_MATCH
            renames[(league, name)] = match if accepted else name
            name_map.append({"league": league, "source_name": name, "uefa_name": match if accepted else "",
                             "best_candidate": match or "", "score": round(score, 2)})
    for row in domestic:
        row["home"] = renames[(row["competition"], row["home"])]
        row["away"] = renames[(row["competition"], row["away"])]

    # Latest rounds and uncovered leagues, read from saved results pages.
    recent, notes = recent_rows(domestic, name_map, clubs)
    for note in notes:
        print(note if note.startswith("CHECK") else "LEFT OUT " + note)
    print(f"{len(recent)} matches added from data/recent")
    domestic += recent

    attach_xg(domestic, name_map)

    matches = pd.DataFrame(domestic + european).sort_values(["date", "competition", "home"])
    matches["home_implied"], matches["away_implied"] = implied_goals(matches.odds_home, matches.odds_draw, matches.odds_away)
    matches.to_csv(OUT / "matches.csv", index=False)
    pd.DataFrame(fixtures).sort_values(["kickoff_utc", "home"]).to_csv(OUT / "fixtures.csv", index=False)
    pd.DataFrame(name_map).to_csv(OUT / "name_map.csv", index=False)

    print(f"{len(matches)} matches ({len(domestic)} domestic, {len(european)} European), {len(fixtures)} fixtures")
    mapped = sum(1 for r in name_map if r["uefa_name"])
    print(f"{mapped} of {len(name_map)} domestic clubs matched to a UEFA name")


if __name__ == "__main__":
    main()
