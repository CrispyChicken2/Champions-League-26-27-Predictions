"""Extract odds and absences from a Matchday's saved Snapshot pages.

Usage: uv run python -m clpred.snapshot <matchday number>

Reads matchday-NN/source/, writes matchday-NN/odds.csv, correct_score_odds.csv
and absences.csv, using UEFA's club names. Anything that cannot be read is
reported and left out, so later steps can see what is missing.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

from clpred.names import best_match

BOOKMAKER_ROW = re.compile(r"\|([\d.]+)\|\s*\|([\d.]+)\|\s*\|([\d.]+)\|\s*\|\d+\.\d%")
OVER_UNDER_ROW = re.compile(r"Over/Under \+2\.5\|\d+\|[\s|]*([\d.]+)[\s|]*([\d.]+)[\s|]*[\d.]+%")
CORRECT_SCORE_ROW = re.compile(r"\|(\d+):(\d+)\|\d+\|[\s|]*([\d.]+)\|")
HEADING = re.compile(r"\|([^|]+) - ([^|]+)\|\1\|-\|\2\|")
NAME_THRESHOLD = 0.6


def flat(path: Path) -> str:
    return "|" + re.sub(r"\n+", "|", path.read_text("utf-8"))


def club_spellings() -> dict[str, list[str]]:
    """Every 2026/27 league-phase club with the spellings UEFA uses for it."""
    spellings: dict[str, list[str]] = {}
    for match in json.loads(Path("data/history/european_ucl_2027.json").read_text("utf-8")):
        if match.get("round", {}).get("metaData", {}).get("name") != "League Phase":
            continue
        for team in (match["homeTeam"], match["awayTeam"]):
            names = team.get("translations", {})
            spellings[team["internationalName"]] = [
                team["internationalName"],
                names.get("displayOfficialName", {}).get("EN", ""),
                names.get("displayName", {}).get("EN", ""),
            ]
    return spellings


def extract_odds(source: Path, manifest: dict, fixtures: pd.DataFrame, clubs: dict) -> tuple[list, list]:
    odds, correct_scores = [], []
    for name, entry in manifest.items():
        if not (name.startswith("odds_") and name.endswith("_1x2")):
            continue
        stem = name[: -len("_1x2")]
        # The page heading reads "Home - Away|Home|-|Away"; the page title is not always filled in.
        heading = HEADING.search(flat(source / f"{name}.txt"))
        if not heading:
            print(f"UNREADABLE heading for {name}")
            continue
        home_text, away_text = heading.groups()
        home, home_score = best_match(home_text, clubs)
        away, away_score = best_match(away_text, clubs)
        fixture = fixtures[(fixtures.home == home) & (fixtures.away == away)]
        if min(home_score, away_score) < NAME_THRESHOLD or fixture.empty:
            print(f"UNREADABLE fixture for {name}: {home_text} - {away_text} -> {home} - {away}")
            continue
        row = {"home": home, "away": away, "captured_at_utc": entry["captured_at_utc"]}

        books = [tuple(map(float, b)) for b in BOOKMAKER_ROW.findall(flat(source / f"{name}.txt"))]
        if books:
            row.update(odds_home=sum(b[0] for b in books) / len(books), odds_draw=sum(b[1] for b in books) / len(books),
                       odds_away=sum(b[2] for b in books) / len(books), bookmakers=len(books))
        else:
            print(f"MISSING 1X2 odds: {home} - {away}")

        over_under = OVER_UNDER_ROW.search(flat(source / f"{stem}_ou.txt")) if (source / f"{stem}_ou.txt").exists() else None
        if over_under:
            row.update(odds_over_2_5=float(over_under.group(1)), odds_under_2_5=float(over_under.group(2)))
        else:
            print(f"MISSING over/under 2.5 odds: {home} - {away}")
        odds.append(row)

        if (source / f"{stem}_cs.txt").exists():
            for home_goals, away_goals, price in CORRECT_SCORE_ROW.findall(flat(source / f"{stem}_cs.txt")):
                correct_scores.append({"home": home, "away": away, "home_goals": int(home_goals),
                                       "away_goals": int(away_goals), "odds": float(price)})
    return odds, correct_scores


def extract_absences(source: Path, manifest: dict, clubs: dict) -> list:
    absences = []
    for name, entry in manifest.items():
        if not name.startswith("absences_"):
            continue
        club_text = entry["title"].split(" - Suspensions")[0]
        club, score = best_match(club_text, clubs)
        if score < NAME_THRESHOLD:
            print(f"UNREADABLE club for {name}: {club_text} -> {club} ({score:.2f})")
            continue
        lines = [line for line in (source / f"{name}.txt").read_text("utf-8").split("\n") if line.strip()]
        kind = None
        for k, line in enumerate(lines):
            text = line.strip()
            if text.upper().startswith("RISK OF SUSPENSION"):
                break
            if text in ("Injuries", "Suspensions"):
                kind = text
            fields = line.split("\t")
            # A detail row looks like "<tab>27<tab>Broken finger<tab>since<tab>return<tab>missed".
            if kind and len(fields) >= 5 and fields[1].strip().isdigit() and k >= 2:
                absences.append({"club": club, "player": lines[k - 2].strip(), "position": lines[k - 1].strip(),
                                 "kind": kind, "reason": fields[2].strip(), "since": fields[3].strip(),
                                 "expected_return": fields[4].strip(), "captured_at_utc": entry["captured_at_utc"]})
    return absences


def main() -> None:
    matchday = int(sys.argv[1])
    folder = Path(f"matchday-{matchday:02d}")
    source = folder / "source"
    manifest = json.loads((source / "manifest.json").read_text("utf-8"))
    fixtures = pd.read_csv("data/fixtures.csv")
    fixtures = fixtures[fixtures.matchday == matchday]
    clubs = club_spellings()

    odds, correct_scores = extract_odds(source, manifest, fixtures, clubs)
    absences = extract_absences(source, manifest, clubs)
    pd.DataFrame(odds).round(3).to_csv(folder / "odds.csv", index=False)
    pd.DataFrame(correct_scores).to_csv(folder / "correct_score_odds.csv", index=False)
    pd.DataFrame(absences).to_csv(folder / "absences.csv", index=False)
    print(f"{len(odds)} of {len(fixtures)} fixtures with odds, {len(correct_scores)} correct-score prices, "
          f"{len(absences)} absences across {len({a['club'] for a in absences})} clubs")


if __name__ == "__main__":
    main()
