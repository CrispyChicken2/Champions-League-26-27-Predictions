"""What each player contributes to his club, from saved Transfermarkt performance pages.

Usage: uv run python -m clpred.players      rebuilds data/player_seasons.csv

Two measures come out of a club's season table:

  attack importance   the player's goals and assists per appearance, as a share
                      of what the whole team produces per match
  defence importance  for goalkeepers, defenders and defensive midfielders: how
                      much of the available playing time the player is given

Both describe the player when he is fit, so a long injury does not make a
key player look unimportant.
"""

import json
import re
from pathlib import Path

import pandas as pd

from clpred.names import best_match

SOURCE = Path("data/players")
OUT = Path("data/player_seasons.csv")
POSITIONS = {"Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield", "Central Midfield",
             "Attacking Midfield", "Left Midfield", "Right Midfield", "Left Winger", "Right Winger",
             "Second Striker", "Centre-Forward"}
DEFENSIVE = {"Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield"}
DEFENSIVE_PLACES = 6
# Appearances added when averaging, so three good matches do not make a star.
PRIOR_APPEARANCES = 5
STAT_FIELDS = 12
CLUB_THRESHOLD = 0.6


def number(text: str) -> float:
    text = text.replace("'", "").replace(".", "").strip()
    return float(text) if text.isdigit() else 0.0


def parse_page(text: str) -> list[dict]:
    """Players of one performance page: name, position, squad listings, appearances, goals, assists, minutes."""
    lines = [line for line in text.split("\n") if line.strip()]
    players = []
    for k in range(1, len(lines) - 1):
        if lines[k].strip() not in POSITIONS:
            continue
        # The figures sit on one line, or two when the player is flagged as injured.
        fields = [f.strip() for f in lines[k + 1].split("\t") if f.strip()]
        if len(fields) < STAT_FIELDS and k + 2 < len(lines):
            fields += [f.strip() for f in lines[k + 2].split("\t") if f.strip()]
        if len(fields) < STAT_FIELDS or not fields[STAT_FIELDS - 1].endswith("'"):
            continue  # not used that season
        players.append({"player": " ".join(lines[k - 1].split()), "position": lines[k].strip(),
                        "in_squad": number(fields[1]), "appearances": number(fields[2]), "goals": number(fields[3]),
                        "assists": number(fields[4]), "minutes": number(fields[STAT_FIELDS - 1])})
    return players


def read_seasons(clubs: dict[str, list[str]]) -> pd.DataFrame:
    """Every saved club-season, with UEFA's club name. The season is the year it starts in."""
    manifest = json.loads((SOURCE / "manifest.json").read_text("utf-8"))
    rows = []
    for name, entry in manifest.items():
        path = SOURCE / f"{name}.txt"
        if not name.startswith("perf_") or "error" in entry or not path.exists():
            continue
        club, score = best_match(entry.get("title", "").split(" - ")[0], clubs)
        if score < CLUB_THRESHOLD:
            print(f"UNREADABLE club for {name}: {entry.get('title')!r}")
            continue
        players = parse_page(path.read_text("utf-8"))
        if len(players) < 11:
            print(f"THIN table for {name}: {len(players)} players")
        for player in players:
            rows.append({"club": club, "season": int(name.rsplit("_", 1)[1]), **player})
    return pd.DataFrame(rows)


def importance(table: pd.DataFrame) -> pd.DataFrame:
    """Add attack and defence importance to one club's player table (one or more seasons summed)."""
    totals = table.groupby(["player", "position"], as_index=False)[
        ["in_squad", "appearances", "goals", "assists", "minutes"]].sum()
    matches = max(totals.minutes.sum() / (11 * 90), 1.0)
    team_per_match = max((totals.goals + totals.assists).sum() / matches, 0.1)
    totals["attack"] = (totals.goals + totals.assists) / (totals.appearances + PRIOR_APPEARANCES) / team_per_match
    listed = (totals.in_squad + PRIOR_APPEARANCES) * 90
    share = (totals.minutes / listed).where(totals.position.isin(DEFENSIVE), 0.0)
    regulars = share.nlargest(DEFENSIVE_PLACES).sum()
    totals["defence"] = share / regulars if regulars > 0 else 0.0
    return totals


def all_spellings() -> dict[str, list[str]]:
    spellings: dict[str, list[str]] = {}
    for path in Path("data/history").glob("european_*.json"):
        for match in json.loads(path.read_text("utf-8")):
            for team in (match["homeTeam"], match["awayTeam"]):
                names = team.get("translations", {})
                spellings[team["internationalName"]] = [
                    team["internationalName"], names.get("displayOfficialName", {}).get("EN", ""),
                    names.get("displayName", {}).get("EN", "")]
    return spellings


def main() -> None:
    seasons = read_seasons(all_spellings())
    seasons.to_csv(OUT, index=False)
    print(f"{len(seasons)} player-seasons for {seasons.groupby(['club', 'season']).ngroups} club-seasons")


if __name__ == "__main__":
    main()
