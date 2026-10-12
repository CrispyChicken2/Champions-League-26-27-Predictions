"""Market value of each club's best eleven, from saved Transfermarkt squad pages.

Usage: uv run python -m clpred.squad_value     rebuilds data/squad_values.csv from data/squads_history/

The best eleven is one goalkeeper and the ten most valuable outfield players.
"""

import json
import re
from pathlib import Path

import pandas as pd

from clpred.names import best_match

SOURCE = Path("data/squads_history")
OUT = Path("data/squad_values.csv")
GOALKEEPER = "Goalkeeper"
OUTFIELD_PLACES = 10
MARKET_VALUE = re.compile(r"€([\d.]+)(m|k)")
BIRTH_DATE = re.compile(r"\d\d/\d\d/\d{4} \(\d+\)")
POSITIONS = {"Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield", "Central Midfield",
             "Attacking Midfield", "Left Midfield", "Right Midfield", "Left Winger", "Right Winger",
             "Second Striker", "Centre-Forward"}
CLUB_THRESHOLD = 0.6


def parse_squad(text: str) -> pd.DataFrame:
    """Players of one squad page with position and market value in millions of euros."""
    lines = [line for line in text.split("\n") if line.strip()]
    players = []
    for k in range(1, len(lines) - 1):
        if lines[k].strip() not in POSITIONS:
            continue
        # For players flagged as injured the details wrap onto a second line.
        value = MARKET_VALUE.search(lines[k + 1]) or (
            MARKET_VALUE.search(lines[k + 2]) if k + 2 < len(lines) and BIRTH_DATE.search(lines[k + 1]) else None)
        if value:
            amount = float(value.group(1)) * (1.0 if value.group(2) == "m" else 0.001)
            players.append({"player": " ".join(lines[k - 1].split()), "position": lines[k].strip(), "value_m": amount})
    return pd.DataFrame(players, columns=["player", "position", "value_m"])


def best_eleven_value(players: pd.DataFrame) -> float:
    keepers = players[players.position == GOALKEEPER].value_m
    outfield = players[players.position != GOALKEEPER].value_m.nlargest(OUTFIELD_PLACES)
    return float((keepers.max() if len(keepers) else 0.0) + outfield.sum())


def european_spellings() -> dict[str, list[str]]:
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
    clubs = european_spellings()
    manifest = json.loads((SOURCE / "manifest.json").read_text("utf-8"))
    rows = []
    for name, entry in manifest.items():
        path = SOURCE / f"{name}.txt"
        if not name.startswith("squad_") or "error" in entry or not path.exists():
            continue
        club, score = best_match(entry.get("title", "").split(" - ")[0], clubs)
        if score < CLUB_THRESHOLD:
            print(f"UNREADABLE club for {name}: {entry.get('title')!r}")
            continue
        squad = parse_squad(path.read_text("utf-8"))
        if len(squad) < 15:
            print(f"THIN squad for {name}: {len(squad)} players")
            continue
        # Page seasons are named by the year they start in; the rest of the project uses the year they end in.
        rows.append({"club": club, "season": int(name.rsplit("_", 1)[1]) + 1, "players": len(squad),
                     "best_eleven_m": round(best_eleven_value(squad), 3)})
    pd.DataFrame(rows).drop_duplicates(["club", "season"]).to_csv(OUT, index=False)
    print(f"{len(rows)} club-seasons written")


if __name__ == "__main__":
    main()
