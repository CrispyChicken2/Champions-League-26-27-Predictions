"""Download the matchday squads of past European league-phase matches.

Usage: uv run scripts/download_lineups.py

Reads the match lists in data/history/ and asks UEFA's match feed for the
line-up of every finished league-phase match of the replay seasons. Writes
data/lineups.csv: one row per player named in a matchday squad, saying
whether he started or sat on the bench.
"""

import csv
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

HISTORY = Path("data/history")
OUT = Path("data/lineups.csv")
SEASON_YEARS = (2025, 2026)
COMPETITIONS = ("ucl", "uel", "uecl")
HEADERS = {"User-Agent": "Mozilla/5.0"}
PAUSE_SECONDS = 0.5


def main() -> None:
    rows, failed = [], 0
    for competition, year in [(c, y) for c in COMPETITIONS for y in SEASON_YEARS]:
        matches = json.load(io.open(HISTORY / f"european_{competition}_{year}.json", encoding="utf-8"))
        league_phase = [m for m in matches if m["status"] == "FINISHED"
                        and m.get("round", {}).get("metaData", {}).get("name") == "League Phase"]
        for match in league_phase:
            url = f"https://match.uefa.com/v5/matches/{match['id']}/lineups"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
                    lineup = json.loads(response.read())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                failed += 1
                print(f"FAIL {match['id']} {exc}")
                continue
            for side in ("homeTeam", "awayTeam"):
                club = match[side]["internationalName"]
                for role, key in (("start", "field"), ("bench", "bench")):
                    for entry in lineup[side].get(key, []):
                        player = entry["player"]
                        rows.append({"match_id": match["id"], "competition": competition, "date": match["kickOffTime"]["date"], "season": year,
                                     "club": club, "player": player["internationalName"],
                                     "position": player.get("fieldPosition", ""), "role": role})
            time.sleep(PAUSE_SECONDS)
        print(f"{competition}, season ending {year}: {len(league_phase)} matches", flush=True)
    with io.open(OUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} squad entries written, {failed} matches failed")


if __name__ == "__main__":
    main()
