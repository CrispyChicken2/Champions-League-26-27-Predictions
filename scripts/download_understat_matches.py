"""Download line-ups and shots of every played match in the five biggest leagues, from Understat.

Usage: uv run scripts/download_understat_matches.py

For each match: who played and for how long, who replaced whom, and every shot
with its minute and xG. This feeds the player ratings (clpred.player_ratings).
One file per league-season in data/history/understat_matches/, one match per
line; matches already saved are skipped, so the download can be resumed.
"""

import json
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from download_history import OUT, UNDERSTAT_LEAGUES, fetch  # noqa: E402

TARGET = OUT / "understat_matches"
SEASONS = [2022, 2023, 2024, 2025, 2026]
WORKERS = 4
PAUSE_SECONDS = 0.5
ROSTER_FIELDS = ("player_id", "player", "position", "time", "roster_in", "roster_out", "id", "red_card", "xG", "xA",
                 "goals", "assists")
SHOT_FIELDS = ("minute", "xG", "result", "player_id", "h_a")


def compact(match_id: str, data: dict) -> dict:
    # An empty roster or shot list comes as a JSON list rather than an object.
    rosters = {side: [{k: p[k] for k in ROSTER_FIELDS} for p in (players.values() if isinstance(players, dict) else players)]
               for side, players in (data["rosters"] or {"h": [], "a": []}).items()}
    shots = [{k: s[k] for k in SHOT_FIELDS} for side in ("h", "a") for s in (data["shots"] or {}).get(side, [])]
    return {"id": match_id, "rosters": rosters, "shots": shots}


def one(match: dict) -> dict | None:
    for attempt in range(3):
        try:
            data = json.loads(fetch(f"https://understat.com/getMatchData/{match['id']}"))
            time.sleep(PAUSE_SECONDS)
            return {**compact(match["id"], data), "date": match["datetime"],
                    "home": match["h"]["title"], "away": match["a"]["title"]}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            if attempt == 2:
                print(f"FAIL {match['id']} {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for league, code in UNDERSTAT_LEAGUES.items():
        for season in SEASONS:
            path = TARGET / f"{code}_{season}.jsonl"
            done = {json.loads(line)["id"] for line in path.open(encoding="utf-8")} if path.exists() else set()
            listing = json.loads((OUT / f"xg_{code}_{season}.json").read_text("utf-8"))
            todo = [m for m in listing if m.get("isResult") and m["id"] not in done]
            with ThreadPoolExecutor(WORKERS) as pool, path.open("a", encoding="utf-8") as out:
                for result in pool.map(one, todo):
                    if result:
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"ok   {path.name}: {len(done) + len(todo)} played matches", flush=True)


if __name__ == "__main__":
    main()
