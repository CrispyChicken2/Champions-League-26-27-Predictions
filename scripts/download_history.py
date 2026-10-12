"""Download match history: domestic leagues and European club competitions.

Usage: uv run scripts/download_history.py

Domestic results, closing odds and xG come from football-data.co.uk (CSV).
European results come from UEFA's public match feed (JSON).
Files land in data/history/ unchanged; a failed download is reported and skipped.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path("data/history")
HEADERS = {"User-Agent": "Mozilla/5.0"}
PAUSE_SECONDS = 1

# football-data.co.uk season codes, oldest first.
SEASONS = ["2122", "2223", "2324", "2425", "2526", "2627"]
# One file per season: leagues of the 2026/27 Champions League clubs.
MAIN_LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "N1", "B1", "P1", "T1", "G1"]
# One file for all seasons.
EXTRA_LEAGUES = ["AUT", "NOR"]

# UEFA competition ids; seasonYear is the year the season ends in.
UEFA_COMPETITIONS = {"ucl": 1, "uel": 14, "uecl": 2019}
UEFA_SEASON_YEARS = [2022, 2023, 2024, 2025, 2026, 2027]
UEFA_PAGE = 100


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download_domestic() -> None:
    targets = {
        f"domestic_{league}_{season}.csv": f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
        for league in MAIN_LEAGUES
        for season in SEASONS
    }
    targets.update(
        {f"domestic_{league}_all.csv": f"https://www.football-data.co.uk/new/{league}.csv" for league in EXTRA_LEAGUES}
    )
    for name, url in targets.items():
        try:
            data = fetch(url)
            (OUT / name).write_bytes(data)
            print(f"ok   {name} {len(data)} bytes")
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"FAIL {name} {url} {exc}")
        time.sleep(PAUSE_SECONDS)


def download_european() -> None:
    for code, competition_id in UEFA_COMPETITIONS.items():
        for year in UEFA_SEASON_YEARS:
            name = f"european_{code}_{year}.json"
            matches, offset = [], 0
            try:
                while True:
                    url = (
                        "https://match.uefa.com/v5/matches"
                        f"?competitionId={competition_id}&seasonYear={year}"
                        f"&limit={UEFA_PAGE}&offset={offset}&order=ASC"
                    )
                    page = json.loads(fetch(url))
                    matches.extend(page)
                    if len(page) < UEFA_PAGE:
                        break
                    offset += UEFA_PAGE
                    time.sleep(PAUSE_SECONDS)
                (OUT / name).write_text(json.dumps(matches, ensure_ascii=False), "utf-8")
                print(f"ok   {name} {len(matches)} matches")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"FAIL {name} {exc}")
            time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    download_domestic()
    download_european()
