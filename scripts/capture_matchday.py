"""Capture everything a Matchday needs from the web, in one command.

Usage: uv run scripts/capture_matchday.py <matchday number> [--list]

Saves, with a browser:
  data/recent/            latest league results of the 16 leagues (BetExplorer)
  matchday-NN/source/     the Snapshot: odds per match (OddsPortal), absences and
                          squad per club (Transfermarkt)
  data/players/           each club's player performance, this season and last

Run it before the Matchday's first kick-off; running it again refreshes the
Snapshot with the latest odds and absences. `--list` only prints the pages it
would fetch. Needs data/fixtures.csv (from `python -m clpred.build_matches`).
"""

import csv
import io
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture_raw import capture  # noqa: E402

from clpred.names import similarity  # noqa: E402

BETEXPLORER = "https://www.betexplorer.com/football"
ODDSPORTAL = "https://www.oddsportal.com"
TRANSFERMARKT = "https://www.transfermarkt.com"

# Current-season results page of every league in the history.
LEAGUE_PAGES = {
    "E0": "england/premier-league", "D1": "germany/bundesliga", "I1": "italy/serie-a", "SP1": "spain/laliga",
    "F1": "france/ligue-1", "N1": "netherlands/eredivisie", "B1": "belgium/jupiler-pro-league",
    "P1": "portugal/liga-portugal", "T1": "turkey/super-lig", "G1": "greece/super-league",
    "AUT": "austria/bundesliga", "NOR": "norway/eliteserien", "CZE": "czech-republic/chance-liga",
    "SVK": "slovakia/nike-liga", "AZE": "azerbaijan/premier-league", "UKR": "ukraine/premier-league",
}
MATCH_LINK = re.compile(r'href="(/football/h2h/([a-z0-9-]+)-[A-Za-z0-9]{8}/([a-z0-9-]+)-[A-Za-z0-9]{8}/#[^"]+)"')
CLUB_LINK = re.compile(r'href="/([a-z0-9-]+)/startseite/verein/(\d+)')
SLUG_THRESHOLD = 0.6


def fixtures_of(matchday: int) -> list[dict]:
    path = Path("data/fixtures.csv")
    if not path.exists():
        sys.exit("data/fixtures.csv is missing: run `uv run python -m clpred.build_matches` first.")
    rows = [r for r in csv.DictReader(io.open(path, encoding="utf-8")) if int(float(r["matchday"])) == matchday]
    if not rows:
        sys.exit(f"No unplayed fixtures for Matchday {matchday} in data/fixtures.csv.")
    return rows


def odds_targets(listing_html: str, fixtures: list[dict]) -> tuple[dict[str, str], list[str]]:
    """Odds pages for the Matchday's fixtures, and the fixtures no odds link was found for."""
    targets, found = {}, set()
    for link, first, second in set(MATCH_LINK.findall(listing_html)):
        # The link lists the two clubs in alphabetical order, not home then away.
        for fixture in fixtures:
            straight = min(similarity(first, fixture["home"]), similarity(second, fixture["away"]))
            swapped = min(similarity(first, fixture["away"]), similarity(second, fixture["home"]))
            if max(straight, swapped) >= SLUG_THRESHOLD:
                key = f"odds_{first}__{second}"
                targets[f"{key}_1x2"] = ODDSPORTAL + link
                targets[f"{key}_ou"] = ODDSPORTAL + link + ":over-under;2"
                targets[f"{key}_cs"] = ODDSPORTAL + link + ":cs;2"
                found.add((fixture["home"], fixture["away"]))
                break
    missing = [f"{f['home']} - {f['away']}" for f in fixtures if (f["home"], f["away"]) not in found]
    return targets, missing


def club_targets(participants_html: str, season: int) -> tuple[dict[str, str], dict[str, str]]:
    """Snapshot pages (absences, squad) and player performance pages for every participating club."""
    snapshot, players = {}, {}
    for slug, club_id in sorted(set(CLUB_LINK.findall(participants_html))):
        snapshot[f"absences_{slug}"] = f"{TRANSFERMARKT}/{slug}/sperrenundverletzungen/verein/{club_id}"
        # No season in the address: the page then shows the current squad, also for calendar-year leagues.
        snapshot[f"squad_{slug}"] = f"{TRANSFERMARKT}/{slug}/kader/verein/{club_id}/plus/1"
        for year in (season - 1, season):
            players[f"perf_{slug}_{club_id}_{year}"] = (
                f"{TRANSFERMARKT}/{slug}/leistungsdaten/verein/{club_id}/reldata/%26{year}/plus/1")
    return snapshot, players


def main() -> None:
    matchday = int(sys.argv[1])
    list_only = "--list" in sys.argv
    fixtures = fixtures_of(matchday)
    first_day = date.fromisoformat(min(f["date"] for f in fixtures))
    season = first_day.year if first_day.month >= 7 else first_day.year - 1
    source = Path(f"matchday-{matchday:02d}") / "source"
    scratch = Path(tempfile.mkdtemp()) if list_only else source

    # Two index pages tell us which pages to fetch.
    index = {"oddsportal_cl": f"{ODDSPORTAL}/football/europe/champions-league/",
             "tm_participants": f"{TRANSFERMARKT}/uefa-champions-league/teilnehmer/pokalwettbewerb/CL/saison_id/{season}"}
    capture(scratch, index)
    odds, without_odds = odds_targets((scratch / "oddsportal_cl.html").read_text("utf-8"), fixtures)
    snapshot, players = club_targets((scratch / "tm_participants.html").read_text("utf-8"), season)
    recent = {league: f"{BETEXPLORER}/{path}/results/" for league, path in LEAGUE_PAGES.items()}

    print(f"Matchday {matchday}: {len(fixtures)} fixtures, {len(odds) // 3} with an odds page, "
          f"{len(snapshot) // 2} clubs, {len(recent)} leagues")
    for fixture in without_odds:
        print(f"NO ODDS PAGE FOUND for {fixture}")
    if len(snapshot) // 2 != 36:
        print(f"EXPECTED 36 clubs on the participants page, found {len(snapshot) // 2}")
    if list_only:
        for group in (recent, odds, snapshot, players):
            for name, url in group.items():
                print(name, url)
        return

    capture(Path("data/recent"), recent)
    capture(source, {**odds, **snapshot})
    # Last season's tables no longer change; this season's are refreshed every time.
    capture(Path("data/players"), {n: u for n, u in players.items() if n.endswith(f"_{season - 1}")}, skip_existing=True)
    manifest = capture(Path("data/players"), {n: u for n, u in players.items() if n.endswith(f"_{season}")})

    failed = [n for n, e in manifest.items() if "error" in e]
    print(f"Capture finished. {len(failed)} player pages failed." if failed else "Capture finished.")


if __name__ == "__main__":
    main()
