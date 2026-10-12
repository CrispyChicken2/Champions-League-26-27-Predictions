"""Read league results from saved BetExplorer pages in data/recent/.

Two uses: the latest rounds that the main domestic source has not published
yet, and whole leagues it does not cover (Czech Republic, Slovakia,
Azerbaijan, Ukraine). Pages are named <league>[_<season>][_<stage>].txt.
"""

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from clpred.names import best_match

SOURCE = Path("data/recent")
ROW = re.compile(r"\|([^|]+?) - ([^|]+?)\|(\d+):(\d+)[^|]*\|([\d.]+)\|([\d.]+)\|([\d.]+)\|(\d\d\.\d\d\.|Yesterday|Today)")
EXTRA_LEAGUE_COUNTRY = {"CZE": "CZE", "SVK": "SVK", "AZE": "AZE", "UKR": "UKR"}
KNOWN_CLUB_THRESHOLD = 0.72
# Matches below this are accepted but listed, so a wrong pairing gets noticed.
SURE_THRESHOLD = 0.85
UEFA_NAME_THRESHOLD = 0.82


def read_pages() -> list[dict]:
    if not (SOURCE / "manifest.json").exists():
        return []
    manifest = json.loads((SOURCE / "manifest.json").read_text("utf-8"))
    rows = []
    for name, entry in manifest.items():
        path = SOURCE / f"{name}.txt"
        if "error" in entry or not path.exists():
            continue
        parts = name.split("_")
        captured = datetime.fromisoformat(entry["captured_at_utc"]).date()
        first_year = int(parts[1][:4]) if len(parts) > 1 else None
        text = "|" + re.sub(r"[\n\t]+", "|", path.read_text("utf-8"))
        for home, away, home_goals, away_goals, o1, ox, o2, when in ROW.findall(text):
            if when == "Today":
                day = captured
            elif when == "Yesterday":
                day = captured - timedelta(days=1)
            else:
                d, m = int(when[:2]), int(when[3:5])
                # A past season runs July to June; the current page holds this calendar year.
                year = captured.year if first_year is None else (first_year if m >= 7 else first_year + 1)
                day = date(year, m, d)
            rows.append({"date": day.isoformat(), "competition": parts[0], "home": home.strip(), "away": away.strip(),
                         "home_goals": int(home_goals), "away_goals": int(away_goals),
                         "odds_home": float(o1), "odds_draw": float(ox), "odds_away": float(o2)})
    return rows


def recent_rows(domestic: list[dict], name_map: list[dict], uefa_clubs: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Rows to add to the domestic history, in canonical club names, plus notes on anything left out."""
    last_date = {}
    for row in domestic:
        last_date[row["competition"]] = max(last_date.get(row["competition"], ""), row["date"])

    added, notes, seen = [], [], set()
    for row in read_pages():
        league = row["competition"]
        if league in EXTRA_LEAGUE_COUNTRY:
            candidates = {n: c["spellings"] for n, c in uefa_clubs.items() if c["country"] == EXTRA_LEAGUE_COUNTRY[league]}
            threshold, keep_unmatched = UEFA_NAME_THRESHOLD, True
        else:
            if row["date"] <= last_date.get(league, ""):
                continue
            candidates = {}
            for entry in name_map:
                if entry["league"] == league:
                    canonical = entry["uefa_name"] or entry["source_name"]
                    candidates.setdefault(canonical, [canonical]).append(entry["source_name"])
            threshold, keep_unmatched = KNOWN_CLUB_THRESHOLD, False

        names = {}
        for side in ("home", "away"):
            match, score = best_match(row[side], candidates)
            if match is not None and score >= threshold:
                names[side] = match
                if score < SURE_THRESHOLD and not keep_unmatched:
                    notes.append(f"CHECK {league}: {row[side]!r} read as {match!r} ({score:.2f})")
            elif keep_unmatched:
                names[side] = row[side]
            else:
                notes.append(f"{league} {row['date']} {row['home']} - {row['away']}: unknown club {row[side]!r}")
        if len(names) < 2:
            continue
        key = (league, row["date"], names["home"], names["away"])
        if key in seen:
            continue
        seen.add(key)
        added.append({**row, **names, "stage": "", "matchday": "", "season": "", "kickoff_utc": ""})
    return added, sorted(set(notes))
