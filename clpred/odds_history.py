"""Extract past European 1X2 odds (Champions, Europa and Conference League) from saved BetExplorer results pages.

Usage: uv run python -m clpred.odds_history

Reads data/odds_history/be*_<season>_<stage>.txt and data/matches.csv, writes
data/odds_european_history.csv with one row per league-phase match that could be
paired with its odds. Unpaired matches are listed on screen.
"""

import glob
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from clpred.names import similarity

SOURCE = Path("data/odds_history")
ROW = re.compile(r"\|([^|]+?) - ([^|]+?)\|(\d+):(\d+)[^|]*\|([\d.]+)\|([\d.]+)\|([\d.]+)\|(\d\d)\.(\d\d)\.")
PAIR_THRESHOLD = 0.6
COMPETITION_OF_PREFIX = {"be": "ucl", "be-uel": "uel", "be-uecl": "uecl"}


def read_rows() -> pd.DataFrame:
    rows = []
    for path in glob.glob(str(SOURCE / "be*_*.txt")):
        prefix, season = Path(path).stem.split("_")[:2]
        competition = COMPETITION_OF_PREFIX[prefix]
        first_year = int(season[:4])
        text = "|" + re.sub(r"[\n\t]+", "|", Path(path).read_text("utf-8"))
        for home, away, hg, ag, o1, ox, o2, day, month in ROW.findall(text):
            year = first_year if int(month) >= 7 else first_year + 1
            rows.append({
                "competition": competition, "date": date(year, int(month), int(day)), "be_home": home.strip(), "be_away": away.strip(),
                "home_goals": int(hg), "away_goals": int(ag),
                "odds_home": float(o1), "odds_draw": float(ox), "odds_away": float(o2),
            })
    return pd.DataFrame(rows).drop_duplicates()


def main() -> None:
    odds = read_rows()
    matches = pd.read_csv("data/matches.csv", low_memory=False)
    league_phase = matches[matches.competition.isin(COMPETITION_OF_PREFIX.values()) & (matches.stage == "League Phase")]

    paired, unpaired = [], []
    for match in league_phase.itertuples():
        day = date.fromisoformat(match.date)
        # Sources disagree by a day around midnight kick-offs in other time zones.
        near = odds[(odds.competition == match.competition)
                    & (odds.date >= day - timedelta(days=1)) & (odds.date <= day + timedelta(days=1))]
        best, best_score = None, 0.0
        for row in near.itertuples():
            score = min(similarity(match.home, row.be_home), similarity(match.away, row.be_away))
            if score > best_score:
                best, best_score = row, score
        same_score = best is not None and (best.home_goals, best.away_goals) == (match.home_goals, match.away_goals)
        if best is not None and best_score >= PAIR_THRESHOLD and same_score:
            paired.append({"competition": match.competition, "date": match.date, "home": match.home, "away": match.away,
                           "odds_home": best.odds_home, "odds_draw": best.odds_draw, "odds_away": best.odds_away})
        else:
            unpaired.append((match.competition, match.date, match.home, match.away,
                             None if best is None else (best.be_home, best.be_away, round(best_score, 2))))

    pd.DataFrame(paired).to_csv("data/odds_european_history.csv", index=False)
    print(f"{len(paired)} of {len(league_phase)} league-phase matches paired with odds")
    for item in unpaired:
        print("UNPAIRED", item)


if __name__ == "__main__":
    main()
