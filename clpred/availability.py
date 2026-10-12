"""Availability: how much weaker a club is without its absent players.

A club's strength on the pitch is taken from the market value of its best
eleven (one goalkeeper and the ten most valuable outfield players). When
players are absent the next most valuable ones step in, so losing a star
costs the gap to his replacement, not his whole value.

How much strength a given loss of value costs is learned from the clubs
themselves: the slope of fitted strength against the log of best-eleven
value across the Matchday's clubs.
"""

import re
import unicodedata
from datetime import date, datetime

import numpy as np
import pandas as pd

GOALKEEPER = "Goalkeeper"
OUTFIELD_PLACES = 10
# No Availability adjustment may move a club's strength by more than this (log goals).
MAX_ADJUSTMENT = 0.30
# Absences that keep a player out of this competition. Other suspensions may
# apply to a domestic league only, so they are not counted.
COUNTED_SUSPENSIONS = {"No eligibility"}


def plain(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", text).strip()


def best_eleven_value(players: pd.DataFrame) -> float:
    """Summed value of one goalkeeper and the ten most valuable outfield players."""
    keepers = players[players.position == GOALKEEPER].value_m
    outfield = players[players.position != GOALKEEPER].value_m.nlargest(OUTFIELD_PLACES)
    return float((keepers.max() if len(keepers) else 0.0) + outfield.sum())


def absent_players(absences: pd.DataFrame, club: str, match_day: date) -> set[str]:
    """Plain names of the club's players who cannot play on `match_day`."""
    out = set()
    for row in absences[absences.club == club].itertuples():
        if row.kind == "Suspensions" and row.reason not in COUNTED_SUSPENSIONS:
            continue
        back = row.expected_return if isinstance(row.expected_return, str) else ""
        if back:
            try:
                if datetime.strptime(back, "%d/%m/%Y").date() <= match_day:
                    continue
            except ValueError:
                pass
        out.add(plain(row.player))
    return out


def value_loss(squad: pd.DataFrame, absent: set[str]) -> float:
    """Log ratio of the best eleven without the absent players to the full best eleven (zero or negative)."""
    full = best_eleven_value(squad)
    available = best_eleven_value(squad[~squad.player.map(plain).isin(absent)])
    if full <= 0 or available <= 0:
        return 0.0
    return float(np.log(available / full))


def strength_per_log_value(strength: dict[str, float], squads: pd.DataFrame) -> tuple[float, float]:
    """Slope and R² of club strength against log best-eleven value."""
    clubs = [c for c in strength if c in set(squads.club)]
    x = np.array([np.log(best_eleven_value(squads[squads.club == c])) for c in clubs])
    y = np.array([strength[c] for c in clubs])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return float(slope), float(1 - residual.var() / y.var())


def adjustment(squads: pd.DataFrame, absences: pd.DataFrame, club: str, match_day: date, slope: float) -> tuple[float, list[str]]:
    """Change in the club's strength (log goals, zero or negative) and the absent players it rests on."""
    squad = squads[squads.club == club]
    if squad.empty:
        return 0.0, []
    absent = absent_players(absences, club, match_day)
    counted = sorted(squad[squad.player.map(plain).isin(absent)].sort_values("value_m", ascending=False).player)
    change = slope * value_loss(squad, absent)
    return float(max(change, -MAX_ADJUSTMENT)), counted
