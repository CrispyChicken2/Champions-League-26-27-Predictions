"""Availability: how much weaker a club is without its absent players.

A club's absent players are summed into two figures:

  attack missing    the share of the team's goals and assists per match that the
                    absent players usually produce
  defence missing   the share of the regular defensive unit (goalkeeper,
                    defenders, defensive midfielders) that is absent

How much each figure changes the score is learned from past Champions League
matches (`uv run python -m clpred.availability`): the replay's expected goals
are compared with what happened, given who was missing from each matchday
squad. A club missing attack scores less; a club missing defence concedes more.
"""

import json
import re
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from clpred.players import importance

COEFFICIENTS = Path("data/availability.json")
# No Availability adjustment may change a club's goals scored or conceded by more than this (log goals).
MAX_ADJUSTMENT = 0.30
# Absences that keep a player out of this competition. Other suspensions may
# apply to a domestic league only, so they are not counted.
COUNTED_SUSPENSIONS = {"No eligibility"}
PLAYER_THRESHOLD = 0.75


def plain(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", text.replace("-", " ")).strip()


def same_player(a: str, b: str) -> float:
    """How likely two spellings name the same player ("Bremer" and "Gleison Bremer")."""
    a, b = plain(a), plain(b)
    if a == b:
        return 1.0
    if a and b and (set(a.split()) <= set(b.split()) or set(b.split()) <= set(a.split())):
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


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


def missing(players: pd.DataFrame, absent: set[str]) -> tuple[float, float, list[str]]:
    """Attack missing, defence missing and the absent players behind them, for one club's importance table."""
    gone = players[players.player.map(plain).isin(absent)]
    counted = gone[(gone.attack > 0) | (gone.defence > 0)].sort_values(["attack", "defence"], ascending=False)
    return float(gone.attack.sum()), float(gone.defence.sum()), counted.player.tolist()


def rate_changes(home_missing: tuple[float, float], away_missing: tuple[float, float],
                 attack_effect: float, defence_effect: float) -> tuple[float, float]:
    """Change in the log expected goals of the home and away side."""
    cap = lambda x: float(np.clip(x, -MAX_ADJUSTMENT, MAX_ADJUSTMENT))
    home = cap(-attack_effect * home_missing[0]) + cap(defence_effect * away_missing[1])
    away = cap(-attack_effect * away_missing[0]) + cap(defence_effect * home_missing[1])
    return home, away


def effects_in_use() -> dict | None:
    """The attack and defence effects applied to Predictions: the values fitted on the replay."""
    if not COEFFICIENTS.exists():
        return None
    fitted = json.loads(COEFFICIENTS.read_text("utf-8"))
    return {"attack": fitted["attack_effect"], "defence": fitted["defence_effect"]}


def squad_importance(player_seasons: pd.DataFrame, squad: pd.DataFrame, club: str, seasons: tuple[int, ...]) -> pd.DataFrame:
    """Importance of the players currently at the club, from the given seasons' tables."""
    table = player_seasons[(player_seasons.club == club) & player_seasons.season.isin(seasons)]
    current = set(squad[squad.club == club].player.map(plain))
    table = table[table.player.map(plain).isin(current)]
    return importance(table) if not table.empty else pd.DataFrame(columns=["player", "position", "attack", "defence"])


# --- learning the effects from past matches -------------------------------------------------


def historical_missing(lineups: pd.DataFrame, player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Attack and defence missing for every club in every past match with a known line-up."""
    order = {m: k for k, m in enumerate(lineups.sort_values("date").match_id.drop_duplicates())}
    rows = []
    for (club, season), squads in lineups.groupby(["club", "season"]):
        # The season in the line-ups is the year it ends in; player tables use the year it starts in.
        table = player_seasons[(player_seasons.club == club) & (player_seasons.season == season - 1)]
        if table.empty:
            continue
        players = importance(table)
        named = sorted(set(squads.player))
        spelling = {}
        for player in players.player:
            score, best = max((same_player(player, n), n) for n in named)
            if score >= PLAYER_THRESHOLD:
                spelling[player] = best
        match_ids = sorted(squads.match_id.unique(), key=order.get)
        in_squad = {m: set(squads[squads.match_id == m].player) for m in match_ids}
        first = {p: next(k for k, m in enumerate(match_ids) if s in in_squad[m]) for p, s in spelling.items()}
        for k, match_id in enumerate(match_ids):
            absent = {plain(p) for p, s in spelling.items() if s not in in_squad[match_id]
                      and not (first[p] > k and first[p] >= LATE_ARRIVAL_MATCHDAY - 1)}
            attack, defence, _ = missing(players, absent)
            rows.append({"match_id": match_id, "club": club, "attack_missing": attack, "defence_missing": defence})
    return pd.DataFrame(rows)


def log_rates(table: pd.DataFrame, x) -> tuple[pd.Series, pd.Series]:
    """Log expected goals of both sides for a level shift x[0], attack effect x[1] and defence effect x[2]."""
    home = np.log(table.home_rate) + x[0] - x[1] * table.home_attack_missing + x[2] * table.away_defence_missing
    away = np.log(table.away_rate) + x[0] - x[1] * table.away_attack_missing + x[2] * table.home_defence_missing
    return home, away


def poisson_loss(table: pd.DataFrame, x) -> float:
    """Average negative log-likelihood of the goals, up to a constant; lower is better."""
    home, away = log_rates(table, x)
    return float(np.mean(np.exp(home) - table.home_goals * home + np.exp(away) - table.away_goals * away))


def fit_effects(table: pd.DataFrame, with_availability: bool = True) -> np.ndarray:
    """Level shift, attack effect and defence effect that best explain the scores.

    The missing figures must be centred first: the model's expected goals
    already assume a club with its usual number of absentees.
    """
    if not with_availability:
        result = minimize(lambda x: poisson_loss(table, [x[0], 0, 0]), np.zeros(1), method="Nelder-Mead")
        return np.array([result.x[0], 0.0, 0.0])
    result = minimize(lambda x: poisson_loss(table, x), np.zeros(3), method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 2000})
    return result.x


def centred(table: pd.DataFrame, typical: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Missing figures measured against the typical club, so that zero means 'as depleted as usual'."""
    if typical is None:
        typical = {"attack": float(pd.concat([table.home_attack_missing, table.away_attack_missing]).mean()),
                   "defence": float(pd.concat([table.home_defence_missing, table.away_defence_missing]).mean())}
    out = table.copy()
    for side in ("home", "away"):
        out[f"{side}_attack_missing"] -= typical["attack"]
        out[f"{side}_defence_missing"] -= typical["defence"]
    return out, typical


def with_missing(table: pd.DataFrame) -> pd.DataFrame:
    """The replay matches for which both clubs' missing figures are known, with those figures."""
    lineups = pd.read_csv("data/lineups.csv")
    features = historical_missing(lineups, pd.read_csv("data/player_seasons.csv"))
    clubs_of = lineups.groupby("match_id").club.unique()
    dates = lineups.drop_duplicates("match_id").set_index("match_id").date
    match_of = {(dates[match_id], club): match_id for match_id, clubs in clubs_of.items() for club in clubs}
    table = table.assign(match_id=[match_of.get((m.date, m.home)) for m in table.itertuples()])
    for side in ("home", "away"):
        side_features = features.rename(columns={"club": side, "attack_missing": f"{side}_attack_missing",
                                                 "defence_missing": f"{side}_defence_missing"})
        table = table.merge(side_features, on=["match_id", side], how="left")
    return table.dropna(subset=["home_attack_missing", "away_attack_missing"])


def main() -> None:
    from clpred.model import Settings
    from clpred.replay import expected_goals, replay_matches

    matches, targets = replay_matches()
    table = expected_goals(matches, targets, Settings())
    known = with_missing(table)
    print(f"{len(known)} of {len(table)} replay matches have line-ups and player tables for both clubs")
    print(known[["home_attack_missing", "home_defence_missing"]].describe().loc[["mean", "50%", "max"]].round(3).to_string())

    known, typical = centred(known)
    fitted, level_only = fit_effects(known), fit_effects(known, with_availability=False)
    print(f"fitted on all: attack effect {fitted[1]:.3f}, defence effect {fitted[2]:.3f}")
    print(f"loss without Availability {poisson_loss(known, level_only):.5f}, with {poisson_loss(known, fitted):.5f}")

    # Honest check: learn the effects on one season, judge them on the other.
    gains = []
    for test in sorted(known.season.unique()):
        train, held_out = known[known.season != test], known[known.season == test]
        with_effects, without = fit_effects(train), fit_effects(train, with_availability=False)
        gain = poisson_loss(held_out, without) - poisson_loss(held_out, with_effects)
        gains.append(gain)
        print(f"season ending {int(test)}: effects from the other season {with_effects[1]:.3f} / {with_effects[2]:.3f}, "
              f"gain on this season {gain:+.5f}")

    # How sure the effects are: refit on resampled matches.
    rng = np.random.default_rng(0)
    draws = np.array([fit_effects(known.iloc[rng.integers(0, len(known), len(known))]) for _ in range(300)])
    low, high = np.percentile(draws, [5, 95], axis=0)
    print(f"90% range: attack {low[1]:.3f} to {high[1]:.3f}, defence {low[2]:.3f} to {high[2]:.3f}")

    COEFFICIENTS.write_text(json.dumps({
        "attack_effect": float(fitted[1]), "defence_effect": float(fitted[2]),
        "attack_range": [float(low[1]), float(high[1])], "defence_range": [float(low[2]), float(high[2])],
        "matches": int(len(known)), "held_out_gains": gains, "typical_missing": typical,
    }, indent=2), "utf-8")


if __name__ == "__main__":
    main()
