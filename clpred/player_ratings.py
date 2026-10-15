"""Player ratings from who was on the pitch, and what each line-up is worth.

Usage: uv run python -m clpred.player_ratings      choose the ratings' settings on domestic matches

Every match of the five biggest leagues is cut into stretches between
substitutions. In each stretch the expected goals (xG) of each side are
explained by the players on the pitch:

    home xG per 90: exp(base + home_advantage + sum(attack of home players) - sum(defence of away players))

and the same for the away side. Ratings are pulled towards zero (an average
player), so a player needs many minutes before his rating moves far, and
recent matches weigh more. A line-up's attack is then the sum of its starters'
attack ratings, and likewise for defence.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from clpred.availability import same_player
from clpred.names import best_match

SOURCE = Path("data/history/understat_matches")
MATCH_MINUTES = 90.0
HISTORY_DAYS = 365 * 3
TEAM_THRESHOLD = 0.8
PLAYER_THRESHOLD = 0.8
POOL_DAYS = 365          # a starter is looked for among the club's players of the last year
USUAL_MATCHES = 10       # the usual line-up: the club's last ten league matches


@dataclass(frozen=True)
class RatingSettings:
    penalty: float = 30.0          # pull of every rating towards zero
    decay_per_day: float = 0.001   # weight of a stretch halves every ln(2)/decay days


def stretches(match: dict) -> list[dict]:
    """The stretches of one match between substitutions, with players on the pitch and xG of each side."""
    spans = {}
    for side in ("h", "a"):
        for p in match["rosters"][side]:
            minutes = min(float(p["time"]), MATCH_MINUTES)
            if minutes <= 0:
                continue
            start = 0.0 if p["position"] != "Sub" else MATCH_MINUTES - minutes
            spans[(side, p["player_id"])] = (start, start + minutes)
    cuts = sorted({0.0, MATCH_MINUTES} | {t for span in spans.values() for t in span})
    out = []
    for begin, end in zip(cuts, cuts[1:]):
        if end - begin < 1:
            continue
        middle = (begin + end) / 2
        on = {side: [pid for (s, pid), (a, b) in spans.items() if s == side and a <= middle < b] for side in ("h", "a")}
        xg = {side: sum(float(s["xG"]) for s in match["shots"]
                        if s["h_a"] == side and begin <= min(float(s["minute"]), MATCH_MINUTES - 0.5) < end)
              for side in ("h", "a")}
        out.append({"minutes": end - begin, "home_players": on["h"], "away_players": on["a"],
                    "home_xg": xg["h"], "away_xg": xg["a"]})
    return out


def read_matches() -> pd.DataFrame:
    """One row per match: date, Understat team names, starters and stretches."""
    rows = []
    for path in sorted(SOURCE.glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            match = json.loads(line)
            if not (match["rosters"].get("h") and match["rosters"].get("a")):
                continue  # line-ups missing at the source
            starters = {side: [p["player_id"] for p in match["rosters"][side] if p["position"] != "Sub"]
                        for side in ("h", "a")}
            squads = {side: {p["player_id"]: p["player"] for p in match["rosters"][side]} for side in ("h", "a")}
            rows.append({"id": match["id"], "league": path.stem.split("_")[0], "date": pd.Timestamp(match["date"][:10]),
                         "home": match["home"], "away": match["away"], "home_starters": starters["h"],
                         "away_starters": starters["a"], "home_squad": squads["h"], "away_squad": squads["a"],
                         "stretches": stretches(match)})
    return pd.DataFrame(rows).sort_values("date", ignore_index=True)


@dataclass
class Ratings:
    players: dict[str, int]
    attack: np.ndarray
    defence: np.ndarray
    base: float
    home_advantage: float

    def lineup(self, player_ids: list[str]) -> tuple[float, float]:
        """Attack and defence of a line-up; unknown players count as average (zero)."""
        known = [self.players[p] for p in player_ids if p in self.players]
        return float(self.attack[known].sum()), float(self.defence[known].sum())


def fit(matches: pd.DataFrame, as_of: pd.Timestamp, settings: RatingSettings = RatingSettings()) -> Ratings:
    """Ratings from every stretch played strictly before `as_of`."""
    window = matches[(matches.date < as_of) & (matches.date >= as_of - pd.Timedelta(days=HISTORY_DAYS))]
    players: dict[str, int] = {}
    rows_h, rows_a, cols_h, cols_a, minutes, y_home, y_away, age = [], [], [], [], [], [], [], []
    k = 0
    for match in window.itertuples():
        days = (as_of - match.date).days
        for s in match.stretches:
            for pid in s["home_players"]:
                rows_h.append(k)
                cols_h.append(players.setdefault(pid, len(players)))
            for pid in s["away_players"]:
                rows_a.append(k)
                cols_a.append(players.setdefault(pid, len(players)))
            minutes.append(s["minutes"])
            y_home.append(s["home_xg"])
            y_away.append(s["away_xg"])
            age.append(days)
            k += 1
    n = len(players)
    on_home = sparse.csr_matrix((np.ones(len(rows_h)), (rows_h, cols_h)), shape=(k, n))
    on_away = sparse.csr_matrix((np.ones(len(rows_a)), (rows_a, cols_a)), shape=(k, n))
    exposure = np.array(minutes) / MATCH_MINUTES
    weight = np.exp(-settings.decay_per_day * np.array(age))
    y_home, y_away = np.array(y_home), np.array(y_away)

    def loss(x):
        base, advantage, attack, defence = x[0], x[1], x[2:2 + n], x[2 + n:]
        eta_home = base + advantage + on_home @ attack - on_away @ defence
        eta_away = base + on_away @ attack - on_home @ defence
        rate_home, rate_away = exposure * np.exp(eta_home), exposure * np.exp(eta_away)
        value = np.sum(weight * (rate_home - y_home * eta_home + rate_away - y_away * eta_away))
        value += settings.penalty * (attack @ attack + defence @ defence)
        d_home, d_away = weight * (rate_home - y_home), weight * (rate_away - y_away)
        grad = np.empty_like(x)
        grad[0] = d_home.sum() + d_away.sum()
        grad[1] = d_home.sum()
        grad[2:2 + n] = on_home.T @ d_home + on_away.T @ d_away + 2 * settings.penalty * attack
        grad[2 + n:] = -(on_away.T @ d_home) - on_home.T @ d_away + 2 * settings.penalty * defence
        return value, grad

    x0 = np.zeros(2 + 2 * n)
    x0[0] = np.log(1.3)
    result = minimize(loss, x0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    return Ratings(players, result.x[2:2 + n], result.x[2 + n:], float(result.x[0]), float(result.x[1]))


def validation_loss(matches: pd.DataFrame, settings: RatingSettings, start: str, end: str) -> float:
    """Poisson loss of the goals of matches between `start` and `end`, predicted from their actual players."""
    ratings = fit(matches, pd.Timestamp(start), settings)
    test = matches[(matches.date >= start) & (matches.date < end)]
    total, count = 0.0, 0
    for match in test.itertuples():
        for s in match.stretches:
            home_attack, home_defence = ratings.lineup(s["home_players"])
            away_attack, away_defence = ratings.lineup(s["away_players"])
            exposure = s["minutes"] / MATCH_MINUTES
            for y, eta in ((s["home_xg"], ratings.base + ratings.home_advantage + home_attack - away_defence),
                           (s["away_xg"], ratings.base + away_attack - home_defence)):
                total += exposure * np.exp(eta) - y * eta
        count += 1
    return total / count


def understat_team(club: str, teams: list[str]) -> str | None:
    team, score = best_match(club, {t: [t] for t in teams})
    return team if score >= TEAM_THRESHOLD else None


def player_ids(names: list[str], pool: dict[str, str]) -> list[str | None]:
    """Understat ids of UEFA starters, matched by name among the club's recent players; None when not found."""
    out = []
    for name in names:
        best = max(pool.items(), key=lambda item: same_player(name, item[1]), default=None)
        out.append(best[0] if best and same_player(name, best[1]) >= PLAYER_THRESHOLD else None)
    return out


def lineup_changes(matches: pd.DataFrame, ratings: Ratings, team: str, starters: list[str], day: pd.Timestamp) -> dict:
    """How much stronger a line-up is than the club's usual one, in attack and defence."""
    earlier = matches[((matches.home == team) | (matches.away == team)) & (matches.date < day)
                      & (matches.date >= day - pd.Timedelta(days=POOL_DAYS))]
    pool = {}
    for match in earlier.itertuples():
        pool.update(match.home_squad if match.home == team else match.away_squad)
    ids = player_ids(starters, pool)
    usual = [ratings.lineup(m.home_starters if m.home == team else m.away_starters) for m in earlier.tail(USUAL_MATCHES).itertuples()]
    if not usual:
        return {"attack_change": np.nan, "defence_change": np.nan, "matched": 0}
    attack, defence = ratings.lineup([pid for pid in ids if pid])
    return {"attack_change": attack - np.mean([u[0] for u in usual]),
            "defence_change": defence - np.mean([u[1] for u in usual]),
            "matched": sum(pid is not None for pid in ids)}


def replay_lineups(table: pd.DataFrame, settings: RatingSettings = RatingSettings()) -> pd.DataFrame:
    """Line-up changes of both clubs in every replay match, with ratings fitted before its week."""
    matches = read_matches()
    lineups = pd.read_csv("data/lineups.csv")
    starters = lineups[lineups.role == "start"].groupby(["date", "club"]).player.apply(list)
    teams = sorted(set(matches.home) | set(matches.away))
    rows = []
    week = pd.to_datetime(table.date).dt.to_period("W")
    for _, group in table.groupby(week):
        ratings = fit(matches, pd.Timestamp(group.date.min()), settings)
        for m in group.itertuples():
            row = {"index": m.Index}
            for side, club in (("home", m.home), ("away", m.away)):
                team = understat_team(club, teams)
                names = starters.get((m.date, club), [])
                change = (lineup_changes(matches, ratings, team, names, pd.Timestamp(m.date)) if team and names
                          else {"attack_change": np.nan, "defence_change": np.nan, "matched": 0})
                row.update({f"{side}_{k}": v for k, v in change.items()})
            rows.append(row)
        print(f"week of {group.date.min()}: {len(ratings.players)} players rated", flush=True)
    return pd.DataFrame(rows).set_index("index")


def main() -> None:
    matches = read_matches()
    print(f"{len(matches)} matches, {sum(len(s) for s in matches.stretches)} stretches")
    # Chosen on the second half of 2023/24, before any replay season starts.
    for penalty in (3.0, 10.0, 30.0, 100.0):
        for decay in (0.0005, 0.001, 0.002):
            settings = RatingSettings(penalty, decay)
            print(f"penalty {penalty} decay {decay}: loss {validation_loss(matches, settings, '2024-01-01', '2024-07-01'):.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
