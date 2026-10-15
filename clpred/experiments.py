"""Try candidate improvements on the replay and say honestly whether each helps.

Usage: uv run python -m clpred.experiments

Each candidate is learned on one replay season and judged on the other, both
ways round. A candidate is worth keeping only if the blend's exact-score
log-loss drops on both held-out seasons; points are shown too but are noisier.
"""

import io
import json
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from clpred.model import EUROPEAN, Settings, market_distribution, pick, points, score_matrix
from clpred.replay import BLEND_GRID, REPLAY_SEASONS, expected_goals, replay_matches

REST_CAP_DAYS = 7
KNOWN_SCHEDULE_DAYS = 21


def rest_days(matches: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Days since each club's previous match; missing when its domestic schedule is not in the history."""
    played = pd.concat([matches[["date", "home", "competition"]].rename(columns={"home": "club"}),
                        matches[["date", "away", "competition"]].rename(columns={"away": "club"})])
    played["date"] = pd.to_datetime(played.date)
    by_club = {club: group.sort_values("date") for club, group in played.groupby("club")}
    out = {"home_rest": [], "away_rest": []}
    for match in table.itertuples():
        day = pd.Timestamp(match.date)
        for side, club in (("home_rest", match.home), ("away_rest", match.away)):
            earlier = by_club[club][by_club[club].date < day]
            recent = earlier[earlier.date >= day - pd.Timedelta(days=KNOWN_SCHEDULE_DAYS)]
            has_domestic = (~recent.competition.isin(EUROPEAN)).any()
            out[side].append((day - earlier.date.max()).days if has_domestic else np.nan)
    return pd.DataFrame(out, index=table.index)


def kilometres(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (*a, *b))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def travel(table: pd.DataFrame) -> pd.Series:
    """Distance in km from the away club's usual stadium to the match's stadium."""
    home_ground: dict[str, list] = {}
    venue = {}
    for path in Path("data/history").glob("european_*.json"):
        for match in json.load(io.open(path, encoding="utf-8")):
            place = (match.get("stadium") or {}).get("geolocation")
            if not place:
                continue
            spot = (place["latitude"], place["longitude"])
            home_ground.setdefault(match["homeTeam"]["internationalName"], []).append(spot)
            venue[(match["kickOffTime"]["date"], match["homeTeam"]["internationalName"])] = spot
    usual = {club: max(set(spots), key=spots.count) for club, spots in home_ground.items()}
    return pd.Series([kilometres(usual[m.away], venue[(m.date, m.home)])
                      if m.away in usual and (m.date, m.home) in venue else np.nan for m in table.itertuples()],
                     index=table.index)


def fit_terms(train: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """A level shift plus one coefficient per column; a column raises the home rate and lowers the away rate."""
    x = train[columns].to_numpy(float)

    def loss(c):
        shift = x @ c[1:]
        home, away = np.log(train.home_rate) + c[0] + shift, np.log(train.away_rate) + c[0] - shift
        return float(np.sum(np.exp(home) - train.home_goals * home + np.exp(away) - train.away_goals * away))
    return minimize(loss, np.zeros(len(columns) + 1), method="BFGS").x


def with_terms(table: pd.DataFrame, columns: list[str], c: np.ndarray) -> pd.DataFrame:
    shift = table[columns].to_numpy(float) @ c[1:] if columns else 0.0
    return table.assign(home_rate=table.home_rate * np.exp(c[0] + shift), away_rate=table.away_rate * np.exp(c[0] - shift))


def inflate_draws(matrix: np.ndarray, extra: float) -> np.ndarray:
    out = matrix.copy()
    out[np.diag_indices_from(out)] *= 1 + extra
    return out / out.sum()


def judge(table: pd.DataFrame, market: dict, share: float, draws: float = 0.0) -> dict:
    loss, total, exact = 0.0, 0, 0
    for m in table.itertuples():
        blend = share * inflate_draws(score_matrix(m.home_rate, m.away_rate, 0.0), draws) + (1 - share) * market[m.Index]
        actual = (int(m.home_goals), int(m.away_goals))
        loss -= np.log(max(blend[min(actual[0], 10), min(actual[1], 10)], 1e-12))
        picked = pick(blend)
        total += points(picked, actual)
        exact += picked == actual
    return {"log_loss": loss / len(table), "points": total, "exact": exact}


def held_out(table: pd.DataFrame, market: dict, columns: list[str], draw_grid=(0.0,), fixed_share: float | None = None) -> dict:
    """Learn everything on one season, judge on the other; returns the two held-out seasons added up."""
    result = {"log_loss": 0.0, "points": 0, "exact": 0, "per_season": [], "learned": []}
    for test in REPLAY_SEASONS:
        train, test_rows = table[table.season != test], table[table.season == test]
        c = fit_terms(train, columns)
        train, test_rows = with_terms(train, columns, c), with_terms(test_rows, columns, c)
        shares = BLEND_GRID if fixed_share is None else (fixed_share,)
        share, draws = min(((s, d) for s in shares for d in draw_grid),
                           key=lambda sd: judge(train, market, sd[0], sd[1])["log_loss"])
        verdict = judge(test_rows, market, share, draws)
        result["log_loss"] += verdict["log_loss"] * len(test_rows) / len(table)
        result["points"] += verdict["points"]
        result["exact"] += verdict["exact"]
        result["per_season"].append(round(verdict["log_loss"], 4))
        result["learned"].append({"coefficients": [round(float(v), 4) for v in c[1:]], "share": float(share), "draws": draws})
    return result


def availability_held_out(table: pd.DataFrame, market: dict) -> tuple[dict, dict]:
    """Held-out verdicts without and with Availability, on the matches where it is known."""
    from clpred.availability import centred, fit_effects, log_rates, with_missing

    known = with_missing(table.reset_index().rename(columns={"index": "row"})).set_index("row")
    results = []
    for use in (False, True):
        result = {"log_loss": 0.0, "points": 0, "exact": 0, "per_season": [], "learned": []}
        for test in REPLAY_SEASONS:
            train, typical = centred(known[known.season != test])
            test_rows, _ = centred(known[known.season == test], typical)
            x = fit_effects(train, with_availability=use)
            parts = []
            for rows in (train, test_rows):
                home, away = log_rates(rows, x)
                parts.append(rows.assign(home_rate=np.exp(home), away_rate=np.exp(away)))
            share = min(BLEND_GRID, key=lambda s: judge(parts[0], market, s)["log_loss"])
            verdict = judge(parts[1], market, share)
            result["log_loss"] += verdict["log_loss"] * len(test_rows) / len(known)
            result["points"] += verdict["points"]
            result["exact"] += verdict["exact"]
            result["per_season"].append(round(verdict["log_loss"], 4))
            result["learned"].append({"coefficients": [round(float(v), 3) for v in x[1:]], "share": float(share), "draws": 0.0})
        results.append(result)
    print(f"Availability known for {len(known)} matches")
    return results[0], results[1]


def lineup_terms(table: pd.DataFrame, c: np.ndarray) -> pd.DataFrame:
    """Rates moved by how much stronger each club's line-up is than usual (player ratings); unknown counts as usual."""
    f = table[["home_attack_change", "home_defence_change", "away_attack_change", "away_defence_change"]].fillna(0.0)
    home = np.log(table.home_rate) + c[0] + c[1] * f.home_attack_change - c[2] * f.away_defence_change
    away = np.log(table.away_rate) + c[0] + c[1] * f.away_attack_change - c[2] * f.home_defence_change
    return table.assign(home_rate=np.exp(home), away_rate=np.exp(away))


def fit_lineup_terms(train: pd.DataFrame) -> np.ndarray:
    def loss(c):
        moved = lineup_terms(train, c)
        return float(np.sum(moved.home_rate - train.home_goals * np.log(moved.home_rate)
                            + moved.away_rate - train.away_goals * np.log(moved.away_rate)))
    return minimize(loss, np.zeros(3), method="Nelder-Mead").x


def lineup_held_out(table: pd.DataFrame, market: dict) -> dict:
    """Held-out verdict with line-up strength from player ratings."""
    result = {"log_loss": 0.0, "points": 0, "exact": 0, "per_season": [], "learned": []}
    for test in REPLAY_SEASONS:
        train, test_rows = table[table.season != test], table[table.season == test]
        c = fit_lineup_terms(train)
        train, test_rows = lineup_terms(train, c), lineup_terms(test_rows, c)
        share = min(BLEND_GRID, key=lambda s: judge(train, market, s)["log_loss"])
        verdict = judge(test_rows, market, share)
        result["log_loss"] += verdict["log_loss"] * len(test_rows) / len(table)
        result["points"] += verdict["points"]
        result["exact"] += verdict["exact"]
        result["per_season"].append(round(verdict["log_loss"], 4))
        result["learned"].append({"coefficients": [round(float(v), 3) for v in c], "share": float(share), "draws": 0.0})
    return result


def show(name: str, result: dict, reference: dict | None = None) -> None:
    better = ""
    if reference:
        both = all(a < b for a, b in zip(result["per_season"], reference["per_season"]))
        better = "  BETTER on both seasons" if both else "  not better on both seasons"
    print(f"{name:34s} log-loss {result['log_loss']:.4f} {result['per_season']}  points {result['points']}  "
          f"exact {result['exact']}{better}")
    print(f"{'':34s} learned {result['learned']}")


def main() -> None:
    matches, targets = replay_matches()
    table = expected_goals(matches, targets, Settings())
    market = {m.Index: market_distribution(m.odds_home, m.odds_draw, m.odds_away, rho=0.0) for m in table.itertuples()}

    rest = rest_days(matches, table)
    gap = rest.home_rest.clip(upper=REST_CAP_DAYS) - rest.away_rest.clip(upper=REST_CAP_DAYS)
    table["rest_gap"] = gap.fillna(0.0)
    table["short_rest_gap"] = ((rest.away_rest <= 3).astype(float) - (rest.home_rest <= 3).astype(float)).fillna(0.0)
    distance = travel(table)
    table["travel"] = (np.log1p(distance / 1000) - np.log1p(distance / 1000).mean()).fillna(0.0)
    print(f"{len(table)} matches; rest known for both clubs in {int(gap.notna().sum())}, travel in {int(distance.notna().sum())}")

    values = pd.read_csv("data/squad_values.csv").set_index(["club", "season"]).best_eleven_m
    log_value = {key: np.log(v) for key, v in values.items() if v > 0}
    gap = pd.Series([log_value.get((m.home, m.season), np.nan) - log_value.get((m.away, m.season), np.nan)
                     for m in table.itertuples()], index=table.index)
    # Only the part of the value gap the ratings do not already see can add anything.
    rating_gap = np.log(table.home_rate) - np.log(table.away_rate)
    known = gap.notna()
    slope, intercept = np.polyfit(rating_gap[known], gap[known], 1)
    table["value_gap"] = (gap - gap.mean()).fillna(0.0)
    table["value_surprise"] = (gap - (slope * rating_gap + intercept)).fillna(0.0)
    print(f"squad value known for both clubs in {int(known.sum())} matches; "
          f"correlation with the ratings' gap {np.corrcoef(rating_gap[known], gap[known])[0, 1]:.2f}")

    reference = held_out(table, market, [])
    show("current method", reference)
    show("+ rest-day gap", held_out(table, market, ["rest_gap"]), reference)
    show("+ short-rest gap (3 days or fewer)", held_out(table, market, ["short_rest_gap"]), reference)
    show("+ travel distance", held_out(table, market, ["travel"]), reference)
    show("+ squad-value gap", held_out(table, market, ["value_gap"]), reference)
    show("+ squad value beyond the ratings", held_out(table, market, ["value_surprise"]), reference)
    without, with_it = availability_held_out(table, market)
    show("same matches, no Availability", without)
    show("+ Availability (attack, defence)", with_it, without)
    show("+ more or fewer draws", held_out(table, market, [], draw_grid=(-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2)), reference)


if __name__ == "__main__":
    main()
