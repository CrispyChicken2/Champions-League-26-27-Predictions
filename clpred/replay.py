"""Replay past Champions League league phases to tune and judge the model.

Usage: uv run python -m clpred.replay            judge the current settings
       uv run python -m clpred.replay --tune     search for better settings

Every match of the 2024/25 and 2025/26 league phases is predicted using only
matches played before its Matchday, then compared with what happened and with
the bookmaker Baseline of the time.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from clpred.model import Settings, fit, market_distribution, outcome_probabilities, pick, points, score_matrix

REPLAY_SEASONS = (2025, 2026)
BLEND_GRID = np.round(np.arange(0, 1.01, 0.05), 2)
RHO_GRID = (0.0, -0.03, -0.06, -0.09, -0.12)


def replay_matches() -> pd.DataFrame:
    matches = pd.read_csv("data/matches.csv", low_memory=False)
    odds = pd.read_csv("data/odds_ucl_history.csv")
    league_phase = matches[(matches.competition == "ucl") & (matches.stage == "League Phase")
                           & matches.season.isin(REPLAY_SEASONS)]
    return matches, league_phase.merge(odds, on=["date", "home", "away"], suffixes=("_none", ""))


def expected_goals(matches: pd.DataFrame, targets: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Model expected goals for every replay match, fitted Matchday by Matchday."""
    rows, start = [], None
    for (season, matchday), group in targets.groupby(["season", "matchday"]):
        model = fit(matches, group.date.min(), settings, start)
        start = model.solution
        for match in group.itertuples():
            home_rate, away_rate = model.expected_goals(match.home, match.away)
            rows.append({"index": match.Index, "home_rate": home_rate, "away_rate": away_rate})
    return targets.join(pd.DataFrame(rows).set_index("index"))


def judge(table: pd.DataFrame, rho: float, blend: float, market: list[np.ndarray]) -> dict:
    """Score the replay for one low-score correction and one model/market blend."""
    log_loss = {"model": 0.0, "baseline": 0.0, "blend": 0.0}
    rps = dict.fromkeys(log_loss, 0.0)
    total = dict.fromkeys(log_loss, 0)
    exact = dict.fromkeys(log_loss, 0)
    for match, market_matrix in zip(table.itertuples(), market):
        actual = (int(match.home_goals), int(match.away_goals))
        model_matrix = score_matrix(match.home_rate, match.away_rate, rho)
        candidates = {"model": model_matrix, "baseline": market_matrix,
                      "blend": blend * model_matrix + (1 - blend) * market_matrix}
        result = np.array([actual[0] > actual[1], actual[0] == actual[1], actual[0] < actual[1]], float)
        for name, matrix in candidates.items():
            log_loss[name] -= np.log(max(matrix[min(actual[0], 10), min(actual[1], 10)], 1e-12))
            cumulative = np.cumsum(outcome_probabilities(matrix)) - np.cumsum(result)
            rps[name] += float(np.sum(cumulative[:2] ** 2) / 2)
            picked = pick(matrix)
            total[name] += points(picked, actual)
            exact[name] += picked == actual
    n = len(table)
    return {name: {"log_loss": log_loss[name] / n, "rps": rps[name] / n, "points": total[name], "exact": exact[name]}
            for name in log_loss}


def market_matrices(table: pd.DataFrame, rho: float) -> list[np.ndarray]:
    return [market_distribution(m.odds_home, m.odds_draw, m.odds_away, rho=rho) for m in table.itertuples()]


def model_log_loss(table: pd.DataFrame, rho: float) -> float:
    loss = 0.0
    for m in table.itertuples():
        matrix = score_matrix(m.home_rate, m.away_rate, rho)
        loss -= np.log(max(matrix[min(int(m.home_goals), 10), min(int(m.away_goals), 10)], 1e-12))
    return loss / len(table)


def tune(matches: pd.DataFrame, targets: pd.DataFrame, settings: Settings) -> Settings:
    """Improve one setting at a time, keeping a change only if the model's log-loss drops."""
    grid = {
        "decay_per_day": (0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004),
        "ucl_weight": (1.0, 1.5, 2.0, 3.0, 4.0),
        "shots_share": (0.0, 0.15, 0.3, 0.45, 0.6),
        "venue_penalty": (5.0, 20.0, 60.0, 200.0),
        "strength_penalty": (0.3, 1.0, 3.0),
    }
    best = model_log_loss(expected_goals(matches, targets, settings), settings.low_score_rho)
    print(f"start {best:.4f} {settings}")
    for _ in range(2):
        for name, values in grid.items():
            for value in values:
                if value == getattr(settings, name):
                    continue
                candidate = replace(settings, **{name: value})
                loss = model_log_loss(expected_goals(matches, targets, candidate), candidate.low_score_rho)
                print(f"  {name}={value}: {loss:.4f}", flush=True)
                if loss < best - 1e-5:
                    best, settings = loss, candidate
            print(f"after {name}: {best:.4f} {settings}", flush=True)
    return settings


def main() -> None:
    matches, targets = replay_matches()
    settings = Settings()
    if "--tune" in sys.argv:
        settings = tune(matches, targets, settings)
    table = expected_goals(matches, targets, settings)

    rho = min(RHO_GRID, key=lambda r: model_log_loss(table, r))
    market = market_matrices(table, rho)
    blend = min(BLEND_GRID, key=lambda b: judge(table, rho, b, market)["blend"]["log_loss"])
    verdict = judge(table, rho, blend, market)
    print(f"\n{len(table)} matches, settings {settings}")
    print(f"best low-score correction {rho}, best model share in blend {blend}")
    print(pd.DataFrame(verdict).T.round(4).to_string())
    summary = {"matches": len(table), "model_share": float(blend), "low_score_rho": float(rho), "verdict": verdict}
    Path("data/replay_summary.json").write_text(json.dumps(summary, indent=2, default=float), "utf-8")

    # Honest check of the blend: choose its share on one season, judge it on the other.
    for test in REPLAY_SEASONS:
        train_rows, test_rows = (table.season != test).to_numpy(), (table.season == test).to_numpy()
        train_market = [m for m, keep in zip(market, train_rows) if keep]
        test_market = [m for m, keep in zip(market, test_rows) if keep]
        share = min(BLEND_GRID, key=lambda b: judge(table[train_rows], rho, b, train_market)["blend"]["log_loss"])
        print(f"\nseason ending {test}, model share {share} chosen on the other season")
        print(pd.DataFrame(judge(table[test_rows], rho, share, test_market)).T.round(4).to_string())


if __name__ == "__main__":
    main()
