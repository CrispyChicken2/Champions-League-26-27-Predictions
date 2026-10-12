"""The score model: club strengths fitted on history, turned into a Score Distribution.

Each club has an attack strength, a defence strength and a Venue Effect. The
expected goals of a match are

    home: exp(base + home_advantage + attack[home] - defence[away] + venue[home] / 2)
    away: exp(base                  + attack[away] - defence[home] - venue[home] / 2)

with a separate base and home advantage for European matches. Strengths are
fitted by weighted Poisson likelihood: recent matches weigh more, Champions
League matches weigh more, and thin evidence is pulled towards the average.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

EUROPEAN = {"ucl", "uel", "uecl"}
MAX_GOALS = 10
HISTORY_DAYS = 365 * 4


@dataclass(frozen=True)
class Settings:
    # Defaults are the values tuned on the 2024/25 and 2025/26 replay (see clpred.replay).
    decay_per_day: float = 0.001      # weight of a match halves every ln(2)/decay days
    form_boost: float = 0.0           # extra weight of a match played today, fading with form_decay_per_day
    form_decay_per_day: float = 0.02  # how fast that extra weight fades (0.02 halves it every 5 weeks)
    ucl_weight: float = 2.0           # weight of a Champions League match relative to a domestic one
    shots_share: float = 0.0          # share of the target taken from shots on target instead of goals
    xg_share: float = 0.0             # share of the target taken from expected goals (xG) where known
    strength_penalty: float = 0.3     # pull of attack/defence towards the average
    venue_penalty: float = 200.0      # pull of the Venue Effect towards zero
    low_score_rho: float = 0.0       # Dixon-Coles correction for 0-0, 1-0, 0-1, 1-1


@dataclass
class Model:
    clubs: dict[str, int]
    attack: np.ndarray
    defence: np.ndarray
    venue: np.ndarray
    base: float
    home_advantage: float
    european_base: float
    european_home_advantage: float
    weight: np.ndarray                # total match weight behind each club
    rho: float

    def expected_goals(self, home: str, away: str, european: bool = True) -> tuple[float, float]:
        i, j = self.clubs[home], self.clubs[away]
        base = self.base + (self.european_base if european else 0.0)
        advantage = self.home_advantage + (self.european_home_advantage if european else 0.0)
        log_home = base + advantage + self.attack[i] - self.defence[j] + self.venue[i] / 2
        log_away = base + self.attack[j] - self.defence[i] - self.venue[i] / 2
        return float(np.exp(log_home)), float(np.exp(log_away))

    def score_distribution(self, home: str, away: str, european: bool = True) -> np.ndarray:
        return score_matrix(*self.expected_goals(home, away, european), self.rho)


def score_matrix(home_goals: float, away_goals: float, rho: float) -> np.ndarray:
    """Probability of every scoreline; rows are home goals, columns away goals."""
    goals = np.arange(MAX_GOALS + 1)
    matrix = np.outer(poisson.pmf(goals, home_goals), poisson.pmf(goals, away_goals))
    matrix[0, 0] *= 1 - home_goals * away_goals * rho
    matrix[0, 1] *= 1 + home_goals * rho
    matrix[1, 0] *= 1 + away_goals * rho
    matrix[1, 1] *= 1 - rho
    matrix = np.clip(matrix, 0, None)
    return matrix / matrix.sum()


def fit(matches: pd.DataFrame, as_of: str, settings: Settings, start: np.ndarray | None = None) -> Model:
    """Fit club strengths on every match played strictly before `as_of`."""
    as_of_day = pd.Timestamp(as_of)
    dates = pd.to_datetime(matches["date"])
    age = (as_of_day - dates).dt.days.to_numpy()
    kept = (age > 0) & (age <= HISTORY_DAYS)
    data = matches[kept]
    age = age[kept]

    names = sorted(set(data["home"]) | set(data["away"]))
    clubs = {name: k for k, name in enumerate(names)}
    n = len(names)
    home = data["home"].map(clubs).to_numpy()
    away = data["away"].map(clubs).to_numpy()
    european = data["competition"].isin(EUROPEAN).to_numpy().astype(float)

    weight = np.exp(-settings.decay_per_day * age)
    # Form: extra weight on the last few weeks, on top of the slow decay.
    weight = weight * (1 + settings.form_boost * np.exp(-settings.form_decay_per_day * age))
    is_ucl = (data["competition"] == "ucl").to_numpy()
    weight = weight * np.where(is_ucl, settings.ucl_weight, np.where(european > 0, np.sqrt(settings.ucl_weight), 1.0))

    # Target: goals, partly replaced by shots on target scaled to goals where
    # known. Shots are a steadier sign of how a team played than its goals.
    home_goals = data["home_goals"].to_numpy(float)
    away_goals = data["away_goals"].to_numpy(float)
    home_shots = data["home_shots_on_target"].to_numpy(float)
    away_shots = data["away_shots_on_target"].to_numpy(float)
    known = ~np.isnan(home_shots) & ~np.isnan(away_shots)
    shots = home_shots[known].sum() + away_shots[known].sum()
    goals_per_shot = (home_goals[known].sum() + away_goals[known].sum()) / shots if shots > 0 else 0.0
    share = settings.shots_share
    y_home = np.where(known, (1 - share) * home_goals + share * goals_per_shot * np.nan_to_num(home_shots), home_goals)
    y_away = np.where(known, (1 - share) * away_goals + share * goals_per_shot * np.nan_to_num(away_shots), away_goals)

    # Likewise for expected goals (xG), where a source provides them.
    if "home_xg" in data and settings.xg_share > 0:
        home_xg, away_xg = data["home_xg"].to_numpy(float), data["away_xg"].to_numpy(float)
        has_xg = ~np.isnan(home_xg) & ~np.isnan(away_xg)
        y_home = np.where(has_xg, (1 - settings.xg_share) * y_home + settings.xg_share * np.nan_to_num(home_xg), y_home)
        y_away = np.where(has_xg, (1 - settings.xg_share) * y_away + settings.xg_share * np.nan_to_num(away_xg), y_away)

    def unpack(x):
        return x[0], x[1], x[2], x[3], x[4:4 + n], x[4 + n:4 + 2 * n], x[4 + 2 * n:]

    def loss(x):
        base, advantage, eu_base, eu_advantage, attack, defence, venue = unpack(x)
        log_home = base + advantage + european * (eu_base + eu_advantage) + attack[home] - defence[away] + venue[home] / 2
        log_away = base + european * eu_base + attack[away] - defence[home] - venue[home] / 2
        rate_home, rate_away = np.exp(log_home), np.exp(log_away)
        value = np.sum(weight * (rate_home - y_home * log_home + rate_away - y_away * log_away))
        value += settings.strength_penalty * (attack @ attack + defence @ defence) + settings.venue_penalty * (venue @ venue)

        d_home = weight * (rate_home - y_home)
        d_away = weight * (rate_away - y_away)
        grad = np.empty_like(x)
        grad[0] = d_home.sum() + d_away.sum()
        grad[1] = d_home.sum()
        grad[2] = (european * (d_home + d_away)).sum()
        grad[3] = (european * d_home).sum()
        grad[4:4 + n] = (np.bincount(home, d_home, n) + np.bincount(away, d_away, n) + 2 * settings.strength_penalty * attack)
        grad[4 + n:4 + 2 * n] = (-np.bincount(away, d_home, n) - np.bincount(home, d_away, n) + 2 * settings.strength_penalty * defence)
        grad[4 + 2 * n:] = np.bincount(home, (d_home - d_away) / 2, n) + 2 * settings.venue_penalty * venue
        return value, grad

    x0 = np.zeros(4 + 3 * n)
    x0[0], x0[1] = np.log(1.3), 0.2
    if start is not None and len(start) == len(x0):
        x0 = start
    result = minimize(loss, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "maxfun": 4000})
    base, advantage, eu_base, eu_advantage, attack, defence, venue = unpack(result.x)
    club_weight = np.bincount(home, weight, n) + np.bincount(away, weight, n)
    model = Model(clubs, attack, defence, venue, base, advantage, eu_base, eu_advantage, club_weight, settings.low_score_rho)
    model.solution = result.x
    model.converged = bool(result.success)
    return model


def market_distribution(odds_home: float, odds_draw: float, odds_away: float,
                        odds_over: float | None = None, odds_under: float | None = None,
                        rho: float = -0.05) -> np.ndarray:
    """The Score Distribution implied by bookmaker odds (the Baseline).

    The bookmaker's margin is removed, then the pair of expected goals whose
    scorelines best reproduce the odds is found. With only 1X2 odds the draw
    price fixes how many goals are expected; over/under 2.5 sharpens it.
    """
    inverse = np.array([1 / odds_home, 1 / odds_draw, 1 / odds_away])
    targets = list(inverse / inverse.sum())
    if odds_over and odds_under:
        over = (1 / odds_over) / (1 / odds_over + 1 / odds_under)
        targets += [over]

    def miss(log_rates):
        matrix = score_matrix(np.exp(log_rates[0]), np.exp(log_rates[1]), rho)
        got = list(outcome_probabilities(matrix))
        if len(targets) == 4:
            totals = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
            got += [matrix[totals > 2.5].sum()]
        return float(np.sum((np.array(got) - np.array(targets)) ** 2))

    best = min((minimize(miss, np.log(start), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-12})
                for start in ([1.5, 1.1], [2.5, 0.7], [0.7, 2.5])), key=lambda r: r.fun)
    return score_matrix(np.exp(best.x[0]), np.exp(best.x[1]), rho)


def outcome_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """Probability of home win, draw, away win."""
    return float(np.tril(matrix, -1).sum()), float(np.trace(matrix)), float(np.triu(matrix, 1).sum())


def pick(matrix: np.ndarray, exact_points: float = 3, outcome_points: float = 1) -> tuple[int, int]:
    """The scoreline with the highest expected points."""
    home_win, draw, away_win = outcome_probabilities(matrix)
    outcome = np.where(np.tril(np.ones_like(matrix), -1) > 0, home_win, away_win)
    np.fill_diagonal(outcome, draw)
    expected = exact_points * matrix + outcome_points * (outcome - matrix)
    i, j = np.unravel_index(np.argmax(expected), matrix.shape)
    return int(i), int(j)


def points(picked: tuple[int, int], actual: tuple[int, int], exact_points: int = 3, outcome_points: int = 1) -> int:
    if picked == actual:
        return exact_points
    return outcome_points if np.sign(picked[0] - picked[1]) == np.sign(actual[0] - actual[1]) else 0
