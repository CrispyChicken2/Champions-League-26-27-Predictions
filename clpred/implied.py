"""Expected goals implied by bookmaker 1X2 odds, for many matches at once.

The odds of a match say how strong the bookmakers thought each side was on
the day, which is a steadier measure of a club than the goals it happened to
score. Each match's odds are turned into the pair of expected goals whose
home-win and away-win probabilities sit closest to the odds.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import poisson

MAX_GOALS = 12
GRID = np.arange(0.15, 4.51, 0.02)


def _table() -> tuple[cKDTree, np.ndarray]:
    goals = np.arange(MAX_GOALS + 1)
    pmf = poisson.pmf(goals[None, :], GRID[:, None])               # rate x goals
    joint = pmf[:, None, :, None] * pmf[None, :, None, :]           # home rate x away rate x home goals x away goals
    home_win = joint[:, :, np.tril_indices(MAX_GOALS + 1, -1)[0], np.tril_indices(MAX_GOALS + 1, -1)[1]].sum(-1)
    away_win = joint[:, :, np.triu_indices(MAX_GOALS + 1, 1)[0], np.triu_indices(MAX_GOALS + 1, 1)[1]].sum(-1)
    rates = np.stack(np.meshgrid(GRID, GRID, indexing="ij"), -1).reshape(-1, 2)
    return cKDTree(np.stack([home_win.ravel(), away_win.ravel()], 1)), rates


def implied_goals(odds_home: pd.Series, odds_draw: pd.Series, odds_away: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Expected home and away goals implied by the odds; NaN where odds are missing."""
    inverse = np.stack([1 / odds_home.to_numpy(float), 1 / odds_draw.to_numpy(float), 1 / odds_away.to_numpy(float)], 1)
    known = np.isfinite(inverse).all(1) & (inverse > 0).all(1)
    fair = inverse[known] / inverse[known].sum(1, keepdims=True)   # the bookmaker's margin removed
    tree, rates = _table()
    _, nearest = tree.query(fair[:, [0, 2]])
    home, away = np.full(len(inverse), np.nan), np.full(len(inverse), np.nan)
    home[known], away[known] = rates[nearest, 0], rates[nearest, 1]
    return home, away
