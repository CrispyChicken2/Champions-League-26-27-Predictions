# Champions League 2026/27 score predictions

Predicts the score of every league-phase match, Matchday by Matchday (2 to 8), and measures the Picks against the bookmakers. Terms are defined in [CONTEXT.md](CONTEXT.md).

## Layout

- `matchday-NN/` — one folder per Matchday: `source/` (the Snapshot: raw saved pages), `odds.csv`, `absences.csv`, `predictions.csv`, `score_distributions.csv`, `rationale.json`, `report.html`.
- `data/` — `matches.csv` (all history), `fixtures.csv`, `odds_ucl_history.csv`, `replay_summary.json`, plus saved source pages in `recent/` and `odds_history/`. `data/history/` is re-downloadable and not in git.
- `clpred/` — the pipeline. `scripts/` — downloading and page capture.

## Running a Matchday

Two commands, run from the project folder:

```
uv run scripts/run_matchday.py N             before kick-off: capture, predict, write the report (about 35 minutes)
uv run scripts/run_matchday.py N --results   after the matches: fill in the Results
```

The first can be run again on a match day for fresher odds and absences; matches already kicked off keep their earlier Prediction. It does not write the Rationale texts (`matchday-NN/rationale.json`).

The steps it runs, for when one of them needs repeating on its own:

1. `uv run scripts/download_history.py` — domestic CSVs, UEFA's match feed and Understat xG.
2. `uv run python -m clpred.build_matches` — builds `data/matches.csv` and `data/fixtures.csv`. Read every `CHECK` and `LEFT OUT` line: they are club names matched with doubt or not at all.
3. `uv run scripts/capture_matchday.py N` — with a browser: latest league rounds into `data/recent/`, the Snapshot (odds, absences, squads) into `matchday-NN/source/`, player performance into `data/players/`. Add `--list` to see the pages without fetching them.
4. `uv run python -m clpred.build_matches` again, then `uv run python -m clpred.players`.
5. `uv run python -m clpred.snapshot N` — extracts `odds.csv`, `absences.csv` and `squads.csv`. Squad pages of calendar-year leagues (Norway) must be captured without a season in the URL.
6. `uv run python -m clpred.predict N` — writes the Predictions, with Availability applied.
7. Write `matchday-NN/rationale.json` (two or three sentences per match).
8. `uv run python -m clpred.report N` — writes `report.html`. 

`uv run python -m clpred.replay [--tune]` replays the 2024/25 and 2025/26 league phases to judge or re-tune the model.

`uv run scripts/download_lineups.py` then `uv run python -m clpred.availability` re-learns how much absent players change the score, from past line-ups.

`uv run python -m clpred.experiments` tests candidate improvements (rest days, travel, squad value, draw frequency, Availability) on held-out replay seasons. Settings that were tested and left at zero because they did not help: `shots_share`, `xg_share`, `odds_share`, `form_boost` in `clpred/model.py`.

Also tested after Matchday 2 and rejected (held-out blend log-loss 2.9511, 650 points, 85 exact for the current method):

| Idea | Log-loss | Points | Exact |
|---|---|---|---|
| More goals in the market part (factor learned per season) | 2.9521 | 648 | 84 |
| One strength per league from European results | 2.9590 | 640 | 86 |
| Negative binomial instead of Poisson (wider spread of scores) | 2.9544 or worse | 647–651 | 84–87 |
| Ratings that change over time (Kalman filter) | 2.9528 | 641 | 82 |
| Half static, half changing ratings | 2.9516 | 648 | 85 |
| Line-up strength from player ratings (actual starters against usual starters) | 2.9525 | 648 | 85 |

Player ratings (`clpred/player_ratings.py`, data from `uv run scripts/download_understat_matches.py`) rate every player of the five biggest leagues from the xG his team creates and concedes while he is on the pitch (7,378 matches cut at each substitution). They measure team strength, but on 1,700 domestic matches the actual starters predicted xG no better than the club's usual starters (loss 1.6375 against 1.6370), and the replay coefficients changed sign between seasons. They are not used in the predictions.

`uv run pytest` runs the automated tests (club-name matching, the model, Availability, the page readers).

Package installs need `uv add --system-certs` on this machine.
