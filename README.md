# Champions League 2026/27 score predictions

Predicts the score of every league-phase match, Matchday by Matchday (2 to 8), and measures the Picks against the bookmakers. Terms are defined in [CONTEXT.md](CONTEXT.md).

## Layout

- `matchday-NN/` — one folder per Matchday: `source/` (the Snapshot: raw saved pages), `odds.csv`, `absences.csv`, `predictions.csv`, `score_distributions.csv`, `rationale.json`, `report.html`.
- `data/` — `matches.csv` (all history), `fixtures.csv`, `odds_ucl_history.csv`, `replay_summary.json`, plus saved source pages in `recent/` and `odds_history/`. `data/history/` is re-downloadable and not in git.
- `clpred/` — the pipeline. `scripts/` — downloading and page capture.

## Running a Matchday

Run from the project folder, in this order.

1. `uv run scripts/download_history.py` — domestic CSVs and UEFA's match feed.
2. `uv run scripts/capture_raw.py data/recent <league>=<url> ...` — BetExplorer results pages for the rounds the CSVs lag behind (see `data/recent/manifest.json` for the URLs used last time).
3. `uv run python -m clpred.build_matches` — rebuilds `data/matches.csv`. Read every `CHECK` and `LEFT OUT` line: they are club names matched with doubt or not at all.
4. `uv run scripts/capture_raw.py matchday-NN/source <name>=<url> ...` — the Snapshot, before the first kick-off: OddsPortal 1X2, over/under and correct-score pages per match, Transfermarkt absences and squad per club (`matchday-02/source/manifest.json` shows the naming).
5. `uv run python -m clpred.snapshot N` — extracts `odds.csv` and `absences.csv`.
6. `uv run python -m clpred.predict N` — writes the Predictions.
7. Write `matchday-NN/rationale.json` (two or three sentences per match).
8. `uv run python -m clpred.report N` — writes `report.html`. Run steps 1, 3 and 8 again after the matches to fill in the Results.

`uv run python -m clpred.replay [--tune]` replays the 2024/25 and 2025/26 league phases to judge or re-tune the model.

Package installs need `uv add --system-certs` on this machine.
