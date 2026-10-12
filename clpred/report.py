"""Write the HTML report of a Matchday.

Usage: uv run python -m clpred.report <matchday number>

Reads matchday-NN/predictions.csv, rationale.json and absences.csv, and the
match table for any Results already known. Writes matchday-NN/report.html.
"""

import html
import json
import sys
from pathlib import Path

import pandas as pd

from clpred.model import points

STYLE = """
body { font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color: #1c1c1c; line-height: 1.45; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { padding: 0.35rem 0.5rem; border-bottom: 1px solid #ddd; text-align: left; }
th { background: #f3f3f3; } td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.match { border: 1px solid #ddd; border-radius: 6px; padding: 0.8rem 1rem; margin: 0.8rem 0; }
.match h3 { margin: 0 0 0.3rem; font-size: 1rem; } .muted { color: #666; font-size: 0.85rem; }
.exact { color: #0a7a2f; font-weight: 600; } .outcome { color: #8a6d00; font-weight: 600; } .miss { color: #b3261e; font-weight: 600; }
.warn { color: #b3261e; }
"""


def percent(value: float) -> str:
    return f"{value:.0%}" if pd.notna(value) else "n/a"


def result_of(pick: str, home_goals: int, away_goals: int) -> tuple[str, int]:
    picked = tuple(int(x) for x in pick.split("-"))
    earned = points(picked, (home_goals, away_goals))
    return {3: "exact", 1: "outcome"}.get(earned, "miss"), earned


def main() -> None:
    matchday = int(sys.argv[1])
    folder = Path(f"matchday-{matchday:02d}")
    predictions = pd.read_csv(folder / "predictions.csv")
    rationale = json.loads((folder / "rationale.json").read_text("utf-8")) if (folder / "rationale.json").exists() else {}
    absences = pd.read_csv(folder / "absences.csv") if (folder / "absences.csv").exists() else pd.DataFrame(columns=["club"])
    matches = pd.read_csv("data/matches.csv", low_memory=False)
    played = matches[(matches.competition == "ucl") & (matches.date >= predictions.date.min())]
    scores = {(m.home, m.away): (int(m.home_goals), int(m.away_goals)) for m in played.itertuples()}
    replay = json.loads(Path("data/replay_summary.json").read_text("utf-8")) if Path("data/replay_summary.json").exists() else None

    rows, cards = [], []
    totals = {"ours": 0, "baseline": 0, "exact": 0, "outcome": 0, "known": 0}
    for p in predictions.itertuples():
        score = scores.get((p.home, p.away))
        result_cell, result_line = '<td class="muted">not played</td>', ""
        if score:
            verdict, earned = result_of(p.pick, *score)
            totals["known"] += 1
            totals["ours"] += earned
            totals["exact"] += verdict == "exact"
            totals["outcome"] += verdict != "miss"
            if isinstance(p.baseline_pick, str):
                totals["baseline"] += result_of(p.baseline_pick, *score)[1]
            result_cell = f'<td class="{verdict}">{score[0]}-{score[1]} · {verdict} · {earned} pt</td>'
            result_line = f'<p>Result: <span class="{verdict}">{score[0]}-{score[1]}, {verdict}, {earned} points</span></p>'
        baseline = p.baseline_pick if isinstance(p.baseline_pick, str) else "n/a"
        rows.append(
            f"<tr><td>{html.escape(p.home)} – {html.escape(p.away)}</td><td><b>{p.pick}</b></td>"
            f'<td class="n">{percent(p.confidence_pick)}</td><td class="n">{percent(p.confidence_outcome)}</td>'
            f'<td class="n">{percent(p.prob_home)} / {percent(p.prob_draw)} / {percent(p.prob_away)}</td>'
            f"<td>{baseline}</td>{result_cell}</tr>"
        )
        out = []
        for club in (p.home, p.away):
            names = absences[absences.club == club].player.drop_duplicates().tolist()
            if names:
                out.append(f"{html.escape(club)}: {html.escape(', '.join(names))}")
        degraded = f'<p class="warn">Missing input: {html.escape(p.degraded)}</p>' if isinstance(p.degraded, str) and p.degraded else ""
        cards.append(
            f'<div class="match"><h3>{html.escape(p.home)} – {html.escape(p.away)}: {p.pick}</h3>'
            f'<p class="muted">Kick-off {p.kickoff_utc} · Confidence: {percent(p.confidence_pick)} for the exact score, '
            f"{percent(p.confidence_outcome)} for the outcome · likeliest scores: {p.top_scores}</p>"
            f"<p>{html.escape(rationale.get(f'{p.home}|{p.away}', 'No Rationale written.'))}</p>"
            f'<p class="muted">Home / draw / away: model {percent(p.model_prob_home)} / {percent(p.model_prob_draw)} / {percent(p.model_prob_away)}'
            + (f", Baseline {percent(p.baseline_prob_home)} / {percent(p.baseline_prob_draw)} / {percent(p.baseline_prob_away)}" if baseline != "n/a" else "")
            + f" · expected goals {p.expected_home_goals} – {p.expected_away_goals}</p>"
            + (f'<p class="muted">Listed absences (not yet used by the model): {"; ".join(out)}</p>' if out else "")
            + degraded + result_line + "</div>"
        )

    if totals["known"]:
        summary = (f"<p><b>Results so far:</b> {totals['known']} of {len(predictions)} matches played. "
                   f"Our Picks: {totals['ours']} points ({totals['exact']} exact, {totals['outcome']} right outcome). "
                   f"Baseline Picks: {totals['baseline']} points.</p>")
    else:
        summary = "<p><b>Results:</b> no match of this Matchday has been played yet.</p>"

    replay_note = ""
    if replay:
        v = replay["verdict"]
        replay_note = (
            f"<h2>How much to trust this</h2><p>Replayed on the {replay['matches']} league-phase matches of 2024/25 and 2025/26, "
            f"predicting each from earlier matches only: our blend scored {v['blend']['points']:.0f} points with "
            f"{v['blend']['exact']:.0f} exact scores; the bookmaker Baseline scored {v['baseline']['points']:.0f} with "
            f"{v['baseline']['exact']:.0f}; the model alone {v['model']['points']:.0f} with {v['model']['exact']:.0f}. "
            f"The blend gives slightly better probabilities for exact scores than the Baseline (log-loss "
            f"{v['blend']['log_loss']:.3f} against {v['baseline']['log_loss']:.3f}, lower is better) but is not better at "
            f"calling the winner. In the replay the blend hit {v['blend']['exact'] / replay['matches']:.0%} of exact scores and "
            f"{(v['blend']['points'] - 2 * v['blend']['exact']) / replay['matches']:.0%} of outcomes. "
            f"Nothing here has yet shown an edge over the bookmakers that would justify betting.</p>"
        )

    captured = predictions.odds_captured_at_utc.dropna()
    snapshot_note = f"Odds captured {captured.min()[:16]} to {captured.max()[:16]} UTC." if len(captured) else "No odds in the Snapshot."
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Champions League 2026/27 – Matchday {matchday}</title>
<style>{STYLE}</style></head><body>
<h1>Champions League 2026/27 – Matchday {matchday} Predictions</h1>
<p class="muted">{snapshot_note} History used: every match up to the day before the first kick-off.
Pick = scoreline with the highest expected points (3 exact, 1 right outcome). Confidence = probability given before kick-off.</p>
{summary}
<table><tr><th>Match</th><th>Pick</th><th class="n">Confidence, score</th><th class="n">Confidence, outcome</th>
<th class="n">Home / draw / away</th><th>Baseline Pick</th><th>Result</th></tr>
{''.join(rows)}</table>
{replay_note}
<h2>Match by match</h2>
{''.join(cards)}
</body></html>"""
    (folder / "report.html").write_text(page, "utf-8")
    print(f"wrote {folder / 'report.html'} ({totals['known']} Results known)")


if __name__ == "__main__":
    main()
