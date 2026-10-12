"""Run a whole Matchday from start to finish.

Usage: uv run scripts/run_matchday.py <matchday number>            predict (before kick-off)
       uv run scripts/run_matchday.py <matchday number> --results  fill in the Results (after the matches)

Predicting downloads the history, captures the Snapshot, and writes
matchday-NN/predictions.csv and report.html. It can be run again on a match
day for fresher odds and absences: matches that have already kicked off keep
their earlier Prediction.

The Rationale texts in matchday-NN/rationale.json are not written by this
command; the report says "No Rationale written." for matches without one.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step(title: str, *command: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, *command], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"Stopped: '{title}' failed. Fix the problem above and run the command again.")
    print(f"--- done in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        sys.exit(__doc__)
    matchday = sys.argv[1]
    folder = f"matchday-{int(matchday):02d}"

    step("Download match history", "scripts/download_history.py")
    step("Build match table", "-m", "clpred.build_matches")
    if "--results" in sys.argv:
        step("Write report with Results", "-m", "clpred.report", matchday)
        print(f"\nOpen {folder}\\report.html")
        return

    step("Capture the Snapshot (about 30 minutes)", "scripts/capture_matchday.py", matchday)
    step("Rebuild match table with the latest rounds", "-m", "clpred.build_matches")
    step("Read player performance", "-m", "clpred.players")
    step("Extract odds, absences and squads", "-m", "clpred.snapshot", matchday)
    step("Predict", "-m", "clpred.predict", matchday)
    step("Write report", "-m", "clpred.report", matchday)
    print(f"\nOpen {folder}\\report.html")
    print("Read the CHECK, LEFT OUT, MISSING and UNREADABLE lines above: each names an input that could not be used.")


if __name__ == "__main__":
    main()
