# Champions League Score Prediction

Predicting the score of every match of the 2026/27 UEFA Champions League league phase, Matchday by Matchday, and measuring how good those predictions are.

## Language

**Matchday**:
One round of the league phase in which each of the 36 clubs plays one match (18 matches). The project covers Matchdays 2 to 8.
_Avoid_: Journey, journée, round, gameweek

**Score Distribution**:
The probability assigned to every possible scoreline of a match.
_Avoid_: Odds, probabilities

**Pick**:
The single scoreline chosen from a Score Distribution as the predicted score of a match.
_Avoid_: Guess, tip

**Prediction**:
A Score Distribution together with its Pick for one match, fixed before kick-off.
_Avoid_: Forecast

**Confidence**:
The probability a Prediction gives, before kick-off, to its Pick and to its outcome (home win, draw, away win).
_Avoid_: Accuracy, certainty

**Result**:
The post-match verdict on a Pick: exact hit, outcome hit, or miss.
_Avoid_: Accuracy, score

**Calibration**:
How closely Confidence matches the frequency of hits over many Predictions.
_Avoid_: Accuracy, reliability

**Availability**:
Which players a club cannot field for a match through injury, suspension or rotation, valued by their importance to the team.
_Avoid_: Health, fitness, injuries

**Snapshot**:
The data captured for a Matchday before its first kick-off and never changed afterwards; a Prediction may only use its Matchday's Snapshot and earlier history.
_Avoid_: Scrape, dump, capture

**Venue Effect**:
A club's own measured difference between playing at home and playing away, distinct from the average home advantage shared by all clubs.
_Avoid_: Home advantage, home weight, home factor

**Soft Adjustment**:
A small, capped change to a Prediction based on morale and news, recorded separately so its effect can be measured.
_Avoid_: Feeling, sentiment, morale factor

**Rationale**:
The two or three sentences attached to a Prediction explaining why the Pick came out as it did.
_Avoid_: Explanation, comment, analysis

**Baseline**:
The Prediction implied by bookmaker odds alone, against which the project's own Predictions are compared.
_Avoid_: Benchmark, reference
