import pandas as pd

from clpred.predict import with_earlier


def table(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["home", "away", "pick"])


def test_first_run_saves_the_new_predictions(tmp_path):
    new = table(("Arsenal", "Lille", "2-0"))
    assert with_earlier(new, tmp_path / "predictions.csv").equals(new)


def test_matches_already_kicked_off_keep_their_prediction(tmp_path):
    path = tmp_path / "predictions.csv"
    table(("Arsenal", "Lille", "2-0"), ("Roma", "Real Madrid", "1-2")).to_csv(path, index=False)
    # Second day of the Matchday: only Roma - Real Madrid is still to be played.
    merged = with_earlier(table(("Roma", "Real Madrid", "1-1")), path)
    picks = {(r.home, r.away): r.pick for r in merged.itertuples()}
    assert picks == {("Arsenal", "Lille"): "2-0", ("Roma", "Real Madrid"): "1-1"}
