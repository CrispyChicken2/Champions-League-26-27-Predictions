from clpred.names import best_match, normalise, similarity


def test_accents_and_club_words_are_ignored():
    assert normalise("FC Bayern München") == normalise("Bayern Munich")
    assert similarity("Bodo/Glimt", "Bodø/Glimt") > 0.9


def test_manchester_clubs_are_not_confused():
    clubs = {"Man City": ["Man City", "Manchester City"], "Man Utd": ["Man Utd", "Manchester United"]}
    assert best_match("Manchester Utd", clubs)[0] == "Man Utd"
    assert best_match("Man City", clubs)[0] == "Man City"


def test_nottingham_is_not_tottenham():
    clubs = {"Tottenham": ["Tottenham"], "Nott'm Forest": ["Nott'm Forest"]}
    name, score = best_match("Nottingham", clubs)
    assert name == "Nott'm Forest" and score == 1.0


def test_abbreviations_from_other_sources_resolve():
    clubs = {"GNK Dinamo": ["GNK Dinamo"], "Leipzig": ["Leipzig", "RB Leipzig"], "Atleti": ["Atleti", "Atlético de Madrid"]}
    assert best_match("Din. Zagreb", clubs)[0] == "GNK Dinamo"
    assert best_match("RasenBallsport Leipzig", clubs)[0] == "Leipzig"
    assert best_match("Atl. Madrid", clubs)[0] == "Atleti"


def test_unrelated_names_score_low():
    assert similarity("Empoli", "Napoli") < 0.72
