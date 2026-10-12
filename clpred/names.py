"""Matching club names across sources.

Every source spells clubs differently ("Man City", "Manchester City",
"Manchester City FC"). The canonical name is UEFA's international name.
"""

import re
import unicodedata
from difflib import SequenceMatcher

# Words that carry no identity and only confuse matching.
NOISE = {"fc", "cf", "sc", "ac", "as", "fk", "sk", "bk", "if", "afc", "ssc", "rc", "cd", "club", "de", "kv", "1907"}

# Spellings no fuzzy match gets right, as normalised source name -> normalised UEFA name.
ALIASES = {
    "man city": "manchester city",
    "man united": "manchester united",
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "nottm forest": "nottingham forest",
    "nott m forest": "nottingham forest",
    "nottingham": "nottingham forest",
    "st liege": "standard liege",
    "standard": "standard liege",
    "ath madrid": "atletico madrid",
    "atl madrid": "atletico madrid",
    "atleti": "atletico madrid",
    "ath bilbao": "athletic club",
    "sociedad": "real sociedad",
    "betis": "real betis",
    "celta": "celta vigo",
    "espanol": "espanyol",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "milan": "ac milan",
    "roma": "as roma",
    "paris sg": "paris saint germain",
    "psg": "paris saint germain",
    "dortmund": "borussia dortmund",
    "m gladbach": "borussia monchengladbach",
    "leverkusen": "bayer leverkusen",
    "ein frankfurt": "eintracht frankfurt",
    "bayern": "bayern munchen",
    "bayern munich": "bayern munchen",
    "leipzig": "rb leipzig",
    "rb leipzig": "rb leipzig",
    "stuttgart": "vfb stuttgart",
    "sp lisbon": "sporting cp",
    "sporting": "sporting cp",
    "sp braga": "braga",
    "porto": "fc porto",
    "psv": "psv eindhoven",
    "club brugge": "club brugge",
    "st gilloise": "union sg",
    "buyuksehyr": "istanbul basaksehir",
    "olympiakos": "olympiacos",
    "olympiacos piraeus": "olympiacos",
    "salzburg": "salzburg",
    "bodo glimt": "bodo glimt",
    "sabah baku": "sabah",
    "slavia prague": "slavia praha",
    "sparta prague": "sparta praha",
    "lask linz": "lask",
    "shakhtar": "shakhtar donetsk",
    "crvena zvezda": "crvena zvezda",
    "red star belgrade": "crvena zvezda",
    "fc copenhagen": "copenhagen",
    "kairat almaty": "kairat",
    "tns": "the new saints",
    "o ljubljana": "olimpija",
    "rasenballsport leipzig": "rb leipzig",
    "cologne": "koln",
    "wolverhampton wanderers": "wolves",
    "din zagreb": "gnk dinamo",
}


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = text.replace("ø", "o")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [w for w in text.split() if w not in NOISE]
    joined = " ".join(words) or text.strip()
    return ALIASES.get(joined, joined)


def similarity(a: str, b: str) -> float:
    a, b = normalise(a), normalise(b)
    if a == b:
        return 1.0
    score = SequenceMatcher(None, a, b).ratio()
    # A name wholly contained in the other ("Lille" in "Lille OSC") is a strong hint.
    if a and b and (set(a.split()) <= set(b.split()) or set(b.split()) <= set(a.split())):
        score = max(score, 0.9)
    return score


def best_match(name: str, candidates: dict[str, list[str]]) -> tuple[str | None, float]:
    """Return the canonical name whose known spellings best match `name`.

    `candidates` maps a canonical name to every spelling known for it.
    """
    best, best_score = None, 0.0
    for canonical, spellings in candidates.items():
        score = max(similarity(name, s) for s in spellings)
        if score > best_score:
            best, best_score = canonical, score
    return best, best_score
