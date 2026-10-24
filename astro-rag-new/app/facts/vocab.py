"""
Vocabulary/lookup tables for the rule-based question parser in
`app/facts/parser.py`. Deliberately simple and inspectable -- this is the
"deterministic retrieval context" approach the project's architecture doc
recommends for a first version (don't hand raw chart JSON + free text to an
LLM and ask it to figure out what to retrieve; extract facts with rules
first, only reach for the LLM at the generation step).
"""
from __future__ import annotations

import re

PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]

PLANET_ALIASES: dict[str, str] = {
    "sun": "Sun", "surya": "Sun",
    "moon": "Moon", "chandra": "Moon",
    "mars": "Mars", "mangal": "Mars", "kuja": "Mars",
    "mercury": "Mercury", "budh": "Mercury", "budha": "Mercury",
    "jupiter": "Jupiter", "guru": "Jupiter", "brihaspati": "Jupiter",
    "venus": "Venus", "shukra": "Venus",
    "saturn": "Saturn", "shani": "Saturn",
    "rahu": "Rahu",
    "ketu": "Ketu",
}

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati",
]

DASHA_WORDS = [
    "mahadasha", "mahadasa", "maha dasha", "maha dasa", "maha-dasha",
    "maha-dasa", "dasha", "dasa", "antardasha", "antardasa",
    "antar dasha", "antar dasa", "period", "bhukti",
]
TIMING_WORDS = ["when", "this year", "next year", "soon", "upcoming", "timing", "date", "period", "future", "next few", "this month", "this quarter"]
ASPECT_WORDS = ["aspect", "drishti", "aspects"]

_WORD_NUMBERS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10, "eleventh": 11, "11th": 11, "twelfth": 12, "12th": 12,
}

_HOUSE_NUM_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)?\s*house\b")
_HOUSE_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(_WORD_NUMBERS.keys(), key=len, reverse=True)) + r")\s*house\b"
)

# Open-ended "what does my X look like" topics -> which houses are classically
# examined for that life area. This is what lets a question with no explicit
# planet/house/sign still produce a useful, personalized retrieval context.
TOPIC_HOUSES: dict[str, list[int]] = {
    # career / work
    "career": [10, 6, 2], "job": [10, 6], "jobs": [10, 6], "profession": [10],
    "work": [10, 6], "promotion": [10], "business": [7, 10], "boss": [10],
    "salary": [2, 11], "success": [10, 11], "fame": [10], "status": [10],
    # marriage / relationships
    "marriage": [7, 2, 8], "married": [7, 2, 8], "marry": [7, 2, 8], "wedding": [7],
    "spouse": [7], "wife": [7], "husband": [7], "partner": [7], "divorce": [7, 8],
    "relationship": [7, 5], "relationships": [7, 5], "love": [5, 7], "dating": [5, 7],
    "romance": [5, 7],
    # money
    "wealth": [2, 11], "money": [2, 11], "finance": [2, 11], "finances": [2, 11],
    "rich": [2, 11, 9, 5], "wealthy": [2, 11], "income": [2, 11], "savings": [2, 11],
    "struggling": [6, 8, 12],
    "gains": [11], "loss": [12, 8], "debt": [6],
    # health
    "health": [1, 6, 8], "illness": [6, 8], "disease": [6, 8], "surgery": [6, 8],
    "longevity": [8], "accident": [8],
    # education
    "education": [4, 5, 9], "study": [4, 5, 9], "studies": [4, 5, 9], "exam": [4, 5],
    "exams": [4, 5], "degree": [4, 5, 9], "college": [4, 5, 9], "university": [4, 5, 9],
    # children / family
    "children": [5], "child": [5], "kids": [5], "baby": [5], "pregnancy": [5],
    "family": [2, 4], "mother": [4], "father": [9], "siblings": [3], "brother": [3],
    "sister": [3], "younger": [3], "older": [11], "home": [4], "property": [4], "land": [4],
    "company": [10, 6], "companies": [10, 6], "multinational": [10, 12, 9],
    "country": [4, 12, 9], "distant": [12, 9], "far": [12, 9],
    "undertaking": [3, 10], "undertakings": [3, 10], "bold": [3, 1],
    "courage": [3, 1], "changing": [10, 3, 8], "future": [1, 9, 10, 11],
    "belong": [4, 12, 9], "famous": [10, 1, 11],
    # travel / spirituality
    "travel": [3, 9, 12], "abroad": [12, 9], "foreign": [12, 9], "overseas": [12, 9],
    "settle": [4, 12], "immigration": [12, 9],
    "spirituality": [9, 12], "spiritual": [9, 12], "moksha": [12], "meditation": [12],
    # self
    "personality": [1], "character": [1], "nature": [1], "appearance": [1],
    # everyday and domain terms
    "financial": [2, 11], "comfort": [4, 2, 11], "comfortably": [4],
    "support": [11, 4, 2], "supportive": [7, 11, 4], "poor": [2, 11, 12],
    "millionaire": [2, 11, 9, 5], "inherit": [8, 2, 4], "inheritance": [8, 2, 4],
    "borrow": [6, 11, 2], "succeed": [10, 11, 5], "media": [3, 5, 10],
    "design": [3, 5, 10], "communication": [3, 2, 10],
    "communications": [3, 2, 10], "venture": [3, 7, 10, 11],
    "startup": [3, 10, 11], "entrepreneur": [3, 7, 10, 11],
    "youtube": [3, 10, 11], "channel": [3, 10], "technology": [5, 10, 11],
    "engineering": [4, 5, 10], "government": [10, 6, 1],
    "private": [10, 6, 7], "sector": [10, 6],
    "son": [5], "daughter": [5], "conception": [5], "fertility": [5],
    "offspring": [5], "properties": [4], "own house": [4], "houses": [4],
    "domestic": [4], "instability": [4, 8], "move": [4, 9, 12],
    "relocate": [4, 9, 12], "relocation": [4, 9, 12],
    "long life": [8, 1, 3], "lifespan": [8, 1], "live past": [8, 1],
    "partnership": [7], "romantic": [5, 7], "name": [7, 1],
}

_HOUSE_TOPIC_PHRASES: dict[str, int] = {
    "self house": 1, "ascendant house": 1,
    "wealth house": 2, "money house": 2, "finance house": 2,
    "family house": 2, "speech house": 2,
    "siblings house": 3, "courage house": 3, "efforts house": 3,
    "home house": 4, "property house": 4, "mother house": 4, "education house": 4,
    "children house": 5, "progeny house": 5, "intelligence house": 5,
    "romance house": 5,
    "health house": 6, "enemies house": 6, "debt house": 6, "service house": 6,
    "marriage house": 7, "spouse house": 7, "partnership house": 7,
    "longevity house": 8, "transformation house": 8, "inheritance house": 8,
    "luck house": 9, "fortune house": 9, "dharma house": 9, "father house": 9,
    "career house": 10, "profession house": 10, "karma house": 10, "fame house": 10,
    "gains house": 11, "income house": 11, "friends house": 11,
    "loss house": 12, "foreign house": 12, "moksha house": 12,
}


def find_house_numbers(text_lower: str) -> list[int]:
    found = {int(m.group(1)) for m in _HOUSE_NUM_RE.finditer(text_lower)}
    for m in _HOUSE_WORD_RE.finditer(text_lower):
        found.add(_WORD_NUMBERS[m.group(1)])
    for phrase, house_number in _HOUSE_TOPIC_PHRASES.items():
        if phrase in text_lower:
            found.add(house_number)
    return sorted(h for h in found if 1 <= h <= 12)


def find_planets(text_lower: str) -> list[str]:
    found = []
    for alias, canonical in PLANET_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text_lower) and canonical not in found:
            found.append(canonical)
    return found


def find_signs(text_lower: str) -> list[str]:
    return [s for s in SIGNS if re.search(rf"\b{s.lower()}\b", text_lower)]


def find_nakshatras(text_lower: str) -> list[str]:
    return [n for n in NAKSHATRAS if n.lower() in text_lower]


def find_topics(text_lower: str) -> list[str]:
    return [t for t in TOPIC_HOUSES if re.search(rf"\b{t}\b", text_lower)]


def mentions_any(text_lower: str, words: list[str]) -> bool:
    return any(w in text_lower for w in words)
