import pytest


@pytest.fixture
def sample_chart():
    """A trimmed-down chart shaped like the real chart-engine's response,
    with just enough fields for the facts/retrieval layers to work with."""
    return {
        "planets": [
            {
                "body": "Mars",
                "position": {"zodiac_sign": "Gemini"},
                "placement": {"house": 10},
                "nakshatra": {"name": "Ardra", "pada": 2},
                "motion": {"retrograde": False},
            },
            {
                "body": "Moon",
                "position": {"zodiac_sign": "Aries"},
                "placement": {"house": 8},
                "nakshatra": {"name": "Ashwini", "pada": 4},
                "motion": {"retrograde": False},
            },
        ],
        "houses": [
            {"number": 10, "sign": "Gemini", "lord": "Mercury",
             "occupants": ["Mars", "Venus", "Saturn"], "lord_placement": {"house": 8}},
            {"number": 7, "sign": "Pisces", "lord": "Jupiter",
             "occupants": [], "lord_placement": {"house": 12}},
        ],
        "aspects": [
            {"relationship": "aspect", "from": "Mars", "to": "Ketu", "aspect_type": "8th"},
            {"relationship": "aspect", "from": "Mars", "to": "Venus", "aspect_type": "4th"},
        ],
        "dashas": {"current_period": {"mahadasha": "Sun", "antardasha": "Rahu"}},
        "chart_summary": {
            "ascendant_sign": "Virgo",
            "moon_sign": "Aries",
            "moon_nakshatra": "Ashwini",
            "moon_pada": 4,
            "sun_sign": "Taurus",
        },
        "natural_language_facts": ["Mars is in Gemini at 12°25'41\"."],
    }
