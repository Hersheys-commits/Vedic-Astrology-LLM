"""Canonical zodiac data. Pure data, no calculation."""

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

# Vedic (Parashari) sign rulership. Modern rulers are deliberately not used.
SIGN_LORDS = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}

SIGN_ELEMENT = {
    "Aries": "fire", "Leo": "fire", "Sagittarius": "fire",
    "Taurus": "earth", "Virgo": "earth", "Capricorn": "earth",
    "Gemini": "air", "Libra": "air", "Aquarius": "air",
    "Cancer": "water", "Scorpio": "water", "Pisces": "water",
}

SIGN_MODALITY = {
    "Aries": "movable", "Cancer": "movable", "Libra": "movable", "Capricorn": "movable",
    "Taurus": "fixed", "Leo": "fixed", "Scorpio": "fixed", "Aquarius": "fixed",
    "Gemini": "dual", "Virgo": "dual", "Sagittarius": "dual", "Pisces": "dual",
}


def sign_name(sign_index: int) -> str:
    """sign_index is 0-based: 0 = Aries."""
    return SIGNS[sign_index % 12]


def sign_lord(sign: str) -> str:
    return SIGN_LORDS[sign]
