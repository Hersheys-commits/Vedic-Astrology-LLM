"""Exaltation, debilitation, own-sign and moolatrikona data (Parashari)."""

EXALTATION = {
    "Sun": {"sign": "Aries", "degree": 10.0},
    "Moon": {"sign": "Taurus", "degree": 3.0},
    "Mars": {"sign": "Capricorn", "degree": 28.0},
    "Mercury": {"sign": "Virgo", "degree": 15.0},
    "Jupiter": {"sign": "Cancer", "degree": 5.0},
    "Venus": {"sign": "Pisces", "degree": 27.0},
    "Saturn": {"sign": "Libra", "degree": 20.0},
    "Rahu": {"sign": "Taurus", "degree": 20.0},   # tradition-dependent
    "Ketu": {"sign": "Scorpio", "degree": 20.0},  # tradition-dependent
}

DEBILITATION = {
    "Sun": {"sign": "Libra", "degree": 10.0},
    "Moon": {"sign": "Scorpio", "degree": 3.0},
    "Mars": {"sign": "Cancer", "degree": 28.0},
    "Mercury": {"sign": "Pisces", "degree": 15.0},
    "Jupiter": {"sign": "Capricorn", "degree": 5.0},
    "Venus": {"sign": "Virgo", "degree": 27.0},
    "Saturn": {"sign": "Aries", "degree": 20.0},
    "Rahu": {"sign": "Scorpio", "degree": 20.0},
    "Ketu": {"sign": "Taurus", "degree": 20.0},
}

OWN_SIGNS = {
    "Sun": ["Leo"],
    "Moon": ["Cancer"],
    "Mars": ["Aries", "Scorpio"],
    "Mercury": ["Gemini", "Virgo"],
    "Jupiter": ["Sagittarius", "Pisces"],
    "Venus": ["Taurus", "Libra"],
    "Saturn": ["Capricorn", "Aquarius"],
}

# Moolatrikona ranges: (sign, start_degree, end_degree)
MOOLATRIKONA = {
    "Sun": ("Leo", 0.0, 20.0),
    "Moon": ("Taurus", 4.0, 30.0),
    "Mars": ("Aries", 0.0, 12.0),
    "Mercury": ("Virgo", 16.0, 20.0),
    "Jupiter": ("Sagittarius", 0.0, 10.0),
    "Venus": ("Libra", 0.0, 15.0),
    "Saturn": ("Aquarius", 0.0, 20.0),
}
