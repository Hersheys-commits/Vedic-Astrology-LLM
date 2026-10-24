"""Body definitions, Vedic aspect rules and Vimshottari constants."""

import swisseph as swe

# Order matters: this is the order bodies appear in the output JSON.
# swe_id is resolved at runtime; Ketu is derived from Rahu (Rahu + 180).
BODIES = [
    {"name": "Sun",     "swe_id": swe.SUN,     "kind": "graha"},
    {"name": "Moon",    "swe_id": swe.MOON,    "kind": "graha"},
    {"name": "Mars",    "swe_id": swe.MARS,    "kind": "graha"},
    {"name": "Mercury", "swe_id": swe.MERCURY, "kind": "graha"},
    {"name": "Jupiter", "swe_id": swe.JUPITER, "kind": "graha"},
    {"name": "Venus",   "swe_id": swe.VENUS,   "kind": "graha"},
    {"name": "Saturn",  "swe_id": swe.SATURN,  "kind": "graha"},
    {"name": "Rahu",    "swe_id": None,        "kind": "node"},   # mean/true node
    {"name": "Ketu",    "swe_id": None,        "kind": "node"},   # Rahu + 180
]

BODY_NAMES = [b["name"] for b in BODIES]

NODE_SWE_ID = {
    "mean": swe.MEAN_NODE,
    "true": swe.TRUE_NODE,
}

# Optional outer planets, disabled by default in the Vedic configuration.
OPTIONAL_BODIES = [
    {"name": "Uranus",  "swe_id": swe.URANUS,  "kind": "extra"},
    {"name": "Neptune", "swe_id": swe.NEPTUNE, "kind": "extra"},
    {"name": "Pluto",   "swe_id": swe.PLUTO,   "kind": "extra"},
]

# ---------------------------------------------------------------------------
# Vedic graha drishti (sign/house based special aspects).
# Values are house distances counted inclusively from the aspecting planet.
# ---------------------------------------------------------------------------
ASPECT_RULES = {
    "default": [7],
    "Mars": [4, 7, 8],
    "Jupiter": [5, 7, 9],
    "Saturn": [3, 7, 10],
    "Rahu": [5, 7, 9],   # tradition-dependent; configurable
    "Ketu": [5, 7, 9],   # tradition-dependent; configurable
}

ASPECT_LABEL = {
    3: "3rd", 4: "4th", 5: "5th", 7: "7th", 8: "8th", 9: "9th", 10: "10th",
}

# Bodies that do not cast drishti in the default configuration.
ASPECTING_BODIES = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]

# ---------------------------------------------------------------------------
# Vimshottari dasha
# ---------------------------------------------------------------------------
VIMSHOTTARI_SEQUENCE = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]

VIMSHOTTARI_YEARS = {
    "Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7,
    "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17,
}

VIMSHOTTARI_TOTAL_YEARS = 120
SIDEREAL_YEAR_DAYS = 365.25  # civil-year convention used for dasha arithmetic

# Combustion orbs in degrees (distance from the Sun), tradition-dependent.
COMBUSTION_ORB = {
    "Moon": 12.0, "Mars": 17.0, "Mercury": 14.0,
    "Jupiter": 11.0, "Venus": 10.0, "Saturn": 15.0,
}
