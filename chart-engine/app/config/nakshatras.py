"""27 nakshatras with their Vimshottari lords.

Each nakshatra spans 13 deg 20 min = 13.333... degrees.
Each pada spans 3 deg 20 min = 3.333... degrees.
"""

NAKSHATRA_SPAN = 360.0 / 27.0      # 13.333333...
PADA_SPAN = NAKSHATRA_SPAN / 4.0   # 3.3333333...

NAKSHATRAS = [
    {"index": 1,  "name": "Ashwini",          "lord": "Ketu"},
    {"index": 2,  "name": "Bharani",          "lord": "Venus"},
    {"index": 3,  "name": "Krittika",         "lord": "Sun"},
    {"index": 4,  "name": "Rohini",           "lord": "Moon"},
    {"index": 5,  "name": "Mrigashira",       "lord": "Mars"},
    {"index": 6,  "name": "Ardra",            "lord": "Rahu"},
    {"index": 7,  "name": "Punarvasu",        "lord": "Jupiter"},
    {"index": 8,  "name": "Pushya",           "lord": "Saturn"},
    {"index": 9,  "name": "Ashlesha",         "lord": "Mercury"},
    {"index": 10, "name": "Magha",            "lord": "Ketu"},
    {"index": 11, "name": "Purva Phalguni",   "lord": "Venus"},
    {"index": 12, "name": "Uttara Phalguni",  "lord": "Sun"},
    {"index": 13, "name": "Hasta",            "lord": "Moon"},
    {"index": 14, "name": "Chitra",           "lord": "Mars"},
    {"index": 15, "name": "Swati",            "lord": "Rahu"},
    {"index": 16, "name": "Vishakha",         "lord": "Jupiter"},
    {"index": 17, "name": "Anuradha",         "lord": "Saturn"},
    {"index": 18, "name": "Jyeshtha",         "lord": "Mercury"},
    {"index": 19, "name": "Mula",             "lord": "Ketu"},
    {"index": 20, "name": "Purva Ashadha",    "lord": "Venus"},
    {"index": 21, "name": "Uttara Ashadha",   "lord": "Sun"},
    {"index": 22, "name": "Shravana",         "lord": "Moon"},
    {"index": 23, "name": "Dhanishta",        "lord": "Mars"},
    {"index": 24, "name": "Shatabhisha",      "lord": "Rahu"},
    {"index": 25, "name": "Purva Bhadrapada", "lord": "Jupiter"},
    {"index": 26, "name": "Uttara Bhadrapada","lord": "Saturn"},
    {"index": 27, "name": "Revati",           "lord": "Mercury"},
]

NAKSHATRA_BY_INDEX = {n["index"]: n for n in NAKSHATRAS}
