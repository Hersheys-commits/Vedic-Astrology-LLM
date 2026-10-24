from app.config.planets import VIMSHOTTARI_YEARS
from app.config.settings import ChartSettings
from app.engine.dasha import calculate_vimshottari

SETTINGS = ChartSettings(include_antardasha=True)
JD = 2453142.878467252  # 2004-05-17 09:05 UTC


def test_lord_comes_from_moon_nakshatra():
    d = calculate_vimshottari(0.0, JD, SETTINGS)          # exact start of Ashwini
    assert d["birth_mahadasha"]["lord"] == "Ketu"
    assert abs(d["birth_mahadasha"]["balance_years"] - 7.0) < 1e-6


def test_balance_is_proportional():
    span = 360.0 / 27.0
    d = calculate_vimshottari(span / 2, JD, SETTINGS)       # halfway through Ashwini
    assert abs(d["birth_mahadasha"]["balance_years"] - 3.5) < 1e-6


def test_sequence_and_total():
    d = calculate_vimshottari(5.0, JD, SETTINGS)
    lords = [m["lord"] for m in d["mahadasha"]]
    assert lords[:3] == ["Ketu", "Venus", "Sun"]
    assert sum(m["duration_years"] for m in d["mahadasha"]) == 120


def test_antardasha_sums_to_mahadasha():
    d = calculate_vimshottari(5.0, JD, SETTINGS)
    maha = d["mahadasha"][1]
    total = sum(a["duration_years"] for a in maha["antardasha"])
    assert abs(total - VIMSHOTTARI_YEARS[maha["lord"]]) < 1e-5  # 9 rounded entries
