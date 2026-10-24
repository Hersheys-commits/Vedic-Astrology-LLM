from app.config.nakshatras import NAKSHATRA_SPAN, PADA_SPAN
from app.engine.nakshatra import nakshatra_of


def test_boundaries():
    assert nakshatra_of(0.0)["name"] == "Ashwini"
    assert nakshatra_of(0.0)["pada"] == 1
    assert nakshatra_of(NAKSHATRA_SPAN - 0.001)["name"] == "Ashwini"
    assert nakshatra_of(NAKSHATRA_SPAN - 0.001)["pada"] == 4
    assert nakshatra_of(NAKSHATRA_SPAN)["name"] == "Bharani"


def test_last_nakshatra():
    n = nakshatra_of(359.99)
    assert n["name"] == "Revati"
    assert n["index"] == 27
    assert n["pada"] == 4


def test_pada_width():
    assert nakshatra_of(PADA_SPAN * 2 + 0.01)["pada"] == 3


def test_known_lords():
    assert nakshatra_of(10.0)["lord"] == "Ketu"       # Ashwini
    assert nakshatra_of(30.0)["lord"] == "Sun"        # Krittika (26.67-40)
    assert nakshatra_of(225.0)["lord"] == "Saturn"    # Anuradha (213.33-226.67)


def test_wraparound_normalisation():
    assert nakshatra_of(360.0)["name"] == "Ashwini"
    assert nakshatra_of(-1.0)["name"] == "Revati"
