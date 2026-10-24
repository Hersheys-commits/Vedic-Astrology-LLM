from app.config.settings import ChartSettings
from app.engine.houses import build_houses, place_planets_in_houses
from app.utils.angles import house_distance


def test_whole_sign_starts_at_lagna():
    houses = build_houses(9, ChartSettings())  # Capricorn ascendant
    assert houses[0]["sign"] == "Capricorn"
    assert houses[0]["lord"] == "Saturn"
    assert houses[1]["sign"] == "Aquarius"
    assert houses[9]["sign"] == "Libra"
    assert houses[9]["lord"] == "Venus"


def test_house_distance_is_inclusive():
    assert house_distance(0, 0) == 1
    assert house_distance(9, 3) == 7      # Capricorn -> Cancer is the 7th
    assert house_distance(11, 0) == 2


def test_placement_links_both_directions():
    settings = ChartSettings()
    houses = build_houses(9, settings)
    planets = [{"body": "Mars", "position": {"sign_index": 3}}]   # Cancer
    place_planets_in_houses(planets, 9, houses)
    assert planets[0]["placement"]["house"] == 7
    assert "Mars" in houses[6]["occupants"]
