import json

import pytest

from app.config.settings import ChartSettings
from app.engine.calculator import generate_chart
from app.utils.errors import ChartError

REFERENCE = dict(
    birth_date="2004-05-17",
    birth_time="14:35:00",
    latitude=25.4358,
    longitude=81.8463,
    location_name="Prayagraj, Uttar Pradesh, India",
    as_of_date="2026-01-01",
)


def chart():
    return generate_chart(**REFERENCE)


def test_pipeline_passes_validation():
    c = chart()
    assert c["validation"]["status"] == "passed"
    assert c["time"]["utc_datetime"] == "2004-05-17T09:05:00Z"
    assert c["time"]["timezone"]["iana"] == "Asia/Kolkata"


def test_output_is_json_serialisable():
    json.dumps(chart())


def test_cross_references_are_explicit():
    c = chart()
    for planet in c["planets"]:
        house = c["houses"][planet["placement"]["house"] - 1]
        assert planet["body"] in house["occupants"]
        assert planet["placement"]["house_sign"] == house["sign"]


def test_fact_keys_present():
    c = chart()
    assert any(k.startswith("planet:sun|sign:") for k in c["fact_keys"])
    assert any(k.startswith("house:10|lord:") for k in c["fact_keys"])
    assert c["natural_language_facts"]


def test_chart_id_is_deterministic():
    assert chart()["chart_id"] == chart()["chart_id"]


def test_same_input_same_output():
    a, b = chart(), chart()
    a["provenance"].pop("computed_at_utc")
    b["provenance"].pop("computed_at_utc")
    assert a == b


def test_invalid_time_rejected():
    with pytest.raises(ChartError) as exc:
        generate_chart("2004-05-17", "25:99", 25.4, 81.8)
    assert exc.value.code == "INVALID_TIME"


def test_invalid_coordinates_rejected():
    with pytest.raises(ChartError) as exc:
        generate_chart("2004-05-17", "14:35", 200.0, 81.8)
    assert exc.value.code == "INVALID_COORDINATES"


def test_southern_hemisphere_and_dst_zone():
    c = generate_chart(
        "1999-12-25", "23:45:00", -33.8688, 151.2093,
        location_name="Sydney, Australia", as_of_date="2026-01-01",
    )
    assert c["time"]["timezone"]["iana"] == "Australia/Sydney"
    assert c["time"]["timezone"]["is_dst"] is True
    assert c["validation"]["status"] == "passed"


def test_settings_are_reported_in_output():
    c = generate_chart(**REFERENCE, settings=ChartSettings(ayanamsa="raman"))
    assert c["calculation_settings"]["ayanamsa"] == "raman"
    assert c["calculation_settings"]["ayanamsa_value_degrees"] > 0
