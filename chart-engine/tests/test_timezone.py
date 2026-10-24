from datetime import date, time

import pytest

from app.astronomy.timezone import resolve_timezone, to_utc
from app.utils.errors import ChartError


def test_resolve_india():
    assert resolve_timezone(25.4358, 81.8463) == "Asia/Kolkata"


def test_ist_offset_applied():
    block = to_utc(date(2004, 5, 17), time(14, 35), "Asia/Kolkata")
    assert block["timezone"]["utc_offset"] == "+05:30"
    assert block["utc_datetime"] == "2004-05-17T09:05:00Z"
    assert block["timezone"]["is_dst"] is False


def test_historical_offset_is_not_todays_offset():
    # The USA observed war time / different DST rules historically; zoneinfo
    # must apply the offset in force on the birth date.
    block = to_utc(date(1970, 7, 1), time(12, 0), "America/New_York")
    assert block["timezone"]["utc_offset"] == "-04:00"
    winter = to_utc(date(1970, 1, 15), time(12, 0), "America/New_York")
    assert winter["timezone"]["utc_offset"] == "-05:00"


def test_ambiguous_local_time_requires_choice():
    with pytest.raises(ChartError) as exc:
        to_utc(date(2020, 11, 1), time(1, 30), "America/New_York")
    assert exc.value.code == "AMBIGUOUS_LOCAL_TIME"


def test_ambiguous_local_time_resolved():
    first = to_utc(date(2020, 11, 1), time(1, 30), "America/New_York", "first_occurrence")
    second = to_utc(date(2020, 11, 1), time(1, 30), "America/New_York", "second_occurrence")
    assert first["utc_datetime"] != second["utc_datetime"]


def test_nonexistent_local_time_rejected():
    with pytest.raises(ChartError) as exc:
        to_utc(date(2020, 3, 8), time(2, 30), "America/New_York")
    assert exc.value.code == "NONEXISTENT_LOCAL_TIME"
