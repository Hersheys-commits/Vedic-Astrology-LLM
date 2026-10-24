"""Vimshottari dasha. Independent deterministic module.

Input:  Moon sidereal longitude + birth Julian Day (UT)
Output: balance at birth, mahadasha timeline, optional antardashas.
"""

from __future__ import annotations

from app.astronomy.julian import julian_day_to_utc_string
from app.config.planets import (
    SIDEREAL_YEAR_DAYS,
    VIMSHOTTARI_SEQUENCE,
    VIMSHOTTARI_TOTAL_YEARS,
    VIMSHOTTARI_YEARS,
)
from app.engine.nakshatra import nakshatra_of


def _date_of(jd: float) -> str:
    return julian_day_to_utc_string(jd)[:10]


def _sequence_from(lord: str) -> list[str]:
    start = VIMSHOTTARI_SEQUENCE.index(lord)
    return [VIMSHOTTARI_SEQUENCE[(start + i) % 9] for i in range(9)]


def _antardashas(maha_lord: str, maha_start_jd: float, maha_years: float) -> list[dict]:
    periods = []
    cursor = maha_start_jd
    for lord in _sequence_from(maha_lord):
        years = maha_years * VIMSHOTTARI_YEARS[lord] / VIMSHOTTARI_TOTAL_YEARS
        end = cursor + years * SIDEREAL_YEAR_DAYS
        periods.append(
            {
                "lord": lord,
                "start": _date_of(cursor),
                "end": _date_of(end),
                "duration_years": round(years, 6),
            }
        )
        cursor = end
    return periods


def calculate_vimshottari(moon_longitude: float, birth_jd_ut: float, settings) -> dict:
    """Full Vimshottari structure for the chart JSON."""
    moon_nak = nakshatra_of(moon_longitude)
    lord = moon_nak["lord"]
    fraction_traversed = moon_nak["fraction_traversed"]

    full_years = VIMSHOTTARI_YEARS[lord]
    balance_years = full_years * (1.0 - fraction_traversed)
    elapsed_years = full_years - balance_years

    # The first mahadasha notionally began before birth.
    first_start_jd = birth_jd_ut - elapsed_years * SIDEREAL_YEAR_DAYS

    timeline: list[dict] = []
    cursor = first_start_jd
    emitted_years = 0.0
    for i, maha_lord in enumerate(_sequence_from(lord)):
        years = float(VIMSHOTTARI_YEARS[maha_lord])
        end = cursor + years * SIDEREAL_YEAR_DAYS
        entry = {
            "lord": maha_lord,
            "start": _date_of(cursor),
            "end": _date_of(end),
            "duration_years": years,
            "starts_before_birth": i == 0 and elapsed_years > 0,
        }
        if settings.include_antardasha:
            entry["antardasha"] = _antardashas(maha_lord, cursor, years)
        timeline.append(entry)
        cursor = end
        emitted_years += years
        if emitted_years >= settings.dasha_years:
            break

    return {
        "system": "vimshottari",
        "reference": {
            "moon_nakshatra": moon_nak["name"],
            "moon_nakshatra_lord": lord,
            "moon_pada": moon_nak["pada"],
            "fraction_of_nakshatra_traversed": round(fraction_traversed, 9),
        },
        "birth_mahadasha": {
            "lord": lord,
            "balance_years": round(balance_years, 6),
            "elapsed_years": round(elapsed_years, 6),
            "ends": _date_of(birth_jd_ut + balance_years * SIDEREAL_YEAR_DAYS),
        },
        "mahadasha": timeline,
        "conventions": {
            "year_length_days": SIDEREAL_YEAR_DAYS,
            "total_cycle_years": VIMSHOTTARI_TOTAL_YEARS,
            "sequence": VIMSHOTTARI_SEQUENCE,
        },
    }


def current_period(vimshottari: dict, as_of_date: str) -> dict | None:
    """Which mahadasha (and antardasha, if present) is running on a given date."""
    for maha in vimshottari["mahadasha"]:
        if maha["start"] <= as_of_date < maha["end"]:
            result = {"as_of": as_of_date, "mahadasha": maha["lord"]}
            for antar in maha.get("antardasha", []):
                if antar["start"] <= as_of_date < antar["end"]:
                    result["antardasha"] = antar["lord"]
                    break
            return result
    return None
