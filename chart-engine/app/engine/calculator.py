"""generate_chart(): the single entry point of the deterministic chart engine.

Pipeline (see plan section 35):
  validate -> resolve timezone -> local to UTC -> Julian Day -> ephemeris ->
  sidereal positions -> angles -> houses -> nakshatras -> aspects -> dasha ->
  derived facts -> assemble JSON -> validate chart.

No LLM call, no randomness, no network access happens here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone as dt_timezone

from app.astronomy.julian import utc_to_julian_day
from app.astronomy.swisseph import (
    ayanamsa_value,
    ephemeris_mode,
    swiss_ephemeris_version,
)
from app.astronomy.timezone import resolve_timezone, to_utc
from app.config.settings import (
    AYANAMSA_LABEL,
    ENGINE_VERSION,
    SCHEMA_VERSION,
    ChartSettings,
)
from app.engine.angles import calculate_angles
from app.engine.aspects import calculate_aspects
from app.engine.dasha import calculate_vimshottari, current_period
from app.engine.derived import (
    annotate_combustion,
    annotate_dignity,
    chart_summary,
)
from app.engine.facts import (
    build_fact_ids,
    build_fact_keys,
    build_natural_language_facts,
)
from app.engine.houses import (
    annotate_house_lord_placement,
    build_houses,
    place_planets_in_houses,
)
from app.utils.errors import ChartError
from app.utils.validation import parse_birth_date, parse_birth_time, validate_coordinates

log = logging.getLogger("chart_engine")


def _chart_id(payload: dict) -> str:
    """Deterministic id: identical input + settings always yields the same id."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _validate_chart(chart: dict) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "passed" if ok else "failed", "detail": detail})

    planets = chart["planets"]
    check("all_bodies_present", len(planets) >= 9, f"{len(planets)} bodies computed")
    check(
        "longitudes_normalised",
        all(0.0 <= p["position"]["sidereal_longitude"] < 360.0 for p in planets),
        "every longitude in [0,360)",
    )
    check(
        "houses_1_to_12",
        [h["number"] for h in chart["houses"]] == list(range(1, 13)),
        "house numbering is contiguous",
    )
    rahu = next((p for p in planets if p["body"] == "Rahu"), None)
    ketu = next((p for p in planets if p["body"] == "Ketu"), None)
    if rahu and ketu:
        diff = abs(
            (ketu["position"]["sidereal_longitude"] - rahu["position"]["sidereal_longitude"]) % 360.0 - 180.0
        )
        check("nodes_opposed", diff < 1e-6, f"Rahu/Ketu separation error {diff:.3e} deg")
    check(
        "nakshatra_padas_valid",
        all(1 <= p["nakshatra"]["pada"] <= 4 for p in planets),
        "every pada in 1..4",
    )
    total = sum(len(h["occupants"]) for h in chart["houses"])
    check("all_planets_placed", total == len(planets), f"{total} placements for {len(planets)} bodies")

    status = "passed" if all(c["status"] == "passed" for c in checks) else "failed"
    return {"status": status, "checks": checks}


def generate_chart(
    birth_date: str,
    birth_time: str,
    latitude: float,
    longitude: float,
    location_name: str | None = None,
    location_source: str = "coordinates",
    timezone_id: str | None = None,
    dst_choice: str | None = None,
    settings: ChartSettings | None = None,
    as_of_date: str | None = None,
) -> dict:
    """Compute a complete, LLM-ready Vedic chart as a plain dict."""
    settings = settings or ChartSettings()

    # 1-2. Validate input -------------------------------------------------
    b_date = parse_birth_date(birth_date)
    b_time = parse_birth_time(birth_time)
    lat, lon = validate_coordinates(latitude, longitude)
    log.info("chart_request_received date=%s time=%s", b_date, b_time)

    # 3-4. Resolve location and timezone ---------------------------------
    timezone_source = "provided"
    if timezone_id is None:
        timezone_id = resolve_timezone(lat, lon)
        timezone_source = "timezonefinder"
    log.info("timezone_resolved tz=%s source=%s", timezone_id, timezone_source)

    # 5-7. Local civil time -> UTC ---------------------------------------
    time_block = to_utc(b_date, b_time, timezone_id, dst_choice)
    warnings = list(time_block["warnings"])
    log.info("utc_datetime_created utc=%s", time_block["utc_datetime"])

    # 8. UTC -> Julian Day ------------------------------------------------
    jd_ut = utc_to_julian_day(time_block["utc_dt"])
    log.info("julian_day_created jd=%.9f", jd_ut)

    # 9. Ephemeris configuration -----------------------------------------
    mode = ephemeris_mode()
    if mode == "moshier":
        warnings.append(
            {
                "code": "MOSHIER_EPHEMERIS_IN_USE",
                "message": (
                    "No Swiss Ephemeris .se1 files found; using the built-in Moshier "
                    "ephemeris. Accuracy is typically sub-arcsecond for modern dates, "
                    "but place the .se1 files in data/ephe for full precision."
                ),
            }
        )

    # 10-11. Planetary positions -----------------------------------------
    from app.engine.planets import calculate_planets  # local import keeps module graph flat

    planets = calculate_planets(jd_ut, settings)
    log.info("sidereal_positions_calculated n=%d", len(planets))

    # 12. Angles ----------------------------------------------------------
    angles = calculate_angles(jd_ut, lat, lon, settings)
    cusps = angles.pop("_cusps")
    ascendant = angles["ascendant"]

    # 13-14. Houses and placements ---------------------------------------
    houses = build_houses(ascendant["sign_index"], settings)
    place_planets_in_houses(planets, ascendant["sign_index"], houses)
    annotate_house_lord_placement(houses, planets)
    log.info("houses_calculated system=%s", settings.house_system)

    # 18-19. Derived states and aspects -----------------------------------
    annotate_dignity(planets)
    annotate_combustion(planets)
    aspects = calculate_aspects(planets, houses)

    # 20. Vimshottari ------------------------------------------------------
    moon = next((p for p in planets if p["body"] == "Moon"), None)
    if moon is None:
        raise ChartError("CALCULATION_FAILED", "Moon position missing; cannot compute Vimshottari dasha.")
    dashas = calculate_vimshottari(moon["position"]["sidereal_longitude"], jd_ut, settings)
    as_of = as_of_date or datetime.now(dt_timezone.utc).date().isoformat()
    running = current_period(dashas, as_of)
    log.info("dasha_calculated lord=%s", dashas["birth_mahadasha"]["lord"])

    # 21. Assemble ---------------------------------------------------------
    input_block = {
        "birth_date": b_date.isoformat(),
        "birth_time_local": b_time.isoformat(),
        "location": {
            "name": location_name,
            "latitude": lat,
            "longitude": lon,
        },
    }
    chart = {
        "schema_version": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "chart_id": _chart_id({"input": input_block, "settings": settings.model_dump(), "tz": timezone_id}),
        "input": input_block,
        "time": {
            "timezone": time_block["timezone"],
            "local_datetime": time_block["local_datetime"],
            "utc_datetime": time_block["utc_datetime"],
            "julian_day_ut": round(jd_ut, 9),
        },
        "calculation_settings": {
            **settings.model_dump(),
            "ayanamsa_label": AYANAMSA_LABEL[settings.ayanamsa],
            "ayanamsa_value_degrees": round(ayanamsa_value(jd_ut), 6),
        },
        "location_resolution": {
            "source": location_source,
            "timezone_source": timezone_source,
        },
        "angles": angles,
        "house_cusps_raw": cusps,
        "houses": houses,
        "planets": planets,
        "aspects": aspects,
        "dashas": {"vimshottari": dashas, "current_period": running},
        "chart_summary": chart_summary(planets, houses, ascendant),
        "warnings": warnings,
        "provenance": {
            "ephemeris": "Swiss Ephemeris" if mode == "swiss" else "Moshier (built into Swiss Ephemeris)",
            "ephemeris_mode": mode,
            "calculation_engine": "pyswisseph",
            "pyswisseph_version": swiss_ephemeris_version(),
            "computed_at_utc": datetime.now(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }

    # 22. Chart-level validation ------------------------------------------
    chart["validation"] = _validate_chart(chart)

    # Retrieval-facing views
    chart["fact_keys"] = build_fact_keys(planets, houses, ascendant, dashas)
    chart["fact_ids"] = build_fact_ids(planets, houses, aspects, dashas)
    if settings.include_natural_language_facts:
        chart["natural_language_facts"] = build_natural_language_facts(
            planets, houses, ascendant, aspects, dashas
        )
    log.info("chart_validation_%s chart_id=%s", chart["validation"]["status"], chart["chart_id"])
    return chart
