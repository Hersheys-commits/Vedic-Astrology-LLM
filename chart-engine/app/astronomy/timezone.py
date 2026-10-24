"""Location -> IANA timezone -> historically correct UTC instant.

This module is the most error-prone part of any chart engine, so every
decision it makes is reported back in the chart JSON.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.utils.errors import ChartError
from app.utils.formatting import format_offset

_finder = None


def _timezone_finder():
    """Lazily construct TimezoneFinder (it loads a boundary dataset)."""
    global _finder
    if _finder is None:
        from timezonefinder import TimezoneFinder

        _finder = TimezoneFinder()
    return _finder


def resolve_timezone(latitude: float, longitude: float) -> str:
    """lat/lon -> IANA timezone id, e.g. 'Asia/Kolkata'."""
    tf = _timezone_finder()
    tz_name = tf.timezone_at(lat=latitude, lng=longitude)
    if tz_name is None:
        tz_name = tf.certain_timezone_at(lat=latitude, lng=longitude)
    if tz_name is None:
        raise ChartError(
            "TIMEZONE_NOT_FOUND",
            "No IANA timezone could be resolved for these coordinates "
            "(likely open ocean). Supply `timezone` explicitly in the request.",
            latitude=latitude,
            longitude=longitude,
        )
    return tz_name


def load_zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ChartError("TIMEZONE_NOT_FOUND", f"Unknown IANA timezone id {tz_name!r}.")


def _is_ambiguous(naive: datetime, zone: ZoneInfo) -> bool:
    """The same wall clock time maps to two UTC instants (clocks went back)."""
    return naive.replace(tzinfo=zone, fold=0).utcoffset() != naive.replace(tzinfo=zone, fold=1).utcoffset()


def _is_nonexistent(naive: datetime, zone: ZoneInfo) -> bool:
    """The wall clock time was skipped (clocks went forward)."""
    aware = naive.replace(tzinfo=zone)
    roundtrip = aware.astimezone(ZoneInfo("UTC")).astimezone(zone).replace(tzinfo=None)
    return roundtrip != naive


def to_utc(
    birth_date: date,
    birth_time: time,
    tz_name: str,
    dst_choice: str | None = None,
) -> dict:
    """Combine local civil date/time with a timezone and produce the UTC instant.

    dst_choice: None | 'first_occurrence' | 'second_occurrence'
    Raises AMBIGUOUS_LOCAL_TIME / NONEXISTENT_LOCAL_TIME when the caller must decide.
    """
    zone = load_zone(tz_name)
    naive = datetime.combine(birth_date, birth_time)
    warnings: list[dict] = []

    if _is_nonexistent(naive, zone):
        raise ChartError(
            "NONEXISTENT_LOCAL_TIME",
            "The selected local birth time never occurred: clocks moved forward "
            "over this wall-clock time in this timezone.",
            requires_user_choice=True,
            timezone=tz_name,
        )

    fold = 0
    if _is_ambiguous(naive, zone):
        if dst_choice not in ("first_occurrence", "second_occurrence"):
            raise ChartError(
                "AMBIGUOUS_LOCAL_TIME",
                "The selected local birth time occurs twice due to a timezone transition.",
                requires_user_choice=True,
                options=["first_occurrence", "second_occurrence"],
                timezone=tz_name,
            )
        fold = 0 if dst_choice == "first_occurrence" else 1
        warnings.append(
            {
                "code": "AMBIGUOUS_LOCAL_TIME_RESOLVED",
                "message": f"Ambiguous local time resolved using {dst_choice}.",
            }
        )

    local_dt = naive.replace(tzinfo=zone, fold=fold)
    offset = local_dt.utcoffset() or timedelta(0)
    dst = local_dt.dst() or timedelta(0)
    utc_dt = local_dt.astimezone(ZoneInfo("UTC"))

    # Pre-1900 offsets in the IANA database are Local Mean Time approximations.
    if birth_date.year < 1900:
        warnings.append(
            {
                "code": "HISTORICAL_TIMEZONE_UNCERTAIN",
                "message": (
                    "Pre-1900 offsets in the IANA database are local mean time "
                    "approximations; the UTC conversion may be uncertain by minutes."
                ),
            }
        )
    if offset.total_seconds() % 60 != 0:
        warnings.append(
            {
                "code": "NON_MINUTE_UTC_OFFSET",
                "message": "The historical UTC offset is not a whole number of minutes (LMT era).",
            }
        )

    return {
        "timezone": {
            "iana": tz_name,
            "utc_offset": format_offset(offset),
            "utc_offset_seconds": int(offset.total_seconds()),
            "is_dst": dst.total_seconds() != 0,
            "abbreviation": local_dt.tzname(),
        },
        "local_datetime": local_dt.isoformat(),
        "utc_datetime": utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "utc_dt": utc_dt,
        "warnings": warnings,
    }
