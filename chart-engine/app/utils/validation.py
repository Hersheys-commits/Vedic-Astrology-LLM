"""Input validation. Runs before any astronomical calculation."""

from __future__ import annotations

from datetime import date, time

from app.config.settings import MAX_YEAR, MIN_YEAR
from app.utils.errors import ChartError


def parse_birth_date(value: str) -> date:
    try:
        d = date.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        raise ChartError("INVALID_DATE", f"Birth date must be ISO format YYYY-MM-DD, got {value!r}.")
    if not (MIN_YEAR <= d.year <= MAX_YEAR):
        raise ChartError(
            "UNSUPPORTED_DATE",
            f"Birth year {d.year} is outside the supported range {MIN_YEAR}-{MAX_YEAR}.",
        )
    return d


def parse_birth_time(value: str) -> time:
    """Accepts HH:MM or HH:MM:SS. 14:35 -> 14:35:00."""
    raw = (value or "").strip()
    parts = raw.split(":")
    if len(parts) == 2:
        raw = raw + ":00"
    elif len(parts) != 3:
        raise ChartError("INVALID_TIME", f"Birth time must be HH:MM or HH:MM:SS, got {value!r}.")
    try:
        t = time.fromisoformat(raw)
    except ValueError:
        raise ChartError("INVALID_TIME", f"Birth time must be a valid local civil time, got {value!r}.")
    return t


def validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        raise ChartError("INVALID_COORDINATES", "Latitude and longitude must be numbers.")
    if not (-90.0 <= lat <= 90.0):
        raise ChartError("INVALID_COORDINATES", f"Latitude must be between -90 and 90, got {lat}.")
    if not (-180.0 <= lon <= 180.0):
        raise ChartError("INVALID_COORDINATES", f"Longitude must be between -180 and 180, got {lon}.")
    return lat, lon
