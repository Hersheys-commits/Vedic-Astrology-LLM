"""Human-readable formatting helpers. Numeric values are always kept alongside."""

from __future__ import annotations

from datetime import timedelta


def format_dms(degrees: float) -> str:
    """12.3821 -> 12°22'56\"."""
    total_seconds = round(abs(degrees) * 3600.0)
    d, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    sign = "-" if degrees < 0 else ""
    return f"{sign}{d}\u00b0{m:02d}'{s:02d}\""


def format_offset(offset: timedelta) -> str:
    """timedelta(hours=5, minutes=30) -> '+05:30'."""
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    return f"{sign}{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def slug(value: str) -> str:
    """'Purva Phalguni' -> 'purva_phalguni'. Used for deterministic fact keys."""
    return value.strip().lower().replace(" ", "_")


def years_to_dhm(years: float) -> str:
    """Render a dasha balance as 'Xy Ym Zd'."""
    total_days = years * 365.25
    y = int(total_days // 365.25)
    rem = total_days - y * 365.25
    m = int(rem // 30.4375)
    d = int(round(rem - m * 30.4375))
    return f"{y}y {m}m {d}d"
