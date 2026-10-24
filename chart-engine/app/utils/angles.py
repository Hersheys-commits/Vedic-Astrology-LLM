"""Circular-angle helpers. Every longitude in the engine passes through here."""

from __future__ import annotations


def norm360(angle: float) -> float:
    """Normalise any angle into [0, 360)."""
    a = angle % 360.0
    if a < 0:
        a += 360.0
    # Guard against float artefacts like 359.9999999999999 -> keep as is, but
    # never allow exactly 360.0 to leak through.
    if a >= 360.0:
        a -= 360.0
    return a


def angular_separation(a: float, b: float) -> float:
    """Shortest separation between two longitudes, in [0, 180]."""
    d = abs(norm360(a) - norm360(b)) % 360.0
    return 360.0 - d if d > 180.0 else d


def forward_distance(frm: float, to: float) -> float:
    """Zodiacal (anticlockwise) distance from `frm` to `to`, in [0, 360)."""
    return norm360(to - frm)


def sign_index_of(longitude: float) -> int:
    """0-based sign index: 0 = Aries."""
    return int(norm360(longitude) // 30.0) % 12


def degree_in_sign(longitude: float) -> float:
    return norm360(longitude) % 30.0


def house_distance(from_sign_index: int, to_sign_index: int) -> int:
    """Inclusive house count from one sign to another: same sign = 1."""
    return ((to_sign_index - from_sign_index) % 12) + 1
