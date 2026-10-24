"""Thin, explicit wrapper around pyswisseph.

Nothing above this module imports swisseph directly, so the ephemeris choice,
sidereal mode and flags live in exactly one place.
"""

from __future__ import annotations

import os
import threading

import swisseph as swe

from app.config.settings import EPHEMERIS_PATH
from app.utils.errors import ChartError

_lock = threading.Lock()
_initialised = False
_ephemeris_mode = "moshier"

AYANAMSA_MODES = {
    "lahiri": swe.SIDM_LAHIRI,
    "raman": swe.SIDM_RAMAN,
    "krishnamurti": swe.SIDM_KRISHNAMURTI,
    "yukteshwar": swe.SIDM_YUKTESHWAR,
    "fagan_bradley": swe.SIDM_FAGAN_BRADLEY,
}

HOUSE_SYSTEM_CODES = {
    "whole_sign": b"W",
    "equal": b"A",
    "placidus": b"P",
}


def _has_ephemeris_files() -> bool:
    if not EPHEMERIS_PATH or not os.path.isdir(EPHEMERIS_PATH):
        return False
    return any(f.lower().endswith(".se1") for f in os.listdir(EPHEMERIS_PATH))


def init_ephemeris() -> str:
    """Configure the ephemeris once per process. Returns the active mode."""
    global _initialised, _ephemeris_mode
    with _lock:
        if _initialised:
            return _ephemeris_mode
        if _has_ephemeris_files():
            swe.set_ephe_path(EPHEMERIS_PATH)
            _ephemeris_mode = "swiss"
        else:
            # Built-in Moshier analytical ephemeris: no data files required.
            _ephemeris_mode = "moshier"
        _initialised = True
        return _ephemeris_mode


def ephemeris_mode() -> str:
    return init_ephemeris()


def base_flags(sidereal: bool) -> int:
    mode = init_ephemeris()
    flags = swe.FLG_SWIEPH if mode == "swiss" else swe.FLG_MOSEPH
    flags |= swe.FLG_SPEED
    if sidereal:
        flags |= swe.FLG_SIDEREAL
    return flags


def set_sidereal_mode(ayanamsa: str) -> None:
    try:
        swe.set_sid_mode(AYANAMSA_MODES[ayanamsa], 0, 0)
    except KeyError:
        raise ChartError("CONFIGURATION_ERROR", f"Unsupported ayanamsa {ayanamsa!r}.")


def calc_body(jd_ut: float, swe_id: int, sidereal: bool) -> dict:
    """Raw astronomical values for one body. No Vedic semantics here."""
    try:
        values, retflags = swe.calc_ut(jd_ut, swe_id, base_flags(sidereal))
    except Exception as exc:
        raise ChartError("CALCULATION_FAILED", f"Swiss Ephemeris calculation failed: {exc}")
    return {
        "longitude": values[0],
        "latitude": values[1],
        "distance_au": values[2],
        "speed_longitude": values[3],
        "speed_latitude": values[4],
        "speed_distance": values[5],
        "return_flags": retflags,
    }


def calc_houses(jd_ut: float, latitude: float, longitude: float, house_system: str, sidereal: bool) -> dict:
    """Returns house cusps and the ascmc array (Ascendant, MC, ...)."""
    code = HOUSE_SYSTEM_CODES.get(house_system)
    if code is None:
        raise ChartError("CONFIGURATION_ERROR", f"Unsupported house system {house_system!r}.")
    flags = swe.FLG_SIDEREAL if sidereal else 0
    try:
        cusps, ascmc = swe.houses_ex(jd_ut, latitude, longitude, code, flags)
    except Exception as exc:
        raise ChartError("CALCULATION_FAILED", f"House calculation failed: {exc}")
    return {"cusps": list(cusps), "ascendant": ascmc[0], "midheaven": ascmc[1], "ascmc": list(ascmc)}


def ayanamsa_value(jd_ut: float) -> float:
    init_ephemeris()
    return swe.get_ayanamsa_ut(jd_ut)


def swiss_ephemeris_version() -> str:
    return swe.version


def node_body_id(node_type: str) -> int:
    return swe.MEAN_NODE if node_type == "mean" else swe.TRUE_NODE
