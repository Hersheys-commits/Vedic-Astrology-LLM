"""Engine-wide configuration. Nothing here is hidden inside calculation code."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"
ENGINE_VERSION = "0.1.0"

BASE_DIR = Path(__file__).resolve().parents[2]

# Directory holding Swiss Ephemeris .se1 files. If empty, the engine falls back
# to the built-in Moshier ephemeris (no data files needed, slightly lower precision).
EPHEMERIS_PATH = os.getenv("EPHEMERIS_PATH", str(BASE_DIR / "data" / "ephe"))

# Supported input range (Moshier is reliable roughly 3000 BC - 3000 AD).
MIN_YEAR = int(os.getenv("MIN_YEAR", "1700"))
MAX_YEAR = int(os.getenv("MAX_YEAR", "2100"))

# Nominatim / geocoding
GEOCODER_URL = os.getenv("GEOCODER_URL", "https://nominatim.openstreetmap.org")
GEOCODER_USER_AGENT = os.getenv("GEOCODER_USER_AGENT", "vedic-chart-engine/0.1 (contact: set-your-email)")
GEOCODER_TIMEOUT = float(os.getenv("GEOCODER_TIMEOUT", "10"))


class ChartSettings(BaseModel):
    """Explicit calculation choices. Change these instead of editing engine code."""

    tradition: Literal["vedic"] = "vedic"
    zodiac: Literal["sidereal", "tropical"] = "sidereal"
    ayanamsa: Literal["lahiri", "raman", "krishnamurti", "yukteshwar", "fagan_bradley"] = "lahiri"
    house_system: Literal["whole_sign", "equal", "placidus"] = "whole_sign"
    node_type: Literal["mean", "true"] = "mean"

    include_outer_planets: bool = False
    include_antardasha: bool = Field(
        default=False,
        description="Nest 2nd-level dashas inside each mahadasha. Validate mahadasha first.",
    )
    dasha_years: int = Field(default=120, description="How many years of mahadasha timeline to emit.")
    include_natural_language_facts: bool = True


AYANAMSA_LABEL = {
    "lahiri": "Lahiri (Chitrapaksha)",
    "raman": "B.V. Raman",
    "krishnamurti": "Krishnamurti (KP)",
    "yukteshwar": "Yukteshwar",
    "fagan_bradley": "Fagan/Bradley",
}
