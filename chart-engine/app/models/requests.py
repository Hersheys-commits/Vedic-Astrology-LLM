"""Request contracts. Pydantic validates shape; app.utils.validation validates meaning."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.config.settings import ChartSettings


class LocationInput(BaseModel):
    mode: Literal["coordinates", "map_click", "geocoded"] = "coordinates"
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    display_name: Optional[str] = None


class ChartRequest(BaseModel):
    birth_date: str = Field(..., examples=["2004-05-17"])
    birth_time: str = Field(..., examples=["14:35:00"])
    location: LocationInput
    timezone: Optional[str] = Field(
        default=None,
        description="IANA id. Leave empty to resolve it from the coordinates.",
    )
    dst_choice: Optional[Literal["first_occurrence", "second_occurrence"]] = None
    as_of_date: Optional[str] = Field(
        default=None, description="Date used to report the running dasha. Defaults to today (UTC)."
    )
    settings: ChartSettings = ChartSettings()

    model_config = {
        "json_schema_extra": {
            "example": {
                "birth_date": "2004-05-17",
                "birth_time": "14:35:00",
                "location": {
                    "mode": "map_click",
                    "latitude": 25.4358,
                    "longitude": 81.8463,
                    "display_name": "Prayagraj, Uttar Pradesh, India",
                },
                "settings": {
                    "tradition": "vedic",
                    "zodiac": "sidereal",
                    "ayanamsa": "lahiri",
                    "house_system": "whole_sign",
                    "node_type": "mean",
                },
            }
        }
    }


class GeocodeRequest(BaseModel):
    query: str = Field(..., min_length=2, examples=["MNNIT Prayagraj"])
    limit: int = Field(default=5, ge=1, le=10)


class ReverseGeocodeRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
