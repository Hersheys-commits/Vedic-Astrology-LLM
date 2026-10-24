"""Predictable, machine-readable error contract."""

from __future__ import annotations

from typing import Any


class ChartError(Exception):
    """Raised for every expected failure in the pipeline.

    Codes:
        INVALID_DATE, INVALID_TIME, INVALID_COORDINATES, LOCATION_NOT_FOUND,
        TIMEZONE_NOT_FOUND, AMBIGUOUS_LOCAL_TIME, NONEXISTENT_LOCAL_TIME,
        EPHEMERIS_NOT_AVAILABLE, UNSUPPORTED_DATE, CALCULATION_FAILED,
        CONFIGURATION_ERROR
    """

    http_status = 400

    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra

    def to_dict(self) -> dict:
        payload = {"code": self.code, "message": self.message}
        payload.update(self.extra)
        return {"error": payload}
