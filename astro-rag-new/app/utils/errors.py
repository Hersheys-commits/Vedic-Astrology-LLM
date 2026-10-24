"""
Single exception type for the whole service, mirroring the chart-engine's
`ChartError` convention: everything raised anywhere in the pipeline carries a
machine-readable `code`, and `app/main.py` is the only place that turns it
into an HTTP response.
"""
from __future__ import annotations


class RAGError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}
