#!/usr/bin/env python3
"""Freeze a chart as a regression fixture.

Run this once a chart has been manually verified against a trusted source, then
commit the fixture. tests/test_reference_charts.py replays every fixture and
fails if the engine output ever drifts.

    python scripts/generate_reference_chart.py --id prayagraj_2004 \
        --date 2004-05-17 --time 14:35 --lat 25.4358 --lon 81.8463
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.calculator import generate_chart   # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--time", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--tz", default=None)
    args = p.parse_args()

    chart = generate_chart(
        birth_date=args.date, birth_time=args.time,
        latitude=args.lat, longitude=args.lon,
        timezone_id=args.tz, as_of_date="2026-01-01",
    )
    fixture = {
        "input": {
            "birth_date": args.date, "birth_time": args.time,
            "latitude": args.lat, "longitude": args.lon, "timezone": args.tz,
        },
        "expected": {
            "utc_datetime": chart["time"]["utc_datetime"],
            "julian_day_ut": chart["time"]["julian_day_ut"],
            "ascendant_sign": chart["angles"]["ascendant"]["sign"],
            "ascendant_longitude": chart["angles"]["ascendant"]["longitude"],
            "planets": {
                p["body"]: {
                    "longitude": p["position"]["sidereal_longitude"],
                    "sign": p["position"]["zodiac_sign"],
                    "house": p["placement"]["house"],
                    "nakshatra": p["nakshatra"]["name"],
                    "pada": p["nakshatra"]["pada"],
                }
                for p in chart["planets"]
            },
            "birth_mahadasha": chart["dashas"]["vimshottari"]["birth_mahadasha"]["lord"],
        },
    }
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / f"{args.id}.json"
    path.write_text(json.dumps(fixture, indent=2))
    print(f"Fixture written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
