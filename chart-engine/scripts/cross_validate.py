#!/usr/bin/env python3
"""Compare engine output against an independently computed chart.

Expected-value file format (positions in decimal degrees of sidereal longitude,
or "Sign D M S" strings):

    {
      "source": "Prokerala API",
      "input": {"birth_date": "2004-05-17", "birth_time": "14:35:00",
                "latitude": 25.4358, "longitude": 81.8463},
      "ascendant": "Virgo 9 14 41",
      "planets": {"Sun": 32.878, "Moon": "Aries 12 44 15"}
    }

Usage:
    python scripts/cross_validate.py expected.json --tolerance 0.0167
(0.0167 degrees = 1 arc-minute.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.signs import SIGNS                  # noqa: E402
from app.engine.calculator import generate_chart    # noqa: E402
from app.utils.angles import angular_separation     # noqa: E402


def to_degrees(value) -> float:
    """Accept 32.878 or 'Taurus 2 52 44'."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).replace("\u00b0", " ").replace("'", " ").replace('"', " ").split()
    sign = parts[0].capitalize()
    d = float(parts[1]) if len(parts) > 1 else 0.0
    m = float(parts[2]) if len(parts) > 2 else 0.0
    s = float(parts[3]) if len(parts) > 3 else 0.0
    return SIGNS.index(sign) * 30.0 + d + m / 60.0 + s / 3600.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("expected_file")
    ap.add_argument("--tolerance", type=float, default=1 / 60, help="degrees (default 1 arc-minute)")
    args = ap.parse_args()

    spec = json.loads(
        Path(args.expected_file).read_text(encoding="utf-8-sig")
    )
    chart = generate_chart(
        birth_date=spec["input"]["birth_date"],
        birth_time=spec["input"]["birth_time"],
        latitude=spec["input"]["latitude"],
        longitude=spec["input"]["longitude"],
        timezone_id=spec["input"].get("timezone"),
        as_of_date="2026-01-01",
    )

    rows, failures = [], 0
    if "ascendant" in spec:
        got = chart["angles"]["ascendant"]["longitude"]
        want = to_degrees(spec["ascendant"])
        diff = angular_separation(got, want)
        failures += diff > args.tolerance
        rows.append(("Ascendant", got, want, diff))

    computed = {p["body"]: p["position"]["sidereal_longitude"] for p in chart["planets"]}
    for body, expected in spec.get("planets", {}).items():
        got = computed.get(body)
        if got is None:
            rows.append((body, float("nan"), to_degrees(expected), float("nan")))
            failures += 1
            continue
        want = to_degrees(expected)
        diff = angular_separation(got, want)
        failures += diff > args.tolerance
        rows.append((body, got, want, diff))

    print(f"\nCross-validation against: {spec.get('source', 'unknown source')}")
    print(f"tolerance: {args.tolerance:.6f} deg ({args.tolerance * 60:.2f} arc-min)\n")
    print(f"{'Body':<12}{'engine':>12}{'expected':>12}{'diff(deg)':>12}{'':>4}")
    print("-" * 52)
    for name, got, want, diff in rows:
        flag = "OK" if diff <= args.tolerance else "FAIL"
        print(f"{name:<12}{got:>12.5f}{want:>12.5f}{diff:>12.6f}  {flag}")
    agreement = (len(rows) - failures) / len(rows) * 100 if rows else 0.0
    print(f"\nagreement rate: {agreement:.1f}%  ({len(rows) - failures}/{len(rows)})")
    print("Differences usually mean a different ayanamsa, node type, or timezone "
          "assumption on one side. Log the configuration of both engines.\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
