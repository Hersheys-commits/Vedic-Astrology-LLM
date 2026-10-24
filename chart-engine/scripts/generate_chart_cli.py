#!/usr/bin/env python3
"""Command-line chart generation. No server needed.

Examples
--------
python scripts/generate_chart_cli.py --date 2004-05-17 --time 14:35 \
    --lat 25.4358 --lon 81.8463 --name "Prayagraj, India"

python scripts/generate_chart_cli.py --date 2004-05-17 --time 14:35 \
    --lat 25.4358 --lon 81.8463 --out chart.json --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import ChartSettings          # noqa: E402
from app.engine.calculator import generate_chart       # noqa: E402
from app.utils.errors import ChartError                # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deterministic Vedic chart engine (CLI)")
    p.add_argument("--date", required=True, help="Birth date, YYYY-MM-DD")
    p.add_argument("--time", required=True, help="Local civil birth time, HH:MM or HH:MM:SS")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--name", default=None, help="Display name of the birth place")
    p.add_argument("--tz", default=None, help="IANA timezone; resolved from coordinates if omitted")
    p.add_argument("--dst-choice", choices=["first_occurrence", "second_occurrence"], default=None)
    p.add_argument("--ayanamsa", default="lahiri",
                   choices=["lahiri", "raman", "krishnamurti", "yukteshwar", "fagan_bradley"])
    p.add_argument("--node-type", default="mean", choices=["mean", "true"])
    p.add_argument("--antardasha", action="store_true", help="Include 2nd-level dashas")
    p.add_argument("--as-of", default=None, help="Date for the running-dasha report (YYYY-MM-DD)")
    p.add_argument("--out", default=None, help="Write the chart JSON to this file")
    p.add_argument("--quiet", action="store_true", help="Suppress the human-readable table")
    return p


def print_table(chart: dict) -> None:
    t, s = chart["time"], chart["chart_summary"]
    print(f"\nChart {chart['chart_id']}  schema v{chart['schema_version']}  engine v{chart['engine_version']}")
    print(f"  local  {t['local_datetime']}   tz {t['timezone']['iana']} ({t['timezone']['utc_offset']})")
    print(f"  utc    {t['utc_datetime']}   JD {t['julian_day_ut']}")
    print(f"  ayanamsa {chart['calculation_settings']['ayanamsa_label']} "
          f"= {chart['calculation_settings']['ayanamsa_value_degrees']}\u00b0")
    print(f"\n  Lagna {s['ascendant_sign']} {chart['angles']['ascendant']['formatted']} "
          f"| Moon {s['moon_sign']} ({s['moon_nakshatra']} pada {s['moon_pada']}) | Sun {s['sun_sign']}")

    print(f"\n  {'Body':<9}{'Sign':<13}{'Degree':<12}{'H':<4}{'Nakshatra':<20}{'Pada':<6}{'State'}")
    print("  " + "-" * 78)
    for p in chart["planets"]:
        state = []
        if p["motion"]["retrograde"]:
            state.append("R")
        if p["dignity"]["status"] != "neutral":
            state.append(p["dignity"]["status"])
        if p.get("combustion", {}).get("combust"):
            state.append("combust")
        print(f"  {p['body']:<9}{p['position']['zodiac_sign']:<13}"
              f"{p['position']['formatted_degree']:<12}{p['placement']['house']:<4}"
              f"{p['nakshatra']['name']:<20}{p['nakshatra']['pada']:<6}{' '.join(state)}")

    print(f"\n  {'House':<7}{'Sign':<13}{'Lord':<10}Occupants")
    print("  " + "-" * 60)
    for h in chart["houses"]:
        print(f"  {h['number']:<7}{h['sign']:<13}{h['lord']:<10}{', '.join(h['occupants']) or '-'}")

    v = chart["dashas"]["vimshottari"]
    bm = v["birth_mahadasha"]
    print(f"\n  Vimshottari: birth mahadasha {bm['lord']}, balance "
          f"{bm['balance_years']:.3f} years, ends {bm['ends']}")
    for m in v["mahadasha"][:9]:
        print(f"    {m['lord']:<9}{m['start']} -> {m['end']}")
    if chart["dashas"]["current_period"]:
        cp = chart["dashas"]["current_period"]
        line = f"  Running on {cp['as_of']}: mahadasha {cp['mahadasha']}"
        if "antardasha" in cp:
            line += f", antardasha {cp['antardasha']}"
        print(line)

    if chart["warnings"]:
        print("\n  Warnings:")
        for w in chart["warnings"]:
            print(f"    [{w['code']}] {w['message']}")
    print(f"\n  Validation: {chart['validation']['status']}  "
          f"({len(chart['fact_keys'])} fact keys, {len(chart.get('natural_language_facts', []))} NL facts)\n")


def main() -> int:
    args = build_parser().parse_args()
    settings = ChartSettings(
        ayanamsa=args.ayanamsa,
        node_type=args.node_type,
        include_antardasha=args.antardasha,
    )
    try:
        chart = generate_chart(
            birth_date=args.date,
            birth_time=args.time,
            latitude=args.lat,
            longitude=args.lon,
            location_name=args.name,
            location_source="cli",
            timezone_id=args.tz,
            dst_choice=args.dst_choice,
            settings=settings,
            as_of_date=args.as_of,
        )
    except ChartError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 1

    if args.out:
        Path(args.out).write_text(
            json.dumps(chart, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Chart written to {args.out}")
    if not args.quiet:
        print_table(chart)
    elif not args.out:
        print(json.dumps(chart, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
