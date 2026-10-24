# Vedic Chart Engine

Deterministic Vedic (Jyotisha) chart engine. It answers exactly one question —
**"what is in the chart?"** — and answers it with facts only: no LLM, no
interpretation, no randomness anywhere in the pipeline.

It is Layer 1 of the AI-assisted astrology interpretation system. The JSON it
produces is the factual input for the RAG layer and, after that, the generation
layer.

```
birth date + local time + lat/lon
        ↓  validation
        ↓  IANA timezone (timezonefinder + zoneinfo, historical offsets)
        ↓  UTC  →  Julian Day
        ↓  Swiss Ephemeris (sidereal, Lahiri)
        ↓  signs · whole-sign houses · nakshatra + pada · aspects · Vimshottari
        ↓
structured chart JSON  +  fact_keys  +  natural_language_facts
```

---

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional
```

Python 3.11+ is required (`zoneinfo` and `datetime.fold` are used directly).

No ephemeris data files are needed to start: the engine falls back to the
Moshier ephemeris built into pyswisseph and says so in every chart's `warnings`
and `provenance`. See `data/ephe/README.md` to switch to the full Swiss
Ephemeris files.

## 2. Run it

**As a CLI (no server):**

```bash
python scripts/generate_chart_cli.py \
    --date 2004-05-17 --time 14:35 \
    --lat 25.4358 --lon 81.8463 \
    --name "Prayagraj, India" --antardasha
```

Add `--out chart.json` to save the full JSON, `--quiet` to suppress the table.

**As an API:**

```bash
uvicorn app.main:app --reload
```

| URL | What it is |
|---|---|
| `http://localhost:8000/` | Leaflet test console (map click + address search) |
| `http://localhost:8000/docs` | Swagger UI |
| `POST /chart` | the only endpoint the RAG layer needs |
| `POST /location/geocode` | address → candidate coordinates (Nominatim) |
| `POST /location/reverse` | coordinates → display name |
| `POST /location/timezone` | coordinates → IANA timezone (offline) |
| `GET /health` | ephemeris mode, engine and schema versions |

```bash
curl -s localhost:8000/chart -H 'Content-Type: application/json' -d '{
  "birth_date": "2004-05-17",
  "birth_time": "14:35:00",
  "location": {"mode": "map_click", "latitude": 25.4358, "longitude": 81.8463,
               "display_name": "Prayagraj, India"},
  "settings": {"ayanamsa": "lahiri", "house_system": "whole_sign", "node_type": "mean"}
}' | python -m json.tool
```

**As a library:**

```python
from app.config.settings import ChartSettings
from app.engine.calculator import generate_chart

chart = generate_chart(
    birth_date="2004-05-17", birth_time="14:35:00",
    latitude=25.4358, longitude=81.8463,
    location_name="Prayagraj, India",
    settings=ChartSettings(ayanamsa="lahiri", include_antardasha=True),
)
```

## 3. What comes out

Top-level keys of the chart JSON (`schema_version` 1.0):

| Key | Contents |
|---|---|
| `input` | exactly what was submitted, unchanged |
| `time` | IANA timezone, UTC offset in force **on the birth date**, local and UTC datetimes, Julian Day |
| `calculation_settings` | tradition, zodiac, ayanamsa (+ its numeric value), house system, node type |
| `angles` | ascendant and midheaven with sign, degree, formatted degree, nakshatra |
| `houses` | 12 objects: sign, lord, occupants, aspects received, where the lord sits |
| `planets` | 9 bodies with `position`, `placement`, `nakshatra`, `motion`, `dignity`, `combustion`, `raw` |
| `aspects` | graha drishti and conjunctions, with rule name and raw angular distance |
| `dashas` | Vimshottari balance at birth, mahadasha timeline, optional antardasha, running period |
| `chart_summary` | lagna / moon / sun, retrograde list, planets per house |
| `fact_keys`, `fact_ids` | normalised retrieval keys, e.g. `planet:mars|sign:cancer|house:7` |
| `natural_language_facts` | template-generated factual sentences for prompting |
| `validation` | per-check results (nodes opposed, padas valid, every planet placed, …) |
| `warnings`, `provenance` | DST resolutions, ephemeris mode, versions |

Both directions of every relation are written out, so the next layer never has
to infer cross-references: `planets[i].placement.house` and
`houses[n].occupants` both exist.

## 4. Configuration

Everything tradition-dependent lives in `app/config/`, not inside calculation
code:

- `settings.py` — `ChartSettings`: zodiac, ayanamsa, house system, node type
- `signs.py` — sign list, Vedic sign lords
- `nakshatras.py` — the 27 nakshatras and their Vimshottari lords
- `planets.py` — bodies, aspect rules (`Mars: [4,7,8]` …), Vimshottari years, combustion orbs
- `dignities.py` — exaltation, debilitation, own sign, moolatrikona

Change the tradition there; the engine records whatever you chose inside every
chart so old charts stay reproducible.

## 5. Tests and validation

```bash
pytest -q
```

Covers timezone/DST edge cases (ambiguous and nonexistent local times),
Julian-day round trips, nakshatra boundaries, whole-sign placement,
Vimshottari arithmetic, API error contract, and full-pipeline determinism.

**Freeze a verified chart as a regression fixture:**

```bash
python scripts/generate_reference_chart.py --id my_chart \
    --date 2004-05-17 --time 14:35 --lat 25.4358 --lon 81.8463
```

`tests/test_reference_charts.py` replays every fixture and fails on drift of
more than one arc-second.

**Cross-check against an external source** (Prokerala, AstrologyAPI, RoxyAPI,
or a printed chart) by writing the expected positions into a JSON file:

```bash
python scripts/cross_validate.py expected.json --tolerance 0.0167
```

It prints a per-body difference table and an agreement rate — that is the
"validation report" deliverable of Phase 2.

## 6. Error contract

Errors return HTTP 400 with a stable code:

```json
{"error": {"code": "AMBIGUOUS_LOCAL_TIME",
           "message": "The selected local birth time occurs twice due to a timezone transition.",
           "requires_user_choice": true,
           "options": ["first_occurrence", "second_occurrence"]}}
```

Codes: `INVALID_DATE`, `INVALID_TIME`, `INVALID_COORDINATES`,
`LOCATION_NOT_FOUND`, `TIMEZONE_NOT_FOUND`, `AMBIGUOUS_LOCAL_TIME`,
`NONEXISTENT_LOCAL_TIME`, `EPHEMERIS_NOT_AVAILABLE`, `UNSUPPORTED_DATE`,
`CALCULATION_FAILED`, `CONFIGURATION_ERROR`.

Resolve an ambiguous time by resending with `"dst_choice": "first_occurrence"`.

## 7. Where the RAG layer plugs in

Index your doctrine corpus by the same key shape the engine emits:

```
planet:mars|house:7          → passages about Mars in the 7th
planet:mars|sign:cancer      → passages about Mars in Cancer
house:10|lord:venus          → passages about the 10th lord being Venus
planet:moon|nakshatra:rohini|pada:2
```

Then retrieval is an exact metadata lookup first, with embedding search only as
a fallback or re-ranking step — which is the design in Section 2.2 of the
system plan. Nothing in this repository should ever call an LLM.

## 8. Docker

```bash
docker build -t chart-engine .
docker run -p 8000:8000 chart-engine
```

## 9. Known limitations

- Whole-sign is the only house model wired into the semantic layer; equal and
  Placidus cusps are computed and returned in `house_cusps_raw` but no house
  objects are built from them yet.
- Vimshottari goes two levels deep (mahadasha, antardasha). Pratyantardasha is
  deliberately not implemented until the first two are validated.
- Divisional charts (D9 navamsa and the rest), ashtakavarga, shadbala and yogas
  are not implemented — each belongs in its own deterministic module under
  `app/engine/`.
- Rahu/Ketu aspect rules and their exaltation signs are tradition-dependent;
  the defaults in `app/config/` are one common choice, not a universal one.
- Pre-1900 births carry a `HISTORICAL_TIMEZONE_UNCERTAIN` warning: IANA offsets
  before then are local-mean-time approximations.
