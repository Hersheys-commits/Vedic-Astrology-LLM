#!/usr/bin/env python3
"""Build the Astro RAG corpus from the four supplied OCR PDFs.

This is intentionally structure-aware.  It preserves PDF-page provenance,
chapter/Adhyaya boundaries, sloka markers where the OCR exposes them, and
source-level metadata.  It does NOT ask an LLM to rewrite or "correct" the
source text.

Expected input directory (default: rag/source_pdfs):
  brihat hora sastra OCR(1).pdf
  Brihat Jataka 2nd Ed. by V Subrahmanya Sastri_OCR(1).pdf
  Jataka Parijata_OCR(1).pdf
  Mantreswaras-Phaladeepika_OCR(1).pdf

Outputs:
  rag/processed/corpus.jsonl
  rag/processed/pages.jsonl
  rag/processed/corpus_manifest.json
  rag/evaluation/retrieval_gold.jsonl

Run:
  python scripts/build_corpus.py
  python scripts/build_corpus.py --input-dir /path/to/pdfs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.facts import vocab

DEFAULT_INPUT_DIR = PROJECT_ROOT / "rag" / "source_pdfs"
CORPUS_OUT = PROJECT_ROOT / "rag" / "processed" / "corpus.jsonl"
PAGES_OUT = PROJECT_ROOT / "rag" / "processed" / "pages.jsonl"
MANIFEST_OUT = PROJECT_ROOT / "rag" / "processed" / "corpus_manifest.json"
GOLD_OUT = PROJECT_ROOT / "rag" / "evaluation" / "retrieval_gold.jsonl"


@dataclass(frozen=True)
class BookConfig:
    book_id: str
    title: str
    author: str
    translator: str | None
    edition: str | None
    filename: str
    chapter_prefix: str
    main_start_page: int
    max_chapter: int
    index_stop_hint: int | None = None
    sign_aliases: dict[str, tuple[str, ...]] | None = None


BOOKS = (
    BookConfig(
        book_id="bphs",
        title="Brihat Parasara Hora Sastra",
        author="Maharshi Parasara",
        translator="Girish Chand Sharma",
        edition="Vol. I; first published 1994, reprints 1995/1999",
        filename="brihat hora sastra OCR(1).pdf",
        chapter_prefix="chapter",
        main_start_page=12,
        max_chapter=47,
    ),
    BookConfig(
        book_id="brihat_jataka",
        title="Brihat Jataka",
        author="Varahamihira",
        translator="V. Subrahmanya Sastri",
        edition="Second Edition, 1956",
        filename="Brihat Jataka 2nd Ed. by V Subrahmanya Sastri_OCR(1).pdf",
        chapter_prefix="chapter",
        main_start_page=36,
        max_chapter=28,
    ),
    BookConfig(
        book_id="jataka_parijata",
        title="Jataka Parijata",
        author="Vaidyanatha Dikshita",
        translator="V. Subrahmanya Sastri",
        edition=None,
        filename="Jataka Parijata_OCR(1).pdf",
        chapter_prefix="adhyaya",
        main_start_page=29,
        max_chapter=18,
        index_stop_hint=512,
    ),
    BookConfig(
        book_id="phaladeepika",
        title="Phaladeepika",
        author="Mantreswara",
        translator="V. Subrahmanya Sastri",
        edition=None,
        filename="Mantreswaras-Phaladeepika_OCR(1).pdf",
        chapter_prefix="adhyaya",
        main_start_page=38,
        max_chapter=28,
        index_stop_hint=469,
    ),
)

PLANET_CANONICAL = vocab.PLANET_ALIASES
SIGN_ALIASES = {
    "Aries": ("aries", "mesha", "mesa"),
    "Taurus": ("taurus", "vrishabha", "vrishabh", "vrisha"),
    "Gemini": ("gemini", "mithuna", "mithun"),
    "Cancer": ("cancer", "kataka", "karka", "karkata"),
    "Leo": ("leo", "simha", "sinha"),
    "Virgo": ("virgo", "kanya"),
    "Libra": ("libra", "tula"),
    "Scorpio": ("scorpio", "vrischika", "vrishchika", "vrishika"),
    "Sagittarius": ("sagittarius", "dhanus", "dhanu"),
    "Capricorn": ("capricorn", "makara", "makar"),
    "Aquarius": ("aquarius", "kumbha", "kumbh"),
    "Pisces": ("pisces", "meena", "mina"),
}
NAK_ALIASES = {n: (n.lower(),) for n in vocab.NAKSHATRAS}

ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10, "eleventh": 11, "11th": 11, "twelfth": 12, "12th": 12,
}

HOUSE_PATTERNS = [
    re.compile(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:house|bhava|bhavam)\b", re.I),
    re.compile(r"\b(" + "|".join(sorted(map(re.escape, ORDINALS), key=len, reverse=True)) + r")\s*(?:house|bhava|bhavam)\b", re.I),
    re.compile(r"\b(?:lord|lordship|owner)\s+(?:of|of the)?\s*(\d{1,2})(?:st|nd|rd|th)?\b", re.I),
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:lord|lordship)\b", re.I),
]

HOUSE_LORD_PATTERNS = [
    re.compile(r"\b(?:lord|lordship|owner)\s+of\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s*(?:house|bhava|bhavam)?\b", re.I),
    re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:house\s+)?(?:lord|lordship)\b", re.I),
    re.compile(r"\b(" + "|".join(sorted(map(re.escape, ORDINALS), key=len, reverse=True)) + r")\s+(?:house\s+)?(?:lord|lordship)\b", re.I),
]

SLOKA_RE = re.compile(
    r"^\s*(?:Slokas?|Sioka|Sleoka|Stoka|Staka|SI\.|Si\.|Sl\.|S1\.|S1oka)\s*"
    r"(\d{1,4})(?:\s*[-–—]\s*(\d{1,4}))?",
    re.I,
)
NUMERIC_RULE_RE = re.compile(r"^\s*(\d{1,4})(?:\s*[-–]\s*(\d{1,4}))?\.\s*(.*)$")

HEADER_CHARS = re.compile(r"[A-Za-z0-9]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def int_to_roman(number: int) -> str:
    values = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
              (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
              (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    out = []
    for value, glyph in values:
        while number >= value:
            out.append(glyph)
            number -= value
    return "".join(out)


def roman_to_int(value: str, expected: range | None = None) -> int | None:
    """Parse Roman numerals and tolerate small OCR substitutions."""
    raw = re.sub(r"[^A-Za-z0-9]", "", value.strip().upper())
    if raw.isdigit():
        n = int(raw)
        if not (1 <= n <= 60):
            return None
        if expected is not None and n not in expected:
            return None
        return n

    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if raw and all(ch in vals for ch in raw):
        total = 0
        prev = 0
        for ch in reversed(raw):
            cur = vals[ch]
            total += -cur if cur < prev else cur
            prev = cur
        if 1 <= total <= 60:
            # When context is available, prefer a nearby plausible chapter
            # over a valid but wildly implausible OCR numeral (e.g. XLV for
            # an actual XIV heading).
            if expected and total not in expected:
                pass
            else:
                return total

    candidates = list(expected or range(1, 61))
    if not candidates:
        return None
    best_n, best_d = None, 999
    for n in candidates:
        d = _edit_distance(raw, int_to_roman(n))
        if d < best_d:
            best_n, best_d = n, d
    # One or two character OCR errors are common in these scanned headings.
    return best_n if best_n is not None and best_d <= 2 else None


_DEV_CHAPTER_STEMS = {
    1: ("प्रथम",),
    2: ("द्वितीय", "दितीय"),
    3: ("तृतीय", "ततीय", "सुतीय"),
    4: ("चतुर्थ", "चतुर्थो", "चतुथ", "चतुथा", "चतुद", "चवुर", "चदु"),
    5: ("पञ्चम", "पंचम", "पञ्चमो", "पश्चम"),
    6: ("षष्ठ", "षष्ट", "पष्ठ", "षट्ठ", "षष्ठो"),
    7: ("सप्तम", "सप्तमो", "सस्तमो"),
    8: ("अष्टम", "अष्टमो", "अश्रद", "अच्रम"),
    9: ("नवम", "नवमो", "नवभो"),
    10: ("दशम", "दशमो", "दरामो", "द्रामो"),
    11: ("एकादश", "एकादशो", "एकादरो", "दादशो"),
    12: ("द्वादश", "द्वादशो", "दादश", "दादशो"),
    13: ("त्रयोदश", "त्रयोदशो", "त्रयोद", "जयोदश", "भयोदश"),
    14: ("चतुर्दश", "चतुर्दशो", "चतुर्द", "चतुद", "चपरुद", "चदुद", "चतुर्दो"),
    15: ("पञ्चदश", "पञ्चदशो", "पञ्चद", "पञ्चदरो"),
    16: ("षोडश", "षोडशो", "पोडश", "पोडशो"),
    17: ("सप्तदश", "सप्तदशो", "उप्तद", "उप्तदक्ष", "सप्तृश"),
    18: ("अष्टादश", "अष्टादशो", "अष्टदश", "अश्रदश"),
    19: ("एकोनविंश", "एकोनविष", "ऊन्नविंश", "ऊनविंश", "एकरानविरा"),
    20: ("विंश", "विंशो", "विशो", "विरऽ"),
    21: ("एकविंश", "एकविज्ञ", "एकविश", "एकविज"),
    22: ("द्वाविंश", "द्वाविंशो", "दाविश", "दाविंश"),
    23: ("त्रयोविंश", "त्रयोविं", "त्रयोपिंश", "त्रयोर्धिश", "श्रथोधिश"),
    24: ("चतुर्विंश", "चतुर्विश", "चतुर्विक्ष", "चदुर्िश", "चतुर्विंक"),
    25: ("पञ्चविंश", "पञ्चविग", "पञ्चविश", "पर्विश", "पर्विंश"),
    26: ("षड्विंश", "षड्वि", "षडधि", "षट्विंश", "षड्‌विश"),
    27: ("सप्तविंश", "सप्तवश", "सप्र्षिं"),
    28: ("अष्टाविंश", "अष्टार्वि", "अष्टाविरो", "अष्टर्विश"),
}



def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


MOJIBAKE_MARKERS = ("à¤", "à¥", "Ã", "Â", "â€")

def _repair_mojibake(text: str) -> str:
    """Best-effort repair for UTF-8 bytes mis-decoded as Latin-1/CP1252.

    The supplied PDFs currently extract clean Devanagari through PyMuPDF, but
    this guard makes the corpus builder resilient to a viewer/extractor that
    hands it common UTF-8 mojibake. The raw OCR is still preserved separately.
    """
    if not text or not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    candidates = [text]
    for enc in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(enc).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    def quality(candidate: str) -> tuple[float, int]:
        dev = devanagari_ratio(candidate)
        moj = sum(candidate.count(m) for m in MOJIBAKE_MARKERS)
        return dev, -moj
    return max(candidates, key=quality)


def norm_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    line = line.replace("\u00a0", " ").replace("\u00ad", "")
    line = _repair_mojibake(line)
    line = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", line)
    line = re.sub(r"\s+", " ", line.strip())
    line = line.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return line


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(bool(DEVANAGARI.match(c)) for c in chars) / len(chars)


def detect_chapter(lines: list[str], cfg: BookConfig, current_chapter: int | None = None) -> tuple[int, int, str] | None:
    """Find a chapter transition while resisting repeated running headers."""
    max_lines = min(len(lines), 30)
    prefix = "chapter" if cfg.chapter_prefix == "chapter" else "adhyaya"
    expected = range(current_chapter + 1, current_chapter + 6) if current_chapter is not None else range(1, 61)
    explicit_marker: tuple[int, str, str] | None = None

    def dev_heading(text: str) -> bool:
        return bool(re.search(r"(?:अध्याय|ऽध्याय|ध्याय|भ्याय)[^\n]{0,10}(?:ः|$)", text))

    def joined(i: int) -> str:
        nxt = lines[i + 1] if i + 1 < len(lines) and i + 1 < 20 else ""
        return f"{lines[i]} {nxt}".strip()

    def is_chapter_prefix(line: str) -> re.Match[str] | None:
        if prefix == "chapter":
            return re.fullmatch(r"\s*[‘'\"“”()\[\]{}-–—]*\s*(?:chapter)\s*[-–—]?\s*([^\s.]+)\.?\s*", line, re.I)
        return re.fullmatch(r"\s*[‘'\"“”()\[\]{}-–—]*\s*(?:adhyaya|adbyaya)\s*(?:[-–—]?\s*([^\s.]+)\.?\s*)?", line, re.I)

    # 1) Full English heading. Short forms such as "Adh. XII" are running
    # headers and intentionally do not count as a chapter transition.
    for i in range(max_lines):
        line = lines[i]
        m = is_chapter_prefix(line)
        if not m:
            continue
        token = m.group(1) or ""
        explicit_marker = (i, line, token)
        if token:
            raw = re.sub(r"[^A-Za-z0-9]", "", token.upper())
            # Only canonical Roman numerals or ordinary numeric chapter labels
            # are trusted without contextual evidence.
            direct = roman_to_int(raw)
            if direct is not None and 1 <= direct <= cfg.max_chapter:
                if current_chapter is None:
                    return direct, i, line
                if direct == current_chapter:
                    # Repeated running chapter title on the next page. It is
                    # not a new chapter; do not fall through to the malformed
                    # heading recovery logic.
                    return None
                if direct in expected and (raw.isdigit() or int_to_roman(direct) == raw):
                    return direct, i, line
        break

    # 2) Devanagari chapter-title headings. Recognizable ordinals are strong
    # evidence when their number is different from the current running header.
    dev_candidates: list[tuple[int | None, int, str]] = []
    for i in range(min(20, len(lines))):
        text = joined(i)
        if not dev_heading(text):
            continue
        hinted = None
        for n, stems in _DEV_CHAPTER_STEMS.items():
            if any(stem in text for stem in stems):
                hinted = n
                break
        dev_candidates.append((hinted, i, text))
        if hinted is not None:
            if current_chapter is None:
                return hinted, i, lines[i]
            if current_chapter < hinted <= min(current_chapter + 5, cfg.max_chapter):
                return hinted, i, lines[i]

    # 3) A malformed English heading plus a Devanagari title is a reliable
    # chapter transition in the scanned editions (e.g. CHAPTER XiIil).
    if explicit_marker is not None and current_chapter is not None:
        if current_chapter < cfg.max_chapter and any(dev_heading(x) for _, i, x in dev_candidates):
            return current_chapter + 1, explicit_marker[0], explicit_marker[1]

    # 4) A malformed English numeral with no Devanagari ordinal can still be
    # recovered when it is within two edits of the next plausible Roman label
    # (e.g. "Adhyaya XIL" -> XII).
    if explicit_marker is not None and current_chapter is not None:
        token = explicit_marker[2]
        raw = re.sub(r"[^A-Za-z0-9]", "", token.upper())
        if raw:
            scored = sorted((_edit_distance(raw, int_to_roman(n)), n) for n in expected if n <= cfg.max_chapter)
            if scored and scored[0][0] <= 2 and scored[0][1] != current_chapter:
                return scored[0][1], explicit_marker[0], explicit_marker[1]

    # 5) Some chapter openings have no usable English heading at all, but the
    # previous chapter is explicitly closed near the top of the page. Use the
    # next Devanagari chapter title in that special case.
    if current_chapter is not None and current_chapter < cfg.max_chapter and any(re.search(r"thus ends the .*adhyaya|ends the .*adhyaya", x, re.I) for x in lines[:24]):
        for hinted, i, text in dev_candidates:
            if hinted == current_chapter + 1:
                return hinted, i, lines[i]
        for hinted, i, text in dev_candidates:
            if hinted is None:
                return current_chapter + 1, i, lines[i]

    return None


def detect_index_page(lines: list[str]) -> bool:
    top = [x.lower().strip().replace(".", "") for x in lines[:18]]
    return any(x == "index" or x.startswith("index ") for x in top)


def repeated_line_candidates(page_lines: list[list[str]]) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in page_lines:
        seen = set()
        for line in lines:
            n = norm_line(line)
            if not n or len(n) > 100:
                continue
            seen.add(n.lower())
        counts.update(seen)
    threshold = max(8, int(len(page_lines) * 0.03))
    candidates = set()
    for line, count in counts.items():
        if count < threshold:
            continue
        # Running headers/footers are normally short and contain the book or
        # chapter header, not substantive prose.
        if len(line) <= 70:
            candidates.add(line)
    return candidates


def clean_page(lines: list[str], repeated: set[str]) -> list[str]:
    out: list[str] = []
    for idx, raw in enumerate(lines):
        line = norm_line(raw)
        if not line:
            continue
        low = line.lower()
        if low in repeated and len(line) <= 100:
            continue
        # Page numbers sitting at the very top/bottom are layout noise. Do not
        # strip numeric lines inside the body.
        if idx < 2 and re.fullmatch(r"\d{1,4}", line):
            continue
        if idx >= max(0, len(lines) - 2) and re.fullmatch(r"\d{1,4}", line):
            continue
        out.append(line)
    return out


def guess_printed_page(lines: list[str]) -> int | None:
    for line in (lines[:5] + lines[-5:]):
        if re.fullmatch(r"\d{1,4}", line.strip()):
            n = int(line)
            if 1 <= n <= 2000:
                return n
        m = re.search(r"(?:^|\s)(\d{1,4})\s*$", line)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 2000:
                return n
    return None


def chapter_title(lines: list[str], heading_idx: int) -> str:
    parts = []
    for line in lines[heading_idx + 1 : heading_idx + 7]:
        if not line:
            continue
        if SLOKA_RE.match(line) or DEVANAGARI.search(line):
            break
        # Avoid swallowing a paragraph of prose as the title.
        if len(parts) >= 3:
            break
        parts.append(line.strip(" :.-"))
    return " ".join(parts).strip()


def line_is_numeric_rule(line: str) -> bool:
    m = NUMERIC_RULE_RE.match(line)
    if not m:
        return False
    rest = m.group(3).strip()
    if not rest:
        return True
    # A real rule marker usually begins an English translation. This filter
    # prevents most bare printed-page numbers from becoming chunks.
    return bool(re.search(r"\b(?:If|When|The|A|An|There|Should|Now|In|For|He|She|Where|During|From|According|Whatever|Any)\b", rest, re.I))


def detect_markers(lines: list[str], use_numeric_fallback: bool) -> list[tuple[int, int | None, int | None, str]]:
    markers = []
    for i, line in enumerate(lines):
        m = SLOKA_RE.match(line)
        if m:
            markers.append((i, int(m.group(1)), int(m.group(2)) if m.group(2) else None, "sloka"))
    if markers or not use_numeric_fallback:
        return markers
    for i, line in enumerate(lines):
        if line_is_numeric_rule(line):
            m = NUMERIC_RULE_RE.match(line)
            if not m:
                continue
            markers.append((i, int(m.group(1)), int(m.group(2)) if m.group(2) else None, "numbered_rule"))
    return markers


def collect_preceding_verse(lines: list[str], marker_idx: int) -> list[str]:
    picked: list[str] = []
    empty_run = 0
    for i in range(marker_idx - 1, max(-1, marker_idx - 16), -1):
        line = lines[i].strip()
        if DEVANAGARI.search(line) and devanagari_ratio(line) >= 0.20:
            picked.append(line)
            empty_run = 0
            if len(picked) >= 10:
                break
        else:
            empty_run += 1
            if picked and empty_run >= 2:
                break
    return list(reversed(picked))



def _house_mentions(text: str) -> list[tuple[int, int, int]]:
    """Return (house_no, start_offset, end_offset) mentions in text."""
    out: list[tuple[int, int, int]] = []
    low = text.lower()
    for pat in HOUSE_PATTERNS[:2]:
        for m in pat.finditer(low):
            token = m.group(1)
            n = int(token) if token.isdigit() else ORDINALS.get(token.lower())
            if n and 1 <= n <= 12:
                out.append((n, m.start(), m.end()))
    for phrase, n in {
        "lagna": 1, "tanu bhava": 1, "dhanabhava": 2, "dhana bhava": 2,
        "sahaja bhava": 3, "bandhu bhava": 4, "putra bhava": 5,
        "ari bhava": 6, "ripu bhava": 6, "jaya bhava": 7, "randhra bhava": 8,
        "bhagya bhava": 9, "karma bhava": 10, "labha bhava": 11, "vyaya bhava": 12,
    }.items():
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", low):
            out.append((n, m.start(), m.end()))
    out.sort(key=lambda mention: mention[1])
    return out


def _direct_relations(text: str) -> dict[str, list[str]]:
    """Extract explicit astrological relationship pairs from source text.

    Entity co-occurrence is deliberately not enough. A chunk may mention Mars
    while actually describing the 5th Lord in the 10th. The precision index
    therefore requires placement/aspect language linking the entities.
    """
    low = text.lower()
    out: dict[str, set[str]] = {
        "planet_house_pairs": set(),
        "planet_house_example_pairs": set(),
        "planet_sign_pairs": set(),
        "planet_nakshatra_pairs": set(),
        "planet_aspect_pairs": set(),
        "lord_house_pairs": set(),
    }

    collapsed = re.sub(r"\s+", " ", low).strip()
    segments = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", collapsed) if seg.strip()]
    planet_aliases = dict(PLANET_CANONICAL)
    sign_aliases = {alias: canonical for canonical, aliases in SIGN_ALIASES.items() for alias in aliases}

    placement_signal = r"(?:is|are|was|were|be|being|been)?\s*(?:occup\w*|situat\w*|plac\w*|posit\w*|remain\w*|in|at|on)"

    def is_role_specific(seg: str, alias: str) -> bool:
        """True when the planet is explicitly introduced as a house lord."""
        patterns = (
            rf"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d{{1,2}}(?:st|nd|rd|th)?)\s+(?:house\s+)?lord\s+{re.escape(alias)}\b",
            rf"\b{re.escape(alias)}\b\s*,?\s*(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d{{1,2}}(?:st|nd|rd|th)?)\s+(?:house\s+)?lord\b",
        )
        return any(re.search(pattern, seg) for pattern in patterns)

    def is_example_context(seg: str) -> bool:
        return bool(re.search(r"\b(?:example|given horoscope|second horoscope|following horoscope|horoscope is|in the given|here the)\b", seg))

    for seg in segments:
        example_context = is_example_context(seg)

        # Planet -> house. The placement verb must occur between the planet and
        # the house mention, which avoids plain co-occurrence.
        for house, hstart, hend in _house_mentions(seg):
            house_text_start = max(0, hend - 1)
            house_window_start = max(0, hend - 100)
            house_window = seg[house_window_start:min(len(seg), hend + 15)]
            for alias, canonical in planet_aliases.items():
                for pm in re.finditer(rf"\b{re.escape(alias)}\b", house_window):
                    local_after = house_window[pm.end():]
                    if not re.search(rf"{placement_signal}\s+(?:the\s+)?(?:\d{{1,2}}(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+house\b", local_after):
                        # Also permit direct "Mars in the 10th house" when the
                        # number was matched by a named-house detector.
                        if not re.search(rf"{placement_signal}\s+(?:the\s+)?(?:house|bhava)\b", local_after):
                            continue
                    if is_role_specific(seg, alias) or example_context:
                        out["planet_house_example_pairs"].add(f"{canonical}|{house}")
                    else:
                        out["planet_house_pairs"].add(f"{canonical}|{house}")

            # Reverse construction: "the 10th house is occupied by Mars".
            reverse = seg[hend:min(len(seg), hend + 70)]
            if re.search(r"\b(?:is|are|was|were)?\s*(?:occup\w*|contain\w*|has|have|filled)\s+(?:by|with)?\s*$", reverse):
                pass  # handled by the explicit planet scan below when possible
            for alias, canonical in planet_aliases.items():
                if re.search(rf"\b(?:occup\w*|contain\w*|has|with)\b[^.?!]{{0,45}}\b{re.escape(alias)}\b", reverse):
                    if is_role_specific(seg, alias) or example_context:
                        out["planet_house_example_pairs"].add(f"{canonical}|{house}")
                    else:
                        out["planet_house_pairs"].add(f"{canonical}|{house}")

        # Planet -> sign. Require the placement language to sit between the
        # planet and sign; this prevents cross-product errors such as Mars|Aries
        # when the source actually says "Sun in Aries ... Mars in Leo".
        for sign_alias, sign in sign_aliases.items():
            for sm in re.finditer(rf"\b{re.escape(sign_alias)}\b", seg):
                before = seg[max(0, sm.start() - 75):sm.start()]
                for alias, canonical in planet_aliases.items():
                    if not re.search(rf"\b{re.escape(alias)}\b{placement_signal}[^.?!]{{0,50}}$", before):
                        continue
                    if is_role_specific(seg, alias) or example_context:
                        continue
                    out["planet_sign_pairs"].add(f"{canonical}|{sign}")

        # Planet-nakshatra relation.
        for nak in vocab.NAKSHATRAS:
            nl = nak.lower()
            for nm in re.finditer(rf"\b{re.escape(nl)}\b", seg):
                before = seg[max(0, nm.start() - 90):nm.start()]
                if "nakshatra" not in before and "asterism" not in before:
                    continue
                for alias, canonical in planet_aliases.items():
                    if re.search(rf"\b{re.escape(alias)}\b[^.?!]{{0,70}}(?:nakshatra|asterism)?[^.?!]{{0,20}}$", before):
                        out["planet_nakshatra_pairs"].add(f"{canonical}|{nak}")

        # Explicit aspect wording, e.g. "Mars's 4th aspect" or "4th aspect of Mars".
        for n in (3, 4, 5, 7, 8, 9, 10):
            ordinal = {3:"3rd",4:"4th",5:"5th",7:"7th",8:"8th",9:"9th",10:"10th"}[n]
            if re.search(rf"\b{re.escape(ordinal)}\s+(?:house\s+)?aspect\b", seg):
                for alias, canonical in planet_aliases.items():
                    if re.search(rf"\b{re.escape(alias)}\b", seg):
                        out["planet_aspect_pairs"].add(f"{canonical}|{ordinal}")

        # House-lord placement stays separate from planet-in-house.
        lord_matches = list(re.finditer(
            r"\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d{1,2}(?:st|nd|rd|th)?)\s+(?:house\s+)?lord\b",
            seg,
        ))
        if lord_matches:
            houses = _house_mentions(seg)
            for lm in lord_matches:
                token = lm.group(1)
                lord_no = int(re.match(r"\d+", token).group()) if token[0].isdigit() else ORDINALS.get(token)
                if not lord_no:
                    continue
                for dest, hstart, hend in houses:
                    if hstart < lm.end() or abs(lm.start() - hend) > 140:
                        continue
                    between = seg[lm.end():hstart]
                    if re.search(r"\b(?:this|that)\s+(?:situation|case)\b", between):
                        continue
                    if re.search(
                        r"\b(?:first|second|third|fourth|fifth|sixth|seventh|"
                        r"eighth|ninth|tenth|eleventh|twelfth|\d{1,2}(?:st|nd|rd|th)?)"
                        r"\s+(?:house\s+)?lord\b",
                        between,
                    ):
                        continue
                    if re.search(
                        r"\b(?:in|occup\w*|situat\w*|plac\w*|posit\w*)\b",
                        between,
                    ):
                        out["lord_house_pairs"].add(f"{lord_no}|{dest}")
                        break

    return {key: sorted(values) for key, values in out.items()}


def _house_numbers_in_text(text: str) -> set[int]:
    numbers = set()
    low = text.lower()
    for pat in HOUSE_PATTERNS[:2]:
        for m in pat.finditer(low):
            token = m.group(1)
            n = int(token) if token.isdigit() else ORDINALS.get(token.lower())
            if n:
                numbers.add(n)
    return numbers


def extract_entities(text: str) -> dict[str, Any]:
    low = text.lower()
    planets: list[str] = []
    for alias, canonical in PLANET_CANONICAL.items():
        if re.search(rf"\b{re.escape(alias)}\b", low) and canonical not in planets:
            planets.append(canonical)

    signs: list[str] = []
    for canonical, aliases in SIGN_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", low) for alias in aliases):
            signs.append(canonical)

    nakshatras: list[str] = []
    for canonical, aliases in NAK_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", low) for alias in aliases):
            nakshatras.append(canonical)

    houses: set[int] = set()
    for pat in HOUSE_PATTERNS:
        for m in pat.finditer(low):
            token = m.group(1)
            if token.isdigit():
                n = int(token)
            else:
                n = ORDINALS.get(token.lower())
            if n and 1 <= n <= 12:
                houses.add(n)

    # Special house terminology frequently occurring in these books.
    named_house = {
        "lagna": 1, "tanu bhava": 1, "dhanabhava": 2, "dhana bhava": 2,
        "sahaja bhava": 3, "bandhu bhava": 4, "putra bhava": 5,
        "ari bhava": 6, "ripu bhava": 6, "jaya bhava": 7, "randhra bhava": 8,
        "bhagya bhava": 9, "karma bhava": 10, "labha bhava": 11, "vyaya bhava": 12,
    }
    for phrase, n in named_house.items():
        if phrase in low:
            houses.add(n)

    lord_of_houses: set[int] = set()
    for pat in HOUSE_LORD_PATTERNS:
        for m in pat.finditer(low):
            token = m.group(1)
            n = int(token) if token.isdigit() else ORDINALS.get(token.lower())
            if n and 1 <= n <= 12:
                lord_of_houses.add(n)

    dasha_lords: list[str] = []
    if re.search(r"\b(?:dasa|dasha|mahadasha|antar(?:dasa|dasha)|bhukti)\b", low):
        dasha_lords = planets[:]

    aspect_types: list[str] = []
    if re.search(r"\b(?:aspect|drishti|dristi|aspects)\b", low):
        for n, word in ((3,"3rd"),(4,"4th"),(5,"5th"),(7,"7th"),(8,"8th"),(9,"9th"),(10,"10th")):
            if re.search(rf"\b{re.escape(word)}\s+(?:house\s+)?aspect\b", low) or re.search(rf"\b{n}(?:st|nd|rd|th)?\s+(?:house\s+)?drishti\b", low):
                aspect_types.append(word)
        if not aspect_types:
            aspect_types = ["7th"]

    topics = vocab.find_topics(low)
    relations = []
    rel_map = {
        "conjunction": ("conjunction", "conjunct", "conjoint", "associated with", "association"),
        "aspect": ("aspect", "drishti", "dristi"),
        "exaltation": ("exaltation", "exalted", "uchcha"),
        "debilitation": ("debilitation", "debilitated", "neecha", "depression sign"),
        "own_sign": ("own sign", "own house", "swakshetra"),
        "moolatrikona": ("moolatrikona", "mulatrikona"),
        "combustion": ("combust", "combustion"),
        "retrogression": ("retrograde", "retrogression"),
        "dasha_rule": ("dasa", "dasha", "mahadasha", "antardasha", "bhukti"),
        "transit_rule": ("transit", "gochara", "gochar"),
        "yoga_rule": ("yoga", "yogas"),
        "nakshatra_rule": ("nakshatra", "asterism"),
        "varga_rule": ("varga", "navamsa", "dasamsa", "divisional"),
        "ashtakavarga_rule": ("ashtakavarga", "ashtaka varga"),
        "karaka_rule": ("karaka", "karakatwa"),
    }
    for name, keys in rel_map.items():
        if any(k in low for k in keys):
            relations.append(name)

    relations_direct = _direct_relations(text)
    return {
        "planets": sorted(planets, key=planets.index),
        "signs": sorted(signs, key=signs.index),
        "houses": sorted(houses),
        "lord_of_houses": sorted(lord_of_houses),
        "nakshatras": sorted(nakshatras, key=nakshatras.index),
        "dasha_lords": sorted(dasha_lords, key=dasha_lords.index),
        "aspect_types": aspect_types,
        "topics": topics,
        "relations": relations,
        **relations_direct,
    }


def scalar_or_none(values: list[Any]) -> Any:
    return values[0] if len(values) == 1 else None


def approx_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def make_retrieval_text(meta: dict[str, Any], text: str) -> str:
    src = meta["source"]
    parts = [
        f"Source: {src['book']}",
        f"Author: {src.get('author') or 'unknown'}",
        f"Chapter: {meta.get('chapter_no') or '?'}{(' - ' + meta['chapter_title']) if meta.get('chapter_title') else ''}",
    ]
    if meta.get("sloka_start") is not None:
        end = meta.get("sloka_end") or meta["sloka_start"]
        parts.append(f"Sloka: {meta['sloka_start'] if end == meta['sloka_start'] else f'{meta["sloka_start"]}-{end}'}")
    if meta.get("topics"):
        parts.append("Topics: " + ", ".join(meta["topics"]))
    if meta.get("planets"):
        parts.append("Planets: " + ", ".join(meta["planets"]))
    if meta.get("houses"):
        parts.append("Houses: " + ", ".join(map(str, meta["houses"])))
    if meta.get("signs"):
        parts.append("Signs: " + ", ".join(meta["signs"]))
    if meta.get("planet_house_pairs"):
        parts.append("Direct planet-house relations: " + ", ".join(meta["planet_house_pairs"]))
    if meta.get("planet_sign_pairs"):
        parts.append("Direct planet-sign relations: " + ", ".join(meta["planet_sign_pairs"]))
    if meta.get("lord_house_pairs"):
        parts.append("Direct lord-house relations: " + ", ".join(meta["lord_house_pairs"]))
    return " | ".join(parts) + "\n" + text


def source_meta(cfg: BookConfig, source_sha: str, pdf_page_start: int, pdf_page_end: int, printed_start: int | None, printed_end: int | None, chapter_no: int | None, chapter_title_: str | None, sloka_start: int | None, sloka_end: int | None) -> dict[str, Any]:
    return {
        "book": cfg.title,
        "book_id": cfg.book_id,
        "author": cfg.author,
        "translator": cfg.translator,
        "edition": cfg.edition,
        "pdf_sha256": source_sha,
        "pdf_page_start": pdf_page_start,
        "pdf_page_end": pdf_page_end,
        "printed_page_start": printed_start,
        "printed_page_end": printed_end,
        "chapter": chapter_no,
        "chapter_title": chapter_title_,
        "sloka": sloka_start if sloka_end in (None, sloka_start) else f"{sloka_start}-{sloka_end}",
        "type": "user_provided_ocr_pdf",
    }


def build_book(cfg: BookConfig, pdf_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_sha = sha256(pdf_path)
    doc = fitz.open(pdf_path)
    raw_pages = [p.get_text("text").splitlines() for p in doc]
    stop_page = cfg.index_stop_hint or len(doc) + 1
    for idx, lines in enumerate(raw_pages[cfg.main_start_page - 1 :], start=cfg.main_start_page):
        normalized = [norm_line(x) for x in lines if norm_line(x)]
        if detect_index_page(normalized):
            stop_page = min(stop_page, idx)
            break
    raw_main = raw_pages[cfg.main_start_page - 1 : stop_page - 1]
    repeated = repeated_line_candidates(raw_main)

    pages: list[dict[str, Any]] = []
    current_chapter: int | None = None
    current_title: str | None = None
    chapter_pages: dict[int, list[dict[str, Any]]] = defaultdict(list)
    chapter_first_page: dict[int, int] = {}
    chapter_last_page: dict[int, int] = {}

    for rel_i, raw_lines in enumerate(raw_main):
        pdf_page = cfg.main_start_page + rel_i
        lines = [norm_line(x) for x in raw_lines if norm_line(x)]
        ch = detect_chapter(lines, cfg, current_chapter)
        if ch and ch[0] <= cfg.max_chapter:
            current_chapter, heading_idx, _heading_text = ch
            current_title = chapter_title(lines, heading_idx) or None
            chapter_first_page.setdefault(current_chapter, pdf_page)
        if current_chapter is None:
            continue
        cleaned = clean_page(lines, repeated)
        printed = guess_printed_page(cleaned)
        page_rec = {
            "book_id": cfg.book_id,
            "pdf_page": pdf_page,
            "printed_page": printed,
            "chapter_no": current_chapter,
            "chapter_title": current_title,
            "text_raw": "\n".join(lines),
            "text_clean": "\n".join(cleaned),
            "source_file_sha256": source_sha,
        }
        pages.append(page_rec)
        chapter_pages[current_chapter].append(page_rec)
        if current_chapter > cfg.max_chapter:
            continue
        chapter_last_page[current_chapter] = pdf_page

    chunks: list[dict[str, Any]] = []
    sibling_by_chapter: dict[tuple[str, int], list[str]] = defaultdict(list)
    id_counts: Counter[str] = Counter()

    for ch_no, ch_pages in sorted(chapter_pages.items()):
        ch_title = next((p["chapter_title"] for p in ch_pages if p.get("chapter_title")), None)
        overview_text = "\n".join(p["text_clean"] for p in ch_pages[:3]).strip()
        if overview_text:
            overview_text = overview_text[:4500]
            meta = {
                "tradition": "vedic",
                "book_id": cfg.book_id,
                "book_title": cfg.title,
                "chapter_no": ch_no,
                "chapter_title": ch_title,
                "chunk_type": "section_overview",
                "layer": "section_overview",
                "parent_id": None,
                "prev_id": None,
                "next_id": None,
                "planets": [], "signs": [], "houses": [], "nakshatras": [], "dasha_lords": [],
                "aspect_types": [], "topics": [], "relations": [],
                "planet_house_pairs": [], "planet_house_example_pairs": [], "planet_sign_pairs": [], "planet_nakshatra_pairs": [],
                "planet_aspect_pairs": [], "lord_house_pairs": [],
                "planet": None, "target_planet": None, "house": None, "sign": None, "nakshatra": None,
                "aspect_type": None, "dasha_lord": None,
                "source": source_meta(cfg, source_sha, ch_pages[0]["pdf_page"], ch_pages[min(2, len(ch_pages)-1)]["pdf_page"], ch_pages[0]["printed_page"], ch_pages[min(2, len(ch_pages)-1)]["printed_page"], ch_no, ch_title, None, None),
            }
            ents = extract_entities(overview_text)
            meta.update(ents)
            meta["planet"] = scalar_or_none(ents["planets"])
            meta["house"] = scalar_or_none(ents["houses"])
            meta["sign"] = scalar_or_none(ents["signs"])
            meta["nakshatra"] = scalar_or_none(ents["nakshatras"])
            meta["dasha_lord"] = scalar_or_none(ents["dasha_lords"])
            chunk_id = f"{cfg.book_id}_ch{ch_no:02d}_overview"
            chunks.append({
                "chunk_id": chunk_id,
                "text": overview_text,
                "retrieval_text": make_retrieval_text(meta, overview_text),
                "metadata": meta,
                "token_count": approx_tokens(overview_text),
            })

        # Flatten the chapter once; this makes slicing each sloka O(1) in the
        # number of pages rather than rescanning all chapter pages for every rule.
        flat: list[dict[str, Any]] = []
        for p in ch_pages:
            lines = p["text_clean"].splitlines()
            for li, line in enumerate(lines):
                flat.append({"page": p, "line_idx": li, "text": line})

        markers: list[dict[str, Any]] = []
        for flat_idx, row in enumerate(flat):
            m = SLOKA_RE.match(row["text"])
            if m:
                markers.append({"flat_idx": flat_idx, "start": int(m.group(1)), "end": int(m.group(2)) if m.group(2) else None, "kind": "sloka"})

        if not markers and cfg.book_id == "bphs":
            for flat_idx, row in enumerate(flat):
                if line_is_numeric_rule(row["text"]):
                    m = NUMERIC_RULE_RE.match(row["text"])
                    if m:
                        markers.append({"flat_idx": flat_idx, "start": int(m.group(1)), "end": int(m.group(2)) if m.group(2) else None, "kind": "numbered_rule"})

        # If the chapter has neither kind of explicit rule marker, use bounded
        # page chunks as a provenance-preserving fallback.
        if not markers:
            for p in ch_pages:
                text = p["text_clean"].strip()
                if not text:
                    continue
                ents = extract_entities(text)
                meta = {
                    "tradition": "vedic", "book_id": cfg.book_id, "book_title": cfg.title,
                    "chapter_no": ch_no, "chapter_title": ch_title, "chunk_type": "prose",
                    "layer": "primary_text", "parent_id": f"{cfg.book_id}_ch{ch_no:02d}_overview",
                    "prev_id": None, "next_id": None, **ents,
                    "planet": scalar_or_none(ents["planets"]), "target_planet": None,
                    "house": scalar_or_none(ents["houses"]), "sign": scalar_or_none(ents["signs"]),
                    "nakshatra": scalar_or_none(ents["nakshatras"]), "aspect_type": scalar_or_none(ents["aspect_types"]),
                    "dasha_lord": scalar_or_none(ents["dasha_lords"]),
                    "source": source_meta(cfg, source_sha, p["pdf_page"], p["pdf_page"], p["printed_page"], p["printed_page"], ch_no, ch_title, None, None),
                }
                cid = f"{cfg.book_id}_ch{ch_no:02d}_p{p['pdf_page']:04d}"
                chunks.append({"chunk_id": cid, "text": text, "retrieval_text": make_retrieval_text(meta, text), "metadata": meta, "token_count": approx_tokens(text)})
                sibling_by_chapter[(cfg.book_id, ch_no)].append(cid)
            continue

        for marker_idx, marker in enumerate(markers):
            start_flat = marker["flat_idx"]
            end_flat = markers[marker_idx + 1]["flat_idx"] if marker_idx + 1 < len(markers) else len(flat)
            seg_lines = [r["text"] for r in flat[start_flat:end_flat]]
            segment = "\n".join(seg_lines).strip()

            # Preserve the Devanagari verse that often appears immediately
            # before the English "Sloka N" marker in these OCR layouts.
            preceding: list[str] = []
            empty_run = 0
            for pos in range(start_flat - 1, max(-1, start_flat - 18), -1):
                line = flat[pos]["text"].strip()
                if DEVANAGARI.search(line) and devanagari_ratio(line) >= 0.20:
                    preceding.append(line)
                    empty_run = 0
                    if len(preceding) >= 10:
                        break
                else:
                    empty_run += 1
                    if preceding and empty_run >= 2:
                        break
            if preceding:
                segment = "\n".join(reversed(preceding)) + "\n" + segment
            if not segment:
                continue

            start_page = flat[start_flat]["page"]["pdf_page"]
            end_page = flat[end_flat - 1]["page"]["pdf_page"] if end_flat > start_flat else start_page
            printed_start = flat[start_flat]["page"]["printed_page"]
            printed_end = flat[end_flat - 1]["page"]["printed_page"] if end_flat > start_flat else printed_start

            ents = extract_entities(segment)
            meta = {
                "tradition": "vedic", "book_id": cfg.book_id, "book_title": cfg.title,
                "chapter_no": ch_no, "chapter_title": ch_title,
                "sloka_start": marker["start"], "sloka_end": marker["end"],
                "chunk_type": "rule" if marker["kind"] == "sloka" else "numbered_rule",
                "layer": "primary_text", "parent_id": f"{cfg.book_id}_ch{ch_no:02d}_overview",
                "prev_id": None, "next_id": None, **ents,
                "planet": scalar_or_none(ents["planets"]), "target_planet": None,
                "house": scalar_or_none(ents["houses"]), "sign": scalar_or_none(ents["signs"]),
                "nakshatra": scalar_or_none(ents["nakshatras"]), "aspect_type": scalar_or_none(ents["aspect_types"]),
                "dasha_lord": scalar_or_none(ents["dasha_lords"]),
                "source": source_meta(cfg, source_sha, start_page, end_page, printed_start, printed_end, ch_no, ch_title, marker["start"], marker["end"]),
            }
            cid = f"{cfg.book_id}_ch{ch_no:02d}_sl{marker['start']:03d}"
            if marker["end"] is not None and marker["end"] != marker["start"]:
                cid += f"_{marker['end']:03d}"
            id_counts[cid] += 1
            if id_counts[cid] > 1:
                cid = f"{cid}_{id_counts[cid]}"
            sibling_by_chapter[(cfg.book_id, ch_no)].append(cid)

            words = segment.split()
            if len(words) <= 950:
                chunks.append({"chunk_id": cid, "text": segment, "retrieval_text": make_retrieval_text(meta, segment), "metadata": meta, "token_count": len(words)})
            else:
                core_words = words[:850]
                core = " ".join(core_words)
                chunks.append({"chunk_id": cid, "text": core, "retrieval_text": make_retrieval_text(meta, core), "metadata": meta, "token_count": len(core_words)})
                for part_idx, pos in enumerate(range(850, len(words), 650), start=1):
                    tail = " ".join(words[pos:pos + 650]).strip()
                    if not tail:
                        continue
                    child_meta = dict(meta)
                    child_meta.update({"chunk_type": "commentary", "layer": "commentary", "parent_id": cid})
                    child_id = f"{cid}_note_{part_idx:02d}"
                    chunks.append({"chunk_id": child_id, "text": tail, "retrieval_text": make_retrieval_text(child_meta, tail), "metadata": child_meta, "token_count": approx_tokens(tail)})

    # Link siblings after every chapter is materialized.
    by_id = {c["chunk_id"]: c for c in chunks}
    for ids in sibling_by_chapter.values():
        ids = [cid for cid in ids if cid in by_id]
        for prev_id, next_id in zip(ids, ids[1:]):
            by_id[prev_id]["metadata"]["next_id"] = next_id
            by_id[next_id]["metadata"]["prev_id"] = prev_id

    manifest = {
        "book_id": cfg.book_id,
        "title": cfg.title,
        "filename": cfg.filename,
        "sha256": source_sha,
        "pdf_pages": len(doc),
        "main_start_pdf_page": cfg.main_start_page,
        "main_end_pdf_page_exclusive": stop_page,
        "detected_chapters": [
            {"chapter_no": n, "title": next((p["chapter_title"] for p in chapter_pages[n] if p.get("chapter_title")), None),
             "pdf_page_start": chapter_first_page[n], "pdf_page_end": chapter_last_page[n]}
            for n in sorted(chapter_pages)
        ],
        "notes": [],
    }
    if cfg.book_id == "jataka_parijata":
        detected = sorted(chapter_pages)
        missing = [n for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 19) if n not in detected]
        if missing:
            manifest["notes"].append(f"Uploaded OCR has no detected main-text headings for Adhyaya {', '.join(map(str, missing))}; coverage should be treated as partial, not complete.")
    return chunks, pages, manifest

def build_gold_set(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate a small corpus-specific sanity set, not a scholarly gold set."""
    by_key: dict[tuple[str, str, Any, Any], list[str]] = defaultdict(list)
    for c in corpus:
        m = c["metadata"]
        if m.get("chunk_type") not in {"rule", "numbered_rule", "commentary"}:
            continue
        for pair in m.get("planet_house_pairs", []):
            try:
                planet, house = pair.split("|", 1)
                by_key[("planet_house", planet, int(house), None)].append(c["chunk_id"])
            except (ValueError, TypeError):
                pass
        for pair in m.get("planet_sign_pairs", []):
            try:
                planet, sign = pair.split("|", 1)
                by_key[("planet_sign", planet, sign, None)].append(c["chunk_id"])
            except (ValueError, TypeError):
                pass
        if m.get("dasha_lord") is not None:
            by_key[("dasha", m["dasha_lord"], None, None)].append(c["chunk_id"])
        if m.get("planet_nakshatra_pairs"):
            for pair in m["planet_nakshatra_pairs"]:
                try:
                    planet, nak = pair.split("|", 1)
                    by_key[("nakshatra", nak, None, None)].append(c["chunk_id"])
                except (ValueError, TypeError):
                    pass
        elif m.get("nakshatra") is not None:
            by_key[("nakshatra", m["nakshatra"], None, None)].append(c["chunk_id"])
        for pair in m.get("planet_aspect_pairs", []):
            try:
                planet, aspect = pair.split("|", 1)
                by_key[("aspect", planet, aspect, None)].append(c["chunk_id"])
            except (ValueError, TypeError):
                pass

    queries = []
    case_no = 1
    templates = {
        "planet_house": lambda a,b: f"What is the effect of {a} in the {b}th house?",
        "planet_sign": lambda a,b: f"What does {a} in {b} indicate?",
        "dasha": lambda a,b: f"What does {a} Mahadasha indicate?",
        "nakshatra": lambda a,b: f"What are the traditional indications of {a} nakshatra?",
        "aspect": lambda a,b: f"What is the {b} aspect of {a}?",
    }
    for (kind, a, b, _), ids in sorted(by_key.items(), key=lambda x: str(x[0])):
        if not ids:
            continue
        if len(queries) >= 80:
            break
        if kind == "planet_house":
            fact = {"type": kind, "planet": a, "house": b}
        elif kind == "planet_sign":
            fact = {"type": kind, "planet": a, "sign": b}
        elif kind == "dasha":
            fact = {"type": kind, "dasha_lord": a}
        elif kind == "nakshatra":
            fact = {"type": kind, "nakshatra": a}
        else:
            fact = {"type": kind, "planet": a, "aspect_type": b}
        queries.append({"id": f"corpus_smoke_{case_no:03d}", "category": kind, "query": templates[kind](a,b), "chart_facts": fact, "expected_chunks": ids[:3]})
        case_no += 1
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()
    input_dir = args.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict[str, Any]] = []
    all_pages: list[dict[str, Any]] = []
    book_manifests: list[dict[str, Any]] = []

    for cfg in BOOKS:
        pdf_path = input_dir / cfg.filename
        if not pdf_path.exists():
            raise SystemExit(f"Missing source PDF for {cfg.book_id}: {pdf_path}")
        print(f"[{cfg.book_id}] reading {pdf_path.name} ...")
        chunks, pages, manifest = build_book(cfg, pdf_path)
        all_chunks.extend(chunks)
        all_pages.extend(pages)
        book_manifests.append(manifest)
        print(f"  pages={manifest['pdf_pages']} main={manifest['main_start_pdf_page']}..{manifest['main_end_pdf_page_exclusive']-1} chapters={len(manifest['detected_chapters'])} chunks={len(chunks)}")
        for note in manifest.get("notes", []):
            print("  NOTE:", note)

    # Stable deterministic ordering.
    all_chunks.sort(key=lambda c: c["chunk_id"])
    all_pages.sort(key=lambda p: (p["book_id"], p["pdf_page"]))
    CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    PAGES_OUT.parent.mkdir(parents=True, exist_ok=True)
    GOLD_OUT.parent.mkdir(parents=True, exist_ok=True)

    with CORPUS_OUT.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            # JSONL keeps the actual source text in a machine-readable form.
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    with PAGES_OUT.open("w", encoding="utf-8") as fh:
        for p in all_pages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    counts = Counter()
    by_book = Counter()
    for c in all_chunks:
        counts[c["metadata"].get("chunk_type", "unknown")] += 1
        by_book[c["metadata"].get("book_id", "unknown")] += 1

    manifest = {
        "schema_version": "2.0",
        "built_from": "the four user-provided OCR PDFs",
        "corpus_path": str(CORPUS_OUT.relative_to(PROJECT_ROOT)),
        "page_inventory_path": str(PAGES_OUT.relative_to(PROJECT_ROOT)),
        "chunk_count": len(all_chunks),
        "chunk_types": dict(sorted(counts.items())),
        "chunks_by_book": dict(sorted(by_book.items())),
        "books": book_manifests,
        "method": {
            "primary_unit": "sloka/rule segment when an OCR sloka marker is available; numbered-rule fallback for BPHS; page/prose fallback otherwise",
            "raw_source_preserved": True,
            "llm_rewriting_during_ingestion": False,
            "semantic_text_field": "retrieval_text",
            "main_text_only": True,
            "index_excluded": True,
        },
    }
    with MANIFEST_OUT.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    gold = build_gold_set(all_chunks)
    with GOLD_OUT.open("w", encoding="utf-8") as fh:
        for row in gold:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks -> {CORPUS_OUT}")
    print(f"Wrote {len(all_pages)} page records -> {PAGES_OUT}")
    print(f"Wrote corpus manifest -> {MANIFEST_OUT}")
    print(f"Wrote {len(gold)} corpus-specific retrieval smoke cases -> {GOLD_OUT}")
    print("Chunk types:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
