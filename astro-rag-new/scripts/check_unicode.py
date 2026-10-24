#!/usr/bin/env python3
"""Check processed corpus for Unicode/OCR display corruption.

Usage: python scripts/check_unicode.py
"""
from __future__ import annotations
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "rag" / "processed" / "corpus.jsonl"
MARKERS = ("à¤", "à¥", "Ã", "Â", "â€")

bad = []
dev_chunks = 0
total = 0
with CORPUS.open(encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        text = row.get("text", "")
        if any(m in text for m in MARKERS):
            bad.append(row["chunk_id"])
        if any("\u0900" <= c <= "\u097f" for c in text):
            dev_chunks += 1

print(f"chunks={total}")
print(f"chunks_with_devanagari={dev_chunks}")
if bad:
    print("FAIL: mojibake markers found in:")
    for cid in bad[:20]:
        print("  ", cid)
    raise SystemExit(1)
print("PASS: no common UTF-8/Latin-1 mojibake markers found in corpus text.")
