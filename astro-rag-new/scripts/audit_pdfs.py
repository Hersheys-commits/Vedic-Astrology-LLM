"""Audit extracted text from the supplied PDFs against the built page inventory.

Run from astro-rag-new:
    python scripts/audit_pdfs.py
    python scripts/audit_pdfs.py --out rag/evaluation/pdf_ocr_audit.json

This checks the existing embedded OCR/text layer and extraction consistency. It
does not claim that the OCR is semantically correct; flagged source pages need
human visual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "rag" / "processed" / "corpus_manifest.json"
PAGES_PATH = PROJECT_ROOT / "rag" / "processed" / "pages.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "rag" / "evaluation" / "pdf_ocr_audit.json"
MOJIBAKE_MARKERS = ("Ã Â", "Ãƒ", "â€", "â€™", "â€œ", "â€“", "â€”", "ï¿½")


def normalize_for_comparison(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def load_page_inventory() -> dict[tuple[str, int], dict[str, Any]]:
    inventory: dict[tuple[str, int], dict[str, Any]] = {}
    with PAGES_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                inventory[(row["book_id"], int(row["pdf_page"]))] = row
    return inventory


def audit_book(book: dict[str, Any], inventory: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    pdf_path = PROJECT_ROOT / "rag" / "source_pdfs" / book["filename"]
    sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    doc = pymupdf.open(pdf_path)
    low_text_pages = []
    empty_pages = []
    mojibake_pages = []
    replacement_pages = []
    inventory_mismatch_pages = []
    missing_inventory_pages = []
    char_counts = []

    for page_num, page in enumerate(doc, start=1):
        # Match the extraction call used by build_corpus.py exactly.
        text = page.get_text("text")
        cleaned = normalize_for_comparison(text)
        char_counts.append(len(cleaned))
        if not cleaned:
            empty_pages.append(page_num)
        elif len(cleaned) < 40:
            low_text_pages.append({"page": page_num, "characters": len(cleaned), "sample": cleaned[:120]})
        if any(marker in text for marker in MOJIBAKE_MARKERS):
            mojibake_pages.append({"page": page_num, "markers": [m for m in MOJIBAKE_MARKERS if m in text]})
        if "\ufffd" in text or "\u25a0" in text:
            replacement_pages.append({"page": page_num, "replacement_characters": text.count("\ufffd"), "black_squares": text.count("\u25a0")})

        stored = inventory.get((book["book_id"], page_num))
        if stored is not None:
            old = normalize_for_comparison(stored.get("text_raw", ""))
            old_tokens = Counter(old.split())
            new_tokens = Counter(cleaned.split())
            denominator = max(sum(old_tokens.values()), sum(new_tokens.values()), 1)
            ratio = sum((old_tokens & new_tokens).values()) / denominator
            if ratio < 0.98:
                inventory_mismatch_pages.append({
                    "page": page_num,
                    "token_overlap": round(ratio, 4),
                    "inventory_characters": len(old),
                    "pdf_characters": len(cleaned),
                    "pdf_sample": cleaned[:180],
                })
        elif len(cleaned) >= 40:
            # The built inventory intentionally excludes front/back matter;
            # this list is a candidate for checking against the manifest range.
            missing_inventory_pages.append(page_num)

    expected_sha = book.get("sha256")
    manifest_mismatch = expected_sha is not None and sha != expected_sha
    return {
        "book_id": book["book_id"],
        "title": book["title"],
        "filename": book["filename"],
        "sha256": sha,
        "manifest_sha256_matches": not manifest_mismatch,
        "manifest_sha256": expected_sha,
        "pdf_pages": len(doc),
        "manifest_pdf_pages": book.get("pdf_pages"),
        "page_count_matches_manifest": len(doc) == book.get("pdf_pages"),
        "pages_with_empty_text": empty_pages,
        "pages_with_under_40_chars": low_text_pages,
        "pages_with_mojibake_markers": mojibake_pages,
        "pages_with_replacement_glyphs": replacement_pages,
        "page_inventory_differences_below_0_98_similarity": inventory_mismatch_pages,
        "pages_with_text_missing_from_main_text_inventory": missing_inventory_pages,
        "extracted_text_characters": sum(char_counts),
        "median_page_characters": sorted(char_counts)[len(char_counts) // 2] if char_counts else 0,
        "main_text_coverage_notes": book.get("notes", []),
        "detected_chapters": book.get("detected_chapters", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = load_page_inventory()
    books = [audit_book(book, inventory) for book in manifest["books"]]
    report = {
        "audit_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_method": "PyMuPDF text extraction compared to pages.jsonl raw extraction, manifest SHA-256, text anomaly checks",
        "interpretation_note": "Automated checks flag extraction anomalies; semantic OCR correctness requires human review of the rendered PDF page.",
        "pdf_count": len(books),
        "manifest_corpus_chunks": manifest.get("chunk_count"),
        "page_inventory_records": len(inventory),
        "books": books,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audited {len(books)} PDFs; {len(inventory)} page inventory records")
    for book in books:
        print(
            f"{book['book_id']}: {book['pdf_pages']} pages; "
            f"sha_match={book['manifest_sha256_matches']}; "
            f"empty={len(book['pages_with_empty_text'])}; "
            f"low_text={len(book['pages_with_under_40_chars'])}; "
            f"ocr_mismatch={len(book['page_inventory_differences_below_0_98_similarity'])}; "
            f"mojibake={len(book['pages_with_mojibake_markers'])}"
        )
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
