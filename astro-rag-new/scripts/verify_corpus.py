#!/usr/bin/env python3
"""Verify that Astro RAG is using the four uploaded OCR PDFs.

This script checks corpus provenance, file hashes, expected book coverage,
chunk counts, and that no sample/placeholder corpus is present in the built
JSONL. It does not modify the corpus.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "rag" / "source_pdfs"
PROCESSED_DIR = ROOT / "rag" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "corpus_manifest.json"
CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl"

EXPECTED_BOOKS = {
    "bphs": 47,
    "brihat_jataka": 28,
    "jataka_parijata": 9,  # supplied OCR begins at Adhyaya X in this edition
    "phaladeepika": 28,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path in (MANIFEST_PATH, CORPUS_PATH):
        if not path.exists():
            raise SystemExit(f"MISSING: {path}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    books = manifest.get("books", [])
    by_id = {b.get("book_id"): b for b in books}

    missing = sorted(set(EXPECTED_BOOKS) - set(by_id))
    extra = sorted(set(by_id) - set(EXPECTED_BOOKS))
    if missing or extra:
        raise SystemExit(f"BOOK SET MISMATCH: missing={missing}, extra={extra}")

    total_chunks = 0
    print("Corpus verification")
    print("=" * 72)
    for book_id, expected_chapters in EXPECTED_BOOKS.items():
        book = by_id[book_id]
        source_file = book.get("filename")
        source_path = SOURCE_DIR / source_file
        if not source_path.exists():
            raise SystemExit(f"MISSING SOURCE PDF: {source_path}")

        actual_hash = sha256(source_path)
        manifest_hash = book.get("sha256")
        if actual_hash != manifest_hash:
            raise SystemExit(
                f"HASH MISMATCH for {book_id}: manifest={manifest_hash}, actual={actual_hash}"
            )

        chapter_count = len(book.get("detected_chapters", []))
        chunk_count = int(manifest.get("chunks_by_book", {}).get(book_id, 0))
        if chapter_count != expected_chapters:
            raise SystemExit(
                f"CHAPTER COUNT MISMATCH for {book_id}: expected={expected_chapters}, got={chapter_count}"
            )
        if chunk_count <= 0:
            raise SystemExit(f"NO CHUNKS for {book_id}")

        total_chunks += chunk_count
        print(f"OK  {book_id:18} chapters={chapter_count:2} chunks={chunk_count:4} sha256={actual_hash[:12]}…")

    seen_books: set[str] = set()
    seen = 0
    bad_provenance = 0
    sample_markers = ("sample_corpus", "placeholder", "synthetic")
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            seen += 1
            meta = row.get("metadata", {})
            book_id = meta.get("book_id")
            seen_books.add(book_id)
            book = by_id.get(book_id)
            source = meta.get("source", {})
            source_file = source.get("file") if isinstance(source, dict) else None
            expected_hash = book.get("sha256") if book else None
            text_blob = json.dumps(row, ensure_ascii=False).lower()

            if not book or source.get("book_id") != book_id or source.get("pdf_sha256") != expected_hash:
                bad_provenance += 1
                print(f"BAD provenance line {line_no}: {book_id} / {source.get('book_id') if isinstance(source, dict) else None}")
            if any(marker in text_blob for marker in sample_markers):
                raise SystemExit(f"SAMPLE/PLACEHOLDER MARKER FOUND at corpus line {line_no}")

    if seen != total_chunks:
        raise SystemExit(f"CHUNK COUNT MISMATCH: manifest={total_chunks}, corpus.jsonl={seen}")
    if seen_books != set(EXPECTED_BOOKS):
        raise SystemExit(f"CORPUS BOOK SET MISMATCH: {sorted(seen_books)}")
    if bad_provenance:
        raise SystemExit(f"PROVENANCE ERRORS: {bad_provenance}")

    print("-" * 72)
    print(f"PASS  {seen} corpus chunks verified from the four uploaded PDFs.")
    print("NOTE  Jataka Parijata is intentionally partial in this OCR upload (9 detected chapters).")


if __name__ == "__main__":
    main()
