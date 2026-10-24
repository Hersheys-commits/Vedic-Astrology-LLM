"""
In-memory corpus store: loads `rag/processed/corpus.jsonl` once, and exposes
two independent lookup paths over the same chunks --

  * exact metadata filtering  (planet / house / sign / nakshatra / aspect_type
    / dasha_lord), for the "we know precisely what we're looking for" case
  * TF-IDF cosine-similarity semantic search, for free-text queries and for
    catching wording that doesn't line up with the metadata schema

`app/retrieval/hybrid.py` combines the two. Both a chunk id + a whole
metadata field are indexed so `/corpus/stats` and evaluation scripts can
introspect the corpus without re-reading the file.

Note on the embedding backend: this ships with scikit-learn TF-IDF rather
than a downloaded sentence-embedding model, specifically so the whole
project installs and runs fully offline with no model download and
deterministic output -- appropriate for a first retrieval baseline. Swap in
a real embedding model later by replacing `_build_semantic_index` /
`semantic_search` without touching anything else (see README).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config.settings import settings

METADATA_FIELDS = [
    "planet", "target_planet", "house", "sign", "nakshatra",
    "aspect_type", "dasha_lord", "lord_of_house", "book_id", "chapter_no", "chunk_type",
    "layer", "topics", "relations",
    "planet_house_pairs", "planet_house_example_pairs", "planet_sign_pairs", "planet_nakshatra_pairs",
    "planet_aspect_pairs", "lord_house_pairs",
]

# The corpus stores rich metadata as list-valued fields (e.g. `planets` and
# `houses`) while the retrieval API intentionally accepts compact singular
# filters. Index both representations so a rule mentioning multiple planets
# can still be retrieved for a specific planet.
FIELD_ALIASES = {
    "planet": "planets",
    "house": "houses",
    "sign": "signs",
    "nakshatra": "nakshatras",
    "aspect_type": "aspect_types",
    "dasha_lord": "dasha_lords",
    "lord_of_house": "lord_of_houses",
    "target_planet": "target_planet",
    "topics": "topics",
    "relations": "relations",
    "planet_house_pairs": "planet_house_pairs",
    "planet_house_example_pairs": "planet_house_example_pairs",
    "planet_sign_pairs": "planet_sign_pairs",
    "planet_nakshatra_pairs": "planet_nakshatra_pairs",
    "planet_aspect_pairs": "planet_aspect_pairs",
    "lord_house_pairs": "lord_house_pairs",
}


class CorpusStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.chunks: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self._field_index: dict[str, dict[str, set[str]]] = {f: {} for f in METADATA_FIELDS}
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._loaded_from: Optional[Path] = None

    # ------------------------------------------------------------------ #
    def load(self, path: Optional[Path] = None) -> None:
        path = Path(path or settings.corpus_path)
        with self._lock:
            chunks = []
            with path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc

            self.chunks = chunks
            self.by_id = {c["chunk_id"]: c for c in chunks}
            self._field_index = {f: {} for f in METADATA_FIELDS}
            for c in chunks:
                meta = c.get("metadata", {})
                for field in METADATA_FIELDS:
                    values: list[Any] = []
                    if field in meta and meta.get(field) is not None:
                        raw = meta.get(field)
                        values.extend(raw if isinstance(raw, list) else [raw])
                    alias = FIELD_ALIASES.get(field)
                    if alias and alias in meta and meta.get(alias) is not None:
                        raw = meta.get(alias)
                        values.extend(raw if isinstance(raw, list) else [raw])
                    for val in values:
                        key = str(val).strip().lower()
                        self._field_index[field].setdefault(key, set()).add(c["chunk_id"])

            self._build_semantic_index()
            self._loaded_from = path

    def _build_semantic_index(self) -> None:
        # Search the contextualized representation, not the bare passage.
        # Source/chapter/sloka/topic labels improve retrieval and still leave
        # the exact source text untouched in `text`.
        texts = [c.get("retrieval_text") or c.get("text", "") for c in self.chunks]
        if not texts:
            self._vectorizer, self._matrix = None, None
            return
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self._matrix = self._vectorizer.fit_transform(texts)

    # ------------------------------------------------------------------ #
    def is_loaded(self) -> bool:
        return bool(self.chunks)

    def stats(self) -> dict[str, int]:
        out = {"total_chunks": len(self.chunks)}
        for field in METADATA_FIELDS:
            out[f"distinct_{field}"] = len(self._field_index[field])
        return out

    def metadata_ids(self, field: str, value: Any) -> set[str]:
        if value is None:
            return set()
        return self._field_index.get(field, {}).get(str(value).strip().lower(), set())

    def metadata_candidate_ids(self, filt: dict[str, Any]) -> set[str]:
        """Return the exact intersection for a metadata filter."""
        id_sets = []
        for field, value in filt.items():
            if value is None:
                continue
            ids = self.metadata_ids(field, value)
            id_sets.append(ids)
        if not id_sets:
            return set()
        return set.intersection(*id_sets) if len(id_sets) > 1 else set(id_sets[0])

    def chunk(self, chunk_id: str) -> Optional[dict[str, Any]]:
        return self.by_id.get(chunk_id)

    @property
    def vectorizer(self):
        return self._vectorizer

    @property
    def matrix(self):
        return self._matrix


_store: Optional[CorpusStore] = None


def get_store() -> CorpusStore:
    global _store
    if _store is None:
        _store = CorpusStore()
        _store.load()
    return _store


def reload_store(path: Optional[Path] = None) -> CorpusStore:
    global _store
    _store = CorpusStore()
    _store.load(path)
    return _store
