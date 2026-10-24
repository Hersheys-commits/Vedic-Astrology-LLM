from __future__ import annotations

from typing import Any

from app.retrieval.corpus_store import CorpusStore


def metadata_search(store: CorpusStore, filt: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
    """
    Exact match on every field present in `filt` (case-insensitive). Every
    field must match -- this is deliberately strict; the semantic leg is
    what handles fuzziness.
    """
    if not filt:
        return []

    id_sets = []
    for field, value in filt.items():
        if value is None:
            continue
        ids = store.metadata_ids(field, value)
        id_sets.append(ids)

    if not id_sets:
        return []

    matched_ids = set.intersection(*id_sets) if len(id_sets) > 1 else id_sets[0]

    results = []
    for cid in matched_ids:
        chunk = store.chunk(cid)
        if chunk:
            results.append(
                {
                    "chunk_id": cid,
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "score": 1.0,
                    "match_type": "metadata",
                }
            )
    # Stable order (chunk_id) so results are reproducible for eval/tests.
    results.sort(key=lambda r: r["chunk_id"])
    return results[:top_k]
