from __future__ import annotations

from typing import Any, Optional

from sklearn.metrics.pairwise import cosine_similarity

from app.retrieval.corpus_store import CorpusStore


def semantic_search(
    store: CorpusStore,
    query_text: str,
    top_k: int = 5,
    candidate_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    if not query_text or store.vectorizer is None or store.matrix is None:
        return []

    query_vec = store.vectorizer.transform([query_text])
    sims = cosine_similarity(query_vec, store.matrix)[0]

    scored = []
    for idx, chunk in enumerate(store.chunks):
        if candidate_ids is not None and chunk["chunk_id"] not in candidate_ids:
            continue
        score = float(sims[idx])
        if score <= 0.0:
            continue
        scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = []
    for score, chunk in scored[:top_k]:
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "score": round(score, 4),
                "match_type": "semantic",
            }
        )
    return results
