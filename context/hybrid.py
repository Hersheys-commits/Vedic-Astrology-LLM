"""
Retrieval strategies, and the fan-out/merge logic that turns "a handful of
chart facts + the user's question" into one ranked passage list.
"""
from __future__ import annotations

import re
from typing import Any

from app.config.settings import settings
from app.facts.parser import normalize_fact
from app.retrieval.corpus_store import CorpusStore, get_store
from app.retrieval.metadata_search import metadata_search
from app.retrieval.semantic_search import semantic_search

NON_RETRIEVABLE_FACT_TYPES = {"house_lookup", "sign_lookup"}
QUESTION_SEMANTIC_TOP_K = 20
QUESTION_SEMANTIC_SCORE_OFFSET = 0.12
QUESTION_SEMANTIC_SCORE_SCALE = 0.65

QUESTION_TOPIC_ALIASES = {
    "job": {"job", "profession", "work", "promotion", "salary", "business", "study", "education"},
    "jobs": {"job", "profession", "work", "promotion", "salary", "business", "study", "education"},
    "career": {"career", "job", "profession", "work", "promotion", "salary", "business", "study", "education"},
    "media": {"media", "design", "communication", "communications", "study", "education", "character", "wealth"},
    "design": {"media", "design", "communication", "communications", "study", "education", "character", "wealth"},
    "communications": {"media", "design", "communication", "communications", "study", "education", "wealth"},
    "venture": {"business", "venture", "startup", "entrepreneur", "wealth", "courage"},
    "marriage": {"marriage", "married", "marry", "wedding", "spouse", "wife", "husband", "relationship"},
    "married": {"marriage", "married", "marry", "wedding", "spouse", "wife", "husband", "relationship"},
    "personality": {"personality", "character", "nature", "appearance"},
    "character": {"personality", "character", "nature", "appearance"},
    "nature": {"personality", "character", "nature"},
}

_PROGENY_QUERY_TERMS = {
    "children", "child", "kids", "baby", "pregnancy", "son", "daughter",
    "conception", "fertility", "offspring", "mother", "father", "family",
    "marriage", "married", "spouse", "wife", "husband",
}
_PROGENY_ONLY_CHUNK_TOPICS = {"child", "children", "wife"}


def _question_topic_terms(question: str) -> set[str]:
    terms = set()
    q = question.lower()
    for topic in __import__("app.facts.vocab", fromlist=["find_topics"]).find_topics(q):
        terms.update(QUESTION_TOPIC_ALIASES.get(topic, {topic}))
    return terms


def _is_progeny_or_marriage_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(rf"\b{re.escape(w)}\b", q) for w in _PROGENY_QUERY_TERMS)


def _off_topic_penalty(question: str, metadata: dict[str, Any], chunk_id: str = "") -> float:
    """
    Penalize conception/childbirth-specific passages (e.g. Phaladeepika Ch. 12 on
    biological conception or Jataka Parijata Ch. 13 on loss of children) when the
    user's question is about career, media, wealth, or education rather than children.
    """
    if not question or _is_progeny_or_marriage_question(question):
        return 0.0
    terms = _question_topic_terms(question)
    if not terms:
        return 0.0
    meta_topics = {str(x).lower() for x in metadata.get("topics", []) if x}
    book_id = str(metadata.get("book_id") or (metadata.get("source") or {}).get("book_id") or "").lower()
    chapter_no = metadata.get("chapter_no")

    # Phaladeepika Ch. 12 (Garbhadhana / conception) and Jataka Parijata Ch. 13 (progeny)
    if (book_id == "phaladeepika" and chapter_no == 12) or (book_id == "jataka_parijata" and chapter_no == 13):
        return -0.35

    # If a chunk's only tagged topics are child/wife/loss and have zero overlap with the query
    if meta_topics and meta_topics.issubset({"child", "children", "wife", "loss"}) and not (terms & meta_topics):
        return -0.22
    return 0.0


def _question_relevance_bonus(question: str, metadata: dict[str, Any]) -> float:
    """Small domain reranker so a life-area question favors matching doctrine."""
    if not question:
        return 0.0
    terms = _question_topic_terms(question)
    if not terms:
        return 0.0
    meta_topics = {str(x).lower() for x in metadata.get("topics", [])}
    source = metadata.get("source", {}) or {}
    title = str(metadata.get("chapter_title") or source.get("chapter_title") or "").lower()
    overlap = len(terms & meta_topics)
    bonus = min(0.18, 0.09 * overlap)
    if not overlap and any(re.search(rf"\b{re.escape(t)}\b", title) for t in terms):
        bonus = 0.10
    return bonus


def _loose_metadata_filter(fact: dict[str, Any]) -> dict[str, Any]:
    """Legacy co-occurrence filter used only as a recall fallback."""
    t = fact.get("type")
    if t == "planet_house":
        role = str(fact.get("role") or "")
        m = re.search(r"lord of house (\d+)", role.lower())
        if m:
            return {"house": fact.get("house"), "lord_of_house": int(m.group(1))}
        return {"planet": fact.get("planet"), "house": fact.get("house")}
    if t == "planet_sign":
        return {"planet": fact.get("planet"), "sign": fact.get("sign")}
    if t == "nakshatra":
        f = {"nakshatra": fact.get("nakshatra")}
        if fact.get("planet"):
            f["planet"] = fact["planet"]
        return f
    if t == "aspect":
        return {"planet": fact.get("planet"), "aspect_type": fact.get("aspect_type")}
    if t == "dasha":
        return {"dasha_lord": fact.get("dasha_lord")}
    return {}


def _chart_relation_sets(facts: list[dict[str, Any]] | None) -> tuple[set[str], set[str]]:
    """Extract all valid planet|house and lord|house pairs present in the chart facts."""
    ph_pairs: set[str] = set()
    lh_pairs: set[str] = set()
    for f in facts or []:
        if f.get("type") == "planet_house":
            planet = f.get("planet")
            house = f.get("house")
            role = str(f.get("role") or "")
            m = re.search(r"lord of house (\d+)", role.lower())
            if m and house is not None:
                lh_pairs.add(f"{int(m.group(1))}|{house}")
            if planet and house is not None:
                ph_pairs.add(f"{planet}|{house}")
    return ph_pairs, lh_pairs


def _relation_precision_bonus(
    fact: dict[str, Any],
    metadata: dict[str, Any],
    chart_ph_pairs: set[str] | None = None,
    chart_lh_pairs: set[str] | None = None,
    chunk_id: str = "",
) -> float:
    """
    Strongly reward explicit relationship metadata, while discounting compound
    multi-condition slokas (e.g. 4|10 + 10|4 exchange or Ketu|5 + Mercury|5)
    when their additional required conditions of the SAME relation type are not
    satisfied by the chart.
    """
    t = fact.get("type")
    bonus = 0.0
    matched_pair = None
    if t == "planet_house":
        role = str(fact.get("role") or "")
        role_match = re.search(r"lord of house (\d+)", role.lower())
        if role_match:
            pair = f"{int(role_match.group(1))}|{fact.get('house')}"
            if pair in set(metadata.get("lord_house_pairs", [])):
                bonus += 0.62
                matched_pair = ("lh", pair)
        else:
            pair = f"{fact.get('planet')}|{fact.get('house')}"
            if pair in set(metadata.get("planet_house_pairs", [])):
                bonus += 0.62
                matched_pair = ("ph", pair)
    elif t == "planet_sign":
        pair = f"{fact.get('planet')}|{fact.get('sign')}"
        if pair in set(metadata.get("planet_sign_pairs", [])):
            bonus += 0.62
    elif t == "nakshatra" and fact.get("planet"):
        pair = f"{fact.get('planet')}|{fact.get('nakshatra')}"
        if pair in set(metadata.get("planet_nakshatra_pairs", [])):
            bonus += 0.62
    elif t == "nakshatra" and fact.get("nakshatra") in set(metadata.get("nakshatras", [])):
        bonus += 0.62
    elif t == "aspect":
        pair = f"{fact.get('planet')}|{fact.get('aspect_type')}"
        if pair in set(metadata.get("planet_aspect_pairs", [])):
            bonus += 0.62

    if matched_pair is not None:
        chunk_ph = set(metadata.get("planet_house_pairs", []) or [])
        chunk_lh = set(metadata.get("lord_house_pairs", []) or [])
        known_ph = chart_ph_pairs or ({matched_pair[1]} if matched_pair[0] == "ph" else set())
        known_lh = chart_lh_pairs or ({matched_pair[1]} if matched_pair[0] == "lh" else set())
        if matched_pair[0] == "lh":
            # For a house-lord rule (e.g. bphs_ch26_sl052 with lh=['5|4']), only
            # penalize unmet lord_house_pairs (ignore commentary planet examples).
            unmet_total = len(chunk_lh - known_lh)
        else:
            unmet_total = len(chunk_ph - known_ph) + len(chunk_lh - known_lh)
        if unmet_total > 0:
            bonus -= min(0.24, 0.08 * unmet_total)

        # BPHS Ch. 26 Slokas 1-12 are "Ascendant Lord in Houses 1-12"; if a chunk
        # in bphs_ch26_sl001..012 was tagged with a Karaka planet (e.g. Jupiter|5 in
        # bphs_ch26_sl005) when the Ascendant lord is not in that house, discount it.
        m_asc_sloka = re.match(r"^bphs_ch26_sl0(0[1-9]|1[0-2])$", chunk_id)
        if m_asc_sloka and matched_pair[0] == "ph":
            req_asc_pair = f"1|{int(m_asc_sloka.group(1))}"
            if req_asc_asc := (req_asc_pair not in known_lh):
                bonus -= 0.30

    if metadata.get("chunk_type") in {"rule", "numbered_rule"}:
        bonus += 0.04
    elif metadata.get("chunk_type") == "section_overview":
        bonus += 0.0
    return bonus


def _score_hit(
    hit: dict[str, Any],
    fact: dict[str, Any],
    question: str,
    *,
    direct: bool,
    chart_ph_pairs: set[str] | None = None,
    chart_lh_pairs: set[str] | None = None,
) -> dict[str, Any]:
    meta = hit.get("metadata", {})
    cid = hit.get("chunk_id", "")
    semantic_score = float(hit.get("score", 0.0))
    if direct:
        score = (
            0.22
            + 0.55 * semantic_score
            + _relation_precision_bonus(fact, meta, chart_ph_pairs, chart_lh_pairs, cid)
            + 0.25 * _question_relevance_bonus(question, meta)
            + _off_topic_penalty(question, meta, cid)
        )
        hit["match_type"] = "relation+semantic"
    else:
        score = (
            0.08
            + 0.48 * semantic_score
            + _question_relevance_bonus(question, meta)
            + _off_topic_penalty(question, meta, cid)
        )
        if meta.get("chunk_type") == "section_overview":
            score += 0.01
        hit["match_type"] = "metadata_fallback+semantic"
    hit["score"] = round(score, 4)
    return hit


def retrieve_for_fact(
    fact: dict[str, Any],
    strategy: str = "hybrid",
    top_k: int = 3,
    store: CorpusStore | None = None,
    query_context: str | None = None,
    chart_ph_pairs: set[str] | None = None,
    chart_lh_pairs: set[str] | None = None,
) -> list[dict[str, Any]]:
    store = store or get_store()
    filt, query_text = normalize_fact(fact)
    question = query_context or ""
    if query_context:
        query_text = f"{query_context.strip()} | {query_text}"

    strict_ids = store.metadata_candidate_ids(filt) if filt else set()

    if strategy == "metadata_only":
        if strict_ids:
            results = []
            for cid in sorted(strict_ids):
                chunk = store.chunk(cid)
                if chunk:
                    results.append({
                        "chunk_id": cid,
                        "text": chunk["text"],
                        "metadata": chunk.get("metadata", {}),
                        "score": round(
                            1.0
                            + _relation_precision_bonus(
                                fact,
                                chunk.get("metadata", {}),
                                chart_ph_pairs,
                                chart_lh_pairs,
                                cid,
                            ),
                            4,
                        ),
                        "match_type": "metadata_relation",
                    })
            results.sort(key=lambda r: (-r["score"], r["chunk_id"]))
            results = results[:top_k]
        else:
            results = []

    elif strategy == "semantic_only":
        results = semantic_search(store, query_text, top_k=top_k)
        for hit in results:
            hit["score"] = round(float(hit["score"]) + _question_relevance_bonus(question, hit.get("metadata", {})), 4)

    else:  # hybrid
        scored: dict[str, dict[str, Any]] = {}

        if strict_ids:
            direct_hits = semantic_search(
                store,
                query_text,
                top_k=max(top_k * 4, settings.per_fact_top_k * 4, 12),
                candidate_ids=strict_ids,
            )
            for hit in direct_hits:
                hit = _score_hit(
                    hit,
                    fact,
                    question,
                    direct=True,
                    chart_ph_pairs=chart_ph_pairs,
                    chart_lh_pairs=chart_lh_pairs,
                )
                scored[hit["chunk_id"]] = hit

        loose_filt = _loose_metadata_filter(fact)
        loose_ids = store.metadata_candidate_ids(loose_filt) if loose_filt else set()
        loose_ids -= strict_ids
        if loose_ids and len(scored) < top_k:
            loose_hits = semantic_search(
                store,
                query_text,
                top_k=max(top_k * 3, 9),
                candidate_ids=loose_ids,
            )
            for hit in loose_hits:
                hit = _score_hit(hit, fact, question, direct=False)
                existing = scored.get(hit["chunk_id"])
                if existing is None or hit["score"] > existing["score"]:
                    scored[hit["chunk_id"]] = hit

        if len(scored) < top_k or not strict_ids:
            global_hits = semantic_search(store, query_text, top_k=max(top_k * 2, 8))
            for hit in global_hits:
                meta = hit.get("metadata", {})
                base = float(hit.get("score", 0.0)) + _question_relevance_bonus(question, meta)
                hit["score"] = round(0.05 + 0.45 * base, 4)
                hit["match_type"] = "semantic_tail"
                existing = scored.get(hit["chunk_id"])
                if existing is None or hit["score"] > existing["score"]:
                    scored[hit["chunk_id"]] = hit

        results = sorted(scored.values(), key=lambda r: (-r["score"], r["chunk_id"]))[:top_k]

    for r in results:
        r["matched_fact"] = fact
    return results


def retrieve_for_facts(
    facts: list[dict[str, Any]],
    question: str,
    strategy: str = "hybrid",
    overall_top_k: int = 5,
    per_fact_top_k: int | None = None,
    store: CorpusStore | None = None,
) -> list[dict[str, Any]]:
    store = store or get_store()
    per_fact_top_k = per_fact_top_k or settings.per_fact_top_k
    chart_ph_pairs, chart_lh_pairs = _chart_relation_sets(facts)

    best_by_id: dict[str, dict[str, Any]] = {}

    for fact in facts:
        if fact.get("type") in NON_RETRIEVABLE_FACT_TYPES:
            continue
        for hit in retrieve_for_fact(
            fact,
            strategy=strategy,
            top_k=per_fact_top_k,
            store=store,
            query_context=question,
            chart_ph_pairs=chart_ph_pairs,
            chart_lh_pairs=chart_lh_pairs,
        ):
            existing = best_by_id.get(hit["chunk_id"])
            if existing is None or hit["score"] > existing["score"]:
                best_by_id[hit["chunk_id"]] = hit

    if strategy != "metadata_only" and question:
        for hit in semantic_search(store, question, top_k=max(per_fact_top_k * 4, QUESTION_SEMANTIC_TOP_K)):
            hit["score"] = round(
                QUESTION_SEMANTIC_SCORE_OFFSET
                + QUESTION_SEMANTIC_SCORE_SCALE * float(hit["score"])
                + _question_relevance_bonus(question, hit.get("metadata", {})),
                4,
            )
            existing = best_by_id.get(hit["chunk_id"])
            if existing is None:
                hit["matched_fact"] = {"type": "raw_question_semantic"}
                hit["match_type"] = "question_semantic"
                best_by_id[hit["chunk_id"]] = hit
            elif hit["score"] > existing["score"]:
                existing["score"] = hit["score"]
                existing["match_type"] = "question_semantic+metadata"

    ranked = sorted(best_by_id.values(), key=lambda r: (-r["score"], r["chunk_id"]))
    seen_sloka_chapters = set()
    for item in ranked[: overall_top_k * 2]:
        metadata = item.get("metadata") or {}
        source = metadata.get("source") or {}
        if metadata.get("chunk_type") == "sloka":
            book_id = metadata.get("book_id") or source.get("book_id")
            chapter_no = metadata.get("chapter_no")
            if book_id is not None and chapter_no is not None:
                seen_sloka_chapters.add((book_id, chapter_no))

    filtered = []
    for item in ranked:
        metadata = item.get("metadata") or {}
        source = metadata.get("source") or {}
        chapter_key = (
            metadata.get("book_id") or source.get("book_id"),
            metadata.get("chapter_no"),
        )
        if (
            metadata.get("chunk_type") == "section_overview"
            and chapter_key[0] is not None
            and chapter_key[1] is not None
            and chapter_key in seen_sloka_chapters
        ):
            continue
        filtered.append(item)

    # Enforce per-fact diversity (max 2 chunks per single matched_fact on Pass 1)
    # when multiple retrievable facts exist, so one planet (like Jupiter|5)
    # cannot consume 3+ slots and crowd out house-lord slokas (like 5|4).
    retrievable_fact_count = sum(
        1 for f in facts if f.get("type") not in NON_RETRIEVABLE_FACT_TYPES
    )
    if retrievable_fact_count <= 1 or len(filtered) <= overall_top_k:
        return filtered[:overall_top_k]

    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    counts_by_fact: dict[str, int] = {}
    for item in filtered:
        mf = item.get("matched_fact") or {}
        fkey = f"{mf.get('type')}:{mf.get('planet')}:{mf.get('house')}:{mf.get('role')}"
        if counts_by_fact.get(fkey, 0) < 2:
            selected.append(item)
            counts_by_fact[fkey] = counts_by_fact.get(fkey, 0) + 1
            if len(selected) == overall_top_k:
                return selected
        else:
            deferred.append(item)

    for item in deferred:
        selected.append(item)
        if len(selected) == overall_top_k:
            break
    return selected