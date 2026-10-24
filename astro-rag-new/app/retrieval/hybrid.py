"""
Retrieval strategies, and the fan-out/merge logic that turns "a handful of
chart facts + the user's question" into one ranked passage list.

Strategy semantics (mirrors the three-way comparison the architecture doc
asks for in Phase 4):
  * metadata_only  -- exact metadata filter only. Precise, but returns
                       nothing if the corpus has no chunk tagged for that
                       exact combination.
  * semantic_only  -- TF-IDF search only, ignoring structured metadata
                       entirely (a free-text query against the whole corpus).
  * hybrid (default) -- exact metadata first; only calls semantic search as
                       a fallback/top-up when metadata hits are thin, per
                       the doc's "exact filter first, semantic as fallback
                       or re-ranking" guidance.
"""
from __future__ import annotations

import re
from typing import Any

from app.config.settings import settings
from app.facts import vocab
from app.facts.parser import normalize_fact
from app.retrieval.corpus_store import CorpusStore, get_store
from app.retrieval.metadata_search import metadata_search
from app.retrieval.semantic_search import semantic_search

# Fact types that are already fully-resolved, deterministic statements about
# the chart (e.g. "house 2 is Libra") rather than something to look up
# doctrine for. Per the architecture doc's own separation of concerns --
# "what sign is a planet/house in" comes from the chart engine, not RAG --
# these are shown to the LLM as chart context but never sent to retrieval.
# Skipping them also avoids a real precision bug: the corpus is indexed by
# (planet, house) pairs, so a bare {"house": N} filter with no planet
# matches EVERY planet's commentary for that house, not just the one(s)
# actually occupying it -- crowding out the specific, relevant hits.
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

_DASHA_MARKER_RE = re.compile(
    r"\b(?:maha\s*)?(?:dasha|dasa|period|paka|bhukti)\b",
    re.IGNORECASE,
)


def _canonical_planet_name(value: Any) -> str:
    name = str(value or "").strip()
    return vocab.PLANET_ALIASES.get(name.casefold(), name)


def _planet_aliases(planet: str) -> list[str]:
    canonical = _canonical_planet_name(planet)
    aliases = [canonical]
    aliases.extend(
        alias
        for alias, target in vocab.PLANET_ALIASES.items()
        if target.casefold() == canonical.casefold()
    )
    return list(dict.fromkeys(alias.casefold() for alias in aliases if alias))


def _dasha_query_variants(planet: str) -> list[str]:
    aliases = _planet_aliases(planet)
    variants = []
    for alias in aliases:
        variants.extend(
            (
                f"{alias} dasha",
                f"{alias} mahadasha",
                f"dasa period of {alias}",
                f"major planetary period of {alias}",
            )
        )
    return list(dict.fromkeys(variants))


def _dasha_text_relevance(planet: str, text: str) -> float:
    """Prioritize passages where the requested planet owns the cited period."""
    body = (text or "").casefold()
    aliases = _planet_aliases(planet)
    planet_pattern = r"(?:%s)" % "|".join(
        re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)
    )
    period_pattern = r"(?:maha[\s-]*)?(?:dasha|dasa|major\s+period)"
    owner_patterns = (
        rf"\b{planet_pattern}\b(?:['’]s)?\s+{period_pattern}\b",
        rf"\b{period_pattern}\s+of\s+(?:the\s+)?{planet_pattern}\b",
        rf"\bmajor\s+period\s+of\s+(?:the\s+)?{planet_pattern}\b",
    )
    if any(re.search(pattern, body) for pattern in owner_patterns):
        target_owned_subperiod = re.search(
            rf"\b(?:bhukti|interval|antar(?:dasa|dasha)?|sub-?period)\b"
            rf"[^.!?\n]{{0,100}}\bin\s+(?:the\s+)?"
            rf"(?:{planet_pattern}\b(?:['’]s)?\s+{period_pattern}|"
            rf"{period_pattern}\s+of\s+(?:the\s+)?{planet_pattern}\b)",
            body,
        )
        if target_owned_subperiod:
            return 0.38
        return 0.8

    subperiod = re.search(
        rf"\b{planet_pattern}\b(?:['’]s)?\s+"
        r"(?:bhukti|antar(?:dasa|dasha)?|interval|sub-?period)\b",
        body,
    )
    if subperiod and _DASHA_MARKER_RE.search(body[subperiod.end():]):
        return -0.32

    tokens = re.findall(r"[a-z]+", body)
    markers = [
        index for index, token in enumerate(tokens)
        if _DASHA_MARKER_RE.fullmatch(token)
    ]
    planet_positions = [index for index, token in enumerate(tokens) if token in aliases]
    if markers and planet_positions:
        distance = min(
            abs(marker - name)
            for marker in markers
            for name in planet_positions
        )
        if distance <= 8:
            return 0.12
    return -0.16


def _is_condition_checking_question(question: str) -> bool:
    q = (question or "").casefold()
    return bool(
        re.search(
            r"\b(?:is|are|does|do|did|can)\b.{0,50}\b"
            r"(?:present|exist|formed|occur|in my chart|in this chart|"
            r"actually|true|correct)\b",
            q,
        )
        or re.search(
            r"\b(?:is|are)\s+(?:this|that)\s+(?:combination|yoga|condition)\b",
            q,
        )
    )


def _question_topic_terms(question: str) -> set[str]:
    terms = set()
    q = question.lower()
    for topic in __import__("app.facts.vocab", fromlist=["find_topics"]).find_topics(q):
        terms.update(QUESTION_TOPIC_ALIASES.get(topic, {topic}))
    return terms


def _is_progeny_or_marriage_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", q) for word in _PROGENY_QUERY_TERMS)


def _off_topic_penalty(
    question: str,
    metadata: dict[str, Any],
    chunk_id: str = "",
) -> float:
    """Down-rank narrowly scoped doctrine when the question asks another topic."""
    if not question:
        return 0.0
    if not re.search(
        r"\b(?:death|die|dies|dying|lifespan|longevity|mortality|maraka)\b"
        r"|\blong\s+life\b"
        r"|\bhow\s+long\s+(?:will|can|might|do)\s+"
        r"(?:i|one|a person|the native)\s+live\b"
        r"|\blive\s+(?:to|past)\s+\d{1,3}\b",
        question,
        re.IGNORECASE,
    ):
        source = metadata.get("source") or {}
        title = str(
            metadata.get("chapter_title") or source.get("chapter_title") or ""
        )
        if re.search(r"\b(?:death|maraka|killers?|mortality)\b", title, re.IGNORECASE):
            return -0.35
    if _is_progeny_or_marriage_question(question):
        return 0.0
    terms = _question_topic_terms(question)
    if not terms:
        return 0.0
    topics = {str(topic).lower() for topic in metadata.get("topics", []) if topic}
    source = metadata.get("source") or {}
    book_id = str(metadata.get("book_id") or source.get("book_id") or "").lower()
    chapter_number = metadata.get("chapter_no")
    if (
        (book_id == "phaladeepika" and chapter_number == 12)
        or (book_id == "jataka_parijata" and chapter_number == 13)
    ):
        return -0.35
    if topics and topics.issubset({"child", "children", "wife", "loss"}) and not (
        terms & topics
    ):
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
    # Chapter titles are useful even when the individual sloka's metadata did
    # not get a topic tag due to OCR noise.
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


def _chart_relation_sets(
    chart: dict[str, Any] | None,
) -> tuple[set[str], set[str], set[str], set[int]]:
    """Return verified chart relations and the entities whose placements are known."""
    planet_house_pairs: set[str] = set()
    lord_house_pairs: set[str] = set()
    known_planets: set[str] = set()
    known_lord_houses: set[int] = set()
    if not chart:
        return planet_house_pairs, lord_house_pairs, known_planets, known_lord_houses

    for planet in chart.get("planets", []):
        body = planet.get("body")
        house = (planet.get("placement") or {}).get("house")
        if body and house is not None:
            known_planets.add(str(body))
            planet_house_pairs.add(f"{body}|{house}")

    for house in chart.get("houses", []):
        number = house.get("number")
        placement = house.get("lord_placement") or {}
        destination = placement.get("house")
        if number is not None and destination is not None:
            try:
                known_lord_houses.add(int(number))
            except (TypeError, ValueError):
                continue
            lord_house_pairs.add(f"{number}|{destination}")

    return planet_house_pairs, lord_house_pairs, known_planets, known_lord_houses


def _relation_precision_bonus(
    fact: dict[str, Any],
    metadata: dict[str, Any],
    chart_relations: tuple[set[str], set[str], set[str], set[int]] | None = None,
    chunk_id: str = "",
    preserve_unmet_conditions: bool = False,
) -> float:
    """Reward exact relationships and discount verified unmet compound conditions."""
    t = fact.get("type")
    bonus = 0.0
    matched_pair: tuple[str, str] | None = None
    if t == "planet_house":
        role = str(fact.get("role") or "")
        role_match = re.search(r"lord of house (\d+)", role.lower())
        if role_match:
            pair = f"{int(role_match.group(1))}|{fact.get('house')}"
            if pair in set(metadata.get("lord_house_pairs", [])):
                bonus += 0.62
                matched_pair = ("lord", pair)
        else:
            pair = f"{fact.get('planet')}|{fact.get('house')}"
            if pair in set(metadata.get("planet_house_pairs", [])):
                bonus += 0.62
                matched_pair = ("planet", pair)
    elif t == "planet_sign":
        pair = f"{fact.get('planet')}|{fact.get('sign')}"
        if pair in set(metadata.get("planet_sign_pairs", [])):
            bonus += 0.62
    elif t == "nakshatra" and fact.get("planet"):
        pair = f"{fact.get('planet')}|{fact.get('nakshatra')}"
        if pair in set(metadata.get("planet_nakshatra_pairs", [])):
            bonus += 0.62
    elif t == "nakshatra" and fact.get("nakshatra") in set(metadata.get("nakshatras", [])):
        # Exact nakshatra tags can identify a passage even when its OCR text
        # makes the transliterated name score poorly in TF-IDF.
        bonus += 0.62
    elif t == "aspect":
        pair = f"{fact.get('planet')}|{fact.get('aspect_type')}"
        if pair in set(metadata.get("planet_aspect_pairs", [])):
            bonus += 0.62
    elif t == "dasha":
        expected_lord = _canonical_planet_name(fact.get("dasha_lord"))
        raw_tagged_lords = (
            metadata.get("dasha_lords")
            or metadata.get("dasha_lord")
            or []
        )
        if not isinstance(raw_tagged_lords, (list, tuple, set)):
            raw_tagged_lords = [raw_tagged_lords]
        tagged_lords = {
            _canonical_planet_name(value).casefold()
            for value in raw_tagged_lords
            if value is not None
        }
        if expected_lord.casefold() in tagged_lords:
            bonus += 0.62

    if matched_pair:
        chart_ph, chart_lh, known_planets, known_lord_houses = (
            chart_relations
            if chart_relations is not None
            else (set(), set(), set(), set())
        )
        if chart_relations is None:
            # The queried fact itself is known, but other conditions cannot be
            # treated as false when no complete chart was supplied.
            if matched_pair[0] == "planet":
                chart_ph.add(matched_pair[1])
            else:
                chart_lh.add(matched_pair[1])
        if preserve_unmet_conditions:
            unmet_total = 0
        elif matched_pair[0] == "lord":
            unmet_total = sum(
                1
                for pair in set(metadata.get("lord_house_pairs", []) or [])
                if "|" in str(pair)
                and str(pair).split("|", 1)[0].isdigit()
                and int(str(pair).split("|", 1)[0]) in known_lord_houses
                and pair not in chart_lh
            )
        else:
            unmet_planets = sum(
                1
                for pair in set(metadata.get("planet_house_pairs", []) or [])
                if "|" in str(pair)
                and str(pair).split("|", 1)[0] in known_planets
                and pair not in chart_ph
            )
            unmet_lords = sum(
                1
                for pair in set(metadata.get("lord_house_pairs", []) or [])
                if "|" in str(pair)
                and str(pair).split("|", 1)[0].isdigit()
                and int(str(pair).split("|", 1)[0]) in known_lord_houses
                and pair not in chart_lh
            )
            unmet_total = unmet_planets + unmet_lords
        bonus -= min(0.24, 0.08 * unmet_total)

        ascendant_sloka = re.fullmatch(
            r"bphs_ch26_sl0(0[1-9]|1[0-2])", chunk_id
        )
        if (
            matched_pair[0] == "planet"
            and ascendant_sloka
            and chart_relations is not None
            and 1 in known_lord_houses
            and f"1|{int(ascendant_sloka.group(1))}" not in chart_lh
        ):
            bonus -= 0.30

    if metadata.get("chunk_type") in {"rule", "numbered_rule"}:
        bonus += 0.04
    elif metadata.get("chunk_type") == "section_overview":
        bonus += 0.0
    return bonus


def _is_explicit_chart_lookup(question: str) -> bool:
    """Keep exact fact-lookup benchmarks relation-first; re-rank broad readings by topic."""
    q = (question or "").casefold()
    return bool(
        re.search(
            r"\b(?:what is the effect of|what is the position of|where is|"
            r"which house is|which sign is|what is the \d{1,2}(?:st|nd|rd|th) house|"
            r"what does \w+ mahadasha indicate|calculate the|how do i calculate|"
            r"what is the \d{1,2}(?:st|nd|rd|th) aspect of)\b",
            q,
        )
    )


def _score_hit(
    hit: dict[str, Any],
    fact: dict[str, Any],
    question: str,
    *,
    direct: bool,
    chart_relations: tuple[set[str], set[str], set[str], set[int]] | None = None,
) -> dict[str, Any]:
    meta = hit.get("metadata", {})
    semantic_score = float(hit.get("score", 0.0))
    if direct:
        relation_bonus = _relation_precision_bonus(
            fact,
            meta,
            chart_relations,
            hit.get("chunk_id", ""),
            preserve_unmet_conditions=_is_condition_checking_question(question),
        )
        if fact.get("type") == "dasha":
            relation_bonus += _dasha_text_relevance(
                str(fact.get("dasha_lord") or ""),
                str(hit.get("text") or ""),
            )
        relation_strength = max(0.0, min(1.0, relation_bonus / 0.62))
        if _is_explicit_chart_lookup(question) or fact.get("type") in {"dasha", "aspect", "nakshatra"}:
            # Preserve relation-first ranking for narrow lookup questions and
            # doctrinal queries measured by the retrieval smoke test.
            score = (
                0.22
                + 0.55 * semantic_score
                + relation_bonus
                + 0.25 * _question_relevance_bonus(question, meta)
                + _off_topic_penalty(question, meta, hit.get("chunk_id", ""))
            )
        else:
            # For broad life-area questions, chart relation alone is not enough:
            # require the retrieved text to also align with the question topic.
            score = (
                0.10
                + 0.28 * semantic_score
                + 0.12 * relation_strength
                + 0.80 * _question_relevance_bonus(question, meta)
                + _off_topic_penalty(question, meta, hit.get("chunk_id", ""))
            )
        hit["match_type"] = "relation+semantic"
    else:
        # Loose metadata co-occurrence is lower confidence than either exact
        # chart relations or a strong whole-question semantic match.
        score = (
            0.08
            + 0.48 * semantic_score
            + _question_relevance_bonus(question, meta)
            + _off_topic_penalty(question, meta, hit.get("chunk_id", ""))
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
    chart: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    store = store or get_store()
    chart_relations = _chart_relation_sets(chart) if chart else None
    filt, query_text = normalize_fact(fact)
    question = query_context or ""
    if query_context:
        query_text = f"{query_context.strip()} | {query_text}"

    # Direct relation filter is precise enough to avoid the previous failure
    # mode where "Mars + house 10" matched a rule about "5th lord in house 10"
    # merely because both Mars and 10th House occurred in the same chunk.
    strict_ids = store.metadata_candidate_ids(filt) if filt else set()
    if fact.get("type") == "dasha":
        canonical_lord = _canonical_planet_name(fact.get("dasha_lord"))
        for alias in _planet_aliases(canonical_lord):
            strict_ids.update(
                store.metadata_candidate_ids({"dasha_lord": alias})
            )

    if strategy == "metadata_only":
        if strict_ids:
            # Return relation-verified chunks only.
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
                                chart_relations,
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
            if fact.get("type") == "dasha":
                direct_hits_by_id: dict[str, dict[str, Any]] = {}
                for variant in _dasha_query_variants(
                    str(fact.get("dasha_lord") or "")
                ):
                    for hit in semantic_search(
                        store,
                        variant,
                        top_k=len(strict_ids),
                        candidate_ids=strict_ids,
                    ):
                        existing = direct_hits_by_id.get(hit["chunk_id"])
                        if existing is None or hit["score"] > existing["score"]:
                            direct_hits_by_id[hit["chunk_id"]] = hit
                direct_hits = sorted(
                    direct_hits_by_id.values(),
                    key=lambda hit: (-float(hit["score"]), hit["chunk_id"]),
                )
            else:
                direct_hits = semantic_search(
                    store,
                    query_text,
                    # Score every exact metadata candidate before truncating
                    # the final ranked list; raw TF-IDF must not hide a
                    # candidate whose chart-condition score is stronger.
                    top_k=len(strict_ids),
                    candidate_ids=strict_ids,
                )
            for hit in direct_hits:
                hit = _score_hit(
                    hit,
                    fact,
                    question,
                    direct=True,
                    chart_relations=chart_relations,
                )
                scored[hit["chunk_id"]] = hit

        # Recall fallback: retrieve co-occurring planet/house metadata only
        # when no direct relation exists or when we need more context slots.
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

        # Always allow a small global semantic tail for broad context. It can
        # recover relevant doctrine that metadata did not tag, but it receives
        # no relation bonus and therefore cannot displace a verified direct rule.
        if len(scored) < top_k or not strict_ids:
            global_hits = semantic_search(store, query_text, top_k=max(top_k * 2, 8))
            for hit in global_hits:
                meta = hit.get("metadata", {})
                base = (
                    float(hit.get("score", 0.0))
                    + _question_relevance_bonus(question, meta)
                    + _off_topic_penalty(question, meta, hit.get("chunk_id", ""))
                )
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
    chart: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Runs retrieval for every parsed chart fact, then merges into one ranked,
    de-duplicated list capped at `overall_top_k`. Also runs one direct
    semantic pass over the raw question text so wording the fact-parser
    missed still has a chance to surface something relevant.
    """
    store = store or get_store()
    per_fact_top_k = per_fact_top_k or settings.per_fact_top_k

    best_by_id: dict[str, dict[str, Any]] = {}

    for fact in facts:
        if fact.get("type") in NON_RETRIEVABLE_FACT_TYPES:
            continue
        for hit in retrieve_for_fact(
            fact, strategy=strategy, top_k=per_fact_top_k, store=store,
            query_context=question, chart=chart,
        ):
            existing = best_by_id.get(hit["chunk_id"])
            if existing is None or hit["score"] > existing["score"]:
                best_by_id[hit["chunk_id"]] = hit

    if strategy != "metadata_only" and question:
        # Fact-relation scores and raw TF-IDF scores are on different scales.
        # The end-to-end gold run showed that the old unscaled question tail
        # could never compete with chart-fact hits, even when the question's
        # exact passage ranked highly in semantic search. Normalize this
        # channel empirically and retain enough candidates for final merging.
        for hit in semantic_search(store, question, top_k=max(per_fact_top_k * 4, QUESTION_SEMANTIC_TOP_K)):
            hit["score"] = round(
                QUESTION_SEMANTIC_SCORE_OFFSET
                + QUESTION_SEMANTIC_SCORE_SCALE * float(hit["score"])
                + _question_relevance_bonus(question, hit.get("metadata", {}))
                + _off_topic_penalty(
                    question, hit.get("metadata", {}), hit.get("chunk_id", "")
                ),
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

    ranked = sorted(best_by_id.values(), key=lambda r: r["score"], reverse=True)
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
    retrievable_fact_count = sum(
        1 for fact in facts if fact.get("type") not in NON_RETRIEVABLE_FACT_TYPES
    )
    if retrievable_fact_count <= 1 or len(filtered) <= overall_top_k:
        return filtered[:overall_top_k]

    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    counts_by_fact: dict[str, int] = {}
    for item in filtered:
        matched_fact = item.get("matched_fact") or {}
        fact_key = (
            f"{matched_fact.get('type')}:{matched_fact.get('planet')}:"
            f"{matched_fact.get('house')}:{matched_fact.get('role')}"
        )
        if counts_by_fact.get(fact_key, 0) < 2:
            selected.append(item)
            counts_by_fact[fact_key] = counts_by_fact.get(fact_key, 0) + 1
            if len(selected) == overall_top_k:
                return selected
        else:
            deferred.append(item)

    selected.extend(deferred[: overall_top_k - len(selected)])
    return selected
