import pytest

from app.retrieval.corpus_store import reload_store
from app.retrieval.hybrid import retrieve_for_fact, retrieve_for_facts


def test_question_semantic_channel_is_downscaled_and_keeps_more_candidates(monkeypatch):
    import app.retrieval.hybrid as hybrid

    def fake_semantic_search(store, query, top_k):
        assert query == "What themes are described?"
        assert top_k >= 20
        return [
            {"chunk_id": "specific", "score": 0.2, "text": "specific passage", "metadata": {}},
            {"chunk_id": "generic", "score": 0.1, "text": "generic passage", "metadata": {}},
        ]

    monkeypatch.setattr(hybrid, "semantic_search", fake_semantic_search)
    results = hybrid.retrieve_for_facts([], "What themes are described?", store=object())
    assert [result["chunk_id"] for result in results] == ["specific", "generic"]
    assert results[0]["score"] == 0.25
    assert results[0]["match_type"] == "question_semantic"


def test_question_semantic_results_drop_overview_when_same_chapter_sloka_is_present(monkeypatch):
    import app.retrieval.hybrid as hybrid

    def fake_semantic_search(store, query, top_k):
        return [
            {
                "chunk_id": "overview",
                "score": 0.9,
                "text": "overview",
                "metadata": {
                    "chunk_type": "section_overview",
                    "book_id": "book-a",
                    "chapter_no": 3,
                },
            },
            {
                "chunk_id": "sloka",
                "score": 0.7,
                "text": "sloka",
                "metadata": {
                    "chunk_type": "sloka",
                    "book_id": "book-a",
                    "chapter_no": 3,
                },
            },
            {
                "chunk_id": "other-overview",
                "score": 0.6,
                "text": "other overview",
                "metadata": {
                    "chunk_type": "section_overview",
                    "book_id": "book-b",
                    "chapter_no": 3,
                },
            },
            {
                "chunk_id": "unattributed-overview",
                "score": 0.5,
                "text": "unattributed overview",
                "metadata": {"chunk_type": "section_overview", "chapter_no": 3},
            },
        ]

    monkeypatch.setattr(hybrid, "semantic_search", fake_semantic_search)
    results = retrieve_for_facts(
        [], "What themes are described?", overall_top_k=3, store=object()
    )

    assert [result["chunk_id"] for result in results] == [
        "sloka",
        "other-overview",
        "unattributed-overview",
    ]


def test_exact_nakshatra_relation_surfaces_source_passage():
    store = reload_store()
    results = retrieve_for_facts(
        [{"type": "nakshatra", "nakshatra": "Rohini"}],
        "What qualities does Brihat Jataka associate with Rohini?",
        overall_top_k=5,
        store=store,
    )
    assert "brihat_jataka_ch16_sl002" in {result["chunk_id"] for result in results}


def test_hybrid_returns_metadata_hit_when_available():
    store = reload_store()
    fact = {"type": "planet_house", "planet": "Mars", "house": 10}
    results = retrieve_for_fact(fact, strategy="hybrid", top_k=3, store=store)
    assert results
    assert "Mars" in results[0]["metadata"].get("planets", [])
    assert 10 in results[0]["metadata"].get("houses", [])


def test_semantic_only_finds_something_relevant():
    store = reload_store()
    fact = {"type": "planet_house", "planet": "Mars", "house": 10}
    results = retrieve_for_fact(fact, strategy="semantic_only", top_k=3, store=store)
    assert results
    assert all(r["match_type"] == "semantic" for r in results)


def test_metadata_only_empty_for_impossible_combo():
    store = reload_store()
    fact = {"type": "planet_house", "planet": "Mars", "house": 13}
    results = retrieve_for_fact(fact, strategy="metadata_only", top_k=3, store=store)
    assert results == []


def test_multi_fact_merge_and_cap():
    store = reload_store()
    facts = [
        {"type": "planet_house", "planet": "Mars", "house": 10},
        {"type": "planet_sign", "planet": "Mars", "sign": "Gemini"},
    ]
    results = retrieve_for_facts(facts, question="What does Mars mean here?", overall_top_k=5, store=store)
    assert 0 < len(results) <= 5
    ids = [r["chunk_id"] for r in results]
    assert len(ids) == len(set(ids))  # no duplicates


def test_hybrid_prefers_explicit_mars_in_tenth_rule():
    store = reload_store()
    fact = {"type": "planet_house", "planet": "Mars", "house": 10}
    results = retrieve_for_fact(
        fact,
        strategy="hybrid",
        top_k=5,
        store=store,
        query_context="What does Mars in the 10th house mean?",
    )
    assert results
    top = results[0]
    assert "Mars|10" in top["metadata"].get("planet_house_pairs", [])
    # This chunk only mentions Mars and the 10th house while describing the
    # 5th lord in the 10th; it must not outrank an explicit Mars-in-10th rule.
    assert top["chunk_id"] != "bphs_ch26_sl058"


def test_compound_relation_bonus_discounts_only_chart_verified_unmet_conditions():
    import app.retrieval.hybrid as hybrid

    chart = {
        "planets": [
            {"body": "Venus", "placement": {"house": 3}},
            {"body": "Ketu", "placement": {"house": 5}},
            {"body": "Mercury", "placement": {"house": 4}},
        ],
        "houses": [
            {
                "number": 4,
                "lord": "Mercury",
                "lord_placement": {"house": 4},
            },
            {
                "number": 5,
                "lord": "Venus",
                "lord_placement": {"house": 3},
            },
            {
                "number": 10,
                "lord": "Jupiter",
                "lord_placement": {"house": 4},
            },
        ],
    }
    chart_relations = hybrid._chart_relation_sets(chart)
    exact = hybrid._relation_precision_bonus(
        {"type": "planet_house", "planet": "Venus", "house": 3},
        {
            "planet_house_pairs": ["Venus|3"],
            "chunk_type": "numbered_rule",
        },
        chart_relations,
    )
    partial_compound = hybrid._relation_precision_bonus(
        {"type": "planet_house", "planet": "Ketu", "house": 5},
        {
            "planet_house_pairs": ["Ketu|5", "Mercury|5"],
            "chunk_type": "numbered_rule",
        },
        chart_relations,
    )

    assert exact == 0.66
    assert partial_compound == pytest.approx(0.58)
    assert hybrid._relation_precision_bonus(
        {"type": "planet_house", "planet": "Ketu", "house": 5},
        {
            "planet_house_pairs": ["Ketu|5", "Mercury|5"],
            "chunk_type": "numbered_rule",
        },
    ) == 0.66


def test_house_lord_bonus_ignores_example_planet_relation_metadata():
    import app.retrieval.hybrid as hybrid

    chart_relations = hybrid._chart_relation_sets({
        "planets": [{"body": "Venus", "placement": {"house": 3}}],
        "houses": [
            {
                "number": 5,
                "lord": "Venus",
                "lord_placement": {"house": 4},
            }
        ],
    })
    score = hybrid._relation_precision_bonus(
        {
            "type": "planet_house",
            "planet": "Venus",
            "house": 4,
            "role": "lord of house 5",
        },
        {
            "lord_house_pairs": ["5|4"],
            "planet_house_pairs": ["Venus|4"],
            "chunk_type": "numbered_rule",
        },
        chart_relations,
    )

    assert score == 0.66


def test_dasha_aliases_use_canonical_metadata_and_surface_period_rules():
    from app.facts.parser import fact_to_metadata_filter, fact_to_query_text
    from app.retrieval.hybrid import _dasha_query_variants

    fact = {"type": "dasha", "dasha_lord": "Guru"}

    assert fact_to_metadata_filter(fact) == {"dasha_lord": "Jupiter"}
    assert "Jupiter" in fact_to_query_text(fact)
    assert "dasa period of guru" in _dasha_query_variants("Jupiter")
    assert "jupiter mahadasha" in _dasha_query_variants("Jupiter")

    store = reload_store()
    results = retrieve_for_fact(fact, query_context="What does Guru dasa indicate?", store=store)
    assert results
    assert any(
        "Jupiter" in (
            result["metadata"].get("dasha_lords")
            or [result["metadata"].get("dasha_lord")]
        )
        for result in results
    )


def test_dasha_text_relevance_prefers_explicit_planet_period_language():
    import app.retrieval.hybrid as hybrid

    explicit = hybrid._dasha_text_relevance(
        "Jupiter", "During the dasa of Jupiter, the result is favorable."
    )
    unrelated = hybrid._dasha_text_relevance(
        "Jupiter", "Jupiter aspects the house; the period is discussed elsewhere."
    )

    assert explicit > unrelated


def test_condition_checking_question_preserves_unactivated_rule_candidates():
    import app.retrieval.hybrid as hybrid

    chart_relations = hybrid._chart_relation_sets({
        "planets": [
            {"body": "Mars", "placement": {"house": 4}},
            {"body": "Moon", "placement": {"house": 2}},
        ],
        "houses": [],
    })
    metadata = {
        "planet_house_pairs": ["Mars|4", "Moon|4"],
        "chunk_type": "numbered_rule",
    }
    fact = {"type": "planet_house", "planet": "Mars", "house": 4}

    interpretation_score = hybrid._relation_precision_bonus(
        fact, metadata, chart_relations
    )
    checking_score = hybrid._relation_precision_bonus(
        fact,
        metadata,
        chart_relations,
        preserve_unmet_conditions=hybrid._is_condition_checking_question(
            "Is this combination present in my chart?"
        ),
    )

    assert checking_score > interpretation_score


def test_retrieval_order_is_stable_for_identical_dasha_inputs():
    store = reload_store()
    fact = {"type": "dasha", "dasha_lord": "Jupiter"}
    first = retrieve_for_fact(
        fact,
        query_context="What does Jupiter Mahadasha indicate?",
        top_k=5,
        store=store,
    )
    second = retrieve_for_fact(
        fact,
        query_context="What does Jupiter Mahadasha indicate?",
        top_k=5,
        store=store,
    )

    assert [result["chunk_id"] for result in first] == [
        result["chunk_id"] for result in second
    ]


def test_ascendant_lord_sloka_is_discounted_when_metadata_tags_only_a_karaka():
    import app.retrieval.hybrid as hybrid

    chart = {
        "planets": [{"body": "Jupiter", "placement": {"house": 5}}],
        "houses": [
            {"number": 1, "lord": "Moon", "lord_placement": {"house": 2}},
        ],
    }
    score = hybrid._relation_precision_bonus(
        {"type": "planet_house", "planet": "Jupiter", "house": 5},
        {
            "planet_house_pairs": ["Jupiter|5"],
            "chunk_type": "numbered_rule",
        },
        hybrid._chart_relation_sets(chart),
        "bphs_ch26_sl005",
    )

    assert score == pytest.approx(0.36)


def test_progeny_chapters_are_penalized_only_for_unrelated_questions():
    import app.retrieval.hybrid as hybrid

    question = "Can I succeed in media, design, communications, or starting a venture?"
    conception_chapter = {
        "book_id": "phaladeepika",
        "chapter_no": 12,
        "topics": ["wife", "children"],
    }
    progeny_chapter = {
        "book_id": "jataka_parijata",
        "chapter_no": 13,
        "topics": ["loss", "children"],
    }

    assert hybrid._off_topic_penalty(question, conception_chapter) == -0.35
    assert hybrid._off_topic_penalty(question, progeny_chapter) == -0.35
    assert hybrid._off_topic_penalty(
        "Will I have children?",
        conception_chapter,
    ) == 0.0


def test_death_specific_chapters_are_penalized_unless_lifespan_is_the_topic():
    import app.retrieval.hybrid as hybrid

    chapter = {"chapter_title": "Maraka Planets (Killers)"}

    assert hybrid._off_topic_penalty(
        "What is the effect of Jupiter in the 3rd house?",
        chapter,
    ) == -0.35
    assert hybrid._off_topic_penalty(
        "Does my chart indicate a long life?",
        chapter,
    ) == 0.0


def test_retrieval_limits_single_fact_to_two_passages_when_other_facts_exist(
    monkeypatch,
):
    import app.retrieval.hybrid as hybrid

    jupiter_fact = {"type": "planet_house", "planet": "Jupiter", "house": 5}
    venus_lord_fact = {
        "type": "planet_house",
        "planet": "Venus",
        "house": 4,
        "role": "lord of house 5",
    }

    def fake_retrieve_for_fact(fact, **_kwargs):
        candidates = (
            [("j1", 0.9), ("j2", 0.8), ("j3", 0.7)]
            if fact == jupiter_fact
            else [("lord-1", 0.65), ("lord-2", 0.6)]
        )
        return [
            {
                "chunk_id": chunk_id,
                "score": score,
                "text": chunk_id,
                "metadata": {"chunk_type": "numbered_rule"},
                "matched_fact": fact,
            }
            for chunk_id, score in candidates
        ]

    monkeypatch.setattr(hybrid, "retrieve_for_fact", fake_retrieve_for_fact)
    monkeypatch.setattr(hybrid, "semantic_search", lambda *_args, **_kwargs: [])

    results = hybrid.retrieve_for_facts(
        [jupiter_fact, venus_lord_fact],
        "What does my chart suggest?",
        overall_top_k=4,
        store=object(),
    )

    assert [item["chunk_id"] for item in results] == [
        "j1",
        "j2",
        "lord-1",
        "lord-2",
    ]


def test_metadata_only_requires_explicit_relationship():
    store = reload_store()
    results = retrieve_for_fact(
        {"type": "planet_house", "planet": "Mars", "house": 10},
        strategy="metadata_only",
        top_k=10,
        store=store,
    )
    assert results
    assert all("Mars|10" in r["metadata"].get("planet_house_pairs", []) for r in results)


def test_corpus_text_is_not_mojibake():
    store = reload_store()
    bad = []
    for chunk in store.chunks:
        text = chunk.get("text", "")
        if any(m in text for m in ("à¤", "à¥", "Ã©", "Â", "â€")):
            bad.append(chunk["chunk_id"])
            if len(bad) >= 5:
                break
    assert not bad, f"Mojibake found in corpus chunks: {bad}"
