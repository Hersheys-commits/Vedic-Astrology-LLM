from app.generation.openvino_npu_client import NPU_SYSTEM_PROMPT
from app.generation.prompt import CORE_GROUNDING_POLICY, SYSTEM_PROMPT, build_prompt


def test_prompt_allows_cautious_interpretive_synthesis():
    assert "Synthesize only when the chart fact is explicitly supplied" in SYSTEM_PROMPT
    assert "Do not use pretrained astrology knowledge" in SYSTEM_PROMPT
    assert "TRADITIONAL SYNTHESIS" in SYSTEM_PROMPT
    assert "UNSUPPORTED DOCTRINE" in SYSTEM_PROMPT
    assert "Do not begin with a limitation" in SYSTEM_PROMPT
    assert "For narrow factual questions, answer in 2–5 sentences" in SYSTEM_PROMPT
    assert "Do not append a generic astrologer disclaimer" in SYSTEM_PROMPT
    assert "Use D-1 chart facts unless D-9, D-10" in SYSTEM_PROMPT
    assert CORE_GROUNDING_POLICY in SYSTEM_PROMPT
    assert CORE_GROUNDING_POLICY in NPU_SYSTEM_PROMPT


def test_prompt_requires_personality_reading_structure():
    assert "For personality questions specifically" in SYSTEM_PROMPT
    assert "3–5" in SYSTEM_PROMPT
    assert "3–5 concise supporting points" in SYSTEM_PROMPT


def test_prompt_preserves_rule_conditions_and_avoids_neighboring_formula_confusion():
    assert "Do not transfer a rule about a house lord to a planet in that house" in SYSTEM_PROMPT
    assert "Do not blend it with a neighboring rule" in SYSTEM_PROMPT
    assert "Never turn them into guidance or a prediction about the user" in SYSTEM_PROMPT
    assert "Avoid vague combined references such as \"Passage 2/4/5\"" in SYSTEM_PROMPT
    assert "Do not use a chapter or sloka number as a passage number" in SYSTEM_PROMPT
    assert "Verify each condition in a retrieved passage" in SYSTEM_PROMPT
    assert "If none of the retrieved passages is relevant" in SYSTEM_PROMPT
    assert "Never silently swap placements" in SYSTEM_PROMPT
    assert "Do not assert mutually exclusive yogas" in SYSTEM_PROMPT
    assert "Do not invent Sanskrit names" in SYSTEM_PROMPT


def test_built_prompt_keeps_chart_and_passage_grounding():
    chart = {
        "ascendant": {"sign": "Gemini"},
        "moon": {"sign": "Cancer", "nakshatra": "Ashlesha", "pada": 4},
        "sun": {"sign": "Virgo"},
    }
    chart_facts = [
        {"type": "planet_house", "planet": "Mercury", "house": 4, "role": "ascendant lord"},
    ]
    passages = [
        {
            "text": "The fourth house relates to domestic comforts and popularity.",
            "match_type": "metadata",
            "metadata": {
                "source": {"book": "Brihat Jataka", "pdf_page_start": 10},
                "chapter_no": 4,
                "sloka_start": 12,
            },
        }
    ]

    system, user = build_prompt(chart, chart_facts, passages, "How is my personality?")

    assert system == SYSTEM_PROMPT
    assert "Mercury: house=H4; role=ascendant lord" in user
    assert "Passage 1 — source: Brihat Jataka | Chapter 4 | Sloka 12 | PDF p. 10" in user
    assert "How is my personality?" in user


def test_built_prompt_warns_when_retrieval_has_no_chart_relation_or_direct_evidence():
    chart_facts = [
        {"type": "planet_house", "planet": "Moon", "house": 7},
    ]
    passages = [
        {
            "text": "A broad chapter overview with no rule for the question.",
            "match_type": "question_semantic",
            "matched_fact": {"type": "raw_question_semantic"},
            "metadata": {"house": [7], "chunk_type": "section_overview"},
        }
    ]

    _, user = build_prompt({}, chart_facts, passages, "Why do friends borrow money?")

    assert "If no retrieved passage directly answers this question, abstain" in user
    assert user.index("RETRIEVED REFERENCE PASSAGES") < user.index(
        "QUESTION\nWhy do friends borrow money?"
    )
    assert "do not infer relevance from shared keywords or chapter titles" in user
    assert "RETRIEVAL RELEVANCE WARNING" not in user


def test_built_prompt_omits_retrieval_warning_for_exact_chart_relation_match():
    fact = {"type": "planet_house", "planet": "Moon", "house": 7}
    passages = [
        {
            "text": "The Moon in the seventh house is associated with partnership.",
            "match_type": "relation+semantic",
            "matched_fact": fact,
            "metadata": {"planet_house_pairs": ["Moon|7"], "chunk_type": "sloka"},
        }
    ]

    _, user = build_prompt({}, [fact], passages, "What does this suggest?")

    assert "If no retrieved passage directly answers this question, abstain" not in user


def test_built_prompt_does_not_trust_wrong_or_unprovided_chart_relation():
    fact = {"type": "planet_house", "planet": "Moon", "house": 7}
    passages = [
        {
            "text": "A passage about the Moon in the seventh house.",
            "match_type": "relation+semantic",
            "matched_fact": fact,
            "metadata": {"planet_house_pairs": ["Moon|8"]},
        }
    ]

    _, user = build_prompt({}, [], passages, "What does this suggest?")

    assert "If no retrieved passage directly answers this question, abstain" in user


def test_built_prompt_warns_when_retrieval_scores_are_weak():
    fact = {"type": "planet_house", "planet": "Jupiter", "house": 5}
    passage = {
        "text": "A weakly related passage.",
        "score": 0.2621,
        "match_type": "question_semantic",
        "matched_fact": {"type": "raw_question_semantic"},
        "metadata": {},
    }

    _, user = build_prompt({}, [fact], [passage], "What does my chart indicate?")

    assert "RETRIEVAL EVIDENCE WARNING" in user
    assert "No passage has a strong relevance score" in user
    assert "do not use pretrained astrology knowledge to fill gaps" in user


def test_built_prompt_accepts_a_strong_retrieval_score_without_exact_metadata():
    passage = {
        "text": "A source passage with a strong score.",
        "score": 0.72,
        "match_type": "question_semantic",
        "metadata": {},
    }

    _, user = build_prompt({}, [], [passage], "What does my chart indicate?")

    assert "RETRIEVAL EVIDENCE WARNING" not in user


def test_built_prompt_recognizes_supported_exact_chart_relation_metadata():
    cases = [
        (
            {"type": "planet_house", "planet": "Moon", "house": 7},
            {"planet_house_pairs": ["moon|7"]},
        ),
        (
            {
                "type": "planet_house",
                "planet": "Mars",
                "house": 4,
                "role": "lord of house 10",
            },
            {"lord_house_pairs": ["10|4"]},
        ),
        (
            {"type": "planet_sign", "planet": "Jupiter", "sign": "Pisces"},
            {"planet_sign_pairs": ["jupiter|pisces"]},
        ),
        (
            {"type": "nakshatra", "planet": "Moon", "nakshatra": "Rohini"},
            {"planet_nakshatra_pairs": ["moon|rohini"]},
        ),
        (
            {"type": "aspect", "planet": "Mars", "aspect_type": "7th"},
            {"planet_aspect_pairs": ["mars|7th"]},
        ),
        (
            {"type": "dasha", "dasha_lord": "Venus"},
            {"dasha_lords": ["venus"]},
        ),
    ]

    for fact, metadata in cases:
        passage = {
            "text": "A relevant source rule.",
            "matched_fact": fact,
            "metadata": metadata,
        }
        _, user = build_prompt({}, [fact], [passage], "What does this suggest?")
        assert "If no retrieved passage directly answers this question, abstain" not in user


def test_built_prompt_includes_available_twelve_house_map_without_inventing_missing_houses():
    chart = {
        "houses": [
            {
                "number": 1,
                "sign": "Gemini",
                "lord": "Mercury",
                "occupants": ["Moon"],
                "lord_placement": {"house": 4},
            },
            {
                "number": 2,
                "sign": "Cancer",
                "lord": "Moon",
                "occupants": [],
                "lord_placement": {"house": 1},
            },
        ]
    }

    _, user = build_prompt(chart, [], [], "What does my chart suggest?")

    assert "H1 sign=Gemini lord=Mercury occupants=Moon" in user
    assert "H2 sign=Cancer lord=Moon occupants=none" in user
    assert "H3 sign=not supplied lord=not supplied" in user
    assert "H12 sign=not supplied lord=not supplied" in user


def test_built_prompt_includes_planet_placements_and_house_lordships():
    chart = {
        "planets": [
            {
                "body": "Mercury",
                "position": {"zodiac_sign": "Virgo"},
                "placement": {"house": 4},
            },
            {
                "body": "Venus",
                "position": {"zodiac_sign": "Leo"},
                "placement": {"house": 3},
            },
        ],
        "houses": [
            {"number": 1, "sign": "Gemini", "lord": "Mercury", "occupants": []},
            {"number": 4, "sign": "Virgo", "lord": "Mercury", "occupants": ["Mercury"]},
            {"number": 5, "sign": "Libra", "lord": "Venus", "occupants": []},
        ],
    }

    _, user = build_prompt(chart, [], [], "What does my chart suggest?")

    assert "Mercury sign=Virgo house=H4 lord_of=H1,H4" in user
    assert "Venus sign=Leo house=H3 lord_of=H5" in user
    assert all(f"H{number} sign=" in user for number in range(1, 13))


def test_built_prompt_flags_passage_conditions_that_chart_disproves():
    chart = {
        "planets": [
            {
                "body": "Mercury",
                "position": {"zodiac_sign": "Virgo"},
                "placement": {"house": 4},
            }
        ],
        "houses": [
            {
                "number": 4,
                "sign": "Virgo",
                "lord": "Mercury",
                "occupants": ["Mercury"],
                "lord_placement": {"house": 4},
            }
        ],
    }
    passage = {
        "text": "If Mercury is in the fifth house and the fourth lord is in the tenth.",
        "metadata": {
            "planet_house_pairs": ["Mercury|5"],
            "lord_house_pairs": ["4|10"],
        },
    }

    _, user = build_prompt(chart, [], [passage], "What does this rule mean?")

    assert "Mercury is in H4 Virgo (NOT H5)" in user
    assert "H4 lord (Mercury) is in H4 (NOT H10" in user
    assert "Do not claim these unmet conditions apply." in user


def test_built_prompt_binds_house_lord_rule_to_chart_planet_and_disambiguates():
    chart = {
        "planets": [
            {
                "body": "Jupiter",
                "position": {"zodiac_sign": "Virgo"},
                "placement": {"house": 5},
            },
            {
                "body": "Mercury",
                "position": {"zodiac_sign": "Leo"},
                "placement": {"house": 4},
            },
        ],
        "houses": [
            {
                "number": 5,
                "sign": "Virgo",
                "lord": "Mercury",
                "lord_placement": {"house": 4},
            },
            {
                "number": 11,
                "sign": "Pisces",
                "lord": "Jupiter",
                "lord_placement": {"house": 5},
            },
        ],
    }
    passage = {
        "text": "If the 11th lord is placed in the 5th house.",
        "metadata": {"lord_house_pairs": ["11|5"]},
    }

    _, user = build_prompt(
        chart,
        [],
        [passage],
        "What does the 11th lord in the 5th indicate?",
    )

    assert (
        "[Chart Binding: H11 lord Jupiter is placed in H5 Virgo; "
        "distinguish this from H5 lord Mercury, placed in H4 Leo.]"
    ) in user


def test_built_prompt_explains_planet_sign_and_corrects_false_house_premise():
    chart = {
        "planets": [
            {
                "body": "Saturn",
                "position": {"zodiac_sign": "Cancer"},
                "placement": {"house": 3},
            }
        ],
        "houses": [
            {
                "number": 7,
                "sign": "Scorpio",
                "lord": "Mars",
                "occupants": [],
            }
        ],
    }
    chart_facts = [
        {
            "type": "planet_house",
            "planet": "Saturn",
            "house": 3,
            "sign": "Cancer",
            "queried_house_mismatch": 7,
        },
        {
            "type": "house_lookup",
            "house": 7,
            "sign": "Scorpio",
            "lord": "Mars",
            "occupants": [],
        },
    ]

    _, user = build_prompt(
        chart,
        chart_facts,
        [],
        "Will I have a stable marriage despite Saturn being in my 7th house?",
    )

    assert "Saturn: house=H3; sign=Cancer" in user
    assert "Saturn is in house 3, NOT in house 7" in user
    assert "H7: sign=Scorpio; lord=Mars" in user
    assert "occupants=none" in user
