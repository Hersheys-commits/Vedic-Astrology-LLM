from app.facts.parser import parse_question
from app.facts.parser import _ordered_topic_houses
from app.facts.vocab import find_house_numbers, find_topics


def test_explicit_planet_and_house(sample_chart):
    facts = parse_question("What is the effect of Mars in the 10th house?", sample_chart)
    assert {"type": "planet_house", "planet": "Mars", "house": 10} in facts


def test_explicit_planet_and_sign(sample_chart):
    facts = parse_question("What does Mars in Gemini mean?", sample_chart)
    assert {"type": "planet_sign", "planet": "Mars", "sign": "Gemini"} in facts


def test_word_form_house_number(sample_chart):
    facts = parse_question("Mars placed in the tenth house", sample_chart)
    assert {"type": "planet_house", "planet": "Mars", "house": 10} in facts


def test_planet_only_uses_actual_chart_placement(sample_chart):
    facts = parse_question("Tell me about my Moon", sample_chart)
    types_planets = [(f["type"], f.get("planet")) for f in facts]
    assert ("planet_house", "Moon") in types_planets
    assert ("planet_sign", "Moon") in types_planets
    house_fact = next(f for f in facts if f["type"] == "planet_house")
    assert house_fact["house"] == 8


def test_dasha_question(sample_chart):
    facts = parse_question("What does Sun Mahadasha mean for me?", sample_chart)
    assert {"type": "dasha", "dasha_lord": "Sun"} in facts


def test_dasa_spelling_selects_named_dasha_not_current_period(sample_chart):
    facts = parse_question("What does Venus's dasa mean in Brihat Jataka?", sample_chart)
    assert {"type": "dasha", "dasha_lord": "Venus"} in facts


def test_timing_with_marriage_topic_keeps_marriage_houses_and_current_dasha(sample_chart):
    facts = parse_question("Can my current dasha tell the date I will get married?", sample_chart)
    houses = {fact["house"] for fact in facts if fact["type"] == "house_lookup"}
    # The compact fixture includes house 7 only; the full chart service returns all 12.
    assert 7 in houses
    assert {"type": "dasha", "dasha_lord": "Sun"} in facts


def test_llm_topic_intents_are_augmented_with_deterministic_topic_facts(sample_chart, monkeypatch):
    from types import SimpleNamespace
    import app.facts.intent as intent

    sample_chart["houses"].extend([
        {"number": 2, "sign": "Taurus", "lord": "Venus", "occupants": [], "lord_placement": {"house": 10}},
        {"number": 6, "sign": "Aquarius", "lord": "Saturn", "occupants": [], "lord_placement": {"house": 10}},
    ])
    monkeypatch.setattr(intent, "get_llm_client", lambda: SimpleNamespace(mode="openai"))
    monkeypatch.setattr(intent, "extract_facts_llm", lambda *args, **kwargs: [
        {"type": "house_lookup", "house": 10, "sign": "Gemini"}
    ])
    facts, source = intent.resolve_facts("What career themes might be considered from my chart?", sample_chart)
    houses = {fact["house"] for fact in facts if fact["type"] == "house_lookup"}
    assert {2, 6, 10}.issubset(houses)
    assert source == "llm+rules"
    deterministic_facts = parse_question(
        "What career themes might be considered from my chart?", sample_chart
    )
    assert facts[:len(deterministic_facts)] == deterministic_facts


def test_dasha_question_no_planet_uses_current_period(sample_chart):
    facts = parse_question("What is my current dasha period about?", sample_chart)
    assert {"type": "dasha", "dasha_lord": "Sun"} in facts


def test_topic_question_maps_to_houses(sample_chart):
    facts = parse_question("What does my career look like?", sample_chart)
    # career -> houses [10, 6, 2]; house 10 has occupants Mars, Venus, Saturn in the fixture
    house_lookup = [f for f in facts if f["type"] == "house_lookup" and f["house"] == 10]
    assert house_lookup
    planet_house_facts = [f for f in facts if f["type"] == "planet_house" and f["house"] == 10]
    assert any(
        fact["planet"] == "Mars" and fact["house"] == 10
        for fact in planet_house_facts
    )


def test_everyday_vocabulary_maps_to_relevant_house_topics():
    topics = find_topics(
        "Why do multinational companies offer distant jobs far from my country, "
        "and what capacity for bold undertakings do I have?"
    )
    assert {"multinational", "companies", "distant", "far", "country", "bold", "undertakings"} <= set(topics)


def test_everyday_domain_topics_map_to_relevant_houses():
    topics = find_topics(
        "financial comfort support succeed media design communications venture "
        "son daughter properties domestic long life"
    )
    assert {
        "financial", "comfort", "support", "succeed", "media", "design",
        "communications", "venture", "son", "daughter", "properties",
        "domestic", "long life",
    } <= set(topics)


def test_topic_phrases_resolve_explicit_house_numbers():
    assert find_house_numbers("wealth house") == [2]
    assert find_house_numbers("marriage house") == [7]


def test_planet_and_life_area_question_includes_both_fact_sets(sample_chart):
    sample_chart["houses"].extend([
        {"number": 2, "sign": "Libra", "lord": "Venus", "occupants": []},
        {"number": 11, "sign": "Cancer", "lord": "Moon", "occupants": []},
    ])

    facts = parse_question("Can Saturn improve my wealth?", sample_chart)

    houses = {fact["house"] for fact in facts if fact["type"] == "house_lookup"}
    assert {2, 11} <= houses


def test_personal_planet_house_misconception_uses_chart_placement(sample_chart):
    sample_chart["planets"].append({
        "body": "Saturn",
        "position": {"zodiac_sign": "Cancer"},
        "placement": {"house": 3},
        "nakshatra": {"name": "Pushya", "pada": 1},
        "motion": {"retrograde": False},
    })
    facts = parse_question(
        "Will I have a stable marriage despite Saturn being in my 7th house?",
        sample_chart,
    )

    assert any(
        fact["type"] == "planet_house"
        and fact["planet"] == "Saturn"
        and fact["house"] == 3
        for fact in facts
    )
    assert {"type": "planet_sign", "planet": "Saturn", "sign": "Cancer"} in facts
    assert {"type": "planet_house", "planet": "Saturn", "house": 7} not in facts
    assert any(
        fact.get("type") == "planet_house"
        and fact["planet"] == "Saturn"
        and fact.get("queried_house_mismatch") == 7
        for fact in facts
    )
    assert any(fact.get("type") == "house_lookup" and fact["house"] == 7 for fact in facts)


def test_topic_house_expansion_is_capped():
    from app.facts.parser import MAX_TOPIC_HOUSES

    houses = _ordered_topic_houses(
        "wealth marriage career education health",
        ["wealth", "marriage", "career", "education", "health"],
    )

    assert len(houses) == MAX_TOPIC_HOUSES == 4


def test_domain_topic_houses_precede_generic_success_word(sample_chart):
    question = (
        "Can I succeed in media, design, communications, or starting my own venture?"
    )
    topics = find_topics(question.lower())

    assert _ordered_topic_houses(question.lower(), topics) == [3, 5, 10, 2]

    sample_chart["houses"].extend([
        {"number": 2, "sign": "Libra", "lord": "Venus", "occupants": []},
        {"number": 3, "sign": "Scorpio", "lord": "Mars", "occupants": ["Moon"]},
        {"number": 5, "sign": "Capricorn", "lord": "Saturn", "occupants": []},
    ])
    facts = parse_question(question, sample_chart)
    topic_houses = [
        fact["house"]
        for fact in facts
        if fact.get("type") == "house_lookup"
    ]

    assert topic_houses == [3, 5, 10, 2]


def test_explicit_planet_limits_supplemental_topic_houses(sample_chart):
    facts = parse_question("What does Saturn suggest for my career?", sample_chart)
    topic_houses = {
        fact["house"]
        for fact in facts
        if fact.get("type") == "house_lookup"
    }

    assert len(topic_houses) <= 2


def test_fallback_when_nothing_matches(sample_chart):
    facts = parse_question("What can you tell me in general?", sample_chart)
    assert len(facts) > 0
    types = {f["type"] for f in facts}
    assert types & {"sign_lookup", "dasha"}


def test_aspect_question(sample_chart):
    facts = parse_question("What is the significance of Mars's aspects?", sample_chart)
    assert any(f["type"] == "aspect" and f["planet"] == "Mars" for f in facts)


def test_timing_marriage_adds_current_dasha(sample_chart):
    facts = parse_question("When will I get married?", sample_chart)
    assert {"type": "dasha", "dasha_lord": "Sun"} in facts


def test_timing_job_adds_current_dasha(sample_chart):
    facts = parse_question("When will I get my job?", sample_chart)
    assert {"type": "dasha", "dasha_lord": "Sun"} in facts
