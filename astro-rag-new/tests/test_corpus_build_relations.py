from scripts.build_corpus import _direct_relations


def test_direct_relations_indexes_own_house_lord_placement():
    relations = _direct_relations(
        "If the 10th lord is in the 10th house, the result is strong."
    )

    assert "10|10" in relations["lord_house_pairs"]


def test_direct_relations_still_indexes_other_house_lord_placement():
    relations = _direct_relations(
        "When the 10th lord is in the 4th house, the result is noted."
    )

    assert "10|4" in relations["lord_house_pairs"]


def test_direct_relations_does_not_link_lord_to_later_unrelated_planet_placement():
    relations = _direct_relations(
        "The debilitated 5th Lord is situated in the 12th, 6th or 8th house "
        "and Mercury in the 5th house."
    )

    assert "5|8" in relations["lord_house_pairs"]
    assert "5|5" not in relations["lord_house_pairs"]


def test_direct_relations_ignores_house_mentioned_as_context_not_destination():
    relations = _direct_relations(
        "The 5th Lord is situated in Trik Houses; in this situation the 5th House "
        "will be weakened."
    )

    assert "5|5" not in relations["lord_house_pairs"]
