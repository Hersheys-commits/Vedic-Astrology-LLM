from app.retrieval.corpus_store import reload_store
from app.retrieval.metadata_search import metadata_search


def test_exact_planet_house_match():
    store = reload_store()
    results = metadata_search(store, {"planet": "Mars", "house": 10}, top_k=5)
    assert len(results) >= 1
    assert all("Mars" in r["metadata"].get("planets", []) and 10 in r["metadata"].get("houses", []) for r in results)


def test_no_match_returns_empty():
    store = reload_store()
    # house 13 does not exist
    results = metadata_search(store, {"planet": "Mars", "house": 13}, top_k=5)
    assert results == []


def test_case_insensitive():
    store = reload_store()
    results = metadata_search(store, {"planet": "mars", "house": 10}, top_k=5)
    assert len(results) >= 1


def test_exact_relation_pair_match():
    store = reload_store()
    results = metadata_search(store, {"planet_house_pairs": "Mars|10"}, top_k=10)
    assert results
    assert all("Mars|10" in r["metadata"].get("planet_house_pairs", []) for r in results)
