"""Vedic graha drishti (sign/house-based aspects).

Rules come from configuration, not from hard-coded assumptions, so a different
tradition can be plugged in without touching this file.
"""

from __future__ import annotations

from app.config.planets import ASPECT_LABEL, ASPECT_RULES, ASPECTING_BODIES
from app.utils.angles import angular_separation, house_distance


def _rules_for(body: str) -> list[int]:
    return ASPECT_RULES.get(body, ASPECT_RULES["default"])


def calculate_aspects(planets: list[dict], houses: list[dict]) -> list[dict]:
    """Planet -> planet and planet -> house drishti, with raw angular distance kept."""
    aspects: list[dict] = []
    by_house = {h["number"]: h for h in houses}

    for source in planets:
        name = source["body"]
        if name not in ASPECTING_BODIES:
            continue
        src_sign = source["position"]["sign_index"]
        src_house = source["placement"]["house"]

        for distance in _rules_for(name):
            target_sign = (src_sign + distance - 1) % 12
            target_house = ((src_house + distance - 2) % 12) + 1
            label = ASPECT_LABEL.get(distance, f"{distance}th")

            # planet -> house
            house_obj = by_house[target_house]
            house_obj["aspects_received"].append({"from": name, "aspect_type": label})

            # planet -> planet
            for target in planets:
                if target["body"] == name:
                    continue
                if target["position"]["sign_index"] != target_sign:
                    continue
                aspects.append(
                    {
                        "relationship": "aspect",
                        "from": name,
                        "to": target["body"],
                        "aspect_type": label,
                        "house_distance": distance,
                        "from_house": src_house,
                        "to_house": target_house,
                        "angular_distance": round(
                            angular_separation(
                                source["position"]["sidereal_longitude"],
                                target["position"]["sidereal_longitude"],
                            ),
                            4,
                        ),
                        "rule_source": "sign_based_vedic_drishti",
                    }
                )

    # Conjunction is not drishti, but downstream retrieval usually wants it.
    for i, a in enumerate(planets):
        for b in planets[i + 1 :]:
            if a["position"]["sign_index"] == b["position"]["sign_index"]:
                aspects.append(
                    {
                        "relationship": "conjunction",
                        "from": a["body"],
                        "to": b["body"],
                        "aspect_type": "conjunction",
                        "house_distance": 1,
                        "from_house": a["placement"]["house"],
                        "to_house": b["placement"]["house"],
                        "angular_distance": round(
                            angular_separation(
                                a["position"]["sidereal_longitude"],
                                b["position"]["sidereal_longitude"],
                            ),
                            4,
                        ),
                        "rule_source": "same_sign_occupation",
                    }
                )
    return aspects
