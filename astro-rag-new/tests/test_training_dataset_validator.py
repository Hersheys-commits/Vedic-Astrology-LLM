from rag.evaluation.validate_training_dataset import validate_dataset


def _example():
    return {
        "id": "sample_1",
        "domain": "education",
        "question": "Where is Mercury placed?",
        "chart_facts": ["Mercury: Placed in 4th house (Leo)"],
        "retrieved_context": [
            {
                "chunk_id": "rule_1",
                "source": "Brihat Jataka",
                "text": "Mercury has a stated result in a chart.",
            }
        ],
        "thought_process": ["private analysis is kept outside the target"],
        "final_answer": "Mercury is placed in the 4th house.",
    }


def _corpus():
    return {
        "rule_1": {
            "chunk_id": "rule_1",
            "text": "Mercury has a stated result in a chart.",
            "metadata": {"source": {"book": "Brihat Jataka"}},
        }
    }


def test_dataset_validator_accepts_a_clean_example_without_rewriting_it():
    example = _example()
    report = validate_dataset([example], _corpus())

    assert report["example_count"] == 1
    assert report["errors"] == []
    assert report["possible_chart_fact_mismatches"] == []
    assert example["final_answer"] == "Mercury is placed in the 4th house."


def test_dataset_validator_reports_data_integrity_and_internal_reasoning_failures():
    broken = _example()
    broken["final_answer"] = (
        "**Reasoning:** internal notes. Mercury is placed in the 8th house."
    )
    broken["retrieved_context"][0]["chunk_id"] = "missing_chunk"
    broken.pop("id")

    report = validate_dataset([broken], _corpus())

    assert report["missing_id_count"] == 1
    assert report["internal_reasoning_leaks"]
    assert report["invalid_passage_references"]
    assert report["possible_chart_fact_mismatches"][0]["id"] == "<example 1>"
    assert report["errors"]


def test_dataset_validator_reports_duplicate_prompt_and_target_groups():
    first = _example()
    second = {**_example(), "id": "sample_2"}

    report = validate_dataset([first, second], _corpus())

    assert len(report["duplicate_questions"]) == 1
    assert len(report["duplicate_question_chart_fact_prompts"]) == 1
    assert len(report["duplicate_targets"]) == 1
