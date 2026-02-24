from data.mmlu_pro_processor import sort_distractors


def test_sort_distractors_uses_exact_question_match_before_cleaning():
    entry = {
        "question": "  Which option is correct?  ",
        "answer_index": 0,
        "options": ["  Correct  ", " Human A ", "Synthetic B", "Human C"],
    }

    # Exact-question key must include surrounding spaces to match.
    mmlu_lookup = {"  Which option is correct?  ": {"Human A", "Human C", "  Correct  "}}

    out = sort_distractors(entry, mmlu_lookup)
    assert out is not None
    human, synthetic, gold = out

    # Post-processing strips whitespace after raw matching is done.
    # " Human A " does not raw-match "Human A", so only Human C is treated as human.
    assert human == ["Human C"]
    assert synthetic == ["Human A", "Synthetic B"]
    assert gold == "Correct"


def test_sort_distractors_returns_none_when_question_key_differs_by_whitespace():
    entry = {
        "question": "  Which option is correct?  ",
        "answer_index": 0,
        "options": ["Correct", "Human A", "Synthetic B", "Human C"],
    }

    # No exact-key match because whitespace differs.
    mmlu_lookup = {
        "Which option is correct?": {"Human A", "Human C", "Correct"}
    }

    assert sort_distractors(entry, mmlu_lookup) is None
