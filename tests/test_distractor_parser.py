from unittest.mock import MagicMock, patch

import pytest

from data.augmentor import (
    AugmentorMode,
    GenerationConfig,
    augment_single_dataset,
    parse_generated_distractors,
)


def _base_entry() -> dict:
    return {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "choices_answer": ["Paris"],
        "choices_human": ["Lyon", "Marseille", "Nice"],
    }


def _six_lines(start_letter: str) -> str:
    start_ord = ord(start_letter)
    lines = []
    for i in range(6):
        letter = chr(start_ord + i)
        lines.append(f"{letter}: distractor_{i + 1}")
    return "\n".join(lines)


def test_parser_ignores_non_target_letters():
    response = "\n".join(
        [
            "B: echoed_old_option_1",
            "C: echoed_old_option_2",
            "D: echoed_old_option_3",
            "E: new_1",
            "F: new_2",
            "G: new_3",
            "H: new_4",
            "I: new_5",
            "J: new_6",
        ]
    )
    parsed = parse_generated_distractors(response, expected_count=6, start_letter="E")
    assert parsed == ["new_1", "new_2", "new_3", "new_4", "new_5", "new_6"]


def test_parser_raises_on_missing_target_letters():
    with pytest.raises(ValueError, match="Missing letters"):
        parse_generated_distractors(
            "B: d1\nC: d2",
            expected_count=3,
            start_letter="B",
        )


def test_augment_retries_on_parse_mismatch():
    dataset = [_base_entry()]

    invalid = MagicMock(text="Only one line")
    valid_scratch = MagicMock(text=_six_lines("B"))
    valid_conditioned = MagicMock(text=_six_lines("E"))

    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        invalid,
        valid_scratch,
        valid_conditioned,
        valid_conditioned,
    ]

    config = GenerationConfig(
        mode=AugmentorMode.FROM_SCRATCH,
        model_provider="openai",
        model_name="gpt-5-mini-2025-08-07",
        max_retries=2,
        save_interval=999,
        generate_branching_prefix_columns=False,
    )

    with patch("data.augmentor.get_client", return_value=mock_client):
        result = augment_single_dataset(dataset, config, limit=1)

    assert len(result) == 1
    row = result[0]
    assert len(row["cond_model_q_a_scratch"]) == 6
    assert len(row["cond_model_q_a_dhuman"]) == 6
    assert len(row["cond_model_q_a_dmodel"]) == 6
