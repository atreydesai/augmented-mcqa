from unittest.mock import MagicMock, patch

from data.augmentor import AugmentorMode, GenerationConfig, augment_single_dataset


def _build_entry() -> dict:
    return {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "choices_answer": ["Paris"],
        "choices_human": ["Lyon", "Marseille", "Nice"],
    }


def _lines(start_letter: str, count: int) -> str:
    lines = []
    for idx in range(count):
        letter = chr(ord(start_letter) + idx)
        lines.append(f"{letter}: distractor_{idx + 1}")
    return "\n".join(lines)


def test_final5_generation_populates_a_b_c_d_e_columns_and_traces():
    dataset = [_build_entry()]

    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        MagicMock(text=_lines("B", 3)),
        MagicMock(text=_lines("E", 6)),
        MagicMock(text=_lines("E", 6)),
        MagicMock(text=_lines("B", 9)),
    ]

    config = GenerationConfig(
        mode=AugmentorMode.FINAL5,
        model_provider="openai",
        model_name="gpt-5.2-2025-12-11",
        reasoning_effort="medium",
        max_retries=1,
        save_interval=999,
    )

    with patch("data.augmentor.get_client", return_value=mock_client):
        result = augment_single_dataset(dataset, config, limit=1)

    assert len(result) == 1
    row = result[0]

    assert row["schema_version"] == "final5_v1"

    # A
    assert len(row["human_from_scratch"]) == 3
    # B
    assert len(row["model_from_scratch"]) == 3
    # C
    assert len(row["augment_human"]) == 6
    # D (delta + combined)
    assert len(row["augment_model_delta_6m"]) == 6
    assert len(row["augment_model"]) == 9
    # E
    assert len(row["augment_ablation"]) == 9

    for strategy in [
        "human_from_scratch",
        "model_from_scratch",
        "augment_human",
        "augment_model",
        "augment_ablation",
    ]:
        assert f"{strategy}_model_input" in row
        assert f"{strategy}_model_output" in row
        assert isinstance(row[f"{strategy}_options_randomized"], list)
        assert isinstance(row[f"{strategy}_correct_answer_letter"], str)
