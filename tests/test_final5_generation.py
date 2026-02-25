import json
from unittest.mock import MagicMock, patch

from config import (
    DISTRACTOR_GENERATION_PROMPT_CONDITIONED_TEMPLATE,
    DISTRACTOR_GENERATION_PROMPT_QA_TEMPLATE,
)
from data.augmentor import (
    AugmentorMode,
    GenerationConfig,
    _build_conditioned_prompt,
    _build_q_a_prompt,
    augment_single_dataset,
    parse_generated_distractors,
)


def _build_entry() -> dict:
    return {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "choices_answer": ["Paris"],
        "choices_human": ["Lyon", "Marseille", "Nice"],
    }


def _json_payload(count: int) -> str:
    return json.dumps(
        {
            "distractors": [f"distractor_{idx + 1}" for idx in range(count)],
        }
    )


def test_final5_generation_populates_a_b_c_d_e_columns_and_traces():
    dataset = [_build_entry()]

    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        MagicMock(text=_json_payload(3)),
        MagicMock(text=_json_payload(6)),
        MagicMock(text=_json_payload(6)),
        MagicMock(text=_json_payload(9)),
    ]

    config = GenerationConfig(
        mode=AugmentorMode.FINAL5,
        model_provider="openai",
        model_name="gpt-5.2-2025-12-11",
        reasoning_effort="medium",
        max_retries=2,
        save_interval=999,
    )

    with patch("data.augmentor.get_client", return_value=mock_client):
        result = augment_single_dataset(dataset, config, limit=1)

    assert len(result) == 1
    row = result[0]

    assert row["schema_version"] == "final5_v1"

    assert len(row["human_from_scratch"]) == 3
    assert len(row["model_from_scratch"]) == 3
    assert len(row["augment_human"]) == 6
    assert len(row["augment_model_delta_6m"]) == 6
    assert len(row["augment_model"]) == 9
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


def test_parse_generated_distractors_prefers_json_schema_payload():
    response = json.dumps({"distractors": ["one", "two", "three"]})
    parsed = parse_generated_distractors(response=response, expected_count=3, start_letter="B")
    assert parsed == ["one", "two", "three"]


def test_parse_generated_distractors_accepts_truncated_structured_json():
    response = (
        '{"distractors": ["B: Planetary days will become longer.", '
        '"C: Planetary years will become shorter.", '
        '"D: Planetary gravity will become stronger.",'
    )
    parsed = parse_generated_distractors(response=response, expected_count=3, start_letter="B")
    assert parsed == [
        "Planetary days will become longer.",
        "Planetary years will become shorter.",
        "Planetary gravity will become stronger.",
    ]


def test_parse_generated_distractors_rejects_non_json_output():
    response = "E: first option F: second option G: third option H: fourth option I: fifth option J: sixth option"
    try:
        parse_generated_distractors(response=response, expected_count=6, start_letter="E")
    except ValueError as exc:
        assert "Could not decode structured distractors JSON" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-JSON structured output")


def test_generation_single_parse_retry_without_repair_call():
    dataset = [_build_entry()]

    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        MagicMock(text='{"distractors": ["only_one"]}'),
        MagicMock(text=_json_payload(3)),
        MagicMock(text=_json_payload(6)),
        MagicMock(text=_json_payload(6)),
        MagicMock(text=_json_payload(9)),
    ]

    config = GenerationConfig(
        mode=AugmentorMode.FINAL5,
        model_provider="openai",
        model_name="gpt-5.2-2025-12-11",
        max_retries=3,
        save_interval=999,
    )

    with patch("data.augmentor.get_client", return_value=mock_client):
        result = augment_single_dataset(dataset, config, limit=1)

    assert len(result) == 1
    assert mock_client.generate.call_count == 5


def test_generation_retries_when_model_returns_empty_output():
    dataset = [_build_entry()]

    empty_raw = MagicMock()
    empty_raw.model = "claude-opus-4-6"
    empty_raw.stop_reason = "end_turn"
    empty_raw.content = []

    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        MagicMock(text="", raw_response=empty_raw),
        MagicMock(text=_json_payload(3)),
        MagicMock(text=_json_payload(6)),
        MagicMock(text=_json_payload(6)),
        MagicMock(text=_json_payload(9)),
    ]

    config = GenerationConfig(
        mode=AugmentorMode.FINAL5,
        model_provider="anthropic",
        model_name="claude-opus-4-6",
        max_retries=2,
        save_interval=999,
    )

    with patch("data.augmentor.get_client", return_value=mock_client):
        result = augment_single_dataset(dataset, config, limit=1)

    assert len(result) == 1
    assert mock_client.generate.call_count == 5


def test_prompt_builders_source_templates():
    qa_prompt = _build_q_a_prompt(
        question="Q?",
        answer="A!",
        count=3,
        start_letter="B",
    )
    conditioned_prompt = _build_conditioned_prompt(
        question="Q?",
        answer="A!",
        context_distractors=["d1", "d2", "d3"],
        count=6,
    )

    expected_qa = DISTRACTOR_GENERATION_PROMPT_QA_TEMPLATE.format(
        question="Q?",
        gold_answer="A!",
        count=3,
        target_letters="B, C, D",
    )
    expected_conditioned = DISTRACTOR_GENERATION_PROMPT_CONDITIONED_TEMPLATE.format(
        question="Q?",
        gold_answer="A!",
        existing_options_block="A: A!\nB: d1\nC: d2\nD: d3",
        num_existing_options=4,
        total_options=10,
        total_distractors=9,
        count=6,
        target_letters="E, F, G, H, I, J",
    )

    assert qa_prompt == expected_qa
    assert conditioned_prompt == expected_conditioned


def test_openai_generation_calls_use_response_format_schema():
    dataset = [_build_entry()]
    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        MagicMock(text=_json_payload(3)),
        MagicMock(text=_json_payload(6)),
        MagicMock(text=_json_payload(6)),
        MagicMock(text=_json_payload(9)),
    ]

    config = GenerationConfig(
        mode=AugmentorMode.FINAL5,
        model_provider="openai",
        model_name="gpt-5.2-2025-12-11",
        max_retries=1,
        save_interval=999,
    )

    with patch("data.augmentor.get_client", return_value=mock_client):
        augment_single_dataset(dataset, config, limit=1)

    first_call = mock_client.generate.call_args_list[0].kwargs
    assert "response_format" in first_call
    assert "output_config" not in first_call
    schema = first_call["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["distractors"]["minItems"] == 3
    assert schema["properties"]["distractors"]["maxItems"] == 3


def test_anthropic_generation_calls_use_output_config_schema():
    dataset = [_build_entry()]
    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        MagicMock(text=_json_payload(3)),
        MagicMock(text=_json_payload(6)),
        MagicMock(text=_json_payload(6)),
        MagicMock(text=_json_payload(9)),
    ]

    config = GenerationConfig(
        mode=AugmentorMode.FINAL5,
        model_provider="anthropic",
        model_name="claude-opus-4-6",
        anthropic_thinking={"type": "adaptive"},
        max_retries=1,
        save_interval=999,
    )

    with patch("data.augmentor.get_client", return_value=mock_client):
        augment_single_dataset(dataset, config, limit=1)

    first_call = mock_client.generate.call_args_list[0].kwargs
    assert "output_config" in first_call
    assert "response_format" not in first_call
    assert first_call["output_config"]["format"]["type"] == "json_schema"
    assert first_call["thinking"] == {"type": "adaptive"}
