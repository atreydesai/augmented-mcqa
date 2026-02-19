from unittest.mock import MagicMock, patch

from data.augmentor import AugmentorMode, GenerationConfig, augment_single_dataset


def _fake_generation_text() -> str:
    lines = []
    for idx in range(6):
        letter = chr(ord("B") + idx)
        lines.append(f"{letter}: distractor_{idx + 1}")
    return "\n".join(lines)


def _build_entry() -> dict:
    return {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "choices_answer": ["Paris"],
        "choices_human": ["Lyon", "Marseille", "Nice"],
    }


def test_branching_prefix_generation_is_opt_in():
    dataset = [_build_entry()]
    mock_client = MagicMock()
    mock_client.generate.return_value.text = _fake_generation_text()

    config = GenerationConfig(
        mode=AugmentorMode.FROM_SCRATCH,
        model_provider="openai",
        model_name="gpt-5-mini-2025-08-07",
        max_retries=1,
        save_interval=999,
        generate_branching_prefix_columns=False,
    )

    with (
        patch("data.augmentor.get_client", return_value=mock_client) as get_client_mock,
        patch("data.augmentor._generate_branching_prefix_columns") as branching_mock,
    ):
        result = augment_single_dataset(dataset, config, limit=1)

    assert len(result) == 1
    branching_mock.assert_not_called()
    get_client_mock.assert_called_once_with("gpt-5-mini-2025-08-07")


def test_branching_prefix_generation_runs_when_enabled():
    dataset = [_build_entry()]
    mock_client = MagicMock()
    mock_client.generate.return_value.text = _fake_generation_text()

    config = GenerationConfig(
        mode=AugmentorMode.FROM_SCRATCH,
        model_provider="openai",
        model_name="gpt-5-mini-2025-08-07",
        max_retries=1,
        save_interval=999,
        generate_branching_prefix_columns=True,
    )

    with (
        patch("data.augmentor.get_client", return_value=mock_client),
        patch("data.augmentor._generate_branching_prefix_columns") as branching_mock,
    ):
        augment_single_dataset(dataset, config, limit=1)

    branching_mock.assert_called_once()


def test_reasoning_effort_not_forwarded_for_non_openai():
    dataset = [_build_entry()]
    mock_client = MagicMock()
    mock_client.generate.return_value.text = _fake_generation_text()

    config = GenerationConfig(
        mode=AugmentorMode.FROM_SCRATCH,
        model_provider="local",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        reasoning_effort="minimal",
        max_retries=1,
        save_interval=999,
        generate_branching_prefix_columns=False,
    )

    with (
        patch("data.augmentor.get_client", return_value=mock_client) as get_client_mock,
        patch("data.augmentor._generate_branching_prefix_columns"),
    ):
        augment_single_dataset(dataset, config, limit=1)

    get_client_mock.assert_called_once_with("Qwen/Qwen3-4B-Instruct-2507")


def test_reasoning_effort_forwarded_when_explicit_for_openai():
    dataset = [_build_entry()]
    mock_client = MagicMock()
    mock_client.generate.return_value.text = _fake_generation_text()

    config = GenerationConfig(
        mode=AugmentorMode.FROM_SCRATCH,
        model_provider="openai",
        model_name="gpt-5-mini-2025-08-07",
        reasoning_effort="low",
        max_retries=1,
        save_interval=999,
        generate_branching_prefix_columns=False,
    )

    with (
        patch("data.augmentor.get_client", return_value=mock_client) as get_client_mock,
        patch("data.augmentor._generate_branching_prefix_columns"),
    ):
        augment_single_dataset(dataset, config, limit=1)

    get_client_mock.assert_called_once_with("gpt-5-mini-2025-08-07", reasoning_effort="low")
