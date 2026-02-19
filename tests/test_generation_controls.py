import json
from unittest.mock import MagicMock, patch

from data.augmentor import AugmentorMode, GenerationConfig, augment_single_dataset


def _fake_generation_text() -> str:
    return _fake_generation_text_from("B")


def _fake_generation_text_from(start_letter: str) -> str:
    lines = []
    for idx in range(6):
        letter = chr(ord(start_letter) + idx)
        lines.append(f"{letter}: distractor_{idx + 1}")
    return "\n".join(lines)


def _mock_client_for_three_modes() -> MagicMock:
    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        MagicMock(text=_fake_generation_text_from("B")),  # FROM_SCRATCH
        MagicMock(text=_fake_generation_text_from("E")),  # CONDITIONED_HUMAN
        MagicMock(text=_fake_generation_text_from("E")),  # CONDITIONED_SYNTHETIC
    ]
    return mock_client


def _build_entry() -> dict:
    return {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "choices_answer": ["Paris"],
        "choices_human": ["Lyon", "Marseille", "Nice"],
    }


def test_branching_prefix_generation_is_opt_in():
    dataset = [_build_entry()]
    mock_client = _mock_client_for_three_modes()

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
    mock_client = _mock_client_for_three_modes()

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
    mock_client = _mock_client_for_three_modes()

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
    mock_client = _mock_client_for_three_modes()

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


def test_failed_entry_is_skipped_when_configured():
    dataset = [_build_entry()]
    mock_client = MagicMock()
    mock_client.generate.return_value = MagicMock(text="invalid output")

    config = GenerationConfig(
        mode=AugmentorMode.FROM_SCRATCH,
        model_provider="openai",
        model_name="gpt-5-mini-2025-08-07",
        max_retries=1,
        save_interval=999,
        skip_failed_entries=True,
        generate_branching_prefix_columns=False,
    )

    with patch("data.augmentor.get_client", return_value=mock_client):
        result = augment_single_dataset(dataset, config, limit=1)

    assert len(result) == 0


def test_resume_checkpoint_continues_from_source_index(tmp_path):
    dataset = [_build_entry(), _build_entry()]
    config = GenerationConfig(
        mode=AugmentorMode.FROM_SCRATCH,
        model_provider="openai",
        model_name="gpt-5-mini-2025-08-07",
        max_retries=1,
        save_interval=999,
        generate_branching_prefix_columns=False,
    )

    # First pass: generate one entry to build a realistic checkpoint.
    first_client = _mock_client_for_three_modes()
    with patch("data.augmentor.get_client", return_value=first_client):
        first_result = augment_single_dataset(dataset, config, limit=1)
    checkpoint = tmp_path / "resume.json"
    checkpoint.write_text(json.dumps([first_result[0]], indent=2), encoding="utf-8")

    # Resume pass: should only generate the second row (3 calls: one per mode).
    resume_client = _mock_client_for_three_modes()
    with patch("data.augmentor.get_client", return_value=resume_client):
        resumed_result = augment_single_dataset(
            dataset,
            config,
            resume_from=checkpoint,
        )

    assert len(resumed_result) == 2
    assert resume_client.generate.call_count == 3
    assert resumed_result[0]["_source_index"] == 0
    assert resumed_result[1]["_source_index"] == 1
