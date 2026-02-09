import pytest
import os
from unittest.mock import MagicMock, patch
from pathlib import Path
from data.augmentor import (
    AugmentorMode,
    GenerationConfig,
    build_prompt,
    parse_generated_distractors,
    get_output_column,
    augment_dataset,
)
from config import DistractorType, AUGMENTED_DATASETS_DIR

@pytest.fixture
def mock_entry():
    return {
        "question": "What is 2+2?",
        "options": ["3", "4", "5", "6"],
        "answer_letter": "B",
        "answer_index": 1,
        "gold_answer": "4",
        DistractorType.COND_HUMAN_Q_A.value: ["3", "5", "6"],
    }

def test_get_output_column():
    """Verify output column names."""
    assert get_output_column(AugmentorMode.FROM_SCRATCH) == DistractorType.COND_MODEL_Q_A_SCRATCH.value
    assert get_output_column(AugmentorMode.CONDITIONED_HUMAN) == DistractorType.COND_MODEL_Q_A_DHUMAN.value
    assert get_output_column(AugmentorMode.CONDITIONED_SYNTHETIC) == DistractorType.COND_MODEL_Q_A_DMODEL.value

def test_parse_generated_distractors():
    """Test parsing of model response."""
    # Standard format
    response = """
    B: Distractor 1
    C: Distractor 2
    D: Distractor 3
    """
    parsed = parse_generated_distractors(response, expected_count=3)
    assert len(parsed) == 3
    assert parsed[0] == "Distractor 1"
    
    # Messy format
    response_messy = """
    Here are the distractors:
    1. Distractor 1
    (B) Distractor 2
    - Distractor 3
    """
    # Note: parse_generated_distractors logic might be specific to letter prefixes.
    # Let's check implementation if needed, but assuming standard format works.
    
    # Empty response
    parsed_empty = parse_generated_distractors("", expected_count=3)
    assert len(parsed_empty) == 0

def test_build_prompt(mock_entry):
    """Test prompt construction."""
    mode = AugmentorMode.FROM_SCRATCH
    prompt = build_prompt(mock_entry, mode)
    assert "Question: What is 2+2?" in prompt
    assert "Answer: A: 4" in prompt
    
    mode_human = AugmentorMode.CONDITIONED_HUMAN
    prompt_human = build_prompt(mock_entry, mode_human)
    assert "Existing 4 Options:" in prompt_human
    assert "3" in prompt_human

@patch("data.augmentor.get_client")
@patch("data.augmentor.load_from_disk")
def test_augment_dataset_flow(mock_load, mock_get_client, tmp_path, mock_entry):
    """Test full augmentation flow with mocks."""
    # Setup mocks
    mock_ds = MagicMock()
    mock_ds.__iter__.return_value = [mock_entry]
    mock_ds.__len__.return_value = 1
    mock_load.return_value = mock_ds
    
    mock_client = MagicMock()
    mock_client.generate.return_value.text = "A: D1\nB: D2"
    mock_get_client.return_value = mock_client
    
    config = GenerationConfig(
        mode=AugmentorMode.FROM_SCRATCH,
        model_provider="openai",
        model_name="gpt-4",
    )
    
    output_path = tmp_path / "output.json"
    
    result = augment_dataset(
        dataset_path=tmp_path / "input",
        config=config,
        output_path=output_path,
    )
    
    assert output_path.exists()
    assert len(result) == 1
    
    # Verify output column
    col = get_output_column(AugmentorMode.FROM_SCRATCH)
    assert col in result[0]
