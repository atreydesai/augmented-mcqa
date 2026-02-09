import pytest
from pathlib import Path
from unittest.mock import patch
from data.augmentor import AugmentorMode, GenerationConfig, augment_dataset

@patch("data.augmentor.get_client")
@patch("data.augmentor.load_from_disk")
def test_end_to_end_augmentation(mock_load, mock_get_client, tmp_path):
    """
    Test the full augmentation pipeline with mocks.
    
    Flow:
    1. Load mock dataset
    2. Configure Augmentor
    3. Run augment_dataset
    4. Verify output file structure
    """
    # 1. Mock Data
    dataset_path = tmp_path / "processed_data"
    output_path = tmp_path / "augmented_data.json"
    
    mock_entry = {
        "question": "Integration test?",
        "gold_answer": "Yes",
        "options": ["Yes", "No"],
        "answer_letter": "A",
        "answer_index": 0,
    }
    
    mock_ds = [mock_entry]
    mock_load.return_value = mock_ds
    
    # 2. Mock Model
    mock_client = mock_get_client.return_value
    mock_client.generate.return_value.text = "A: Distractor1\nB: Distractor2"
    
    # 3. Configure
    config = GenerationConfig(
        mode=AugmentorMode.FROM_SCRATCH,
        model_provider="local",
        model_name="test-model",
    )
    
    # 4. Run
    result = augment_dataset(
        dataset_path=dataset_path,
        config=config,
        output_path=output_path,
    )
    
    # 5. Verify
    assert output_path.exists()
    assert len(result) == 1
    
    # Check if new column added
    from config import DistractorType
    assert DistractorType.COND_MODEL_Q_A_SCRATCH.value in result[0]
    
    # Check values
    generated = result[0][DistractorType.COND_MODEL_Q_A_SCRATCH.value]
    assert len(generated) > 0  # Should have parsed something
