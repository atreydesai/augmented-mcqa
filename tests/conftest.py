import pytest
from pathlib import Path
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DistractorType,
    DatasetType,
    DATASET_SCHEMA,
)

@pytest.fixture
def mock_entry_mmlu_pro():
    """Sample MMLU-Pro entry."""
    return {
        "question": "What is 2+2?",
        "options": ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
        "answer": "B",
        "answer_index": 1,
        "category": "math",
        "src": "mmlu",
        DistractorType.COND_HUMAN_Q_A.value: ["3", "5", "6"], # 3 human
        DistractorType.COND_MODEL_Q_A.value: ["7", "8", "9", "10", "11", "12"], # 6 synth
    }

@pytest.fixture
def mock_entry_arc():
    """Sample ARC entry."""
    return {
        "id": "1",
        "question": "Sky color?",
        "options": ["Blue", "Red", "Green", "Yellow"],
        "answerKey": "A",
        "gold_answer": "Blue",
        DistractorType.COND_HUMAN_Q_A.value: ["Red", "Green", "Yellow"],
    }

@pytest.fixture
def mock_dataset_path(tmp_path):
    """Create a dummy dataset file."""
    import json
    path = tmp_path / "dataset.json"
    data = [
        {"question": "Q1", "answer": "A", "options": ["A", "B"]},
        {"question": "Q2", "answer": "B", "options": ["A", "B"]},
    ]
    with open(path, "w") as f:
        json.dump(data, f)
    return path

@pytest.fixture
def mock_model_client():
    """Mock model client."""
    client = MagicMock()
    client.name = "mock-model"
    client.model_id = "mock-model-id"
    
    # Mock generation result
    mock_result = MagicMock()
    mock_result.text = "Generated text"
    mock_result.extract_answer.return_value = "A"
    
    client.generate.return_value = mock_result
    return client
