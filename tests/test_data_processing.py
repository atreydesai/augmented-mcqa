import pytest
from data.mmlu_pro_processor import clean_whitespace, clean_options
from config import DATASET_SCHEMA, DatasetType

def test_clean_whitespace():
    """Test whitespace cleaning utility."""
    assert clean_whitespace("  hello  ") == "hello"
    assert clean_whitespace("\nhello\n") == "hello"
    assert clean_whitespace("hello") == "hello"
    assert clean_whitespace(None) is None

def test_clean_options():
    """Test cleaning a list of options."""
    opts = ["  A  ", "B", "  C"]
    cleaned = clean_options(opts)
    assert cleaned == ["A", "B", "C"]

def test_dataset_schema_integrity():
    """Verify that all schemas have required fields."""
    required_fields = ["question", "options", "num_options"]
    
    for dataset_type, schema in DATASET_SCHEMA.items():
        for field in required_fields:
            assert field in schema, f"{dataset_type} missing {field}"
            
        # Check specific constraints
        if dataset_type == DatasetType.MMLU_PRO:
            assert schema["num_options"] == 10
        elif dataset_type in [DatasetType.MMLU, DatasetType.ARC_EASY]:
            assert schema["num_options"] == 4
