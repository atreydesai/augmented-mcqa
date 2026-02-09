import pytest
import os
from pathlib import Path
from config import (
    DATASETS_DIR,
    RAW_DATASETS_DIR,
    PROCESSED_DATASETS_DIR,
    AUGMENTED_DATASETS_DIR,
    DistractorType,
    get_distractor_column,
)

def test_path_constants():
    """Verify path constants are set correctly."""
    assert isinstance(DATASETS_DIR, Path)
    assert isinstance(RAW_DATASETS_DIR, Path)
    assert isinstance(PROCESSED_DATASETS_DIR, Path)
    assert isinstance(AUGMENTED_DATASETS_DIR, Path)
    
    assert RAW_DATASETS_DIR == DATASETS_DIR / "raw"
    assert PROCESSED_DATASETS_DIR == DATASETS_DIR / "processed"
    assert AUGMENTED_DATASETS_DIR == DATASETS_DIR / "augmented"

def test_distractor_type_enum():
    """Verify DistractorType enum values."""
    assert DistractorType.COND_HUMAN_Q_A.value == "cond_human_q_a"
    assert DistractorType.COND_MODEL_Q_A.value == "cond_model_q_a"
    assert DistractorType.COND_MODEL_Q_A_SCRATCH.value == "cond_model_q_a_scratch"
    assert DistractorType.COND_MODEL_Q_A_DHUMAN.value == "cond_model_q_a_dhuman"
    assert DistractorType.COND_MODEL_Q_A_DMODEL.value == "cond_model_q_a_dmodel"

def test_get_distractor_column():
    """Verify distractor column retrieval."""
    entry = {
        "cond_human_q_a": ["h1", "h2"],
        "cond_model_q_a": ["m1", "m2"],
        "choices_human": ["legacy_h1"],
    }
    
    # Test correct unified access
    assert get_distractor_column(entry, DistractorType.COND_HUMAN_Q_A) == ["h1", "h2"]
    
    # Test legacy fallback
    legacy_entry = {"choices_human": ["legacy_h1"]}
    assert get_distractor_column(legacy_entry, DistractorType.COND_HUMAN_Q_A) == ["legacy_h1"]
    
    # Test empty
    assert get_distractor_column({}, DistractorType.COND_HUMAN_Q_A) == []
