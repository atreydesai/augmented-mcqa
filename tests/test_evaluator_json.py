from pathlib import Path

from solvers.final5_evaluation import _format_json_example
from utils.parsing import extract_answer_letter_from_json


def test_extract_answer_letter_from_json_accepts_valid_json_outputs():
    assert extract_answer_letter_from_json('{"answer": "a"}', "ABCD") == "A"
    assert extract_answer_letter_from_json("```json\n{\"answer\": \"D\"}\n```", "ABCD") == "D"


def test_evaluation_json_example_lists_all_valid_letters():
    assert '"A" | "B" | "C" | "D"' in _format_json_example("ABCD")


def test_evaluate_full_question_prompt_uses_xml_structure_and_json_contract():
    prompt = (Path("prompts") / "evaluate_full_question.txt").read_text(encoding="utf-8")
    assert "<question>" in prompt
    assert "<choices>" in prompt
    assert "<format>" in prompt
    assert 'key "answer"' in prompt
    assert "valid JSON" in prompt


def test_evaluate_choices_only_prompt_uses_xml_structure_and_json_contract():
    prompt = (Path("prompts") / "evaluate_choices_only.txt").read_text(encoding="utf-8")
    assert "<choices>" in prompt
    assert "<format>" in prompt
    assert 'key "answer"' in prompt
    assert "valid JSON" in prompt
