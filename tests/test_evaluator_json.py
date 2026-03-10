from pathlib import Path

from utils.parsing import extract_answer_letter


def test_extract_answer_letter_accepts_plain_letter_outputs():
    assert extract_answer_letter("A", "ABCD") == "A"
    assert extract_answer_letter("Answer: C", "ABCD") == "C"
    assert extract_answer_letter("The answer is (D)", "ABCD") == "D"


def test_plain_text_evaluation_prompt_contains_no_json_contract():
    prompt = (Path("prompts") / "evaluate_full_question.txt").read_text(encoding="utf-8")
    assert "JSON" not in prompt.upper()
    assert "Return only one uppercase letter" in prompt
