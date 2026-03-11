from __future__ import annotations

from pathlib import Path

from inspect_ai.solver import Generate, TaskState, solver

from utils.constants import CHOICE_LABELS, PROMPTS_DIR
from utils.parsing import extract_answer_letter_from_json, format_choice_lines


def _load_prompt(name: str) -> str:
    return (Path(PROMPTS_DIR) / name).read_text(encoding="utf-8").strip()


EVAL_FULL_PROMPT = _load_prompt("evaluate_full_question.txt")
EVAL_CHOICES_PROMPT = _load_prompt("evaluate_choices_only.txt")


def _format_letters(valid_letters: str) -> str:
    return ", ".join(valid_letters)


def _format_json_example(valid_letters: str) -> str:
    choices = " | ".join(f'"{letter}"' for letter in valid_letters) or '"A"'
    return "{\n  \"answer\": " + choices + "\n}"


@solver
def final5_evaluation_solver(mode: str):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        question = str(state.metadata.get("question", "") or state.input_text)
        choices = [getattr(choice, "value", choice) for choice in state.choices]
        valid_letters = CHOICE_LABELS[: len(choices)]
        template = EVAL_CHOICES_PROMPT if mode == "choices_only" else EVAL_FULL_PROMPT
        prompt = template.format(
            question=question,
            choices=format_choice_lines(choices),
            letters=_format_letters(valid_letters),
            json_example=_format_json_example(valid_letters),
        )
        state.user_prompt.text = prompt
        state = await generate(state)
        response_text = str(state.output.completion or "")
        prediction = extract_answer_letter_from_json(response_text, valid_letters)
        state.metadata["evaluation"] = {
            "prompt": prompt,
            "raw_output": response_text,
            "prediction": prediction,
        }
        return state

    return solve
