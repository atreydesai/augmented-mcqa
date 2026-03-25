from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path
from typing import Any

from inspect_ai.solver import Generate, TaskState, solver

from utils.constants import CHOICE_LABELS, GENERATION_RETRY_LIMIT, PROMPTS_DIR
from utils.parsing import LabeledParseError, format_choice_lines, parse_distractors
from utils.recipes import get_setting_recipe
from utils.scheduler_state import SCHEDULABLE_GENERATION_STRATEGIES


class GenerationParseError(LabeledParseError):
    def __init__(self, message: str, *, prompt: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.prompt = prompt
        self.attempts = attempts


def _load_prompt(name: str) -> str:
    return (Path(PROMPTS_DIR) / name).read_text(encoding="utf-8").strip()


def _prompt_template(name: str | None) -> str:
    if not name:
        return ""
    return _load_prompt(name)


def _stable_seed(sample_id: str, setting: str) -> int:
    digest = hashlib.sha256(f"{sample_id}:{setting}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _shuffle_options(sample_id: str, setting: str, answer: str, distractors: list[str]) -> tuple[list[str], str]:
    options = [answer, *distractors]
    rng = random.Random(_stable_seed(sample_id, setting))
    rng.shuffle(options)
    gold_index = options.index(answer)
    return options, CHOICE_LABELS[gold_index]


def _fresh_state(state: TaskState, prompt: str) -> TaskState:
    working_state = copy.deepcopy(state)
    working_state.user_prompt.text = prompt
    working_state.output.completion = ""
    return working_state


def _format_json_example(count: int) -> str:
    items = ",\n".join(f'    "incorrect answer choice {idx}"' for idx in range(1, count + 1))
    return '{\n  "distractors": [\n' + items + "\n  ]\n}"


async def _call_and_parse(
    *,
    state: TaskState,
    generate: Generate,
    prompt: str,
    count: int,
    forbidden: list[str],
) -> tuple[list[str], str, list[dict[str, Any]]]:
    last_error: Exception | None = None
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, GENERATION_RETRY_LIMIT + 1):
        working_state = _fresh_state(state, prompt)
        working_state = await generate(working_state)
        raw_output = str(working_state.output.completion or "")
        attempts.append({"attempt": attempt, "prompt": prompt, "output": raw_output})
        try:
            return parse_distractors(raw_output, count, forbidden=forbidden), raw_output, attempts
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise GenerationParseError(
        str(last_error or "generation failed"),
        prompt=prompt,
        attempts=attempts,
    )


@solver
def final5_generation_solver(strategy: str):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if strategy not in SCHEDULABLE_GENERATION_STRATEGIES:
            raise ValueError(f"Unknown schedulable generation strategy: {strategy}")

        recipe = get_setting_recipe(strategy)
        human_recipe = get_setting_recipe("human_from_scratch")
        metadata = dict(state.metadata or {})
        sample_id = str(metadata["sample_id"])
        question = str(metadata.get("question", "") or state.input_text)
        answer = str(metadata.get("answer", "")).strip()
        human = [str(item).strip() for item in list(metadata.get("choices_human") or []) if str(item).strip()]

        traces: dict[str, dict[str, Any]] = {}
        result: dict[str, Any] = {
            "status": "success",
            "sample_id": sample_id,
            "dataset_type": metadata.get("dataset_type"),
            "row_index": metadata.get("row_index"),
            "question": question,
            "answer": answer,
            "category": metadata.get("category", ""),
            "generation_strategy": strategy,
        }

        try:
            if len(human) < human_recipe.num_human:
                raise ValueError(f"choices_human must contain at least {human_recipe.num_human} distractors")
            human_distractors = human[: human_recipe.num_human]
            result["human_from_scratch"] = human_distractors
            randomized, correct = _shuffle_options(sample_id, "human_from_scratch", answer, human_distractors)
            result["human_from_scratch_options_randomized"] = randomized
            result["human_from_scratch_correct_answer_letter"] = correct
            traces["human_from_scratch"] = {
                "prompt": f"passthrough: choices_human[:{human_recipe.num_human}]",
                "output": "",
            }

            if recipe.prompt_mode == "qa":
                prompt = _prompt_template(recipe.prompt_template).format(
                    count=recipe.generated_count,
                    question=question,
                    gold_answer=answer,
                    json_example=_format_json_example(recipe.generated_count),
                )
                generated, raw_output, attempts = await _call_and_parse(
                    state=state,
                    generate=generate,
                    prompt=prompt,
                    count=recipe.generated_count,
                    forbidden=[answer],
                )
                result[strategy] = generated
                randomized, correct = _shuffle_options(sample_id, strategy, answer, generated)
                result[f"{strategy}_options_randomized"] = randomized
                result[f"{strategy}_correct_answer_letter"] = correct
                traces[strategy] = {"prompt": prompt, "output": raw_output, "attempts": attempts}
            elif recipe.prompt_mode == "conditioned":
                prerequisite_distractors = [
                    str(item).strip()
                    for item in list(metadata.get("existing_prerequisite_distractors") or [])
                    if str(item).strip()
                ]
                if recipe.conditioned_on == "human_from_scratch":
                    prior_distractors = human_distractors
                    prior_setting = "human_from_scratch"
                elif recipe.conditioned_on == "model_from_scratch":
                    prerequisite_recipe = get_setting_recipe("model_from_scratch")
                    if len(prerequisite_distractors) < prerequisite_recipe.num_model:
                        raise ValueError("augment_model requires existing model_from_scratch distractors")
                    prior_distractors = prerequisite_distractors[: prerequisite_recipe.num_model]
                    prior_setting = "model_from_scratch"
                    result[prior_setting] = prior_distractors
                    randomized, correct = _shuffle_options(sample_id, prior_setting, answer, prior_distractors)
                    result[f"{prior_setting}_options_randomized"] = randomized
                    result[f"{prior_setting}_correct_answer_letter"] = correct
                    traces[prior_setting] = {
                        "prompt": "prerequisite: existing model_from_scratch distractors from prior generation outputs",
                        "output": "\n".join(prior_distractors),
                    }
                else:
                    raise ValueError(f"Unsupported conditioned recipe source: {recipe.conditioned_on}")

                prompt = _prompt_template(recipe.prompt_template).format(
                    count=recipe.generated_count,
                    old_count=1 + len(prior_distractors),
                    question=question,
                    gold_answer=answer,
                    choices=format_choice_lines([answer, *prior_distractors]),
                    json_example=_format_json_example(recipe.generated_count),
                )
                generated, raw_output, attempts = await _call_and_parse(
                    state=state,
                    generate=generate,
                    prompt=prompt,
                    count=recipe.generated_count,
                    forbidden=[answer, *prior_distractors],
                )
                if recipe.conditioned_on == "human_from_scratch":
                    final_distractors = list(generated)
                    shuffle_distractors = [*human_distractors, *final_distractors]
                else:
                    final_distractors = [*prior_distractors, *generated]
                    shuffle_distractors = final_distractors
                result[strategy] = final_distractors
                randomized, correct = _shuffle_options(sample_id, strategy, answer, shuffle_distractors)
                result[f"{strategy}_options_randomized"] = randomized
                result[f"{strategy}_correct_answer_letter"] = correct
                traces[strategy] = {"prompt": prompt, "output": raw_output, "attempts": attempts}
            else:
                raise ValueError(f"Unsupported prompt_mode: {recipe.prompt_mode}")

            result["traces"] = traces
            state.output.completion = "generation-complete"
        except Exception as exc:  # noqa: BLE001
            result["status"] = "error"
            result["error"] = str(exc)
            if isinstance(exc, GenerationParseError):
                traces[strategy] = {
                    "prompt": exc.prompt,
                    "output": exc.attempts[-1]["output"] if exc.attempts else "",
                    "attempts": exc.attempts,
                }
            result["traces"] = traces
            state.output.completion = f"generation-error: {exc}"

        state.metadata["generation"] = result
        return state

    return solve
