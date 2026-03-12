from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path
from typing import Any

from inspect_ai.solver import Generate, TaskState, solver

from utils.constants import CHOICE_LABELS, GENERATION_RETRY_LIMIT, PROMPTS_DIR
from utils.parsing import LabeledParseError, format_choice_lines, parse_labeled_distractors
from utils.scheduler_state import SCHEDULABLE_GENERATION_STRATEGIES


def _load_prompt(name: str) -> str:
    return (Path(PROMPTS_DIR) / name).read_text(encoding="utf-8").strip()


GENERATE_QA_PROMPT = _load_prompt("generate_qa.txt")
GENERATE_CONDITIONED_PROMPT = _load_prompt("generate_conditioned.txt")


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


def _format_json_example(labels: list[str]) -> str:
    return "{\n" + ",\n".join(f'  "{label}": "text for distractor {label}"' for label in labels) + "\n}"


def _format_forbidden_values(values: list[str]) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return "(none)"
    return "\n".join(f"- {value}" for value in cleaned)


def _retry_prompt(base_prompt: str, *, labels: list[str], forbidden: list[str], error: str, raw_output: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "<retry instructions>\n"
        f"Your previous response was invalid: {error}\n"
        f"You must return a fresh valid JSON object with exactly these keys: {', '.join(labels)}.\n"
        "Do not repeat any forbidden value.\n"
        "Do not include commentary, apologies, markdown fences, or any text outside the JSON object.\n"
        "</retry instructions>\n\n"
        "<forbidden values reminder>\n"
        f"{_format_forbidden_values(forbidden)}\n"
        "</forbidden values reminder>\n\n"
        "<previous invalid response>\n"
        f"{raw_output}\n"
        "</previous invalid response>"
    )


async def _call_and_parse(
    *,
    state: TaskState,
    generate: Generate,
    prompt: str,
    labels: list[str],
    forbidden: list[str],
) -> tuple[list[str], str, list[dict[str, Any]]]:
    last_error: Exception | None = None
    current_prompt = prompt
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, GENERATION_RETRY_LIMIT + 1):
        working_state = _fresh_state(state, current_prompt)
        working_state = await generate(working_state)
        raw_output = str(working_state.output.completion or "")
        attempts.append({"attempt": attempt, "prompt": current_prompt, "output": raw_output})
        try:
            return parse_labeled_distractors(raw_output, labels, forbidden=forbidden), raw_output, attempts
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < GENERATION_RETRY_LIMIT:
                current_prompt = _retry_prompt(
                    prompt,
                    labels=labels,
                    forbidden=forbidden,
                    error=str(exc),
                    raw_output=raw_output,
                )
    raise LabeledParseError(str(last_error or "generation failed"))


@solver
def final5_generation_solver(strategy: str):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if strategy not in SCHEDULABLE_GENERATION_STRATEGIES:
            raise ValueError(f"Unknown schedulable generation strategy: {strategy}")

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
            if len(human) < 3:
                raise ValueError("choices_human must contain at least 3 distractors")
            human3 = human[:3]
            result["human_from_scratch"] = human3
            randomized, correct = _shuffle_options(sample_id, "human_from_scratch", answer, human3)
            result["human_from_scratch_options_randomized"] = randomized
            result["human_from_scratch_correct_answer_letter"] = correct
            traces["human_from_scratch"] = {
                "prompt": "passthrough: choices_human[:3]",
                "output": "",
            }

            if strategy == "model_from_scratch":
                prompt_b = GENERATE_QA_PROMPT.format(
                    count=3,
                    labels="B, C, D",
                    question=question,
                    gold_answer=answer,
                    json_example=_format_json_example(["B", "C", "D"]),
                    forbidden_values=_format_forbidden_values([answer]),
                )
                model3, raw_b, attempts_b = await _call_and_parse(
                    state=state,
                    generate=generate,
                    prompt=prompt_b,
                    labels=["B", "C", "D"],
                    forbidden=[answer],
                )
                result["model_from_scratch"] = model3
                randomized, correct = _shuffle_options(sample_id, "model_from_scratch", answer, model3)
                result["model_from_scratch_options_randomized"] = randomized
                result["model_from_scratch_correct_answer_letter"] = correct
                traces["model_from_scratch"] = {"prompt": prompt_b, "output": raw_b, "attempts": attempts_b}
            elif strategy == "augment_human":
                prompt_c = GENERATE_CONDITIONED_PROMPT.format(
                    count=6,
                    labels="E, F, G, H, I, J",
                    question=question,
                    gold_answer=answer,
                    existing_options_block=format_choice_lines([answer, *human3]),
                    json_example=_format_json_example(["E", "F", "G", "H", "I", "J"]),
                    forbidden_values=_format_forbidden_values([answer, *human3]),
                )
                delta_human, raw_c, attempts_c = await _call_and_parse(
                    state=state,
                    generate=generate,
                    prompt=prompt_c,
                    labels=["E", "F", "G", "H", "I", "J"],
                    forbidden=[answer, *human3],
                )
                result["augment_human"] = delta_human
                randomized, correct = _shuffle_options(sample_id, "augment_human", answer, human3 + delta_human)
                result["augment_human_options_randomized"] = randomized
                result["augment_human_correct_answer_letter"] = correct
                traces["augment_human"] = {"prompt": prompt_c, "output": raw_c, "attempts": attempts_c}
            elif strategy == "augment_model":
                model3 = [str(item).strip() for item in list(metadata.get("existing_model_from_scratch") or []) if str(item).strip()]
                if len(model3) < 3:
                    raise ValueError("augment_model requires existing model_from_scratch distractors")
                model3 = model3[:3]
                result["model_from_scratch"] = model3
                randomized, correct = _shuffle_options(sample_id, "model_from_scratch", answer, model3)
                result["model_from_scratch_options_randomized"] = randomized
                result["model_from_scratch_correct_answer_letter"] = correct
                traces["model_from_scratch"] = {
                    "prompt": "prerequisite: existing_model_from_scratch from prior generation logs",
                    "output": "\n".join(model3),
                }

                prompt_d = GENERATE_CONDITIONED_PROMPT.format(
                    count=6,
                    labels="E, F, G, H, I, J",
                    question=question,
                    gold_answer=answer,
                    existing_options_block=format_choice_lines([answer, *model3]),
                    json_example=_format_json_example(["E", "F", "G", "H", "I", "J"]),
                    forbidden_values=_format_forbidden_values([answer, *model3]),
                )
                delta_model, raw_d, attempts_d = await _call_and_parse(
                    state=state,
                    generate=generate,
                    prompt=prompt_d,
                    labels=["E", "F", "G", "H", "I", "J"],
                    forbidden=[answer, *model3],
                )
                combined_model = model3 + delta_model
                result["augment_model"] = combined_model
                randomized, correct = _shuffle_options(sample_id, "augment_model", answer, combined_model)
                result["augment_model_options_randomized"] = randomized
                result["augment_model_correct_answer_letter"] = correct
                traces["augment_model"] = {"prompt": prompt_d, "output": raw_d, "attempts": attempts_d}
            else:
                prompt_e = GENERATE_QA_PROMPT.format(
                    count=9,
                    labels="B, C, D, E, F, G, H, I, J",
                    question=question,
                    gold_answer=answer,
                    json_example=_format_json_example(["B", "C", "D", "E", "F", "G", "H", "I", "J"]),
                    forbidden_values=_format_forbidden_values([answer]),
                )
                ablation, raw_e, attempts_e = await _call_and_parse(
                    state=state,
                    generate=generate,
                    prompt=prompt_e,
                    labels=["B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    forbidden=[answer],
                )
                result["augment_ablation"] = ablation
                randomized, correct = _shuffle_options(sample_id, "augment_ablation", answer, ablation)
                result["augment_ablation_options_randomized"] = randomized
                result["augment_ablation_correct_answer_letter"] = correct
                traces["augment_ablation"] = {"prompt": prompt_e, "output": raw_e, "attempts": attempts_e}

            result["traces"] = traces
            state.output.completion = "generation-complete"
        except Exception as exc:  # noqa: BLE001
            result["status"] = "error"
            result["error"] = str(exc)
            result["traces"] = traces
            state.output.completion = f"generation-error: {exc}"

        state.metadata["generation"] = result
        return state

    return solve
