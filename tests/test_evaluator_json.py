import asyncio
from pathlib import Path

from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.solver import TaskState

from solvers.final5_evaluation import _evaluation_messages, _format_json_example, final5_evaluation_solver
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


def test_evaluation_messages_disable_reasoning_for_nemotron_only():
    nemotron_messages, nemotron_payload = _evaluation_messages(
        model="vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        prompt="Prompt text",
    )
    other_messages, other_payload = _evaluation_messages(
        model="vllm/Qwen/Qwen3-4B-Instruct-2507",
        prompt="Prompt text",
    )

    assert [message.role for message in nemotron_messages] == ["system", "user"]
    assert nemotron_messages[0].text == "/no_think"
    assert nemotron_messages[1].text == "Prompt text"
    assert nemotron_payload == [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "Prompt text"},
    ]

    assert [message.role for message in other_messages] == ["user"]
    assert other_messages[0].text == "Prompt text"
    assert other_payload == [{"role": "user", "content": "Prompt text"}]


def test_nemotron_evaluation_solver_injects_no_think_system_message():
    state = TaskState(
        model="vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        sample_id="sample-1",
        epoch=1,
        input="Original prompt",
        messages=[ChatMessageUser(content="Original prompt")],
        choices=["4", "5", "6", "7"],
        output=ModelOutput(model="vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2"),
        metadata={
            "question": "What is 2+2?",
            "evaluation_model": "vllm/nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        },
        store={},
    )

    async def fake_generate(current_state: TaskState) -> TaskState:
        assert [message.role for message in current_state.messages] == ["system", "user"]
        assert current_state.messages[0].text == "/no_think"
        current_state.output.completion = '{"answer": "A"}'
        return current_state

    solved = asyncio.run(final5_evaluation_solver("full_question")(state, fake_generate))

    assert solved.metadata["evaluation"]["prediction"] == "A"
    assert solved.metadata["evaluation"]["messages"][0] == {"role": "system", "content": "/no_think"}
    assert solved.metadata["evaluation"]["raw_output"] == '{"answer": "A"}'


def test_non_nemotron_evaluation_solver_preserves_single_user_prompt():
    state = TaskState(
        model="vllm/Qwen/Qwen3-4B-Instruct-2507",
        sample_id="sample-1",
        epoch=1,
        input="Original prompt",
        messages=[ChatMessageUser(content="Original prompt")],
        choices=["4", "5", "6", "7"],
        output=ModelOutput(model="vllm/Qwen/Qwen3-4B-Instruct-2507"),
        metadata={
            "question": "What is 2+2?",
            "evaluation_model": "vllm/Qwen/Qwen3-4B-Instruct-2507",
        },
        store={},
    )

    async def fake_generate(current_state: TaskState) -> TaskState:
        assert [message.role for message in current_state.messages] == ["user"]
        current_state.output.completion = '{"answer": "A"}'
        return current_state

    solved = asyncio.run(final5_evaluation_solver("full_question")(state, fake_generate))

    assert solved.metadata["evaluation"]["prediction"] == "A"
    assert solved.metadata["evaluation"]["messages"] == [
        {"role": "user", "content": solved.metadata["evaluation"]["prompt"]}
    ]
