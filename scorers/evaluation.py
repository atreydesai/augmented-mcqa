from __future__ import annotations

from inspect_ai.scorer import Score, mean, scorer, stderr


def _prediction_type(
    prediction: str,
    gold_index: int,
    human_indices: list[int],
    model_indices: list[int],
) -> str:
    if not prediction:
        return "?"
    predicted_index = ord(prediction) - ord("A")
    if predicted_index == gold_index:
        return "G"
    if predicted_index in human_indices:
        return "H"
    if predicted_index in model_indices:
        return "M"
    return "?"


@scorer(name="final5_eval", metrics=[mean(), stderr()])
def final5_evaluation_scorer():
    async def score(state, target):  # noqa: ANN001
        target_letter = str(target.text or "").strip().upper()
        evaluation = dict(state.metadata.get("evaluation", {}) or {})
        prediction = str(evaluation.get("prediction", "") or "").strip().upper()
        gold_index = int(state.metadata.get("gold_index", -1))
        human_indices = [int(i) for i in state.metadata.get("human_option_indices", [])]
        model_indices = [int(i) for i in state.metadata.get("model_option_indices", [])]
        is_correct = bool(prediction) and prediction == target_letter
        metadata = {
            "sample_id": state.metadata.get("sample_id"),
            "dataset_type": state.metadata.get("dataset_type"),
            "question_idx": int(state.metadata.get("row_index", -1)),
            "category": state.metadata.get("category", ""),
            "setting": state.metadata.get("setting"),
            "mode": state.metadata.get("mode"),
            "prediction": prediction,
            "prediction_type": _prediction_type(prediction, gold_index, human_indices, model_indices),
            "gold_answer_letter": target_letter,
            "gold_index": gold_index,
            "selected_human_distractors": list(state.metadata.get("selected_human_distractors", [])),
            "selected_model_distractors": list(state.metadata.get("selected_model_distractors", [])),
            "human_option_indices": human_indices,
            "model_option_indices": model_indices,
            "prompt": evaluation.get("prompt", ""),
            "raw_output": evaluation.get("raw_output", ""),
        }
        return Score(
            value=1.0 if is_correct else 0.0,
            answer=prediction or None,
            explanation=evaluation.get("raw_output", ""),
            metadata=metadata,
        )

    return score
