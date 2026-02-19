import json

from analysis.visualize import load_results_file


def test_visualization_loader_ignores_extra_eval_trace_fields(tmp_path):
    payload = {
        "summary": {
            "accuracy": 0.5,
            "correct": 1,
            "total": 2,
        },
        "results": [
            {
                "question_idx": 0,
                "question": "q",
                "gold_answer": "a",
                "gold_index": 0,
                "model_answer": "A",
                "model_prediction": "A",
                "is_correct": True,
                "prediction_type": "G",
                "eval_options_randomized": ["a", "b", "c", "d"],
                "eval_correct_answer_letter": "A",
                "eval_full_question": "Question: q",
                "eval_model_input": "prompt",
                "eval_model_output": "The answer is (A)",
                "selected_human_distractors": ["b"],
                "selected_model_distractors": ["c", "d"],
                "human_option_indices": [1],
                "model_option_indices": [2, 3],
            }
        ],
    }

    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_results_file(path)
    assert loaded == {"accuracy": 0.5, "correct": 1, "total": 2}
