import json

import pytest

import analysis.visualize as viz
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


def test_branching_plot_anchors_blue_lines_and_title_labels(monkeypatch, tmp_path):
    class _FakeAxis:
        def __init__(self):
            self.plots = []
            self.title = ""
            self.transAxes = object()

        def plot(self, x, y, **kwargs):
            self.plots.append({"x": list(x), "y": list(y), **kwargs})
            return []

        def text(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_title(self, title, *args, **kwargs):
            self.title = title
            return None

        def set_xticks(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

    class _FakeFigure:
        pass

    fake_ax = _FakeAxis()
    monkeypatch.setattr(
        viz,
        "_detect_available_configs",
        lambda *_args, **_kwargs: {"arc_challenge": ["scratch", "dhuman"]},
    )
    monkeypatch.setattr(viz.plt, "subplots", lambda *args, **kwargs: (_FakeFigure(), fake_ax))
    monkeypatch.setattr(viz.plt, "tight_layout", lambda: None)
    monkeypatch.setattr(viz.plt, "close", lambda *_args, **_kwargs: None)

    def _fake_collect_xy(_base_dir, _model, _dataset_type, _source, configs, x_values=None):
        if configs == [(1, 0), (2, 0), (3, 0)]:
            return [1, 2, 3], [0.91, 0.82, 0.73]
        if configs == [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]:
            return [1, 2, 3, 4, 5, 6], [0.89, 0.84, 0.8, 0.75, 0.71, 0.67]
        if configs == [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]:
            return [2, 3, 4, 5, 6], [0.88, 0.85, 0.83, 0.8, 0.78]
        if configs == [(2, 1), (2, 2), (2, 3), (2, 4)]:
            return [3, 4, 5, 6], [0.8, 0.77, 0.75, 0.72]
        if configs == [(3, 1), (3, 2), (3, 3)]:
            return [4, 5, 6], [0.71, 0.69, 0.66]
        return [], []

    monkeypatch.setattr(viz, "_collect_xy", _fake_collect_xy)

    viz.plot_branching_comparison(
        base_dir=tmp_path,
        model="fake-local-model",
        output_dir=None,
        show=False,
        dataset_type="arc_challenge",
        generator_model="fake-generator-model",
        evaluation_model="fake-eval-model",
    )

    assert len(fake_ax.plots) >= 5
    red = next(p for p in fake_ax.plots if p.get("label", "").startswith("Human branch"))
    blue_m0 = next(p for p in fake_ax.plots if p.get("label") == "M0_1..M0_k (no human prefix)")
    blue_h1 = next(p for p in fake_ax.plots if p.get("label") == "D1 + M1_1..M1_k")
    blue_h2 = next(p for p in fake_ax.plots if p.get("label") == "D1+D2 + M2_1..M2_k")
    blue_h3 = next(p for p in fake_ax.plots if p.get("label") == "D1+D2+D3 + M3_1..M3_k")

    assert red["x"] == [1, 2, 3]
    assert blue_m0["x"] == [1, 2, 3, 4, 5, 6]
    assert blue_m0["linestyle"] == "-"
    assert blue_h1["x"][0] == 1 and blue_h1["y"][0] == 0.91
    assert blue_h1["linestyle"] == ":"
    assert blue_h2["x"][0] == 2 and blue_h2["y"][0] == 0.82
    assert blue_h3["x"][0] == 3 and blue_h3["y"][0] == 0.73
    assert "Gen: fake-generator-model | Eval: fake-eval-model" in fake_ax.title


def test_branching_plot_prefers_scratch_source_for_m0_when_available(monkeypatch, tmp_path):
    calls = []

    class _FakeAxis:
        def __init__(self):
            self.transAxes = object()

        def plot(self, *args, **kwargs):
            return []

        def text(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def set_xticks(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

    class _FakeFigure:
        pass

    monkeypatch.setattr(
        viz,
        "_detect_available_configs",
        lambda *_args, **_kwargs: {"arc_challenge": ["scratch", "dhuman"]},
    )
    monkeypatch.setattr(viz.plt, "subplots", lambda *args, **kwargs: (_FakeFigure(), _FakeAxis()))
    monkeypatch.setattr(viz.plt, "tight_layout", lambda: None)
    monkeypatch.setattr(viz.plt, "close", lambda *_args, **_kwargs: None)

    def _fake_collect_xy(_base_dir, _model, _dataset_type, source, configs, x_values=None):
        calls.append((source, tuple(configs)))
        if configs == [(1, 0), (2, 0), (3, 0)]:
            return [1, 2, 3], [0.9, 0.8, 0.7]
        if configs == [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]:
            return [1, 2, 3, 4, 5, 6], [0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        if configs == [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]:
            return [2, 3, 4, 5, 6], [0.78, 0.76, 0.74, 0.72, 0.7]
        if configs == [(2, 1), (2, 2), (2, 3), (2, 4)]:
            return [3, 4, 5, 6], [0.69, 0.67, 0.65, 0.63]
        if configs == [(3, 1), (3, 2), (3, 3)]:
            return [4, 5, 6], [0.6, 0.58, 0.56]
        return [], []

    monkeypatch.setattr(viz, "_collect_xy", _fake_collect_xy)

    viz.plot_branching_comparison(
        base_dir=tmp_path,
        model="fake-local-model",
        output_dir=None,
        show=False,
        dataset_type="arc_challenge",
        generator_model="gen",
        evaluation_model="eval",
    )

    # M0 line should use scratch source when it is available.
    assert ("scratch", ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6))) in calls


def test_branching_plot_raises_when_required_sources_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        viz,
        "_detect_available_configs",
        lambda *_args, **_kwargs: {"arc_challenge": ["dhuman"]},
    )

    with pytest.raises(ValueError, match="missing required sources"):
        viz.plot_branching_comparison(
            base_dir=tmp_path,
            model="fake-local-model",
            output_dir=None,
            show=False,
            dataset_type="arc_challenge",
            generator_model="gen",
            evaluation_model="eval",
        )
