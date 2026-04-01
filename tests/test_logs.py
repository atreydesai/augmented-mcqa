from types import SimpleNamespace

from utils.logs import _normalize_log_summary_from_object


def _score(value: float, *, status: str = "success"):
    return SimpleNamespace(value=value, metadata={"status": status})


def _sample(scores):
    return SimpleNamespace(scores=scores)


def test_log_summary_uses_worst_score_and_failure_status_per_sample():
    log = SimpleNamespace(
        eval=SimpleNamespace(metadata={"kind": "generation"}),
        stats=SimpleNamespace(completed_at="2026-01-01T00:00:00+00:00"),
        status="success",
        samples=[
            _sample(
                {
                    "pass": _score(1.0, status="success"),
                    "fail": _score(0.0, status="failed"),
                }
            )
        ],
    )

    summary = _normalize_log_summary_from_object(log)

    assert summary["score_values"] == [0.0]
    assert summary["sample_statuses"] == ["failed"]
    assert summary["summary_count"] == 1
