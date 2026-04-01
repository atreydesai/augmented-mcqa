from types import SimpleNamespace

from cli.runtime import evaluation_logs_completed, generation_logs_succeeded


def _score(value: float, *, status: str = "success"):
    return SimpleNamespace(value=value, metadata={"status": status})


def _sample(scores):
    return SimpleNamespace(scores=scores)


def _log(samples, *, status: str = "success"):
    return SimpleNamespace(status=status, samples=samples)


def test_generation_logs_succeeded_requires_all_sample_scores_to_pass():
    logs = [
        _log(
            [
                _sample(
                    {
                        "first": _score(1.0),
                        "second": _score(0.0, status="failed"),
                    }
                )
            ]
        )
    ]

    assert generation_logs_succeeded(logs) is False


def test_evaluation_logs_completed_requires_scores_for_every_sample():
    logs = [_log([_sample({"only": _score(1.0)}), _sample({})])]

    assert evaluation_logs_completed(logs) is False
