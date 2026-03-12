from __future__ import annotations

from inspect_ai.scorer import Score, mean, scorer, stderr


@scorer(name="final5_generation", metrics=[mean(), stderr()])
def final5_generation_scorer():
    async def score(state, target):  # noqa: ANN001
        payload = dict(state.metadata.get("generation", {}) or {})
        success = payload.get("status") == "success"
        return Score(
            value=1.0 if success else 0.0,
            answer="ok" if success else "error",
            explanation=payload.get("error"),
            metadata=payload,
        )

    return score
