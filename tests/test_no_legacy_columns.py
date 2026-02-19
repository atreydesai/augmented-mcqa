from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ["config", "data", "scripts", "experiments", "analysis", "models", "tests"]


def _iter_source_files():
    for dirname in SOURCE_DIRS:
        base = PROJECT_ROOT / dirname
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            yield path


def test_forbidden_model_columns_not_used_in_source():
    removed_col = "".join(["leg", "acy", "_choices_", "synthetic"])
    removed_enum = "_".join(["COND", "MODEL", "Q", "A"])
    removed_model_col = "_".join(["cond", "model", "q", "a"])

    forbidden_patterns = [
        re.compile(rf"\b{re.escape(removed_col)}\b"),
        re.compile(rf"\b{re.escape(removed_enum)}\b"),
        re.compile(rf"\b{re.escape(removed_model_col)}\b"),
    ]

    violations = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern.search(text):
                violations.append(f"{path}: {pattern.pattern}")

    assert not violations, "Found forbidden removed model-column references:\n" + "\n".join(violations)
