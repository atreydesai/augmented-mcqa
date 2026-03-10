from importlib import import_module


def test_merge_script_is_no_op_under_inspect_first_refactor(capsys):
    mod = import_module("scripts.06_merge_eval_subshards")
    assert mod.main() == 0
    captured = capsys.readouterr()
    assert "No merge step is required" in captured.out
