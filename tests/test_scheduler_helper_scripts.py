from pathlib import Path


def test_generate_helper_script_exposes_editable_variables():
    path = Path("jobs/submit_generate_scheduler.sh")
    text = path.read_text(encoding="utf-8")

    assert 'RUN_NAME="gen_scheduler_run"' in text
    assert 'GENERATION_STRATEGIES="model_from_scratch,augment_human,augment_model,augment_ablation"' in text
    assert '--questions-per-job "$QUESTIONS_PER_JOB"' in text
    assert 'submit-generate-cluster' in text
    assert 'bash "$submit_script"' in text


def test_evaluate_helper_script_exposes_editable_variables():
    path = Path("jobs/submit_evaluate_scheduler.sh")
    text = path.read_text(encoding="utf-8")

    assert 'RUN_NAME="eval_scheduler_run"' in text
    assert 'SETTINGS="human_from_scratch,model_from_scratch,augment_human,augment_model,augment_ablation"' in text
    assert 'MODES="full_question,choices_only"' in text
    assert '--questions-per-job "$QUESTIONS_PER_JOB"' in text
    assert 'submit-evaluate-cluster' in text
    assert 'bash "$submit_script"' in text
