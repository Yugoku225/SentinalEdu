import json

import pytest

from sentineledu.config import PROJECT_ROOT
from sentineledu.schemas.rubric import Rubric
from sentineledu.services.rubric import (
    RubricError,
    from_simple_model_answer,
    load_rubric,
    rubric_from_legacy_keys,
    rubric_summary,
    save_rubric,
    validate_rubric,
)

SAMPLE = PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json"


def test_sample_rubric_loads():
    r = load_rubric(SAMPLE)
    assert [q.question_id for q in r.questions] == ["Q1", "Q2", "Q3"]
    assert r.max_total == 7
    assert r.question("Q1").criterion("C1").level("partial").marks == 1
    assert "Q1 [3]" in rubric_summary(r)


def test_sample_rubric_questions_match_the_test_set():
    manifest = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "synthetic" / "manifest.json").read_text())
    r = load_rubric(SAMPLE)
    assert [q["question_id"] for q in manifest["questions"]] == [q.question_id for q in r.questions]
    assert [q["prompt"] for q in manifest["questions"]] == [q.prompt for q in r.questions]


def test_invalid_rubric_gives_readable_error():
    data = json.loads(SAMPLE.read_text())
    data["questions"][0]["criteria"][0]["levels"] = [
        {"level_id": "full", "label": "f", "marks": 2, "descriptor": "d"},
        {"level_id": "half", "label": "h", "marks": 1, "descriptor": "d"},
    ]
    with pytest.raises(RubricError) as e:
        validate_rubric(data)
    msg = str(e.value)
    assert msg.startswith("Invalid rubric:")
    assert "questions.0.criteria.0" in msg
    assert "needs a 0-mark level" in msg


def test_sum_mismatch_reported_with_location():
    data = json.loads(SAMPLE.read_text())
    data["questions"][1]["max_marks"] = 5
    with pytest.raises(RubricError, match=r"questions\.1.*add up to 2.*max_marks is 5"):
        validate_rubric(data)


def test_malformed_json_and_missing_file(tmp_path):
    with pytest.raises(RubricError, match="Invalid rubric"):
        validate_rubric("{not json")
    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "nope.json")


def test_from_simple_model_answer_builds_three_levels():
    r = from_simple_model_answer("The quick brown fox jumps over the lazy dog.", 10, assignment_id="legacy",
                                 question_id="sample1")
    q = r.questions[0]
    assert q.max_marks == 10 and q.model_answer.startswith("The quick")
    c = q.criteria[0]
    assert [(lv.level_id, lv.marks) for lv in c.levels] == [("full", 10), ("partial", 5), ("none", 0)]
    assert c.lowest_level.level_id == "none"


def test_partial_level_rounds_to_half_marks():
    r = from_simple_model_answer("x y z", 7)
    assert r.questions[0].criteria[0].level("partial").marks == 3.5
    r = from_simple_model_answer("x y z", 1)
    assert r.questions[0].criteria[0].level("partial").marks == 0.5


def test_simple_rubric_rejects_bad_input():
    with pytest.raises(RubricError, match="empty"):
        from_simple_model_answer("   ", 10)
    with pytest.raises(RubricError, match="positive"):
        from_simple_model_answer("ok", 0)


def test_legacy_keys_become_one_rubric():
    keys = json.loads((PROJECT_ROOT / "data" / "model_answers.json").read_text())
    r = rubric_from_legacy_keys(keys, 10)
    assert [q.question_id for q in r.questions] == list(keys)
    assert r.max_total == 10 * len(keys)


def test_save_and_reload_roundtrip(tmp_path):
    r = load_rubric(SAMPLE)
    p = save_rubric(r, tmp_path / "sub" / "r.json")
    assert load_rubric(p) == r
    assert isinstance(validate_rubric(r), Rubric)
