import pytest

from sentineledu.config import PROJECT_ROOT
from sentineledu.schemas.evaluation import CriterionEvaluation, EvaluationResult, QuestionEvaluation
from sentineledu.services.rubric import load_rubric
from sentineledu.services.storage import NotFoundError, Storage

SAMPLE = PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json"


@pytest.fixture
def storage(tmp_path):
    return Storage(storage_dir=tmp_path / "storage")


@pytest.fixture
def submission(storage, tmp_path):
    page = tmp_path / "p1.jpg"
    page.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
    return storage.create_submission("photosynthesis_basics", [page], student_ref="anon-1")


def result(submission_id: str, total: float, model="gemini-3.5-flash") -> EvaluationResult:
    q = QuestionEvaluation(question_id="Q1", criteria=[CriterionEvaluation(criterion_id="C1", selected_level_id="full",
                                                                            marks=total, evidence_verified=True)],
                           total=total, max_marks=3, sbert_sim_model_answer=0.5, sbert_sim_best_chunk=0.4)
    return EvaluationResult(submission_id=submission_id, questions=[q], total=total, max_total=7,
                            prompt_version="evaluation_v1", model_id=model, extraction_version=1)


def test_rubric_versions(storage):
    r = load_rubric(SAMPLE)
    v1 = storage.save_rubric(r)
    v2 = storage.save_rubric(r.model_copy(update={"title": "edited"}))
    assert (v1.version, v2.version) == (1, 2)
    assert storage.get_rubric(r.assignment_id).title == "edited"
    assert storage.get_rubric(r.assignment_id, 1).title == r.title
    assert storage.list_assignments() == [r.assignment_id]
    with pytest.raises(NotFoundError):
        storage.get_rubric("nope")
    with pytest.raises(NotFoundError):
        storage.get_rubric(r.assignment_id, 9)


def test_evaluations_are_appended_not_overwritten(storage, submission):
    assert storage.latest_evaluation(submission.submission_id) is None
    a = storage.save_evaluation(result(submission.submission_id, 2))
    b = storage.save_evaluation(result(submission.submission_id, 3, model="gemini-3.7-flash"))
    assert a.evaluation_id and b.evaluation_id and a.evaluation_id != b.evaluation_id
    assert storage.get_evaluation(a.evaluation_id).total == 2
    assert storage.latest_evaluation(submission.submission_id).model_id == "gemini-3.7-flash"
    assert [e.total for e in storage.list_evaluations(submission.submission_id)] == [2, 3]


def test_evaluation_needs_known_submission(storage):
    with pytest.raises(NotFoundError):
        storage.save_evaluation(result("ghost", 1))
    with pytest.raises(NotFoundError):
        storage.get_evaluation("ghost")
