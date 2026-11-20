"""Integrity checks inside the workflow (SENTINELEDU_TASKS.md 7.1 and 7.4). Offline, fake agents."""
import copy

from sentineledu.schemas.submission import SubmissionStatus as S
from sentineledu.workflow.assessment_workflow import assess_files
from fakes import FakeAgent
from wf_helpers import PAGE, RUBRIC, doc_draft, eval_draft, judge_draft, make_deps

PAGE2 = PAGE.with_name("syn_03_blank_out_of_order_p1.jpg")
QUIRKY = ("Plants drink water mostly so their stems do not flop over like a wet noodle, and I think the water also "
          "travels up like a lift carrying food salts to every leaf.")
PARAPHRASE = ("Plants take in water mainly so the stem does not flop over like a soggy noodle, and the water also goes up "
              "like an elevator carrying food salts to all the leaves.")


def doc_with_q3(text: str) -> dict:
    d = copy.deepcopy(doc_draft())
    d["answers"][2]["answer_text"] = text
    return d


def eval_for_q3() -> dict:
    d = copy.deepcopy(eval_draft())
    d["questions"][2]["criteria"][0]["evidence_quotes"] = ["carrying food salts to"]
    d["questions"][2]["criteria"][0]["selected_level_id"] = "partial"
    return d


def marks(deps, sid):
    ev = deps.storage.latest_evaluation(sid)
    rep = deps.storage.latest_judge_report(sid)
    return ({q.question_id: {c.criterion_id: c.marks for c in q.criteria} for q in ev.questions}, ev.total,
            rep.recommended_marks(), rep.recommended_total)


def test_same_file_twice_is_flagged_and_gemini_reads_it_once(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft()], ev=[eval_draft(), eval_draft()], jd=[judge_draft(), judge_draft()])
    first = assess_files(RUBRIC.assignment_id, [PAGE], deps, student_ref="S1")
    second = assess_files(RUBRIC.assignment_id, [PAGE], deps, student_ref="S2")
    assert first.status == second.status == S.PENDING_TEACHER_REVIEW
    assert len(deps.document_agent.calls) == 1  # the second upload reused the first extraction
    assert second.steps[1].status == "skipped"
    assert "EXACT_DUPLICATE" in second.steps[0].detail
    ext = deps.storage.get_extraction(second.submission_id)
    assert any("Gemini was not called" in i for i in ext.issues)
    for sid, other in ((second.submission_id, first.submission_id), (first.submission_id, second.submission_id)):
        flags = deps.storage.integrity_flags(sid)
        assert [(f.code, f.other_submission_id) for f in flags] == [("EXACT_DUPLICATE", other)]
    assert any("EXACT_DUPLICATE" in f for f in second.flags)  # carried into the judge report the teacher sees
    assert "EXACT_DUPLICATE" in deps.judge_agent.calls[1]["input"]  # and shown to the Judge as context
    assert "must NOT change any verdict" in deps.judge_agent.calls[1]["input"]


def test_paraphrased_answers_flagged_and_marks_unchanged(tmp_path):
    # Reference run: submission B alone, no cohort.
    alone = make_deps(tmp_path / "alone", doc=[doc_with_q3(PARAPHRASE)], ev=[eval_for_q3()], jd=[judge_draft()])
    b_alone = assess_files(RUBRIC.assignment_id, [PAGE2], alone)
    assert alone.storage.integrity_flags(b_alone.submission_id) == []

    # Same submission B, but after A (different file, quirky original) is already in the class.
    deps = make_deps(tmp_path / "class", doc=[doc_with_q3(QUIRKY), doc_with_q3(PARAPHRASE)],
                     ev=[eval_for_q3(), eval_for_q3()], jd=[judge_draft(), judge_draft()])
    a = assess_files(RUBRIC.assignment_id, [PAGE], deps, student_ref="S1")
    b = assess_files(RUBRIC.assignment_id, [PAGE2], deps, student_ref="S2")
    assert len(deps.document_agent.calls) == 2  # different files: both read
    flags = deps.storage.integrity_flags(b.submission_id)
    assert [(f.code in ("NEAR_DUPLICATE", "SIMILAR"), f.question_id, f.other_submission_id) for f in flags] == [(True, "Q3", a.submission_id)]
    assert [f.other_submission_id for f in deps.storage.integrity_flags(a.submission_id)] == [b.submission_id]
    postcheck = next(s for s in b.steps if s.step == "integrity_postcheck")
    assert "Q3" in postcheck.detail

    # 7.4 acceptance: the duplicate changes nothing about the marks.
    assert marks(deps, b.submission_id) == marks(alone, b_alone.submission_id)
