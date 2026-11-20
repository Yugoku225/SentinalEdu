"""Dashboard pages driven with Streamlit's AppTest (SENTINELEDU_TASKS.md 8.1–8.8). Offline, fake agents.

AppTest cannot drive `st.file_uploader`, so uploads are done through storage and the pages are tested
from there (the Submit page through its "Run again" path, which runs the same workflow code).
"""
import copy
import csv
import io
import json

import pytest
from streamlit.testing.v1 import AppTest

from sentineledu.config import PROJECT_ROOT
from sentineledu.schemas.submission import SubmissionStatus as S
from sentineledu.services.export import results_csv, results_json
from sentineledu.services.rubric import from_simple_model_answer
from sentineledu.ui import state
from sentineledu.workflow.actions import approve
from sentineledu.workflow.assessment_workflow import assess_files
from wf_helpers import A2, PAGE, RUBRIC, doc_draft, eval_draft, judge_draft, make_deps

PAGES = PROJECT_ROOT / "sentineledu" / "ui" / "pages"
PAGE2 = PAGE.with_name("syn_03_blank_out_of_order_p1.jpg")
TIMEOUT = 120


@pytest.fixture
def use_deps():
    """Point the dashboard at the given WorkflowDeps for the duration of a test."""
    def install(deps):
        state.DEPS_FACTORY = lambda: deps
        return deps
    yield install
    state.DEPS_FACTORY = None


def page(name: str, **query) -> AppTest:
    at = AppTest.from_file(str(PAGES / f"{name}.py"), default_timeout=TIMEOUT)
    for k, v in query.items():
        at.query_params[k] = v
    return at.run()


def no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


# --- 8.1 shell -------------------------------------------------------------------------------

def test_shell_shows_all_pages_and_runs_the_legacy_app_unchanged(use_deps, tmp_path):
    use_deps(make_deps(tmp_path))
    at = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=300).run()
    no_errors(at)
    assert at.title[0].value == "Assignments & Rubrics"  # default page
    for p in ["submit", "extraction_review", "teacher_review", "integrity", "results"]:
        at.switch_page(f"sentineledu/ui/pages/{p}.py").run()
        no_errors(at)
    at.switch_page("app.py").run()
    no_errors(at)
    assert [t.value for t in at.title] == ["📝 Automated Handwritten Assignment Assessor"]
    assert [n.label for n in at.sidebar.number_input] == ["Maximum Assignment Score"]


# --- 8.2 assignments & rubrics ----------------------------------------------------------------

def test_rubric_built_in_the_form_loads_identically_through_the_service(use_deps, tmp_path):
    deps = use_deps(make_deps(tmp_path, store_rubric=False))
    at = page("assignments")
    no_errors(at)
    at.number_input(key="qf_n").set_value(2).run()
    at.text_input(key="qf_id").set_value("bio_unit2")
    at.text_input(key="qf_title").set_value("Unit 2 quiz")
    answers = [("Q1", "Define osmosis.", 3.0, "Osmosis is the movement of water across a partially permeable membrane."),
               ("Q2", "Name the site of respiration.", 2.0, "Respiration happens in the mitochondria.")]
    for i, (qid, prompt, marks, model) in enumerate(answers):
        at.text_input(key=f"qf_qid_{i}").set_value(qid)
        at.text_input(key=f"qf_prompt_{i}").set_value(prompt)
        at.number_input(key=f"qf_marks_{i}").set_value(marks)
        at.text_area(key=f"qf_model_{i}").set_value(model)
    button(at, "Save rubric").click().run()
    no_errors(at)
    saved = deps.storage.get_rubric("bio_unit2")
    expected = [from_simple_model_answer(m, mk, assignment_id="bio_unit2", question_id=q, prompt=p).questions[0]
                for q, p, mk, m in answers]
    assert saved.title == "Unit 2 quiz" and saved.version == 1
    assert saved.questions == expected
    assert any("Saved rubric" in s.value for s in at.success)


def test_invalid_rubric_json_shows_readable_error(use_deps, tmp_path):
    deps = use_deps(make_deps(tmp_path, store_rubric=False))
    at = page("assignments")
    bad = json.loads(RUBRIC.model_dump_json())
    bad["questions"][0]["max_marks"] = 99
    at.text_area(key="rubric_json_text").set_value(json.dumps(bad))
    button(at, "Validate and save").click().run()
    assert any("add up to 3" in e.value and "max_marks is 99" in e.value for e in at.error)
    assert deps.storage.list_assignments() == []
    at.text_area(key="rubric_json_text").set_value(RUBRIC.model_dump_json())
    button(at, "Validate and save").click().run()
    no_errors(at)
    assert deps.storage.get_rubric(RUBRIC.assignment_id).questions == RUBRIC.questions


# --- 8.3 submit -------------------------------------------------------------------------------

def test_submit_page_runs_the_workflow_with_progress(use_deps, tmp_path):
    deps = use_deps(make_deps(tmp_path, doc=[doc_draft()], ev=[eval_draft()], jd=[judge_draft()]))
    sid = deps.storage.create_submission(RUBRIC.assignment_id, [PAGE], student_ref="S01").submission_id
    at = page("submit")
    no_errors(at)
    assert any(df.value["submission"].tolist() == [sid] for df in at.dataframe)
    button(at, "Run again").click().run()
    no_errors(at)
    assert deps.storage.get_submission(sid).status == S.PENDING_TEACHER_REVIEW
    assert any("Ready for teacher review" in s.value for s in at.success)
    assert [len(a.calls) for a in (deps.document_agent, deps.evaluation_agent, deps.judge_agent)] == [1, 1, 1]


# --- 8.4 extraction review --------------------------------------------------------------------

def test_correcting_a_flagged_extraction_and_resuming_reaches_review(use_deps, tmp_path):
    deps = use_deps(make_deps(tmp_path, doc=[doc_draft(q2_unreadable=True)], ev=[eval_draft()], jd=[judge_draft()]))
    run = assess_files(RUBRIC.assignment_id, [PAGE], deps, student_ref="S02")
    assert run.status == S.NEEDS_EXTRACTION_REVIEW
    at = page("extraction_review")
    no_errors(at)
    assert any("Q2: answer is unreadable" in w.value for w in at.warning)
    assert button(at, "Save and resume grading").disabled  # no teacher name yet
    at.text_area(key=f"ans_{run.submission_id}_v1_Q2").set_value(A2)
    at.text_input(key="extraction_teacher").set_value("Ms Rao").run()
    button(at, "Save and resume grading").click().run()
    no_errors(at)
    assert deps.storage.get_submission(run.submission_id).status == S.PENDING_TEACHER_REVIEW
    ext = deps.storage.get_extraction(run.submission_id)
    assert ext.version == 2 and ext.corrected_by == "Ms Rao" and ext.answer_for("Q2").answer_text == A2
    assert len(deps.document_agent.calls) == 1


# --- 8.5 teacher review -----------------------------------------------------------------------

@pytest.fixture
def pending(use_deps, tmp_path):
    deps = use_deps(make_deps(tmp_path, doc=[doc_draft()], ev=[eval_draft()], jd=[judge_draft()]))
    run = assess_files(RUBRIC.assignment_id, [PAGE], deps, student_ref="S03")
    assert run.status == S.PENDING_TEACHER_REVIEW
    return deps, run.submission_id


def test_review_page_shows_evidence_and_recommendation(pending):
    deps, sid = pending
    at = page("teacher_review")
    no_errors(at)
    labels = [m.label for m in at.metric]
    assert "Evaluator (Agent 2)" in labels and "Recommended (after Judge)" in labels
    html = " ".join(m.value for m in at.markdown)
    assert "<mark>Oxygen is released</mark>" in html  # evidence quote highlighted in the answer
    tables = [df.value for df in at.dataframe]
    assert any(set(t.columns) >= {"evaluator level", "marks", "evidence verified", "judge", "recommended"} for t in tables)


def test_approve_needs_reviewer_and_reasons_then_finalises(pending):
    deps, sid = pending
    at = page("teacher_review")
    assert button(at, "Approve & Finalise").disabled
    at.text_input(key="reviewer_name").set_value("Ms Rao").run()
    assert not button(at, "Approve & Finalise").disabled
    at.number_input(key=f"final_{sid}_Q3_C1").set_value(1.0).run()  # change a mark ...
    assert button(at, "Approve & Finalise").disabled  # ... so a reason is now required
    assert any("give a reason for Q3/C1" in c.value for c in at.caption)
    at.text_input(key=f"reason_{sid}_Q3_C1").set_value("Second reason is too vague to count").run()
    approve_btn = button(at, "Approve & Finalise")
    assert not approve_btn.disabled
    approve_btn.click().run()
    no_errors(at)
    assert deps.storage.get_submission(sid).status == S.FINALIZED
    d = deps.storage.get_decision(sid)
    assert d.reviewer == "Ms Rao" and d.total == 6
    assert [(o.question_id, o.criterion_id, o.from_marks, o.to_marks) for o in d.overrides] == [("Q3", "C1", 2.0, 1.0)]
    assert any("Finalised" in h.value for h in at.subheader)


# --- 8.6 integrity ----------------------------------------------------------------------------

QUIRKY = ("Plants drink water mostly so their stems do not flop over like a wet noodle, and I think the water also "
          "travels up like a lift carrying food salts to every leaf.")
PARAPHRASE = ("Plants take in water mainly so the stem does not flop over like a soggy noodle, and the water also goes up "
              "like an elevator carrying food salts to all the leaves.")


def doc_q3(text):
    d = copy.deepcopy(doc_draft())
    d["answers"][2]["answer_text"] = text
    return d


def test_integrity_page_shows_planted_pair_and_heatmap(use_deps, tmp_path):
    deps = use_deps(make_deps(tmp_path, doc=[doc_q3(QUIRKY), doc_q3(PARAPHRASE)], ev=[eval_draft(), eval_draft()],
                              jd=[judge_draft(), judge_draft()]))
    a = assess_files(RUBRIC.assignment_id, [PAGE], deps, student_ref="S1")
    b = assess_files(RUBRIC.assignment_id, [PAGE2], deps, student_ref="S2")
    at = page("integrity")
    no_errors(at)
    pairs = at.dataframe[0].value
    assert len(pairs) == 1 and pairs.iloc[0]["question"] == "Q3"
    assert {pairs.iloc[0]["student A"], pairs.iloc[0]["student B"]} == {"S1", "S2"}
    at.selectbox(key="heatmap_question").set_value("Q3").run()
    no_errors(at)
    assert len(at.get("arrow_vega_lite_chart")) == 1
    assert a.submission_id != b.submission_id


# --- 8.7 results & export ---------------------------------------------------------------------

def test_export_matches_the_database(pending):
    deps, sid = pending
    approve(deps.storage, sid, "Ms Rao", changes={("Q1", "C2"): (0.0, "Oxygen named only as 'a gas'")})
    rows = list(csv.DictReader(io.StringIO(results_csv(deps.storage, RUBRIC.assignment_id))))
    d = deps.storage.get_decision(sid)
    assert len(rows) == 1 and rows[0]["submission_id"] == sid and float(rows[0]["total"]) == d.total == 6
    for q, crits in d.final_marks.items():
        assert float(rows[0][q]) == sum(crits.values())
        for c, m in crits.items():
            assert float(rows[0][f"{q}.{c}"]) == m
    assert rows[0]["reviewer"] == "Ms Rao" and rows[0]["overrides"] == "1"
    assert rows[0]["feedback_summary"] == "Excellent work."
    record = json.loads(results_json(deps.storage, RUBRIC.assignment_id))[0]
    assert record["decision"]["overrides"][0]["reason"] == "Oxygen named only as 'a gas'"
    assert record["feedback"]["summary_feedback"] == "Excellent work."

    at = page("results")
    no_errors(at)
    assert at.dataframe[0].value.iloc[0]["total"] == 6
    assert [m.value for m in at.metric][:1] == ["1"]


def test_results_page_lists_only_finalised(pending):
    at = page("results")
    no_errors(at)
    assert any("No finalised submissions yet" in i.value for i in at.info)


# --- 8.8 refresh keeps the selection ----------------------------------------------------------

def test_refresh_reopens_the_same_submission_from_the_url(use_deps, tmp_path):
    deps = use_deps(make_deps(tmp_path, doc=[doc_draft(), doc_q3(QUIRKY)], ev=[eval_draft(), eval_draft()],
                              jd=[judge_draft(), judge_draft()]))
    first = assess_files(RUBRIC.assignment_id, [PAGE], deps, student_ref="S1")
    second = assess_files(RUBRIC.assignment_id, [PAGE2], deps, student_ref="S2")
    at = page("teacher_review")
    assert at.selectbox(key="pick_submission").value == first.submission_id  # default: oldest pending
    at.selectbox(key="pick_submission").set_value(second.submission_id).run()
    assert at.selectbox(key="pick_submission").value == second.submission_id
    assert second.submission_id in str(at.query_params["submission"])  # mirrored into the URL
    fresh = page("teacher_review", assignment=RUBRIC.assignment_id, submission=second.submission_id)  # a new browser session
    no_errors(fresh)
    assert fresh.selectbox(key="pick_submission").value == second.submission_id
