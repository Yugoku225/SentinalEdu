"""Page 2 — Submit (TASKS 8.3): upload a student's pages and run the workflow with live progress."""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from sentineledu.schemas.submission import SubmissionStatus as S
from sentineledu.ui import components, state

st.title("Submit")
deps = state.deps()
storage = deps.storage

assignment = state.pick_assignment()
if not assignment:
    st.stop()

with st.form("submit_form", clear_on_submit=True):
    student = st.text_input("Student reference (anonymised, e.g. S01)")
    files = st.file_uploader("Pages, in order (photos or scans; PDF, JPG, PNG, TIFF)",
                             type=["pdf", "jpg", "jpeg", "png", "tif", "tiff", "webp"], accept_multiple_files=True)
    go = st.form_submit_button("Upload and assess", type="primary")

if go:
    if not files:
        st.error("Add at least one page.")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, f in enumerate(files, start=1):
                p = Path(tmp) / f"{i:02d}_{Path(f.name).name}"
                p.write_bytes(f.getvalue())
                paths.append(p)
            sub = storage.create_submission(assignment, paths, student_ref=student.strip() or None)
        state._remember("submission", sub.submission_id)
        run = components.run_with_progress(sub.submission_id, deps)
        st.session_state["last_run"] = run.model_dump(mode="json")

last = st.session_state.get("last_run")
if last and last["assignment_id"] == assignment:
    from sentineledu.workflow.assessment_workflow import AssessmentRun
    run = AssessmentRun.model_validate(last)
    st.subheader(f"Last submission: {run.submission_id}")
    components.show_run_outcome(run)
    if run.status == S.NEEDS_EXTRACTION_REVIEW and st.button("Open Extraction Review"):
        state.go("extraction_review", submission_id=run.submission_id)
    if run.status == S.PENDING_TEACHER_REVIEW and st.button("Open Teacher Review"):
        state.go("teacher_review", submission_id=run.submission_id)

st.divider()
st.subheader("Submissions in this assignment")
subs = storage.list_submissions(assignment)
if not subs:
    st.caption("None yet.")
else:
    rows = []
    for s in reversed(subs):
        ev = storage.latest_evaluation(s.submission_id)
        rep = storage.latest_judge_report(s.submission_id)
        dec = storage.get_decision(s.submission_id)
        rows.append({"submission": s.submission_id, "student": s.student_ref or "", "status": s.status.value,
                     "evaluator": ev.total if ev else None, "recommended": rep.recommended_total if rep else None,
                     "final": dec.total if dec else None, "uploaded": s.created_at.strftime("%Y-%m-%d %H:%M"),
                     "integrity flags": len(storage.integrity_flags(s.submission_id))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    retry = [s for s in subs if s.status in (S.FAILED, S.EXTRACTED, S.EVALUATED, S.UPLOADED)]
    if retry:
        with st.expander("Retry a stopped submission"):
            by_id = {s.submission_id: s for s in retry}
            pick = st.selectbox("Submission", list(by_id), format_func=lambda i: state.submission_label(by_id[i]),
                                key="retry_pick")
            if st.button("Run again"):
                state._remember("submission", pick)
                run = components.run_with_progress(pick, deps)
                st.session_state["last_run"] = run.model_dump(mode="json")
                st.rerun()
