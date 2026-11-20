"""Page 3 — Extraction Review (TASKS 8.4).

The page images next to editable per-question text. Saving stores a new extraction version
(engine "teacher", the original is kept) and resumes the workflow, which skips Agent 1.
"""
import streamlit as st

from sentineledu.schemas.extraction import ILLEGIBLE_MARKER
from sentineledu.schemas.submission import SubmissionStatus as S
from sentineledu.services.documents import detect_kind
from sentineledu.services.extraction_review import AnswerEdit, extraction_gate
from sentineledu.services.storage import IllegalTransitionError
from sentineledu.ui import components, state
from sentineledu.workflow.actions import submit_correction

st.title("Extraction Review")
deps = state.deps()
storage = deps.storage

assignment = state.pick_assignment()
if not assignment:
    st.stop()

subs = [s for s in storage.list_submissions(assignment) if storage.latest_extraction(s.submission_id) is not None]
order = {S.NEEDS_EXTRACTION_REVIEW: 0, S.EXTRACTED: 1, S.PENDING_TEACHER_REVIEW: 2, S.EVALUATED: 3, S.FINALIZED: 4}
subs.sort(key=lambda s: (order.get(s.status, 5), s.created_at))
waiting = sum(s.status == S.NEEDS_EXTRACTION_REVIEW for s in subs)
st.caption(f"{waiting} submission(s) waiting for a transcription check.")
sub = state.pick_submission(subs)
if sub is None:
    st.info("No transcribed submissions in this assignment yet.")
    st.stop()

rubric = storage.get_rubric(sub.assignment_id)
ext = storage.get_extraction(sub.submission_id)
st.markdown(f"Status {components.status_badge(sub.status)} · extraction v{ext.version} by **{ext.engine}**"
            + (f" (corrected by {ext.corrected_by})" if ext.corrected_by else ""))

gate = extraction_gate(ext, [q.question_id for q in rubric.questions], deps.settings.extraction_review_threshold)
if not gate.ok:
    st.warning("Why this needs checking:\n" + "\n".join(f"- {r}" for r in gate.reasons))
if ext.issues:
    with st.expander("Notes from the reader"):
        for i in ext.issues:
            st.write(f"- {i}")

editable = sub.status in (S.EXTRACTED, S.NEEDS_EXTRACTION_REVIEW, S.EVALUATED, S.PENDING_TEACHER_REVIEW)
left, right = st.columns([1, 1])

with left:
    st.subheader("Pages")
    for n, path in enumerate(storage.submission_paths(sub), start=1):
        kind = detect_kind(path)
        if kind == "pdf":
            st.download_button(f"Page file {n} (PDF)", path.read_bytes(), file_name=path.name, mime="application/pdf",
                               key=f"pdf_{sub.submission_id}_{n}")
        else:
            try:
                st.image(str(path), caption=f"Page {n}", use_container_width=True)
            except Exception:  # noqa: BLE001 - e.g. a TIFF the browser cannot show
                from PIL import Image
                st.image(Image.open(path).convert("RGB"), caption=f"Page {n}", use_container_width=True)

with right:
    st.subheader("Transcription")
    st.caption(f"Type exactly what the student wrote. Use {ILLEGIBLE_MARKER} for words nobody can read.")
    texts = {}
    for q in rubric.questions:
        a = ext.answer_for(q.question_id)
        badge = components.legibility_badge(a.legibility, a.confidence) if a else ":red-badge[missing]"
        st.markdown(f"**{q.question_id}** {badge} — {q.prompt}")
        texts[q.question_id] = st.text_area(q.question_id, value=a.answer_text if a else "", height=110,
                                            key=f"ans_{sub.submission_id}_v{ext.version}_{q.question_id}",
                                            label_visibility="collapsed", disabled=not editable)
    unmatched = st.text_area("Writing not matched to any question (move it into an answer above, then clear it)",
                             value=ext.unmatched_text, height=90, key=f"unmatched_{sub.submission_id}_v{ext.version}",
                             disabled=not editable)
    teacher = st.text_input("Your name", key="extraction_teacher", disabled=not editable)

    if editable:
        changed = [q for q, t in texts.items() if t != (ext.answer_for(q).answer_text if ext.answer_for(q) else "")]
        st.caption(f"{len(changed)} answer(s) edited." + ("" if changed else " Saving with no edits confirms the transcription as correct."))
        if st.button("Save and resume grading", type="primary", disabled=not teacher.strip()):
            try:
                edits = [AnswerEdit(question_id=q, answer_text=texts[q]) for q in changed]
                saved = submit_correction(storage, sub.submission_id, edits, teacher.strip(), unmatched_text=unmatched)
            except IllegalTransitionError as e:
                st.error(str(e))
                st.stop()
            st.success(f"Saved extraction v{saved.version}.")
            run = components.run_with_progress(sub.submission_id, deps, "Resuming the assessment")
            components.show_run_outcome(run)
        if not teacher.strip():
            st.caption("Enter your name to save.")
    else:
        st.info(f"This submission is {sub.status.value}; its transcription can no longer be edited.")

versions = storage.list_extraction_versions(sub.submission_id)
if len(versions) > 1:
    with st.expander(f"Earlier versions ({len(versions) - 1})"):
        for v in versions[:-1]:
            old = storage.get_extraction(sub.submission_id, v)
            st.markdown(f"**v{v}** by {old.engine}" + (f", corrected by {old.corrected_by}" if old.corrected_by else ""))
            for a in old.answers:
                st.text(f"{a.question_id}: {a.answer_text}")

if storage.get_submission(sub.submission_id).status == S.PENDING_TEACHER_REVIEW:
    if st.button("Open Teacher Review", key="to_review"):
        state.go("teacher_review", submission_id=sub.submission_id)
