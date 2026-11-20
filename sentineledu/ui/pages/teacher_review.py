"""Page 4 — Teacher Review (TASKS 8.5).

Everything the teacher needs to approve marks: per criterion the evaluator's level and marks, the quotes
(highlighted in the answer, with whether they verified), cited reference chunks, the Judge's verdict and
recommendation, and all flags. The final-marks form starts from the recommendation; any change needs a
reason, and **Approve & Finalise** stays disabled until the reviewer is named and every change has one.
"""
import pandas as pd
import streamlit as st

from sentineledu.schemas.submission import SubmissionStatus as S
from sentineledu.services.storage import FinalizationError, IllegalTransitionError
from sentineledu.ui import components, state
from sentineledu.workflow.actions import approve, request_reevaluation

st.title("Teacher Review")
deps = state.deps()
storage = deps.storage

assignment = state.pick_assignment()
if not assignment:
    st.stop()
order = {S.PENDING_TEACHER_REVIEW: 0, S.FINALIZED: 1}
subs = [s for s in storage.list_submissions(assignment) if s.status in order]
subs.sort(key=lambda s: (order[s.status], s.created_at))
st.caption(f"{sum(s.status == S.PENDING_TEACHER_REVIEW for s in subs)} submission(s) waiting for approval.")
sub = state.pick_submission(subs)
if sub is None:
    st.info("Nothing to review yet: submissions appear here once they reach PENDING_TEACHER_REVIEW.")
    st.stop()

if msg := st.session_state.pop("review_msg", None):
    st.success(msg)

sid = sub.submission_id
rubric = storage.get_rubric(sub.assignment_id)
ext = storage.get_extraction(sid)
evaluation = storage.latest_evaluation(sid)
report = storage.latest_judge_report(sid)
decision = storage.get_decision(sid)
integrity = storage.integrity_flags(sid)
if evaluation is None or report is None:
    st.error("This submission has no evaluation and judge report. Re-run it from the Submit page.")
    st.stop()
chunk_text = {}
if deps.rag.has_index(sub.assignment_id):
    chunk_text = {c.chunk_id: c for c in deps.rag.chunks(sub.assignment_id)}

# --- header ------------------------------------------------------------------------------------
st.markdown(f"**{sub.student_ref or 'No student ref'}** · `{sid}` · {components.status_badge(sub.status)} · "
            f"graded by {evaluation.model_id}, reviewed by {report.model_id}")
c1, c2, c3 = st.columns(3)
c1.metric("Evaluator (Agent 2)", f"{evaluation.total:g} / {evaluation.max_total:g}")
c2.metric("Recommended (after Judge)", f"{report.recommended_total:g}",
          delta=f"{report.recommended_total - evaluation.total:+g}" if report.recommended_total != evaluation.total else None)
if decision:
    c3.metric("Final (teacher)", f"{decision.total:g}")

flags = report.flags
if flags:
    with st.expander(f"Flags ({len(flags)})", expanded=any(f.severity != "info" for f in flags)):
        for f in flags:
            where = "/".join(x for x in (f.question_id, f.criterion_id) if x)
            st.markdown(f"{components.SEVERITY_ICON.get(f.severity, '•')} **{f.code}**{' · ' + where if where else ''} — {f.message}")
if report.summary_feedback:
    st.info(f"**Feedback for the student:** {report.summary_feedback}")

# --- per question ------------------------------------------------------------------------------
for q in rubric.questions:
    qe = evaluation.question(q.question_id)
    verdicts = {v.criterion_id: v for v in report.verdicts.get(q.question_id, [])}
    rec_q = sum(v.recommended_marks for v in verdicts.values())
    with st.expander(f"{q.question_id} — {q.prompt}  ·  {qe.total:g} → {rec_q:g} / {q.max_marks:g}", expanded=True):
        answer = ext.answer_for(q.question_id)
        quotes = [qt for ce in qe.criteria for qt in ce.evidence_quotes] + \
                 [qt for v in verdicts.values() if v.accepted for qt in v.evidence_quotes]
        if answer:
            st.markdown(f"Student's answer {components.legibility_badge(answer.legibility, answer.confidence)}")
        components.answer_box(answer.answer_text if answer else "", quotes)
        if qe.validator_flags:
            st.caption("Validator: " + ", ".join(qe.validator_flags))

        rows = []
        for ce in qe.criteria:
            c = q.criterion(ce.criterion_id)
            v = verdicts.get(ce.criterion_id)
            rows.append({
                "criterion": f"{ce.criterion_id}: {c.description}" if c else ce.criterion_id,
                "evaluator level": ce.selected_level_id, "marks": ce.marks,
                "evidence verified": "✅" if ce.evidence_verified else ("—" if ce.marks == 0 else "❌"),
                "judge": (v.verdict + (" ✔ applied" if v.accepted else "")) if v else "",
                "recommended": f"{v.recommended_level_id} ({v.recommended_marks:g})" if v else "",
                "max": c.max_marks if c else None,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        for ce in qe.criteria:
            v = verdicts.get(ce.criterion_id)
            lines = []
            if ce.rationale:
                lines.append(f"*Evaluator:* {ce.rationale}")
            if ce.missing_concepts:
                lines.append(f"*Missing:* {', '.join(ce.missing_concepts)}")
            if ce.incorrect_statements:
                lines.append(f"*Incorrect:* {', '.join(ce.incorrect_statements)}")
            if v and v.justification:
                lines.append(f"*Judge ({v.verdict}{' → ' + v.proposed_level_id if v.proposed_level_id else ''}):* {v.justification}")
            if lines:
                st.markdown(f"**{ce.criterion_id}** — " + "  \n".join(lines))
            if ce.reference_chunk_ids:
                cols = st.columns(min(len(ce.reference_chunk_ids), 4))
                for col, cid in zip(cols * 10, ce.reference_chunk_ids):
                    ch = chunk_text.get(cid)
                    with col.popover(f"📄 {cid}"):
                        st.caption(f"{ch.source}, page {ch.page}" if ch else "chunk not in the current index")
                        st.write(ch.text if ch else "")
        fb = report.feedback_for(q.question_id)
        if fb and (fb.strengths or fb.improvements):
            st.markdown("**Feedback** — " + " ".join(f"✔ {x}" for x in fb.strengths) + " " + " ".join(f"➜ {x}" for x in fb.improvements))
        q_flags = [f for f in integrity if f.question_id == q.question_id]
        for f in q_flags:
            st.warning(f"{f.code}: {f.message}")

st.divider()

# --- decision ----------------------------------------------------------------------------------
if decision:
    st.subheader("Finalised")
    st.write(f"Approved by **{decision.reviewer}** on {decision.approved_at:%Y-%m-%d %H:%M} UTC — total {decision.total:g}.")
    for o in decision.overrides:
        st.write(f"- {o.question_id}/{o.criterion_id}: {o.from_marks:g} → {o.to_marks:g} — {o.reason}")
    if decision.teacher_comment:
        st.write(f"Comment: {decision.teacher_comment}")
    st.stop()

st.subheader("Final marks")
st.caption("Pre-filled with the recommendation. Change a mark only with a reason.")
recommended = report.recommended_marks()
changes: dict[tuple[str, str], tuple[float, str]] = {}
missing_reasons = []
for q in rubric.questions:
    for c in q.criteria:
        rec = recommended.get(q.question_id, {}).get(c.criterion_id, 0.0)
        cols = st.columns([2, 1, 3])
        cols[0].markdown(f"**{q.question_id}/{c.criterion_id}** · recommended {rec:g} of {c.max_marks:g}")
        mark = cols[1].number_input("Marks", min_value=0.0, max_value=float(c.max_marks), value=float(rec), step=0.5,
                                    key=f"final_{sid}_{q.question_id}_{c.criterion_id}", label_visibility="collapsed")
        changed = abs(mark - rec) > 1e-6
        reason = cols[2].text_input("Reason", key=f"reason_{sid}_{q.question_id}_{c.criterion_id}",
                                    placeholder="required when you change the mark" if changed else "no change",
                                    label_visibility="collapsed", disabled=not changed)
        if changed:
            changes[(q.question_id, c.criterion_id)] = (mark, reason.strip())
            if not reason.strip():
                missing_reasons.append(f"{q.question_id}/{c.criterion_id}")

total = sum(m for crits in recommended.values() for m in crits.values()) + sum(m - recommended.get(q, {}).get(c, 0.0) for (q, c), (m, _) in changes.items())
st.metric("Final total", f"{total:g} / {rubric.max_total:g}")
reviewer = st.text_input("Reviewer name", key="reviewer_name")
comment = st.text_area("Comment for the record (optional)", key=f"comment_{sid}")

blockers = []
if not reviewer.strip():
    blockers.append("enter the reviewer name")
if missing_reasons:
    blockers.append("give a reason for " + ", ".join(missing_reasons))
if blockers:
    st.caption("To approve: " + "; ".join(blockers) + ".")

left, right = st.columns([1, 1])
if left.button("Approve & Finalise", type="primary", disabled=bool(blockers), key="approve"):
    try:
        d = approve(storage, sid, reviewer.strip(), changes=changes, teacher_comment=comment.strip())
    except (FinalizationError, ValueError) as e:
        st.error(str(e))
    else:
        st.session_state["review_msg"] = f"Finalised {sid} with {d.total:g}/{rubric.max_total:g}."
        st.rerun()
if right.button("Send back for re-evaluation", key="reevaluate"):
    try:
        request_reevaluation(storage, sid)
    except IllegalTransitionError as e:
        st.error(str(e))
    else:
        run = components.run_with_progress(sid, deps, "Re-evaluating")
        components.show_run_outcome(run)
