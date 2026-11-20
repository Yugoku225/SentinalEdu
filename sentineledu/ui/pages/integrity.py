"""Page 5 — Integrity (TASKS 8.6): similarity heatmap per question and flagged pairs side by side.

Flags are informational. They never change marks; the teacher decides what, if anything, to do.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sentineledu.services.duplicate import similarity_matrix
from sentineledu.ui import components, state

st.title("Integrity")
st.caption("Duplicate and near-duplicate answers within an assignment. Flags inform the teacher; they never change marks.")
deps = state.deps()
storage = deps.storage

assignment = state.pick_assignment()
if not assignment:
    st.stop()
rubric = storage.get_rubric(assignment)
subs = {s.submission_id: s for s in storage.list_submissions(assignment)}


def who(sid: str) -> str:
    s = subs.get(sid)
    return f"{s.student_ref or sid}" if s else sid


# --- flagged pairs (one row per pair) ---------------------------------------------------------
flags = storage.assignment_integrity_flags(assignment)
pairs = {}
for f in flags:
    key = (f.code, f.question_id, *sorted((f.submission_id, f.other_submission_id)))
    pairs.setdefault(key, f)
st.subheader(f"Flagged pairs ({len(pairs)})")
if not pairs:
    st.success("No duplicates or near-duplicates flagged in this assignment.")
else:
    table = pd.DataFrame([{
        "flag": f"{components.SEVERITY_ICON.get(f.severity, '')} {code}", "question": qid or "all",
        "student A": who(a), "student B": who(b),
        "similarity": f.similarity, "margin over reference": f.margin,
    } for (code, qid, a, b), f in pairs.items()])
    st.dataframe(table, hide_index=True, use_container_width=True)

    labels = [f"{code} · {qid or 'all'} · {who(a)} ↔ {who(b)}" for code, qid, a, b in pairs]
    pick = st.selectbox("Compare a pair", range(len(labels)), format_func=labels.__getitem__, key="pair_pick")
    code, qid, a, b = list(pairs)[pick]
    st.caption(list(pairs.values())[pick].message)
    cols = st.columns(2)
    for col, sid in zip(cols, (a, b)):
        ext = storage.latest_extraction(sid)
        with col:
            st.markdown(f"**{who(sid)}** · `{sid}`")
            questions = [rubric.question(qid)] if qid else rubric.questions
            for q in questions:
                ans = ext.answer_for(q.question_id) if ext else None
                st.markdown(f"*{q.question_id}*")
                components.answer_box(ans.answer_text if ans else "")

st.divider()

# --- heatmap ----------------------------------------------------------------------------------
st.subheader("Similarity heatmap")
qid = st.selectbox("Question", [q.question_id for q in rubric.questions],
                   format_func=lambda x: f"{x} — {rubric.question(x).prompt}", key="heatmap_question")
with st.spinner("Comparing answers…"):
    members, matrix, texts = similarity_matrix(storage, assignment, qid, settings=deps.settings)
if len(members) < 2:
    st.info("Need at least two transcribed submissions to compare.")
    st.stop()
names = [f"{m.student_ref or m.submission_id} ({m.submission_id[:6]})" for m in members]
flagged_pairs = {tuple(sorted((a, b))) for (code, q, a, b) in pairs if q in (qid, None)}
cells = []
for i, a in enumerate(members):
    for j, b in enumerate(members):
        v = matrix[i, j]
        cells.append({"A": names[i], "B": names[j], "similarity": None if np.isnan(v) else round(float(v), 3),
                      "flagged": tuple(sorted((a.submission_id, b.submission_id))) in flagged_pairs})
data = pd.DataFrame(cells)
base = alt.Chart(data).encode(x=alt.X("A:N", sort=names, title=None), y=alt.Y("B:N", sort=names, title=None))
heat = base.mark_rect().encode(
    color=alt.Color("similarity:Q", scale=alt.Scale(domain=[0, 1], scheme="blues"), title="cosine"),
    tooltip=["A", "B", "similarity", "flagged"])
marks = base.transform_filter(alt.datum.flagged).mark_rect(fill=None, stroke="#d62728", strokeWidth=2.5)
st.altair_chart((heat + marks).properties(height=max(240, 36 * len(names))), use_container_width=True)
st.caption("Red outline = flagged pair. Similarity alone is not evidence of copying: two correct answers are naturally "
           "similar. A flag needs the pair to be clearly closer to each other than to the reference material.")
