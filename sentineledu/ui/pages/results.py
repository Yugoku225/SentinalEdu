"""Page 6 — Results & Export (TASKS 8.7): finalised marks only, as CSV or JSON (with feedback)."""
import pandas as pd
import streamlit as st

from sentineledu.schemas.submission import SubmissionStatus as S
from sentineledu.services.export import results_csv, results_json, results_rows
from sentineledu.ui import state

st.title("Results & Export")
deps = state.deps()
storage = deps.storage

assignment = state.pick_assignment()
if not assignment:
    st.stop()
subs = storage.list_submissions(assignment)
counts = {s: sum(x.status == s for x in subs) for s in S}
c1, c2, c3 = st.columns(3)
c1.metric("Finalised", counts[S.FINALIZED])
c2.metric("Waiting for teacher", counts[S.PENDING_TEACHER_REVIEW])
c3.metric("Needs transcription check", counts[S.NEEDS_EXTRACTION_REVIEW])

rows = results_rows(storage, assignment)
if not rows:
    st.info("No finalised submissions yet. Marks appear here only after a teacher approves them.")
    st.stop()
df = pd.DataFrame(rows)
st.dataframe(df, hide_index=True, use_container_width=True)
st.caption(f"Mean {df['total'].mean():.2f} / {rows[0]['max_total']:g} over {len(rows)} submission(s).")

left, right = st.columns(2)
left.download_button("Download CSV", results_csv(storage, assignment), file_name=f"{assignment}_results.csv",
                     mime="text/csv", type="primary")
right.download_button("Download JSON (with feedback and overrides)", results_json(storage, assignment),
                      file_name=f"{assignment}_results.json", mime="application/json")
