"""Page 1 — Assignments & Rubrics (TASKS 8.2).

An assignment exists once it has a rubric. The rubric can be built with a quick form (one
full/partial/none criterion per question, D-06) or imported as JSON (validated, with readable errors).
Reference documents are indexed for RAG and the chunk count is shown.
"""
import json
import re
from pathlib import Path

import streamlit as st

from sentineledu.services.rag import RagError
from sentineledu.services.rubric import RubricError, from_simple_model_answer, rubric_summary, validate_rubric
from sentineledu.schemas.rubric import Rubric
from sentineledu.ui import state

st.title("Assignments & Rubrics")
deps = state.deps()
storage = deps.storage


def save(rubric: Rubric) -> None:
    saved = storage.save_rubric(rubric)
    state._remember("assignment", saved.assignment_id)
    st.session_state["rubric_saved"] = f"Saved rubric **{saved.assignment_id}** as version {saved.version} ({saved.max_total:g} marks)."
    st.rerun()


if msg := st.session_state.pop("rubric_saved", None):
    st.success(msg)

# --- existing assignment -----------------------------------------------------------------------
assignment = state.pick_assignment("Existing assignment")
if assignment:
    rubric = storage.get_rubric(assignment)
    c1, c2, c3 = st.columns(3)
    c1.metric("Questions", len(rubric.questions))
    c2.metric("Total marks", f"{rubric.max_total:g}")
    c3.metric("Rubric version", rubric.version)
    with st.expander("Rubric", expanded=False):
        st.code(rubric_summary(rubric), language=None)
        st.download_button("Download rubric JSON", rubric.model_dump_json(indent=2), file_name=f"{assignment}_rubric.json",
                           mime="application/json")

    st.subheader("Reference documents")
    rag = deps.rag
    if rag.has_index(assignment):
        sources = rag.sources(assignment)
        st.write(f"**{len(rag.chunks(assignment))} chunks** indexed from {len(sources)} document(s): "
                 + ", ".join(Path(s).name for s in sources))
    else:
        st.caption("No reference documents yet. Grading then relies on the rubric and model answers alone.")
    refs = st.file_uploader("Add reference PDF or TXT", type=["pdf", "txt", "md"], accept_multiple_files=True, key="ref_upload")
    if refs and st.button("Index reference documents"):
        folder = storage.root / "references" / assignment
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in refs:
            dest = folder / Path(f.name).name
            dest.write_bytes(f.getvalue())
            paths.append(dest)
        try:
            n = rag.ingest(assignment, paths)
            st.success(f"Indexed: the assignment now has {n} chunks.")
        except RagError as e:
            st.error(str(e))

st.divider()

# --- new or updated rubric ---------------------------------------------------------------------
st.subheader("Create or update a rubric")
form_tab, json_tab = st.tabs(["Quick form", "Import JSON"])

with form_tab:
    st.caption("Each question gets one criterion with three levels: full, partial (about half) and none. "
               "For several criteria per question, download the JSON, edit it, and import it.")
    n = st.number_input("Number of questions", min_value=1, max_value=15, value=3, step=1, key="qf_n")
    with st.form("quick_rubric"):
        a_id = st.text_input("Assignment id (letters, digits, _ . -)", key="qf_id")
        title = st.text_input("Title", key="qf_title")
        questions = []
        for i in range(int(n)):
            st.markdown(f"**Question {i + 1}**")
            cols = st.columns([1, 4, 1])
            qid = cols[0].text_input("Id", value=f"Q{i + 1}", key=f"qf_qid_{i}")
            prompt = cols[1].text_input("Question", key=f"qf_prompt_{i}")
            marks = cols[2].number_input("Marks", min_value=0.5, value=2.0, step=0.5, key=f"qf_marks_{i}")
            model = st.text_area("Model answer", key=f"qf_model_{i}", height=80)
            questions.append((qid.strip(), prompt.strip(), float(marks), model.strip()))
        submitted = st.form_submit_button("Save rubric", type="primary")
    if submitted:
        problems = []
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", a_id or ""):
            problems.append("The assignment id may only contain letters, digits, '_', '.' and '-'.")
        problems += [f"Question {i + 1}: the question text and model answer are required."
                     for i, (_, p, _, m) in enumerate(questions) if not p or not m]
        if problems:
            for p in problems:
                st.error(p)
        else:
            try:
                parts = [from_simple_model_answer(m, mk, assignment_id=a_id, question_id=q, prompt=p).questions[0]
                         for q, p, mk, m in questions]
                save(validate_rubric({"assignment_id": a_id, "title": title or a_id,
                                      "questions": [json.loads(q.model_dump_json()) for q in parts]}))
            except RubricError as e:
                st.error(str(e))

with json_tab:
    up = st.file_uploader("Rubric JSON file", type=["json"], key="rubric_json_file")
    text = st.text_area("…or paste the JSON", height=200, key="rubric_json_text")
    if st.button("Validate and save", key="rubric_json_save"):
        raw = up.getvalue().decode("utf-8") if up else text
        if not raw.strip():
            st.error("Upload a file or paste the rubric JSON first.")
        else:
            try:
                save(validate_rubric(raw))
            except RubricError as e:
                st.error(str(e))
