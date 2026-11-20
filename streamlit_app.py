"""SentinelEdu dashboard entry point (TASKS 8.1, ARCHITECTURE §10, DECISIONS D-13).

    streamlit run streamlit_app.py

The original evaluator is registered unchanged as the "Legacy Evaluator" page; `streamlit run app.py`
still works on its own.
"""
import streamlit as st

st.set_page_config(page_title="SentinelEdu", page_icon="🛡️", layout="wide")

pages = {
    "SentinelEdu": [
        st.Page("sentineledu/ui/pages/assignments.py", title="Assignments & Rubrics", icon="📚", default=True),
        st.Page("sentineledu/ui/pages/submit.py", title="Submit", icon="📤"),
        st.Page("sentineledu/ui/pages/extraction_review.py", title="Extraction Review", icon="✍️"),
        st.Page("sentineledu/ui/pages/teacher_review.py", title="Teacher Review", icon="✅"),
        st.Page("sentineledu/ui/pages/integrity.py", title="Integrity", icon="🔍"),
        st.Page("sentineledu/ui/pages/results.py", title="Results & Export", icon="📊"),
    ],
    "Original app": [
        st.Page("app.py", title="Legacy Evaluator", icon="📝"),
    ],
}

with st.sidebar:
    st.caption("SentinelEdu · marks are final only after a teacher approves them.")

st.navigation(pages).run()
