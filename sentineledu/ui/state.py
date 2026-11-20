"""Shared state for the dashboard pages (TASKS 8.8).

Pages keep no work in memory: everything is read from SQLite by id, and the current assignment /
submission is mirrored into the URL query string (`?assignment=...&submission=...`), so a browser
refresh or a shared link reopens the same item. `st.session_state` only carries the selection across
page switches (Streamlit 1.50's `switch_page` cannot pass query parameters).
"""
from typing import Callable, Sequence

import streamlit as st

from sentineledu.config import Settings
from sentineledu.schemas.submission import Submission
from sentineledu.services.rag import RagStore
from sentineledu.services.storage import Storage
from sentineledu.workflow.steps import WorkflowDeps

# Tests replace this with a factory that returns WorkflowDeps wired to fake agents and a temp storage.
DEPS_FACTORY: Callable[[], WorkflowDeps] | None = None

PAGES = {
    "assignments": "sentineledu/ui/pages/assignments.py",
    "submit": "sentineledu/ui/pages/submit.py",
    "extraction_review": "sentineledu/ui/pages/extraction_review.py",
    "teacher_review": "sentineledu/ui/pages/teacher_review.py",
    "integrity": "sentineledu/ui/pages/integrity.py",
    "results": "sentineledu/ui/pages/results.py",
}


@st.cache_resource(show_spinner=False)
def _default_deps() -> WorkflowDeps:
    settings = Settings()
    return WorkflowDeps(storage=Storage(settings), settings=settings, rag=RagStore(settings))


def deps() -> WorkflowDeps:
    """WorkflowDeps for this session; stops the page with a readable error if settings are invalid."""
    if DEPS_FACTORY is not None:
        return DEPS_FACTORY()
    try:
        return _default_deps()
    except ValueError as e:  # e.g. GEMINI_API_KEY missing
        st.error(f"SentinelEdu is not configured: {e}")
        st.stop()


def storage() -> Storage:
    return deps().storage


# --- selection that survives refresh --------------------------------------------------------

def _remember(key: str, value: str | None) -> None:
    if value:
        st.query_params[key] = value
        st.session_state[f"selected_{key}"] = value
    elif key in st.query_params:
        del st.query_params[key]


def remembered(key: str) -> str | None:
    return st.query_params.get(key) or st.session_state.get(f"selected_{key}")


def pick_assignment(label: str = "Assignment", *, container=st) -> str | None:
    assignments = storage().list_assignments()
    if not assignments:
        container.info("No assignments yet. Create one on the **Assignments & Rubrics** page.")
        return None
    wanted = remembered("assignment")
    index = assignments.index(wanted) if wanted in assignments else 0
    choice = container.selectbox(label, assignments, index=index, key="pick_assignment")
    _remember("assignment", choice)
    return choice


def submission_label(sub: Submission) -> str:
    who = sub.student_ref or "no student ref"
    return f"{who} · {sub.submission_id} · {sub.status.value}"


def pick_submission(submissions: Sequence[Submission], label: str = "Submission", *, container=st) -> Submission | None:
    if not submissions:
        return None
    by_id = {s.submission_id: s for s in submissions}
    ids = list(by_id)
    wanted = remembered("submission")
    index = ids.index(wanted) if wanted in ids else 0
    choice = container.selectbox(label, ids, index=index, format_func=lambda i: submission_label(by_id[i]),
                                 key="pick_submission")
    _remember("submission", choice)
    return by_id[choice]


def go(page: str, *, submission_id: str | None = None, assignment_id: str | None = None) -> None:
    """Switch page, carrying the selection through session state."""
    if submission_id:
        st.session_state["selected_submission"] = submission_id
    if assignment_id:
        st.session_state["selected_assignment"] = assignment_id
    st.switch_page(PAGES[page])
