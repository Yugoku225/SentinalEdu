"""The dashboard pages with real Gemini on a temporary database (TASKS 8.3, 8.5, 8.7). Run: pytest -m live tests/integration/test_dashboard_live.py"""
import pytest
from streamlit.testing.v1 import AppTest

from sentineledu.config import PROJECT_ROOT, Settings
from sentineledu.schemas.submission import SubmissionStatus as S
from sentineledu.services.export import results_rows
from sentineledu.services.rag import RagStore
from sentineledu.services.storage import Storage
from sentineledu.ui import state
from sentineledu.workflow.steps import WorkflowDeps
from wf_helpers import PAGE, RUBRIC

pytestmark = pytest.mark.live
PAGES = PROJECT_ROOT / "sentineledu" / "ui" / "pages"
NOTES = PROJECT_ROOT / "tests" / "fixtures" / "reference" / "photosynthesis_notes.txt"


def test_submit_review_approve_export_with_gemini(live_settings, tmp_path):
    settings = Settings(storage_dir=tmp_path / "storage")
    deps = WorkflowDeps(storage=Storage(settings), settings=settings, rag=RagStore(settings))
    deps.storage.save_rubric(RUBRIC)
    deps.rag.ingest(RUBRIC.assignment_id, [NOTES])
    sid = deps.storage.create_submission(RUBRIC.assignment_id, [PAGE], student_ref="LIVE1").submission_id
    state.DEPS_FACTORY = lambda: deps
    try:
        at = AppTest.from_file(str(PAGES / "submit.py"), default_timeout=600).run()
        next(b for b in at.button if b.label == "Run again").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert deps.storage.get_submission(sid).status == S.PENDING_TEACHER_REVIEW

        at = AppTest.from_file(str(PAGES / "teacher_review.py"), default_timeout=120).run()
        assert not at.exception
        assert "<mark>" in " ".join(m.value for m in at.markdown)  # real evidence quotes highlighted
        at.text_input(key="reviewer_name").set_value("Live Tester").run()
        next(b for b in at.button if b.label == "Approve & Finalise").click().run()
        assert deps.storage.get_submission(sid).status == S.FINALIZED

        rows = results_rows(deps.storage, RUBRIC.assignment_id)
        assert len(rows) == 1 and rows[0]["reviewer"] == "Live Tester"
        assert rows[0]["total"] == deps.storage.latest_judge_report(sid).recommended_total
    finally:
        state.DEPS_FACTORY = None
