"""Shared pieces for the Phase 6 workflow tests: fake agent answers for the synthetic clean page."""
from pathlib import Path

from sentineledu.config import PROJECT_ROOT, Settings
from sentineledu.services.rubric import load_rubric
from sentineledu.services.storage import Storage
from sentineledu.workflow.steps import WorkflowDeps
from fakes import FakeAgent

RUBRIC = load_rubric(PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json")
PAGE = PROJECT_ROOT / "tests" / "fixtures" / "synthetic" / "syn_01_clean_p1.jpg"

A1 = ("Photosynthesis is the process by which green plants make their own food using sunlight, "
      "carbon dioxide and water. Oxygen is released as a by-product.")
A2 = "The green pigment is chlorophyll. It is found in the chloroplasts of leaf cells."
A3 = "Plants need water for photosynthesis and to carry minerals from the roots to the leaves."


def doc_draft(q2_unreadable: bool = False) -> dict:
    q2 = ({"question_id": "Q2", "answer_text": "[[ILLEGIBLE]]", "legibility": "unreadable", "confidence": 0.2, "page_refs": [1]}
          if q2_unreadable else
          {"question_id": "Q2", "answer_text": A2, "legibility": "clear", "confidence": 0.95, "page_refs": [1]})
    return {"answers": [
        {"question_id": "Q1", "answer_text": A1, "legibility": "clear", "confidence": 0.95, "page_refs": [1]},
        q2,
        {"question_id": "Q3", "answer_text": A3, "legibility": "clear", "confidence": 0.95, "page_refs": [1]},
    ], "unmatched_text": "", "issues": []}


def eval_draft() -> dict:
    return {"questions": [
        {"question_id": "Q1", "criteria": [
            {"criterion_id": "C1", "selected_level_id": "full",
             "evidence_quotes": ["make their own food using sunlight, carbon dioxide and water"], "rationale": "complete"},
            {"criterion_id": "C2", "selected_level_id": "full", "evidence_quotes": ["Oxygen is released"], "rationale": "oxygen"}]},
        {"question_id": "Q2", "criteria": [
            {"criterion_id": "C1", "selected_level_id": "full", "evidence_quotes": ["chlorophyll"], "rationale": "named"},
            {"criterion_id": "C2", "selected_level_id": "full", "evidence_quotes": ["found in the chloroplasts"], "rationale": "located"}]},
        {"question_id": "Q3", "criteria": [
            {"criterion_id": "C1", "selected_level_id": "full",
             "evidence_quotes": ["for photosynthesis", "carry minerals from the roots"], "rationale": "two reasons"}]},
    ]}


def judge_draft() -> dict:
    return {"questions": [
        {"question_id": q, "verdicts": [{"criterion_id": c, "verdict": "agree", "justification": "correct"} for c in cs],
         "strengths": ["Clear and complete."], "improvements": ["Add an example."]}
        for q, cs in (("Q1", ["C1", "C2"]), ("Q2", ["C1", "C2"]), ("Q3", ["C1"]))
    ], "summary_feedback": "Excellent work.", "flags": []}


def make_deps(tmp_path: Path, *, doc=(), ev=(), jd=(), store_rubric: bool = True, **kw) -> WorkflowDeps:
    settings = Settings(_env_file=None, gemini_api_key="test-key", storage_dir=tmp_path / "storage")
    storage = Storage(settings)
    if store_rubric:
        storage.save_rubric(RUBRIC)
    return WorkflowDeps(storage=storage, settings=settings, document_agent=FakeAgent(*doc, name="document-agent"),
                        evaluation_agent=FakeAgent(*ev, name="evaluation-agent"),
                        judge_agent=FakeAgent(*jd, name="judge-agent"), use_cache=False, **kw)


def new_submission(deps: WorkflowDeps, files=(PAGE,)) -> str:
    return deps.storage.create_submission(RUBRIC.assignment_id, list(files), student_ref="anon").submission_id
