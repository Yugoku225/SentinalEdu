from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image as PILImage

import sentineledu.agents.document_agent as doc
from fakes import FakeAgent, ProviderError
from sentineledu.agents.document_agent import (
    DocumentDraft,
    DocumentExtractionError,
    build_prompt,
    extract_submission,
    normalise_draft,
)
from sentineledu.schemas import ExtractedAnswer, ExtractedSubmission
from sentineledu.services.documents import prepare_inputs

ROOT = Path(__file__).resolve().parents[2]
TYPED_PDF = ROOT / "tests" / "fixtures" / "legacy" / "sample1_key.pdf"


@dataclass
class Q:
    question_id: str
    prompt: str


QUESTIONS = [Q("Q1", "Define photosynthesis."), Q("Q2a", "Name the pigment."), Q("Q2b", "Where is it found?")]
QIDS = [q.question_id for q in QUESTIONS]


def a(qid, text="answer", legibility="clear", confidence=0.9, pages=(1,), **kw):
    return {"question_id": qid, "answer_text": text, "legibility": legibility, "confidence": confidence,
            "page_refs": list(pages), **kw}


def norm(answers, unmatched="", page_count=2, **kw):
    draft = DocumentDraft.model_validate({"answers": answers, "unmatched_text": unmatched, "issues": []})
    return normalise_draft(draft, submission_id="S1", expected_ids=QIDS, engine="gemini",
                           page_count=page_count, model_id="m", **kw)


@pytest.fixture
def page(tmp_path):
    p = tmp_path / "page.png"
    PILImage.new("RGB", (100, 100), "white").save(p)
    return p


@pytest.fixture
def settings(offline_settings):
    from pydantic import SecretStr
    return offline_settings.model_copy(update={"offline_mode": False, "gemini_api_key": SecretStr("k")})


# --- normalisation ---

@pytest.mark.parametrize("label, expected", [
    ("Q1", "Q1"), ("q1", "Q1"), ("1", "Q1"), ("Question 1", "Q1"), ("Ans 1", "Q1"),
    ("Q2a", "Q2a"), ("2(a)", "Q2a"), ("q 2 b", "Q2b"), ("Ans. 2b", "Q2b"),
])
def test_question_labels_resolved(label, expected):
    out = norm([a(label)])
    assert [x.question_id for x in out.answers] == [expected]


def test_unknown_label_goes_to_unmatched_text():
    out = norm([a("Q1"), a("Q7", "extra writing")], unmatched="doodle")
    assert [x.question_id for x in out.answers] == ["Q1"]
    assert "doodle" in out.unmatched_text and "[labelled 'Q7'] extra writing" in out.unmatched_text


def test_missing_questions_listed_and_answers_in_rubric_order():
    out = norm([a("Q2b"), a("Q1")])
    assert [x.question_id for x in out.answers] == ["Q1", "Q2b"]
    assert out.missing_question_ids == ["Q2a"]


def test_answer_spanning_pages_is_merged():
    out = norm([a("Q1", "first half", confidence=0.9, pages=[1]),
                a("Q1", "second [[ILLEGIBLE]] half", legibility="partial", confidence=0.7, pages=[2],
                  contains_diagram=True)])
    (q1,) = out.answers
    assert q1.answer_text == "first half\nsecond [[ILLEGIBLE]] half"
    assert (q1.legibility, q1.confidence, q1.page_refs, q1.contains_diagram) == ("partial", 0.7, [1, 2], True)


@pytest.mark.parametrize("text, claimed, expected", [
    ("", "clear", "blank"),
    ("   ", "partial", "blank"),
    ("[[ILLEGIBLE]] [[ILLEGIBLE]].", "partial", "unreadable"),
    ("plants [[ILLEGIBLE]] light", "clear", "partial"),
    ("plants use light", "blank", "clear"),
    ("plants use light", "partial", "partial"),
])
def test_legibility_made_consistent_with_text(text, claimed, expected):
    assert norm([a("Q1", text, legibility=claimed)]).answers[0].legibility == expected


def test_invalid_page_refs_dropped():
    assert norm([a("Q1", pages=[0, 2, 2, 9])], page_count=2).answers[0].page_refs == [2]


def test_ambiguous_canonical_ids_do_not_guess():
    draft = DocumentDraft.model_validate({"answers": [a("1")]})
    out = normalise_draft(draft, submission_id="S", expected_ids=["Q1", "1"], engine="gemini", page_count=1)
    assert [x.question_id for x in out.answers] == ["1"]  # exact match wins
    draft = DocumentDraft.model_validate({"answers": [a("q.1")]})
    out = normalise_draft(draft, submission_id="S", expected_ids=["Q1", "Question1"], engine="gemini", page_count=1)
    assert out.answers == [] and "q.1" in out.unmatched_text


# --- prompt ---

def test_prompt_lists_questions_and_pages(page):
    prompt = build_prompt(QUESTIONS, prepare_inputs([page]), ["page01_scan.png"])
    assert "- Q2a: Name the pigment." in prompt
    assert "1 page(s)" in prompt and "page01_scan.png" in prompt
    assert "<document>" not in prompt


def test_prompt_embeds_typed_pdf_text():
    prompt = build_prompt(QUESTIONS, prepare_inputs([TYPED_PDF]), ["key.pdf"])
    assert "<document>" in prompt and "quick brown fox" in prompt


# --- end to end with a fake agent ---

def test_extract_with_gemini(page, settings):
    agent = FakeAgent({"answers": [a("1", "Plants make food"), a("2a", "", legibility="blank")],
                       "issues": ["Page slightly rotated"]}, model_id="gemini-3.7-flash")
    out = extract_submission("S1", [page], QUESTIONS, agent=agent, settings=settings, use_cache=False)
    assert isinstance(out, ExtractedSubmission)
    assert out.engine == "gemini" and out.model_id == "gemini-3.7-flash" and out.prompt_version == "document_v1"
    assert [x.question_id for x in out.answers] == ["Q1", "Q2a"]
    assert out.missing_question_ids == ["Q2b"]
    assert out.issues == ["Page slightly rotated"]
    call = agent.calls[0]
    assert len(call["images"]) == 1 and call["files"] is None


def test_typed_pdf_engine_is_pdf_text(settings):
    agent = FakeAgent({"answers": [a("Q1", "The quick brown fox")]})
    out = extract_submission("S1", [TYPED_PDF], QUESTIONS, agent=agent, settings=settings, use_cache=False)
    assert out.engine == "pdf_text"
    assert agent.calls[0]["images"] is None and agent.calls[0]["files"] is None


def _fake_trocr(monkeypatch, text="The quick brown dog", conf=0.7):
    calls = []

    def fake(image_path, model_path):
        calls.append(image_path)
        return text, conf
    monkeypatch.setattr("sentineledu.services.ocr_fallback._legacy_extract", fake)
    return calls


def test_gemini_failure_falls_back_to_trocr(page, settings, monkeypatch):
    calls = _fake_trocr(monkeypatch)
    agent = FakeAgent(ProviderError("404 NOT_FOUND model retired"))
    out = extract_submission("S1", [page], QUESTIONS, agent=agent, settings=settings, use_cache=False)
    assert out.engine == "trocr" and len(calls) == 1
    assert out.answers == [] and "The quick brown dog" in out.unmatched_text
    assert out.missing_question_ids == QIDS
    assert "Gemini could not read" in out.issues[0]


def test_offline_mode_never_calls_gemini(page, offline_settings, monkeypatch):
    _fake_trocr(monkeypatch)
    agent = FakeAgent()  # any call would fail
    out = extract_submission("S1", [page], QUESTIONS, agent=agent, settings=offline_settings)
    assert out.engine == "trocr" and agent.calls == []


def test_offline_typed_pdf_keeps_text_without_ocr(offline_settings):
    out = extract_submission("S1", [TYPED_PDF], QUESTIONS, settings=offline_settings)
    assert out.engine == "pdf_text" and "quick brown fox" in out.unmatched_text


def test_scanned_pdf_with_gemini_down_raises(tmp_path, settings):
    scan = tmp_path / "scan.pdf"
    PILImage.new("RGB", (100, 100), "white").save(scan, format="PDF")
    with pytest.raises(DocumentExtractionError, match="only handles images"):
        extract_submission("S1", [scan], QUESTIONS, agent=FakeAgent(ProviderError("400 INVALID_ARGUMENT")),
                           settings=settings, use_cache=False)


def test_fallback_model_noted(page, settings):
    class Fallback(FakeAgent):
        def run(self, input, **kw):
            out = super().run(input, **kw)
            self.fallback_events.append("gemini-3.5-flash")
            return out
    agent = Fallback({"answers": [a("Q1")]})
    agent.fallback_events = []
    out = extract_submission("S1", [page], QUESTIONS, agent=agent, settings=settings, use_cache=False)
    assert out.model_id == "gemini-3.5-flash"
    assert "Read by fallback model gemini-3.5-flash." in out.issues


def test_prompt_file_is_versioned():
    assert doc.PROMPT_VERSION == "document_v1"
    assert "[[ILLEGIBLE]]" in doc.INSTRUCTIONS and "verbatim" in doc.INSTRUCTIONS
