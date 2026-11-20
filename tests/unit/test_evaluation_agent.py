from pathlib import Path

import pytest

from sentineledu.agents.evaluation_agent import (
    NO_LLM_MODEL_ID,
    PROMPT_VERSION,
    EvaluationError,
    QuestionContext,
    build_context,
    build_prompt,
    evaluate,
)
from sentineledu.config import PROJECT_ROOT, Settings
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission
from sentineledu.services.rag import RagStore
from sentineledu.services.rubric import load_rubric
from sentineledu.services.scoring_rules import NOT_ATTEMPTED, UNKNOWN_CITATION, UNSUPPORTED_EVIDENCE, check_marks_are_rubric_derived
from fakes import FakeAgent, ProviderError

RUBRIC = load_rubric(PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json")
NOTES = Path(__file__).resolve().parents[1] / "fixtures" / "reference" / "photosynthesis_notes.txt"

ANSWERS = {
    "Q1": "Plants use light energy to turn carbon dioxide and water into glucose.",
    "Q2": "Chlorophyll, which is found in chloroplasts.",
    "Q3": "",
}


@pytest.fixture
def settings(tmp_path):
    """Online-looking settings (no network is used because the agent is faked)."""
    return Settings(_env_file=None, gemini_api_key="test-key", storage_dir=tmp_path / "storage")


def extraction(answers=ANSWERS) -> ExtractedSubmission:
    return ExtractedSubmission(submission_id="syn_03", engine="gemini", answers=[
        ExtractedAnswer(question_id=q, answer_text=t, legibility="clear" if t else "blank", confidence=0.95)
        for q, t in answers.items()
    ])


def draft(q1_c1_quote="turn carbon dioxide and water into glucose", q1_chunks=("d1.p1.c1",)):
    return {"questions": [
        {"question_id": "Q1", "criteria": [
            {"criterion_id": "C1", "selected_level_id": "full", "evidence_quotes": [q1_c1_quote],
             "reference_chunk_ids": list(q1_chunks), "concepts_present": ["light energy", "carbon dioxide", "water", "glucose"],
             "missing_concepts": [], "incorrect_statements": [], "rationale": "All inputs and the product are named."},
            {"criterion_id": "C2", "selected_level_id": "none", "evidence_quotes": [], "reference_chunk_ids": [],
             "concepts_present": [], "missing_concepts": ["oxygen released"], "incorrect_statements": [],
             "rationale": "Oxygen is not mentioned."},
        ]},
        {"question_id": "Q2", "criteria": [
            {"criterion_id": "C1", "selected_level_id": "full", "evidence_quotes": ["Chlorophyll"], "rationale": "named"},
            {"criterion_id": "C2", "selected_level_id": "full", "evidence_quotes": ["found in chloroplasts"], "rationale": "located"},
        ]},
    ]}


# --- context and prompt ---------------------------------------------------------------------

def test_context_without_index_has_sims_but_no_chunks(settings):
    ctx = build_context(extraction(), RUBRIC, settings=settings)
    assert ctx["Q1"].attempted and ctx["Q1"].chunks == [] and ctx["Q1"].sim_best_chunk == 0.0
    assert 0.5 < ctx["Q1"].sim_model_answer <= 1.0
    assert not ctx["Q3"].attempted and ctx["Q3"].answer_text == ""


def test_context_with_index_retrieves_chunks(settings):
    store = RagStore(settings)
    store.ingest(RUBRIC.assignment_id, [NOTES])
    ctx = build_context(extraction(), RUBRIC, rag=store, settings=settings)
    assert len(ctx["Q1"].chunks) == settings.rag_top_k
    assert any("6.1 What is photosynthesis" in c.text for c in ctx["Q1"].chunks)
    assert ctx["Q1"].sim_best_chunk > 0.3
    assert ctx["Q3"].chunks == []  # blank answers are not looked up


def test_prompt_lists_levels_verbatim_answer_and_skips_blank(settings):
    ctx = build_context(extraction(), RUBRIC, settings=settings, use_rag=False)
    prompt = build_prompt(RUBRIC, ctx)
    assert "Questions to grade: Q1, Q2" in prompt
    assert "### Question Q3" not in prompt
    assert 'level_id "partial"' in prompt and 'criterion_id "C2"' in prompt
    assert "<answer>\n" + ANSWERS["Q1"] + "\n</answer>" in prompt
    assert "do not grade from these" in prompt
    assert "Reference excerpts: none available" in prompt
    assert "marks" not in prompt.split("Student's answer")[1].split("</answer>")[0].lower()


def test_prompt_includes_chunk_ids_when_available():
    ctx = {"Q1": QuestionContext("Q1", True, ANSWERS["Q1"]), "Q2": QuestionContext("Q2", False),
           "Q3": QuestionContext("Q3", False)}
    from sentineledu.services.rag import RetrievedChunk
    ctx["Q1"].chunks = [RetrievedChunk(chunk_id="d1.p1.c1", source="n.txt", page=1, text="Plants make food.", score=0.7)]
    prompt = build_prompt(RUBRIC, ctx)
    assert "[chunk_id d1.p1.c1 | n.txt p.1 | similarity 0.70]" in prompt


# --- evaluate ---------------------------------------------------------------------------------

def test_evaluate_offline_with_fake_agent(settings):
    agent = FakeAgent(draft(), model_id="fake-gemini")
    result = evaluate(extraction(), RUBRIC, agent=agent, settings=settings, use_cache=False)
    assert len(agent.calls) == 1
    assert "Questions to grade: Q1, Q2" in agent.calls[0]["input"]
    assert result.model_id == "fake-gemini" and result.prompt_version == PROMPT_VERSION
    assert result.total == 4 and result.max_total == 7
    q1 = result.question("Q1")
    assert q1.criterion("C1").marks == 2 and q1.criterion("C1").evidence_verified
    assert q1.criterion("C1").reference_chunk_ids == []  # no index: nothing was provided, so the citation is dropped
    assert f"{UNKNOWN_CITATION}:C1" in q1.validator_flags
    assert result.question("Q3").validator_flags == [NOT_ATTEMPTED]
    assert check_marks_are_rubric_derived(result, RUBRIC) == []


def test_citation_kept_when_chunk_was_provided(settings):
    store = RagStore(settings)
    store.ingest(RUBRIC.assignment_id, [NOTES])
    ctx = build_context(extraction(), RUBRIC, rag=store, settings=settings)
    provided = ctx["Q1"].chunk_ids
    agent = FakeAgent(draft(q1_chunks=(provided[0], "d7.p7.c7")))
    result = evaluate(extraction(), RUBRIC, context=ctx, agent=agent, settings=settings, use_cache=False)
    assert result.question("Q1").criterion("C1").reference_chunk_ids == [provided[0]]
    assert result.question("Q1").sbert_sim_best_chunk > 0


def test_fabricated_quote_is_downgraded_end_to_end(settings):
    agent = FakeAgent(draft(q1_c1_quote="Plants make glucose and release oxygen through stomata"))
    result = evaluate(extraction(), RUBRIC, agent=agent, settings=settings, use_cache=False)
    assert result.question("Q1").criterion("C1").marks == 0
    assert f"{UNSUPPORTED_EVIDENCE}:C1" in result.question("Q1").validator_flags
    assert result.total == 2


def test_all_blank_skips_the_llm(settings):
    agent = FakeAgent()  # any call would raise
    result = evaluate(extraction({"Q1": "", "Q2": "", "Q3": ""}), RUBRIC, agent=agent, settings=settings, use_cache=False)
    assert agent.calls == [] and result.model_id == NO_LLM_MODEL_ID
    assert result.total == 0 and all(q.validator_flags == [NOT_ATTEMPTED] for q in result.questions)


def test_offline_mode_refuses_to_grade(offline_settings):
    with pytest.raises(EvaluationError, match="OFFLINE_MODE"):
        evaluate(extraction(), RUBRIC, agent=FakeAgent(draft()), settings=offline_settings, use_cache=False)


def test_llm_failure_becomes_evaluation_error(settings):
    agent = FakeAgent(ProviderError("400 INVALID_ARGUMENT"))
    with pytest.raises(EvaluationError, match="could not grade"):
        evaluate(extraction(), RUBRIC, agent=agent, settings=settings, use_cache=False)


def test_cache_prevents_second_call(settings):
    from sentineledu.agents.runner import ResponseCache
    cache = ResponseCache(settings.cache_dir)
    agent = FakeAgent(draft())
    first = evaluate(extraction(), RUBRIC, agent=agent, settings=settings, cache=cache)
    second = evaluate(extraction(), RUBRIC, agent=agent, settings=settings, cache=cache)
    assert len(agent.calls) == 1 and first.total == second.total
