"""Agent 2 — Answer Evaluation (ARCHITECTURE §4.2, TASKS 4.7, DECISIONS D-06, D-07).

Python pre-computes the context (retrieved reference chunks and SBERT similarities per question),
Gemini picks a rubric level per criterion with quotes and citations in ONE structured call per
submission, and `services.scoring_rules.validate_evaluation` turns that into marks. Blank or
unreadable answers never reach the LLM; they get the lowest level with a NOT_ATTEMPTED flag.

CLI:  python -m sentineledu.agents.evaluation_agent extraction.json data/rubrics/sample_rubric.json [--reference notes.pdf]
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from sentineledu.agents.base import build_agent
from sentineledu.agents.runner import LLMCallError, ResponseCache, run_structured
from sentineledu.config import Settings, get_settings
from sentineledu.schemas.evaluation import EvaluationDraft, EvaluationResult
from sentineledu.schemas.extraction import ExtractedSubmission
from sentineledu.schemas.rubric import Rubric
from sentineledu.services import embeddings
from sentineledu.services.rag import RagError, RagStore, RetrievedChunk
from sentineledu.services.scoring_rules import is_attempted, not_attempted_question, validate_evaluation

log = logging.getLogger("sentineledu.evaluation_agent")

PROMPT_VERSION = "evaluation_v1"
INSTRUCTIONS = (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md").read_text()
NO_LLM_MODEL_ID = "none"  # recorded when every answer was blank/unreadable and Gemini was not called


class EvaluationError(RuntimeError):
    """The submission could not be graded (no LLM available, or Gemini failed after retries)."""


@dataclass
class QuestionContext:
    question_id: str
    attempted: bool
    answer_text: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    sim_model_answer: float = 0.0
    sim_best_chunk: float = 0.0

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]


def build_evaluation_agent(settings: Settings | None = None):
    settings = settings or get_settings()
    return build_agent(name="evaluation-agent", instructions=INSTRUCTIONS, output_schema=EvaluationDraft,
                       model_id=settings.gemini_reasoning_model, settings=settings)


# --- context: RAG + SBERT, computed in Python -------------------------------------------------

def build_context(extraction: ExtractedSubmission, rubric: Rubric, *, rag: RagStore | None = None,
                  settings: Settings | None = None, use_rag: bool = True) -> dict[str, QuestionContext]:
    """Per question: the verbatim answer, the top-k reference chunks and the two SBERT signals."""
    settings = settings or get_settings()
    rag = rag if rag is not None else (RagStore(settings) if use_rag else None)
    has_index = bool(rag and use_rag and rag.has_index(rubric.assignment_id))
    if use_rag and not has_index:
        log.info("No reference index for assignment %s; grading from the rubric alone", rubric.assignment_id)

    context: dict[str, QuestionContext] = {}
    for q in rubric.questions:
        answer = extraction.answer_for(q.question_id)
        if not is_attempted(answer):
            context[q.question_id] = QuestionContext(question_id=q.question_id, attempted=False)
            continue
        assert answer is not None
        text = answer.answer_text.strip()
        ctx = QuestionContext(question_id=q.question_id, attempted=True, answer_text=text)
        ctx.sim_model_answer = embeddings.similarity(text, q.model_answer, settings)
        if has_index:
            try:
                ctx.chunks = rag.retrieve(rubric.assignment_id, f"{q.prompt}\n{text}", settings.rag_top_k)  # type: ignore[union-attr]
            except RagError as e:
                log.warning("Retrieval failed for %s: %s", q.question_id, e)
            if ctx.chunks:
                answer_vec = embeddings.encode(text, settings)[0]
                chunk_vecs = embeddings.encode([c.text for c in ctx.chunks], settings)
                ctx.sim_best_chunk = max(embeddings.cosine(answer_vec, v) for v in chunk_vecs)
        context[q.question_id] = ctx
    return context


# --- prompt ---------------------------------------------------------------------------------

def _rubric_block(rubric: Rubric, question_id: str) -> list[str]:
    q = rubric.question(question_id)
    assert q is not None
    lines = [f"### Question {q.question_id} ({q.max_marks:g} marks): {q.prompt}",
             f"Model answer: {q.model_answer}", "Criteria:"]
    for c in q.criteria:
        lines.append(f"- criterion_id \"{c.criterion_id}\": {c.description}")
        if c.key_concepts:
            lines.append(f"  key concepts: {'; '.join(c.key_concepts)}")
        lines.append("  levels (choose exactly one level_id):")
        for lv in sorted(c.levels, key=lambda lv: -lv.marks):
            lines.append(f"    - level_id \"{lv.level_id}\" ({lv.label}): {lv.descriptor}")
    return lines


def build_prompt(rubric: Rubric, context: Mapping[str, QuestionContext]) -> str:
    attempted = [qid for qid in (q.question_id for q in rubric.questions) if context[qid].attempted]
    lines = [f"Assignment: {rubric.title}",
             "Grade the student's answers below. Return one entry per question listed here, with every "
             "criterion of that question. Questions not listed were blank or unreadable and are not graded.",
             f"Questions to grade: {', '.join(attempted)}", ""]
    for qid in attempted:
        ctx = context[qid]
        lines += _rubric_block(rubric, qid)
        lines += ["", "Student's answer (verbatim, quote from this exactly):", "<answer>", ctx.answer_text, "</answer>"]
        if ctx.chunks:
            lines.append("Reference excerpts (cite by chunk_id only if you relied on them):")
            for c in ctx.chunks:
                lines.append(f"[chunk_id {c.chunk_id} | {c.source} p.{c.page} | similarity {c.score:.2f}]")
                lines.append(c.text)
        else:
            lines.append("Reference excerpts: none available for this assignment.")
        lines.append(f"Advisory similarity signals: sbert_sim_model_answer={ctx.sim_model_answer:.2f}, "
                     f"sbert_sim_best_chunk={ctx.sim_best_chunk:.2f} (do not grade from these).")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- entry point ----------------------------------------------------------------------------

def _not_attempted_result(extraction: ExtractedSubmission, rubric: Rubric, context: Mapping[str, QuestionContext]) -> EvaluationResult:
    questions = [
        not_attempted_question(q, extraction.answer_for(q.question_id),
                               (context[q.question_id].sim_model_answer, context[q.question_id].sim_best_chunk))
        for q in rubric.questions
    ]
    return EvaluationResult(submission_id=extraction.submission_id, questions=questions,
                            total=sum(q.total for q in questions), max_total=rubric.max_total,
                            prompt_version=PROMPT_VERSION, model_id=NO_LLM_MODEL_ID,
                            extraction_version=extraction.version)


def evaluate(
    extraction: ExtractedSubmission,
    rubric: Rubric,
    *,
    context: Mapping[str, QuestionContext] | None = None,
    rag: RagStore | None = None,
    agent=None,
    settings: Settings | None = None,
    cache: ResponseCache | None = None,
    use_cache: bool = True,
) -> EvaluationResult:
    """Grade one extracted submission against the rubric (result is not yet saved)."""
    settings = settings or get_settings()
    if context is None:
        context = build_context(extraction, rubric, rag=rag, settings=settings)
    if not any(ctx.attempted for ctx in context.values()):
        log.info("Nothing to grade for %s: every answer is blank or unreadable", extraction.submission_id)
        return _not_attempted_result(extraction, rubric, context)
    if settings.offline_mode:
        raise EvaluationError("OFFLINE_MODE is on: grading needs Gemini and there is no offline substitute "
                              "that could support its marks with evidence.")

    if cache is None and use_cache:
        cache = ResponseCache(settings.cache_dir)
    agent = agent or build_evaluation_agent(settings)
    try:
        result = run_structured(agent, build_prompt(rubric, context), EvaluationDraft,
                                prompt_version=PROMPT_VERSION, cache=cache)
    except LLMCallError as e:
        raise EvaluationError(f"Gemini could not grade submission {extraction.submission_id}: {e}") from e

    evaluation = validate_evaluation(
        result.output, rubric, extraction,
        provided_chunk_ids={qid: ctx.chunk_ids for qid, ctx in context.items()},
        sims={qid: (ctx.sim_model_answer, ctx.sim_best_chunk) for qid, ctx in context.items()},
        prompt_version=PROMPT_VERSION, model_id=result.model_id,
    )
    flags = evaluation.all_flags
    log.info("[evaluation-agent] %s: %g/%g by %s%s", extraction.submission_id, evaluation.total,
             evaluation.max_total, result.model_id, f" flags={flags}" if flags else "")
    return evaluation


if __name__ == "__main__":
    import argparse

    from sentineledu.services.rubric import load_rubric

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Grade an extracted submission against a rubric.")
    ap.add_argument("extraction", type=Path, help="ExtractedSubmission JSON (e.g. from document_agent)")
    ap.add_argument("rubric", type=Path)
    ap.add_argument("--reference", nargs="*", type=Path, default=[], help="reference PDF/TXT to index first")
    args = ap.parse_args()
    rubric_ = load_rubric(args.rubric)
    extraction_ = ExtractedSubmission.model_validate_json(args.extraction.read_text())
    store = RagStore()
    if args.reference:
        store.ingest(rubric_.assignment_id, args.reference, replace=True)
    print(evaluate(extraction_, rubric_, rag=store).model_dump_json(indent=2))
