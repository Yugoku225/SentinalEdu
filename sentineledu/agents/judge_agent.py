"""Agent 3 — Judge & Feedback (ARCHITECTURE §4.3, TASKS 5.3, DECISIONS D-06).

Python runs the deterministic checks first, then Gemini (as an adversarial second reviewer) sees
the rubric, the student's answers, Agent 2's levels and quotes, the reference chunks and the check
flags, and returns verdicts, feedback and flags in ONE structured call. `scoring_rules.validate_judge`
then accepts only proposals that are valid, evidenced and at most ±1 level, and computes the
recommended marks. The Judge never finalises; the teacher does.

`checks_only_report()` builds a report without the LLM (deterministic flags, evaluator levels kept,
feedback generated from the evaluation) for OFFLINE_MODE or when Gemini is down.
"""
import logging
from pathlib import Path
from typing import Mapping

from sentineledu.agents.base import build_agent
from sentineledu.agents.evaluation_agent import QuestionContext, build_context
from sentineledu.agents.runner import LLMCallError, ResponseCache, run_structured
from sentineledu.config import Settings, get_settings
from sentineledu.schemas.evaluation import EvaluationResult
from sentineledu.schemas.extraction import ExtractedSubmission
from sentineledu.schemas.judge import Flag, JudgeDraft, JudgeReport
from sentineledu.schemas.rubric import Rubric
from sentineledu.services.rag import RagStore
from sentineledu.services.scoring_rules import is_attempted, run_checks, validate_judge

log = logging.getLogger("sentineledu.judge_agent")

PROMPT_VERSION = "judge_v1"
INSTRUCTIONS = (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md").read_text()
NO_LLM_MODEL_ID = "none"


class JudgeError(RuntimeError):
    """Gemini could not review the submission."""


def build_judge_agent(settings: Settings | None = None):
    settings = settings or get_settings()
    return build_agent(name="judge-agent", instructions=INSTRUCTIONS, output_schema=JudgeDraft,
                       model_id=settings.gemini_reasoning_model, settings=settings)


# --- prompt ---------------------------------------------------------------------------------

def _levels_line(criterion) -> str:
    ordered = sorted(criterion.levels, key=lambda lv: lv.marks)
    return " < ".join(f'"{lv.level_id}"' for lv in ordered)


def build_judge_prompt(rubric: Rubric, extraction: ExtractedSubmission, evaluation: EvaluationResult,
                       checks: list[Flag], context: Mapping[str, QuestionContext] | None = None) -> str:
    attempted = [q.question_id for q in rubric.questions if is_attempted(extraction.answer_for(q.question_id))]
    lines = [f"Assignment: {rubric.title}",
             "Review the evaluator's grading of the questions below and write feedback for the student.",
             f"Questions to review: {', '.join(attempted)}", ""]
    if checks:
        lines.append("Deterministic check flags (hints from rule-based checks, verify them yourself):")
        lines += [f"- [{f.severity}] {f.code}: {f.message}" for f in checks]
        lines.append("")
    for qid in attempted:
        q = rubric.question(qid)
        qe = evaluation.question(qid)
        answer = extraction.answer_for(qid)
        assert q is not None and qe is not None and answer is not None
        lines += [f"### Question {qid} ({q.max_marks:g} marks): {q.prompt}", f"Model answer: {q.model_answer}",
                  "Student's answer (verbatim, quote from this exactly):", "<answer>", answer.answer_text.strip(), "</answer>",
                  f"Advisory similarity to the model answer: {qe.sbert_sim_model_answer:.2f} (never decides a level).", "",
                  "Criteria and the evaluator's decisions:"]
        for c in q.criteria:
            ce = qe.criterion(c.criterion_id)
            lines.append(f'- criterion_id "{c.criterion_id}": {c.description}')
            if c.key_concepts:
                lines.append(f"  key concepts: {'; '.join(c.key_concepts)}")
            lines.append(f"  levels in order (lowest first): {_levels_line(c)}")
            for lv in sorted(c.levels, key=lambda lv: -lv.marks):
                lines.append(f'    - "{lv.level_id}" ({lv.label}): {lv.descriptor}')
            if ce is not None:
                lines.append(f'  evaluator chose level "{ce.selected_level_id}"')
                lines.append(f"  evaluator's quotes: {ce.evidence_quotes or 'none'}"
                             + ("" if ce.evidence_verified or ce.marks == 0 else " (NOT found in the answer)"))
                lines.append(f"  evaluator's rationale: {ce.rationale or 'none'}")
                lines.append(f"  concepts present: {ce.concepts_present or 'none'}; missing: {ce.missing_concepts or 'none'}; "
                             f"incorrect: {ce.incorrect_statements or 'none'}")
        if qe.validator_flags:
            lines.append(f"Validator corrections already applied: {qe.validator_flags}")
        if context and context.get(qid) and context[qid].chunks:
            lines.append("Reference excerpts:")
            for ch in context[qid].chunks:
                lines.append(f"[chunk_id {ch.chunk_id} | {ch.source} p.{ch.page}]")
                lines.append(ch.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- entry points ---------------------------------------------------------------------------

def checks_only_report(extraction: ExtractedSubmission, rubric: Rubric, evaluation: EvaluationResult, *,
                       settings: Settings | None = None, reason: str = "") -> JudgeReport:
    """A report without the LLM: deterministic flags, the evaluator's levels kept, generated feedback."""
    settings = settings or get_settings()
    checks = run_checks(evaluation, extraction, rubric, low_confidence=settings.extraction_review_threshold)
    if reason:
        checks.append(Flag(code="JUDGE_NOT_RUN", severity="review", message=reason))
    return validate_judge(JudgeDraft(), evaluation, rubric, extraction, checks=checks,
                          prompt_version=PROMPT_VERSION, model_id=NO_LLM_MODEL_ID)


def judge(
    extraction: ExtractedSubmission,
    rubric: Rubric,
    evaluation: EvaluationResult,
    *,
    context: Mapping[str, QuestionContext] | None = None,
    rag: RagStore | None = None,
    use_rag: bool = True,
    agent=None,
    settings: Settings | None = None,
    cache: ResponseCache | None = None,
    use_cache: bool = True,
) -> JudgeReport:
    """Second-review one graded submission. Raises JudgeError if Gemini is unavailable."""
    settings = settings or get_settings()
    checks = run_checks(evaluation, extraction, rubric, low_confidence=settings.extraction_review_threshold)
    if not any(is_attempted(extraction.answer_for(q.question_id)) for q in rubric.questions):
        log.info("Nothing to review for %s: every answer is blank or unreadable", extraction.submission_id)
        return validate_judge(JudgeDraft(), evaluation, rubric, extraction, checks=checks,
                              prompt_version=PROMPT_VERSION, model_id=NO_LLM_MODEL_ID)
    if settings.offline_mode:
        raise JudgeError("OFFLINE_MODE is on: the judge needs Gemini. Use checks_only_report() for a rule-based review.")

    if context is None and use_rag:
        context = build_context(extraction, rubric, rag=rag, settings=settings, use_rag=use_rag)
    if cache is None and use_cache:
        cache = ResponseCache(settings.cache_dir)
    agent = agent or build_judge_agent(settings)
    prompt = build_judge_prompt(rubric, extraction, evaluation, checks, context)
    try:
        result = run_structured(agent, prompt, JudgeDraft, prompt_version=PROMPT_VERSION, cache=cache)
    except LLMCallError as e:
        raise JudgeError(f"Gemini could not review submission {extraction.submission_id}: {e}") from e

    report = validate_judge(result.output, evaluation, rubric, extraction, checks=checks,
                            prompt_version=PROMPT_VERSION, model_id=result.model_id)
    changed = [f.message for f in report.flags if f.code == "JUDGE_ADJUSTMENT"]
    log.info("[judge-agent] %s: recommends %g/%g (evaluator %g) by %s%s", extraction.submission_id,
             report.recommended_total, report.max_total, evaluation.total, result.model_id,
             f" adjustments={len(changed)}" if changed else "")
    return report


if __name__ == "__main__":
    import argparse

    from sentineledu.services.rubric import load_rubric

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Second-review an evaluation.")
    ap.add_argument("extraction", type=Path)
    ap.add_argument("evaluation", type=Path)
    ap.add_argument("rubric", type=Path)
    args = ap.parse_args()
    report = judge(ExtractedSubmission.model_validate_json(args.extraction.read_text()), load_rubric(args.rubric),
                   EvaluationResult.model_validate_json(args.evaluation.read_text()))
    print(report.model_dump_json(indent=2))
