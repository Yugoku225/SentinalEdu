"""Deterministic marking rules (ARCHITECTURE §4.2, DECISIONS D-06): LLMs choose, Python computes.

`validate_evaluation` turns the LLM's `EvaluationDraft` (levels + quotes + citations) into an
`EvaluationResult` whose marks come only from the rubric levels:

1. `selected_level_id` must exist in the criterion's levels, else lowest level + INVALID_LEVEL.
2. marks = level.marks. The LLM never supplies a number.
3. If marks > 0, at least one evidence quote must fuzzy-match the student's answer
   (difflib ratio >= 0.85 on a sliding window), else lowest level + UNSUPPORTED_EVIDENCE.
4. reference_chunk_ids must be among the chunks that were shown; unknown ids are dropped
   and flagged UNKNOWN_CITATION.
5. Question total = sum of criterion marks <= question.max_marks; assignment total = sum.

Extra rules: a question whose answer is blank/unreadable/missing is NOT_ATTEMPTED (lowest
level everywhere, no LLM output used); a criterion or question the LLM left out gets the
lowest level with MISSING_CRITERION / MISSING_QUESTION; unknown ids are dropped.
"""
import re
from difflib import SequenceMatcher
from typing import Iterable, Mapping

from sentineledu.schemas.evaluation import (
    CriterionDraft,
    CriterionEvaluation,
    EvaluationDraft,
    EvaluationResult,
    QuestionEvaluation,
)
from sentineledu.schemas.extraction import ILLEGIBLE_MARKER, ExtractedAnswer, ExtractedSubmission
from sentineledu.schemas.rubric import Criterion, Question, Rubric

QUOTE_MATCH_THRESHOLD = 0.85
MIN_QUOTE_CHARS = 3
_EPS = 1e-6

# validator flag codes (question-level flags are bare, criterion-level ones are "CODE:criterion_id")
INVALID_LEVEL = "INVALID_LEVEL"
UNSUPPORTED_EVIDENCE = "UNSUPPORTED_EVIDENCE"
UNKNOWN_CITATION = "UNKNOWN_CITATION"
MISSING_CRITERION = "MISSING_CRITERION"
UNKNOWN_CRITERION = "UNKNOWN_CRITERION"
MISSING_QUESTION = "MISSING_QUESTION"
NOT_ATTEMPTED = "NOT_ATTEMPTED"
TOTAL_CLAMPED = "TOTAL_CLAMPED"
NO_EVIDENCE_AT_ZERO = "NO_EVIDENCE_AT_ZERO"  # informational: a 0-mark level with no quote (allowed)

NOT_ATTEMPTED_LEGIBILITY = ("blank", "unreadable")


def normalise(text: str) -> str:
    """Lower-case, drop illegible markers, collapse whitespace and most punctuation."""
    text = text.replace(ILLEGIBLE_MARKER, " ").lower()
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[^\w\s'%^+=/-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_match_ratio(quote: str, answer_text: str) -> float:
    """Best difflib ratio of the quote against same-length windows of the answer (1.0 if contained)."""
    q, a = normalise(quote), normalise(answer_text)
    if len(q) < MIN_QUOTE_CHARS or not a:
        return 0.0
    if q in a:
        return 1.0
    if len(q) >= len(a):
        return SequenceMatcher(None, q, a, autojunk=False).ratio()
    window = len(q)
    step = max(1, window // 5)
    best = 0.0
    for start in range(0, len(a) - window + 1, step):
        best = max(best, SequenceMatcher(None, q, a[start:start + window], autojunk=False).ratio())
        if best >= 0.999:
            break
    # the last window, in case the stepping skipped it
    best = max(best, SequenceMatcher(None, q, a[len(a) - window:], autojunk=False).ratio())
    return best


def verify_quote(quote: str, answer_text: str, threshold: float = QUOTE_MATCH_THRESHOLD) -> bool:
    return quote_match_ratio(quote, answer_text) >= threshold


def is_attempted(answer: ExtractedAnswer | None) -> bool:
    return answer is not None and answer.legibility not in NOT_ATTEMPTED_LEGIBILITY and bool(answer.answer_text.strip())


def compute_marks(criterion: Criterion, level_id: str) -> float:
    level = criterion.level(level_id)
    if level is None:
        raise ValueError(f"criterion {criterion.criterion_id} has no level {level_id!r}")
    return float(level.marks)


def _lowest(criterion: Criterion, draft: CriterionDraft | None, rationale: str) -> CriterionEvaluation:
    lv = criterion.lowest_level
    return CriterionEvaluation(
        criterion_id=criterion.criterion_id, selected_level_id=lv.level_id, marks=float(lv.marks),
        evidence_quotes=list(draft.evidence_quotes) if draft else [], evidence_verified=False,
        reference_chunk_ids=[], concepts_present=list(draft.concepts_present) if draft else [],
        missing_concepts=list(draft.missing_concepts) if draft else [c for c in criterion.key_concepts],
        incorrect_statements=list(draft.incorrect_statements) if draft else [], rationale=rationale,
    )


def validate_criterion(
    criterion: Criterion,
    draft: CriterionDraft | None,
    answer_text: str,
    provided_chunk_ids: Iterable[str],
    flags: list[str],
    *,
    quote_threshold: float = QUOTE_MATCH_THRESHOLD,
) -> CriterionEvaluation:
    cid = criterion.criterion_id
    if draft is None:
        flags.append(f"{MISSING_CRITERION}:{cid}")
        return _lowest(criterion, None, "The evaluator returned nothing for this criterion, so the lowest level applies.")

    # 4. citations must be among the chunks the model was shown
    allowed = set(provided_chunk_ids)
    citations = [c for c in draft.reference_chunk_ids if c in allowed]
    if len(citations) != len(draft.reference_chunk_ids):
        flags.append(f"{UNKNOWN_CITATION}:{cid}")

    # 1-2. the level decides the marks
    level = criterion.level(draft.selected_level_id)
    if level is None:
        flags.append(f"{INVALID_LEVEL}:{cid}")
        out = _lowest(criterion, draft, f"Evaluator chose unknown level {draft.selected_level_id!r}; "
                                        f"lowest level applied. {draft.rationale}".strip())
        return out.model_copy(update={"reference_chunk_ids": citations})
    marks = float(level.marks)

    # 3. awarded marks need a verified quote
    verified = any(verify_quote(q, answer_text, quote_threshold) for q in draft.evidence_quotes)
    if marks > _EPS and not verified:
        flags.append(f"{UNSUPPORTED_EVIDENCE}:{cid}")
        out = _lowest(criterion, draft, f"Level {level.level_id!r} was chosen but no quoted evidence matches the "
                                        f"student's answer; downgraded. {draft.rationale}".strip())
        return out.model_copy(update={"reference_chunk_ids": citations})
    if marks <= _EPS and not draft.evidence_quotes:
        flags.append(f"{NO_EVIDENCE_AT_ZERO}:{cid}")

    return CriterionEvaluation(
        criterion_id=cid, selected_level_id=level.level_id, marks=marks,
        evidence_quotes=list(draft.evidence_quotes), evidence_verified=verified,
        reference_chunk_ids=citations, concepts_present=list(draft.concepts_present),
        missing_concepts=list(draft.missing_concepts), incorrect_statements=list(draft.incorrect_statements),
        rationale=draft.rationale,
    )


def not_attempted_question(question: Question, answer: ExtractedAnswer | None, sims: tuple[float, float] = (0.0, 0.0)) -> QuestionEvaluation:
    reason = ("no answer was found" if answer is None else f"the answer is {answer.legibility}")
    criteria = [_lowest(c, None, f"Not graded: {reason}.") for c in question.criteria]
    return QuestionEvaluation(
        question_id=question.question_id, criteria=criteria, total=sum(c.marks for c in criteria),
        max_marks=question.max_marks, sbert_sim_model_answer=sims[0], sbert_sim_best_chunk=sims[1],
        validator_flags=[NOT_ATTEMPTED],
    )


def validate_question(
    question: Question,
    draft: EvaluationDraft,
    answer: ExtractedAnswer | None,
    provided_chunk_ids: Iterable[str] = (),
    sims: tuple[float, float] = (0.0, 0.0),
    *,
    quote_threshold: float = QUOTE_MATCH_THRESHOLD,
) -> QuestionEvaluation:
    if not is_attempted(answer):
        return not_attempted_question(question, answer, sims)
    assert answer is not None
    flags: list[str] = []
    qdraft = draft.question(question.question_id)
    drafts: dict[str, CriterionDraft] = {}
    if qdraft is None:
        flags.append(MISSING_QUESTION)
    else:
        for cd in qdraft.criteria:
            if question.criterion(cd.criterion_id) is None:
                flags.append(f"{UNKNOWN_CRITERION}:{cd.criterion_id}")
            elif cd.criterion_id not in drafts:  # first occurrence wins
                drafts[cd.criterion_id] = cd
    provided = list(provided_chunk_ids)
    criteria = [
        validate_criterion(c, drafts.get(c.criterion_id), answer.answer_text, provided, flags,
                           quote_threshold=quote_threshold)
        for c in question.criteria
    ]
    total = sum(c.marks for c in criteria)
    if total > question.max_marks + _EPS:  # cannot happen with a valid rubric; defence in depth
        flags.append(TOTAL_CLAMPED)
        total = question.max_marks
    return QuestionEvaluation(
        question_id=question.question_id, criteria=criteria, total=total, max_marks=question.max_marks,
        sbert_sim_model_answer=sims[0], sbert_sim_best_chunk=sims[1], validator_flags=flags,
    )


def validate_evaluation(
    draft: EvaluationDraft,
    rubric: Rubric,
    extraction: ExtractedSubmission,
    provided_chunk_ids: Mapping[str, Iterable[str]] | None = None,
    *,
    sims: Mapping[str, tuple[float, float]] | None = None,
    prompt_version: str,
    model_id: str,
    quote_threshold: float = QUOTE_MATCH_THRESHOLD,
) -> EvaluationResult:
    """Apply rules 1-5 to every question of the rubric and return the validated result."""
    provided_chunk_ids = provided_chunk_ids or {}
    sims = sims or {}
    questions = [
        validate_question(q, draft, extraction.answer_for(q.question_id),
                          provided_chunk_ids.get(q.question_id, ()), sims.get(q.question_id, (0.0, 0.0)),
                          quote_threshold=quote_threshold)
        for q in rubric.questions
    ]
    return EvaluationResult(
        submission_id=extraction.submission_id, questions=questions,
        total=sum(q.total for q in questions), max_total=rubric.max_total,
        prompt_version=prompt_version, model_id=model_id, extraction_version=extraction.version,
    )


def check_marks_are_rubric_derived(result: EvaluationResult, rubric: Rubric) -> list[str]:
    """Audit helper: every mark must equal a rubric level, awarded marks must be verified, totals must add up."""
    problems = []
    for qe in result.questions:
        q = rubric.question(qe.question_id)
        if q is None:
            problems.append(f"{qe.question_id}: not in rubric")
            continue
        for ce in qe.criteria:
            c = q.criterion(ce.criterion_id)
            lv = c.level(ce.selected_level_id) if c else None
            if lv is None or abs(lv.marks - ce.marks) > _EPS:
                problems.append(f"{qe.question_id}/{ce.criterion_id}: marks {ce.marks} do not match a rubric level")
            if ce.marks > _EPS and not ce.evidence_verified:
                problems.append(f"{qe.question_id}/{ce.criterion_id}: awarded without verified evidence")
        if abs(sum(c.marks for c in qe.criteria) - qe.total) > _EPS or qe.total > q.max_marks + _EPS:
            problems.append(f"{qe.question_id}: total {qe.total} inconsistent")
    if abs(sum(q.total for q in result.questions) - result.total) > _EPS or result.total > rubric.max_total + _EPS:
        problems.append(f"assignment total {result.total} inconsistent")
    return problems


# =============================================================================================
# Phase 5 — deterministic checks and the Judge validator (ARCHITECTURE §4.3, TASKS 5.1 / 5.4)
# =============================================================================================

from sentineledu.schemas.judge import (  # noqa: E402  (kept here so the file reads top-down by phase)
    CriterionVerdict,
    Flag,
    FlagDraft,
    JudgeDraft,
    JudgeReport,
    QuestionFeedback,
    VerdictDraft,
)

# deterministic check codes
SUM_MISMATCH = "SUM_MISMATCH"
OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
SBERT_LEVEL_DISAGREEMENT = "SBERT_LEVEL_DISAGREEMENT"
NO_EVIDENCE = "NO_EVIDENCE"
LOW_OCR_CONFIDENCE = "LOW_OCR_CONFIDENCE"
ILLEGIBLE_IN_SCORED_ANSWER = "ILLEGIBLE_IN_SCORED_ANSWER"
DIAGRAM_PRESENT = "DIAGRAM_PRESENT"
VALIDATOR_DOWNGRADE = "VALIDATOR_DOWNGRADE"  # Agent 2's output was corrected by the validator

# judge validator codes
JUDGE_ADJUSTMENT = "JUDGE_ADJUSTMENT"  # an accepted ±1 proposal (teacher sees it)
JUDGE_FLAG = "JUDGE_FLAG"  # the judge flagged a criterion without proposing a level
JUDGE_INVALID_LEVEL = "JUDGE_INVALID_LEVEL"
JUDGE_PROPOSAL_TOO_LARGE = "JUDGE_PROPOSAL_TOO_LARGE"
JUDGE_UNSUPPORTED_PROPOSAL = "JUDGE_UNSUPPORTED_PROPOSAL"
JUDGE_PROPOSAL_NOT_ATTEMPTED = "JUDGE_PROPOSAL_NOT_ATTEMPTED"
JUDGE_UNKNOWN_ID = "JUDGE_UNKNOWN_ID"
JUDGE_FEEDBACK_MISSING = "JUDGE_FEEDBACK_MISSING"
JUDGE_VERDICT_MISSING = "JUDGE_VERDICT_MISSING"

SBERT_HIGH = 0.80  # similarity at/above this with the lowest level chosen is suspicious
SBERT_LOW = 0.30  # similarity below this with the top level chosen is suspicious
DESCRIPTOR_CITE_THRESHOLD = 0.6
_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,40}$")


def _flag(code: str, severity: str, message: str, qid: str | None = None, cid: str | None = None) -> Flag:
    return Flag(code=code, severity=severity, question_id=qid, criterion_id=cid, message=message)


def run_checks(evaluation: EvaluationResult, extraction: ExtractedSubmission, rubric: Rubric, *,
               low_confidence: float = 0.6) -> list[Flag]:
    """Rule-based consistency checks on Agent 2's output, run before the Judge and shown to it."""
    flags: list[Flag] = []
    for qe in evaluation.questions:
        q = rubric.question(qe.question_id)
        qid = qe.question_id
        if q is None:
            flags.append(_flag(OUT_OF_BOUNDS, "high", f"{qid} is not in the rubric", qid))
            continue
        answer = extraction.answer_for(qid)
        attempted = is_attempted(answer)

        crit_sum = sum(c.marks for c in qe.criteria)
        if abs(crit_sum - qe.total) > _EPS:
            flags.append(_flag(SUM_MISMATCH, "high", f"{qid}: criteria add up to {crit_sum:g} but total is {qe.total:g}", qid))
        if qe.total > q.max_marks + _EPS:
            flags.append(_flag(OUT_OF_BOUNDS, "high", f"{qid}: total {qe.total:g} exceeds max {q.max_marks:g}", qid))

        for ce in qe.criteria:
            c = q.criterion(ce.criterion_id)
            cid = ce.criterion_id
            if c is None:
                flags.append(_flag(OUT_OF_BOUNDS, "high", f"{qid}/{cid} is not in the rubric", qid, cid))
                continue
            lv = c.level(ce.selected_level_id)
            if lv is None or abs(lv.marks - ce.marks) > _EPS or ce.marks > c.max_marks + _EPS:
                flags.append(_flag(OUT_OF_BOUNDS, "high",
                                   f"{qid}/{cid}: {ce.marks:g} marks do not match level {ce.selected_level_id!r}", qid, cid))
            if ce.marks > _EPS and not ce.evidence_verified:
                flags.append(_flag(NO_EVIDENCE, "high", f"{qid}/{cid}: {ce.marks:g} marks awarded without verified evidence", qid, cid))
            if not attempted:
                continue
            top, low = max(c.levels, key=lambda l: l.marks), c.lowest_level
            if qe.sbert_sim_model_answer >= SBERT_HIGH and ce.selected_level_id == low.level_id and len(c.levels) > 1:
                flags.append(_flag(SBERT_LEVEL_DISAGREEMENT, "review",
                                   f"{qid}/{cid}: answer is very similar to the model answer "
                                   f"(sim {qe.sbert_sim_model_answer:.2f}) but got the lowest level", qid, cid))
            elif qe.sbert_sim_model_answer < SBERT_LOW and ce.selected_level_id == top.level_id:
                flags.append(_flag(SBERT_LEVEL_DISAGREEMENT, "review",
                                   f"{qid}/{cid}: answer is unlike the model answer "
                                   f"(sim {qe.sbert_sim_model_answer:.2f}) but got the top level", qid, cid))

        for f in qe.validator_flags:
            code = f.split(":")[0]
            if code in (INVALID_LEVEL, UNSUPPORTED_EVIDENCE, MISSING_CRITERION, MISSING_QUESTION):
                flags.append(_flag(VALIDATOR_DOWNGRADE, "review", f"{qid}: evaluator output was corrected ({f})", qid,
                                   f.split(":")[1] if ":" in f else None))

        if attempted and answer is not None:
            if answer.confidence < low_confidence:
                flags.append(_flag(LOW_OCR_CONFIDENCE, "review",
                                   f"{qid}: transcription confidence {answer.confidence:.2f} is below {low_confidence:.2f}", qid))
            if ILLEGIBLE_MARKER in answer.answer_text and qe.total > _EPS:
                flags.append(_flag(ILLEGIBLE_IN_SCORED_ANSWER, "review",
                                   f"{qid}: answer contains illegible spans but was awarded {qe.total:g} marks", qid))
            if answer.contains_diagram:
                flags.append(_flag(DIAGRAM_PRESENT, "info", f"{qid}: the answer includes a diagram that was not transcribed", qid))

    total = sum(qe.total for qe in evaluation.questions)
    if abs(total - evaluation.total) > _EPS:
        flags.append(_flag(SUM_MISMATCH, "high", f"question totals add up to {total:g} but the assignment total is {evaluation.total:g}"))
    if evaluation.total > rubric.max_total + _EPS:
        flags.append(_flag(OUT_OF_BOUNDS, "high", f"assignment total {evaluation.total:g} exceeds max {rubric.max_total:g}"))
    return flags


# --- judge validator -------------------------------------------------------------------------

def level_distance(criterion: Criterion, from_id: str, to_id: str) -> int | None:
    """Number of rubric steps between two levels (levels ordered by marks). None if either is unknown."""
    order = [lv.level_id for lv in sorted(criterion.levels, key=lambda lv: lv.marks)]
    if from_id not in order or to_id not in order:
        return None
    return order.index(to_id) - order.index(from_id)


def cites_descriptor(justification: str, criterion: Criterion, level_id: str | None = None) -> bool:
    """Does the justification point at a rubric level (its id, label or descriptor wording)?"""
    j = normalise(justification)
    if not j:
        return False
    levels = [criterion.level(level_id)] if level_id and criterion.level(level_id) else criterion.levels
    for lv in levels:
        if lv is None:
            continue
        for token in (lv.level_id, lv.label):
            if token and re.search(rf"\b{re.escape(normalise(token))}\b", j):
                return True
        if quote_match_ratio(lv.descriptor, justification) >= DESCRIPTOR_CITE_THRESHOLD:
            return True
    return len(set(normalise(criterion.description).split()) & set(j.split())) >= 4


def _agree(ce: CriterionEvaluation, justification: str, draft: VerdictDraft | None = None) -> CriterionVerdict:
    return CriterionVerdict(criterion_id=ce.criterion_id, verdict="agree", justification=justification,
                            evidence_quotes=list(draft.evidence_quotes) if draft else [], accepted=False,
                            recommended_level_id=ce.selected_level_id, recommended_marks=ce.marks)


def _resolve_verdict(criterion: Criterion, ce: CriterionEvaluation, draft: VerdictDraft | None,
                     answer_text: str, attempted: bool, qid: str, flags: list[Flag]) -> CriterionVerdict:
    cid = criterion.criterion_id
    # The recommendation starts from the rubric level Agent 2 chose, never from its `marks` figure,
    # so a tampered or inconsistent evaluation cannot push a recommendation outside the rubric.
    level = criterion.level(ce.selected_level_id) or criterion.lowest_level
    ce = ce.model_copy(update={"selected_level_id": level.level_id, "marks": float(level.marks)})
    if draft is None:
        flags.append(_flag(JUDGE_VERDICT_MISSING, "info", f"{qid}/{cid}: the judge gave no verdict; evaluator level kept", qid, cid))
        return _agree(ce, "No verdict from the judge; the evaluator's level stands.")

    if draft.verdict == "agree":
        return _agree(ce, draft.justification or "Agrees with the evaluator.", draft)

    if draft.verdict == "flag":
        flags.append(_flag(JUDGE_FLAG, "review", f"{qid}/{cid}: {draft.justification or 'flagged by the judge'}", qid, cid))
        return CriterionVerdict(criterion_id=cid, verdict="flag", justification=draft.justification,
                                evidence_quotes=list(draft.evidence_quotes), accepted=False,
                                recommended_level_id=ce.selected_level_id, recommended_marks=ce.marks)

    # propose_adjustment
    base = CriterionVerdict(criterion_id=cid, verdict="propose_adjustment", proposed_level_id=draft.proposed_level_id,
                            justification=draft.justification, evidence_quotes=list(draft.evidence_quotes),
                            accepted=False, recommended_level_id=ce.selected_level_id, recommended_marks=ce.marks)
    proposed = criterion.level(draft.proposed_level_id or "")
    if proposed is None:
        flags.append(_flag(JUDGE_INVALID_LEVEL, "review",
                           f"{qid}/{cid}: proposed level {draft.proposed_level_id!r} does not exist; kept {ce.selected_level_id!r}", qid, cid))
        return base
    if proposed.level_id == ce.selected_level_id:
        return _agree(ce, draft.justification or "Proposed the same level.", draft)
    if not attempted:
        flags.append(_flag(JUDGE_PROPOSAL_NOT_ATTEMPTED, "review",
                           f"{qid}/{cid}: the answer is blank or unreadable, so no level can be awarded", qid, cid))
        return base
    distance = level_distance(criterion, ce.selected_level_id, proposed.level_id)
    if distance is None or abs(distance) > 1:
        flags.append(_flag(JUDGE_PROPOSAL_TOO_LARGE, "review",
                           f"{qid}/{cid}: judge proposed {ce.selected_level_id!r} -> {proposed.level_id!r} "
                           f"({abs(distance or 0)} levels); only ±1 can be applied. {draft.justification}".strip(), qid, cid))
        return base
    quotes_ok = any(verify_quote(qt, answer_text) for qt in draft.evidence_quotes)
    going_up = proposed.marks > ce.marks
    if going_up and not quotes_ok:
        flags.append(_flag(JUDGE_UNSUPPORTED_PROPOSAL, "review",
                           f"{qid}/{cid}: judge proposed a higher level ({proposed.level_id!r}) without a quote that "
                           f"matches the answer. {draft.justification}".strip(), qid, cid))
        return base
    if not going_up and not quotes_ok and not cites_descriptor(draft.justification, criterion, proposed.level_id):
        flags.append(_flag(JUDGE_UNSUPPORTED_PROPOSAL, "review",
                           f"{qid}/{cid}: judge proposed a lower level ({proposed.level_id!r}) without evidence or a rubric "
                           f"reason. {draft.justification}".strip(), qid, cid))
        return base
    flags.append(_flag(JUDGE_ADJUSTMENT, "review",
                       f"{qid}/{cid}: judge recommends {proposed.level_id!r} ({proposed.marks:g}) instead of "
                       f"{ce.selected_level_id!r} ({ce.marks:g}). {draft.justification}".strip(), qid, cid))
    return base.model_copy(update={"accepted": True, "recommended_level_id": proposed.level_id,
                                   "recommended_marks": float(proposed.marks)})


def _fallback_feedback(qe: QuestionEvaluation, q: Question, attempted: bool, answer: ExtractedAnswer | None,
                       draft=None) -> QuestionFeedback:
    if not attempted:
        why = "no answer was found" if answer is None else f"the answer was {answer.legibility}"
        return QuestionFeedback(question_id=q.question_id, strengths=[],
                                improvements=[f"This question was not graded because {why}. Attempt every question."],
                                next_steps=[f"Revise: {q.prompt}"], overlooked_concepts=[])
    present = [c for ce in qe.criteria for c in ce.concepts_present]
    missing = [c for ce in qe.criteria for c in ce.missing_concepts]
    wrong = [c for ce in qe.criteria for c in ce.incorrect_statements]
    strengths = [f"You covered: {', '.join(dict.fromkeys(present))}."] if present else []
    improvements = (([f"Add: {', '.join(dict.fromkeys(missing))}."] if missing else [])
                    + ([f"Check: {', '.join(dict.fromkeys(wrong))}."] if wrong else []))
    for ce in qe.criteria:  # always say which criteria were met and which were not
        c = q.criterion(ce.criterion_id)
        if c is None:
            continue
        if ce.marks >= c.max_marks - _EPS:
            strengths.append(f"{c.description}: shown.")
        else:
            improvements.append(f"{c.description}: not fully shown.")
    return QuestionFeedback(
        question_id=q.question_id,
        strengths=draft.strengths if draft and draft.strengths else strengths,
        improvements=draft.improvements if draft and draft.improvements else improvements,
        next_steps=draft.next_steps if draft else [],
        overlooked_concepts=draft.overlooked_concepts if draft else [],
    )


def validate_judge(
    draft: JudgeDraft,
    evaluation: EvaluationResult,
    rubric: Rubric,
    extraction: ExtractedSubmission,
    *,
    checks: Iterable[Flag] = (),
    prompt_version: str = "",
    model_id: str = "none",
) -> JudgeReport:
    """Turn the judge's draft into a report whose recommended marks are rubric levels at most ±1 from Agent 2's."""
    flags: list[Flag] = list(checks)
    verdicts: dict[str, list[CriterionVerdict]] = {}
    feedback: list[QuestionFeedback] = []
    known_q = {q.question_id for q in rubric.questions}

    for qd in draft.questions:
        if qd.question_id not in known_q:
            flags.append(_flag(JUDGE_UNKNOWN_ID, "info", f"judge referred to unknown question {qd.question_id!r}"))

    for qe in evaluation.questions:
        q = rubric.question(qe.question_id)
        if q is None:
            continue
        qid = q.question_id
        answer = extraction.answer_for(qid)
        attempted = is_attempted(answer)
        answer_text = answer.answer_text if answer else ""
        qd = draft.question(qid)
        drafts: dict[str, VerdictDraft] = {}
        if qd is not None:
            for vd in qd.verdicts:
                if q.criterion(vd.criterion_id) is None:
                    flags.append(_flag(JUDGE_UNKNOWN_ID, "info", f"{qid}: judge referred to unknown criterion {vd.criterion_id!r}", qid))
                elif vd.criterion_id not in drafts:
                    drafts[vd.criterion_id] = vd
        out = []
        for ce in qe.criteria:
            c = q.criterion(ce.criterion_id)
            if c is None:
                continue
            vd = drafts.get(ce.criterion_id)
            if vd is None and not attempted:
                out.append(_agree(ce, "Not attempted; nothing to review."))
            else:
                out.append(_resolve_verdict(c, ce, vd, answer_text, attempted, qid, flags))
        verdicts[qid] = out

        if attempted and (qd is None or not (qd.strengths or qd.improvements)):
            flags.append(_flag(JUDGE_FEEDBACK_MISSING, "info", f"{qid}: the judge gave no feedback; generated from the evaluation", qid))
            feedback.append(_fallback_feedback(qe, q, attempted, answer, qd))
        elif attempted:
            feedback.append(QuestionFeedback(question_id=qid, strengths=qd.strengths, improvements=qd.improvements,
                                             next_steps=qd.next_steps, overlooked_concepts=qd.overlooked_concepts))
        else:
            feedback.append(_fallback_feedback(qe, q, attempted, answer))

    seen = {(f.code, f.question_id, f.criterion_id) for f in flags}
    for fd in draft.flags:
        code = fd.code.strip().upper().replace(" ", "_")
        if not _CODE_RE.match(code):
            code = "JUDGE_NOTE"
        qid = fd.question_id if fd.question_id in known_q else None
        cid = fd.criterion_id if (qid and rubric.question(qid).criterion(fd.criterion_id or "")) else None
        if (code, qid, cid) in seen or (code, None, None) in seen:
            continue  # the model repeated a flag it was given as context
        seen.add((code, qid, cid))
        flags.append(_flag(code, fd.severity, fd.message, qid, cid))

    recommended_total = sum(v.recommended_marks for vs in verdicts.values() for v in vs)
    return JudgeReport(
        submission_id=evaluation.submission_id, verdicts=verdicts, feedback=feedback,
        summary_feedback=draft.summary_feedback.strip() or _default_summary(evaluation, feedback),
        flags=flags, recommended_total=min(recommended_total, rubric.max_total), max_total=rubric.max_total,
        evaluation_total=evaluation.total, prompt_version=prompt_version, model_id=model_id,
        evaluation_id=evaluation.evaluation_id,
    )


def _default_summary(evaluation: EvaluationResult, feedback: list[QuestionFeedback]) -> str:
    improvements = [i for f in feedback for i in f.improvements][:3]
    head = f"Provisional score {evaluation.total:g}/{evaluation.max_total:g}."
    return head + (" To improve: " + " ".join(improvements) if improvements else "")


def check_recommendations_within_bounds(report: JudgeReport, evaluation: EvaluationResult, rubric: Rubric) -> list[str]:
    """Audit helper for the Phase 5 exit gate: every recommended mark is a rubric level within ±1 of Agent 2's."""
    problems = []
    for qid, vs in report.verdicts.items():
        q = rubric.question(qid)
        qe = evaluation.question(qid)
        if q is None or qe is None:
            problems.append(f"{qid}: not in rubric/evaluation")
            continue
        for v in vs:
            c = q.criterion(v.criterion_id)
            ce = qe.criterion(v.criterion_id)
            lv = c.level(v.recommended_level_id) if c else None
            if lv is None or abs(lv.marks - v.recommended_marks) > _EPS or ce is None:
                problems.append(f"{qid}/{v.criterion_id}: recommended {v.recommended_marks} is not a rubric level")
                continue
            d = level_distance(c, ce.selected_level_id, v.recommended_level_id)
            if d is None or abs(d) > 1:
                problems.append(f"{qid}/{v.criterion_id}: recommendation is {d} levels from the evaluation")
            if v.accepted and v.recommended_level_id == ce.selected_level_id:
                problems.append(f"{qid}/{v.criterion_id}: accepted but unchanged")
        if sum(v.recommended_marks for v in vs) > q.max_marks + _EPS:
            problems.append(f"{qid}: recommended total exceeds max")
    if abs(sum(v.recommended_marks for vs in report.verdicts.values() for v in vs) - report.recommended_total) > _EPS:
        problems.append("recommended_total inconsistent")
    if report.recommended_total > rubric.max_total + _EPS:
        problems.append("recommended_total exceeds max")
    return problems
