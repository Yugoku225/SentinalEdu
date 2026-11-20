"""Small reusable pieces for the dashboard pages."""
import html
import re
from typing import Sequence

import streamlit as st

from sentineledu.schemas.submission import SubmissionStatus
from sentineledu.workflow.assessment_workflow import AssessmentRun, run_assessment
from sentineledu.workflow.steps import WorkflowDeps

STATUS_COLOUR = {
    SubmissionStatus.UPLOADED: "gray", SubmissionStatus.EXTRACTED: "blue",
    SubmissionStatus.NEEDS_EXTRACTION_REVIEW: "orange", SubmissionStatus.EVALUATED: "blue",
    SubmissionStatus.PENDING_TEACHER_REVIEW: "violet", SubmissionStatus.FINALIZED: "green",
    SubmissionStatus.FAILED: "red",
}
LEGIBILITY_COLOUR = {"clear": "green", "partial": "orange", "unreadable": "red", "blank": "gray"}
SEVERITY_ICON = {"high": "🔴", "review": "🟠", "info": "🔵"}
STEP_LABEL = {
    "integrity_precheck": "Duplicate pre-check", "document_understanding": "Reading the pages (Agent 1)",
    "extraction_gate": "Extraction check", "evaluation": "Grading against the rubric (Agent 2)",
    "integrity_postcheck": "Similarity scan across the class", "judge": "Second review and feedback (Agent 3)",
    "mark_pending_review": "Hand over to the teacher",
}
STEP_ICON = {"ok": "✅", "skipped": "⏭️", "stopped": "⏸️", "fallback": "⚠️", "error": "❌"}


def status_badge(status: SubmissionStatus) -> str:
    return f":{STATUS_COLOUR.get(status, 'gray')}-badge[{status.value}]"


def legibility_badge(legibility: str, confidence: float) -> str:
    return f":{LEGIBILITY_COLOUR.get(legibility, 'gray')}-badge[{legibility} · {confidence:.0%}]"


def highlight(text: str, quotes: Sequence[str]) -> str:
    """HTML of `text` (escaped) with every quote found in it marked. Quotes not found are ignored here;
    the review table shows whether each quote was verified."""
    spans: list[tuple[int, int]] = []
    lowered = text.lower()
    for q in quotes:
        q = q.strip().strip(".,;:!?\"'").lower()
        if len(q) < 3:
            continue
        start = lowered.find(q)
        if start < 0:  # tolerate whitespace differences
            m = re.search(r"\s+".join(map(re.escape, q.split())), lowered)
            if not m:
                continue
            start, end = m.span()
        else:
            end = start + len(q)
        spans.append((start, end))
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, pos = [], 0
    for s, e in merged:
        out.append(html.escape(text[pos:s]))
        out.append(f"<mark>{html.escape(text[s:e])}</mark>")
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def answer_box(text: str, quotes: Sequence[str] = ()) -> None:
    body = highlight(text, quotes) if text.strip() else "<i>(blank)</i>"
    st.markdown(f"<div style='border:1px solid rgba(128,128,128,.35);border-radius:6px;padding:.6em .8em'>{body}</div>",
                unsafe_allow_html=True)


def run_with_progress(submission_id: str, deps: WorkflowDeps, label: str = "Running the assessment") -> AssessmentRun:
    """Run the workflow with a live per-step log inside st.status."""
    with st.status(label, expanded=True) as box:
        def progress(_sid: str, step: str, status: str, ms: int, message: str) -> None:
            st.write(f"{STEP_ICON.get(status, '•')} **{STEP_LABEL.get(step, step)}** — {status} "
                     f"({ms / 1000:.1f} s){': ' + message if message else ''}")

        previous = deps.progress
        deps.progress = progress
        try:
            run = run_assessment(submission_id, deps)
        finally:
            deps.progress = previous
        state = "error" if any(s.status == "error" for s in run.steps) else "complete"
        box.update(label=f"{label}: {run.status.value} ({run.duration_ms / 1000:.1f} s)", state=state,
                   expanded=state == "error" or run.status == SubmissionStatus.NEEDS_EXTRACTION_REVIEW)
    return run


def show_run_outcome(run: AssessmentRun) -> None:
    if run.status == SubmissionStatus.PENDING_TEACHER_REVIEW:
        st.success(f"Ready for teacher review: {run.evaluation_total:g}/{run.max_total:g} from the evaluator, "
                   f"{run.recommended_total:g} recommended after the second review.")
    elif run.status == SubmissionStatus.NEEDS_EXTRACTION_REVIEW:
        st.warning(f"The transcription needs checking before grading. {run.reason or ''}")
    elif run.status == SubmissionStatus.FINALIZED:
        st.info("This submission is already finalised.")
    else:
        st.error(f"Stopped at {run.stopped_at}: {run.reason}" if run.stopped_at else f"Status: {run.status.value}")
