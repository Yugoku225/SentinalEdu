"""Export of finalised marks (TASKS 8.7). Only FINALIZED submissions, i.e. marks a teacher approved.

`results_rows` gives one flat row per submission (for CSV and the dashboard table);
`results_json` gives the full nested record including feedback and overrides.
"""
import csv
import io
import json

from sentineledu.schemas.submission import SubmissionStatus
from sentineledu.services.storage import NotFoundError, Storage


def _finalised(storage: Storage, assignment_id: str | None):
    for sub in storage.list_submissions(assignment_id):
        if sub.status != SubmissionStatus.FINALIZED:
            continue
        decision = storage.get_decision(sub.submission_id)
        if decision is None:  # cannot happen through finalize(); skip rather than export a guess
            continue
        try:
            rubric = storage.get_rubric(sub.assignment_id)
        except NotFoundError:
            continue
        yield sub, decision, rubric, storage.latest_judge_report(sub.submission_id)


def results_rows(storage: Storage, assignment_id: str | None = None) -> list[dict]:
    rows = []
    for sub, decision, rubric, report in _finalised(storage, assignment_id):
        row = {
            "submission_id": sub.submission_id, "assignment_id": sub.assignment_id, "student_ref": sub.student_ref or "",
            "total": decision.total, "max_total": rubric.max_total, "reviewer": decision.reviewer,
            "approved_at": decision.approved_at.isoformat(), "overrides": len(decision.overrides),
        }
        for q in rubric.questions:
            marks = decision.final_marks.get(q.question_id, {})
            row[q.question_id] = sum(marks.values())
            for c in q.criteria:
                row[f"{q.question_id}.{c.criterion_id}"] = marks.get(c.criterion_id, 0.0)
        row["teacher_comment"] = decision.teacher_comment
        row["feedback_summary"] = report.summary_feedback if report else ""
        for q in rubric.questions:
            fb = report.feedback_for(q.question_id) if report else None
            row[f"{q.question_id}.feedback"] = " | ".join((fb.strengths + fb.improvements) if fb else [])
        rows.append(row)
    return rows


def results_csv(storage: Storage, assignment_id: str | None = None) -> str:
    rows = results_rows(storage, assignment_id)
    if not rows:
        return ""
    fields = list(dict.fromkeys(k for r in rows for k in r))  # union of columns, first-seen order
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def results_json(storage: Storage, assignment_id: str | None = None) -> str:
    records = []
    for sub, decision, rubric, report in _finalised(storage, assignment_id):
        records.append({
            "submission": json.loads(sub.model_dump_json()),
            "decision": json.loads(decision.model_dump_json()),
            "max_total": rubric.max_total,
            "rubric_version": rubric.version,
            "feedback": json.loads(report.model_dump_json(include={"feedback", "summary_feedback"})) if report else None,
            "integrity_flags": [json.loads(f.model_dump_json()) for f in storage.integrity_flags(sub.submission_id)],
        })
    return json.dumps(records, indent=2)
