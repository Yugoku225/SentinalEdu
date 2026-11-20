"""Measure extraction accuracy against known transcriptions (SENTINELEDU_TASKS.md 3.8).

Metrics per engine:
- CER / WER on answers that should be readable (lower-cased, whitespace-normalised;
  [[ILLEGIBLE]] markers removed before comparing).
- Segmentation accuracy: share of expected questions whose text landed under the right
  question_id (or was correctly reported blank / flagged illegible).
- Flag checks: blank answers reported as blank; deliberately illegible answers flagged.
- Gate rate and latency.

Usage:
  python scripts/eval_ocr.py tests/fixtures/synthetic/manifest.json
  python scripts/eval_ocr.py tests/fixtures/gold/manifest.json --engines gemini trocr --models gemini-3.7-flash gemini-3.5-flash
  add --json results.json to save raw numbers
"""
import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nltk import edit_distance  # noqa: E402

from sentineledu.agents.document_agent import extract_submission  # noqa: E402
from sentineledu.config import Settings  # noqa: E402
from sentineledu.schemas.extraction import ILLEGIBLE_MARKER  # noqa: E402
from sentineledu.services.extraction_review import extraction_gate  # noqa: E402

SEGMENT_CER_LIMIT = 0.5  # text counts as "under the right question" if CER is below this


@dataclass
class Q:
    question_id: str
    prompt: str


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(ILLEGIBLE_MARKER, " ").lower()).strip()


def cer(ref: str, hyp: str) -> float:
    ref, hyp = norm(ref), norm(hyp)
    return edit_distance(ref, hyp) / max(len(ref), 1)


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref).split(), norm(hyp).split()
    return edit_distance(r, h) / max(len(r), 1)


def evaluate(manifest_path: Path, engine: str, model: str | None, threshold: float) -> dict:
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    questions = [Q(**q) for q in manifest["questions"]]
    qids = [q.question_id for q in questions]
    overrides = {"offline_mode": engine == "trocr"}
    if model:
        overrides |= {"gemini_ocr_model": model}
    settings = Settings(**overrides)

    rows, totals = [], {"chars": 0, "char_err": 0, "words": 0, "word_err": 0, "seg_ok": 0, "seg_n": 0,
                        "blank_ok": 0, "blank_n": 0, "illeg_ok": 0, "illeg_n": 0, "gated": 0, "seconds": 0.0}
    for item in manifest["items"]:
        start = time.monotonic()
        out = extract_submission(item["id"], [base / f for f in item["files"]], questions, settings=settings)
        seconds = time.monotonic() - start
        expect = item.get("expect", {})
        blank, illegible = set(expect.get("blank", [])), set(expect.get("illegible", []))
        refs = item["transcription"]

        if engine == "trocr":  # no segmentation: compare whole page text
            ref_all = " ".join(refs[q] for q in qids if q not in illegible and refs.get(q))
            hyp_all = re.sub(r"\[page \d+\]", " ", out.unmatched_text)
            n, e = len(norm(ref_all)), cer(ref_all, hyp_all) * max(len(norm(ref_all)), 1)
            wn = len(norm(ref_all).split())
            totals["chars"] += n; totals["char_err"] += e
            totals["words"] += wn; totals["word_err"] += wer(ref_all, hyp_all) * max(wn, 1)
            per_q = {}
        else:
            per_q = {}
            for qid in qids:
                ref = refs.get(qid, "")
                ans = out.answer_for(qid)
                hyp = ans.answer_text if ans else ""
                if qid in blank:
                    ok = ans is None or ans.legibility == "blank"
                    totals["blank_n"] += 1; totals["blank_ok"] += ok
                    per_q[qid] = "blank OK" if ok else f"blank MISSED ({hyp[:30]!r})"
                elif qid in illegible:
                    ok = ans is not None and ans.legibility in ("partial", "unreadable")
                    totals["illeg_n"] += 1; totals["illeg_ok"] += ok
                    per_q[qid] = f"illegible {'FLAGGED' if ok else 'NOT FLAGGED'} ({ans.legibility if ans else 'missing'})"
                else:
                    c = cer(ref, hyp)
                    ok = ans is not None and c < SEGMENT_CER_LIMIT
                    totals["chars"] += len(norm(ref)); totals["char_err"] += c * len(norm(ref))
                    totals["words"] += len(norm(ref).split())
                    totals["word_err"] += wer(ref, hyp) * len(norm(ref).split())
                    per_q[qid] = f"CER {c:.3f}"
                totals["seg_n"] += 1; totals["seg_ok"] += ok
        gate = extraction_gate(out, qids, threshold)
        totals["gated"] += not gate.ok
        totals["seconds"] += seconds
        rows.append({"item": item["id"], "engine": out.engine, "model": out.model_id, "seconds": round(seconds, 1),
                     "per_question": per_q, "gate_ok": gate.ok, "gate_reasons": gate.reasons,
                     "issues": out.issues})

    n_items = len(manifest["items"])
    summary = {
        "engine": engine, "model": model or (settings.gemini_ocr_model if engine == "gemini" else None),
        "items": n_items,
        "cer": totals["char_err"] / max(totals["chars"], 1),
        "wer": totals["word_err"] / max(totals["words"], 1),
        "segmentation_acc": totals["seg_ok"] / totals["seg_n"] if totals["seg_n"] else None,
        "blank_detected": f"{totals['blank_ok']}/{totals['blank_n']}",
        "illegible_flagged": f"{totals['illeg_ok']}/{totals['illeg_n']}",
        "sent_to_review": f"{totals['gated']}/{n_items}",
        "avg_seconds": totals["seconds"] / n_items,
        "synthetic": manifest.get("synthetic", False),
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--engines", nargs="+", default=["gemini"], choices=["gemini", "trocr"])
    ap.add_argument("--models", nargs="+", default=[None], help="Gemini model ids to compare")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    runs = []
    for engine in args.engines:
        for model in (args.models if engine == "gemini" else [None]):
            print(f"\n=== {engine} {model or ''}".rstrip())
            result = evaluate(args.manifest, engine, model, args.threshold)
            for r in result["rows"]:
                print(f"  {r['item']:28} {r['seconds']:6.1f}s  model={r['model']}  review={'no' if r['gate_ok'] else 'YES'}  "
                      + "  ".join(f"{q}: {v}" for q, v in r["per_question"].items()))
            runs.append(result)

    print("\n| Engine | Model | CER | WER | Segmentation | Blank detected | Illegible flagged | Sent to review | Avg s/submission |")
    print("|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        s = run["summary"]
        seg = f"{s['segmentation_acc']:.0%}" if s["segmentation_acc"] is not None else "n/a"
        print(f"| {s['engine']} | {s['model'] or '-'} | {s['cer']:.3f} | {s['wer']:.3f} | {seg} | "
              f"{s['blank_detected']} | {s['illegible_flagged']} | {s['sent_to_review']} | {s['avg_seconds']:.1f} |")
    if runs and runs[0]["summary"]["synthetic"]:
        print("\nNOTE: synthetic font-rendered pages. These numbers do not measure real handwriting accuracy.")
    if args.json:
        args.json.write_text(json.dumps(runs, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
