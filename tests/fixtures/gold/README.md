# Gold test set (SENTINELEDU_TASKS.md 1.6)

Real handwritten pages with known transcriptions. Used to measure OCR accuracy (Phase 3),
grading accuracy (Phase 4) and duplicate detection (Phase 7). **Anonymise everything**:
no student names or roll numbers on the pages or in file names.

## What to collect
- 10–20 handwritten pages answering the **same 3 questions**.
- Include hard cases: messy writing, a crossed-out word, a partly illegible answer,
  a blank answer, answers out of order, one answer continuing onto a second page.
- Photos (JPG/PNG) or scanned PDFs are both fine.

## Files
Put the pages in this folder, e.g. `s01_p1.jpg`, `s01_p2.jpg`, `s02_p1.jpg`, and create `manifest.json`:

```json
{
  "description": "Real handwritten gold set",
  "synthetic": false,
  "questions": [
    {"question_id": "Q1", "prompt": "Define photosynthesis."},
    {"question_id": "Q2", "prompt": "Name the green pigment in leaves and say where it is found."},
    {"question_id": "Q3", "prompt": "Why do plants need water?"}
  ],
  "items": [
    {
      "id": "s01",
      "files": ["s01_p1.jpg"],
      "transcription": {
        "Q1": "Exactly what the student wrote, spelling mistakes included.",
        "Q2": "...",
        "Q3": ""
      },
      "expect": {"blank": ["Q3"], "illegible": []},
      "teacher_marks": null
    }
  ]
}
```

Rules for `transcription`:
- Type **exactly** what is written, including spelling mistakes. Leave out crossed-out text.
- Use `""` for a blank answer and list that question in `expect.blank`.
- If part of an answer is genuinely unreadable to you, list the question in `expect.illegible`.
- `teacher_marks` can stay `null` until Phase 4, when the rubric exists.

Then run:
```
python scripts/eval_ocr.py tests/fixtures/gold/manifest.json --engines gemini trocr
```
