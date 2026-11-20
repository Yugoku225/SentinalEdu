# SentinelEdu — Measured Results

Numbers from the evaluation scripts, recorded per phase. Raw output is in the JSON files next to this one.

## Accuracy report (TASKS 9.4) — summary as of 2026-11-21

> **Read this first.** Every number below comes from **synthetic** pages (rendered from handwriting fonts)
> or **synthetic** calibration data, with marks assigned by the developer from the rubric, not by a
> teacher. They show the pipeline works as designed. They do **not** measure real handwriting accuracy
> or agreement with a teacher. Those columns stay "pending" until the gold set exists (task 1.6).
> Each command is listed so the table can be re-run on the gold set.

| Metric | SentinelEdu | Comparison | Data | Gold set |
|---|---|---|---|---|
| OCR character error rate (CER) | 0.000 (gemini-3.5-flash) | TrOCR 0.558 | 4 synthetic submissions, 5 pages | pending |
| OCR word error rate (WER) | 0.000 | TrOCR 0.993 | same | pending |
| Answers matched to the right question | 100% (12/12) | TrOCR cannot match answers to questions | same | pending |
| Blank answer detected / illegible answer flagged | 1/1, 1/1 | – | same | pending |
| Mark error, per question / per total | 0.00 / 0.00 | legacy grader 0.56 / 1.07 | 4 synthetic submissions; **developer's placeholder marks** | pending |
| Exact rubric-level agreement / Cohen's κ | 100% (20 criteria) / 1.00 | legacy: no levels | same | pending |
| Awarded marks backed by a verified quote | 17/17; 0 rubric-consistency problems | legacy: no evidence | same | pending |
| Judge: planted grading errors caught | 4/5, 0 false alarms on 4 correct evaluations | – | 5 planted errors in synthetic evaluations | (benchmark is synthetic by design) |
| Judge: recommendations within rubric bounds | 9/9 runs; 300-draft property test | – | synthetic | – |
| Integrity: copied pairs found / precision | 4/4 / 1.00; 2 textbook-copy controls not flagged | – | synthetic class: 10 students, 90 pairs | pending (plant 3–4 pairs) |

Commands:
- `scripts/eval_ocr.py`: OCR and question matching.
- `scripts/eval_grading.py --legacy`: marks against the teacher, and against the legacy grader.
- `scripts/eval_judge.py`: the Judge's planted-error benchmark.
- `scripts/calibrate_integrity.py`: duplicate detection.

### Latency and API calls per submission (live, `gemini-3.5-flash`)

| Path | Gemini requests | Reading the pages | Grading | Second review | End to end |
|---|---|---|---|---|---|
| Normal submission | 3 (one per agent) | 9.6–51.7 s | 10.6–38.5 s | 7.4–16.0 s | 33–117 s |
| Stopped for a transcription check, then corrected | 1, then 2 after the correction | 31.7 s | 10.6 s | cached | 31.8 s + 10.6 s |
| Byte-identical re-upload | 2 (no reading step) | skipped | 9.6 s | 7.4 s | 17.0–50.8 s |
| Demo seed (6 submissions, from a clean database) | about 14 | – | – | – | 286 s in total |

Typical tokens per request:

| Agent | Input | Output |
|---|---|---|
| Document Understanding | about 2,000 | about 350 |
| Answer Evaluation | about 3,600 | about 600–800 |
| Judge & Feedback | about 3,000–3,800 | about 350–800 |

Timings vary a lot with Gemini load. 503 "high demand" errors are retried with backoff, then sent to the fallback model. The free tier allows 20 requests per day per model, which is about 6 submissions.

### Legacy regression
`pytest -m legacy`: 3 passed. The original evaluator still reproduces `tests/fixtures/legacy_baseline.json` exactly, the real TrOCR fallback reads the sample page, and TrOCR runs after FAISS has been loaded (crash fix D-29).

---

## Phase 3 — Document Understanding (2026-11-16)

### Synthetic pages (`tests/fixtures/synthetic/`)
4 submissions, 5 pages, 3 questions each. They cover a clean page, a smudged-illegible answer with
"Ans 1" labels and kept misspellings, a blank answer with answers out of order, and an answer
continuing onto a second page.

> ⚠️ These pages are rendered from handwriting-style **fonts**. They test the pipeline logic
> (segmentation, flags, merging, gate), not real handwriting accuracy. Real accuracy needs the
> gold set (task 1.6).

Command: `python scripts/eval_ocr.py tests/fixtures/synthetic/manifest.json --engines gemini trocr --models gemini-3.7-flash gemini-3.5-flash`
Raw output: `docs/phase3_synthetic_eval.json`

| Engine | Model | CER | WER | Segmentation | Blank detected | Illegible flagged | Sent to review | Avg s/submission |
|---|---|---|---|---|---|---|---|---|
| gemini | gemini-3.7-flash* | 0.000 | 0.000 | 100% | 1/1 | 1/1 | 1/4 | 13.4 |
| gemini | gemini-3.5-flash | 0.000 | 0.000 | 100% | 1/1 | 1/1 | 1/4 | 5.9 |
| trocr | (fallback) | 0.558 | 0.993 | n/a | — | — | 4/4 | 27.1 |

\* 3 of the 4 gemini-3.7-flash requests returned 503 "high demand" and were answered by the
gemini-3.5-flash fallback, so this row mostly measures 3.5-flash. Only `syn_01_clean` was read by 3.7-flash.

Observations:
- Gemini kept the student's misspellings ("Photosynthesys", "sunlite", "chlorophyl") instead of correcting them.
- The smudged answer was reported as `unreadable` and the gate sent that submission to review. The other three passed.
- TrOCR reads printed-style lines poorly on these fonts and cannot split by question (WER ≈ 1.0, partly
  because word boundaries and order are lost). It works only as an emergency fallback, as intended.

### Gold set (real handwriting)
_Pending: task 1.6. Run `python scripts/eval_ocr.py tests/fixtures/gold/manifest.json --engines gemini trocr --models gemini-3.5-flash gemini-3.7-flash`._
Phase 3 exit gate: ≥ 90% segmentation, every illegible sample flagged, Gemini CER < TrOCR CER.

## Phase 4 — Answer Evaluation (2026-11-16)

### Synthetic set, graded from the transcriptions (`tests/fixtures/synthetic/manifest.json`)
Rubric: `data/rubrics/sample_rubric.json` (3 questions, 5 criteria, 7 marks). Reference document indexed for RAG:
`tests/fixtures/reference/photosynthesis_notes.txt`. Model: `gemini-3.5-flash`, one structured call per submission.

> ⚠️ The "teacher" marks in this manifest were assigned by the developer from the sample rubric, not by a
> teacher (see `marks_by` in the manifest). The agreement numbers below show that the agent applies the
> rubric the same way a careful reader of it does; they do not measure agreement with a real teacher.
> That measurement needs the gold set (task 1.6) with a teacher's marks.

Command: `python scripts/eval_grading.py tests/fixtures/synthetic/manifest.json --rubric data/rubrics/sample_rubric.json --reference tests/fixtures/reference/photosynthesis_notes.txt --legacy --json docs/phase4_synthetic_grading.json`
Raw output: `docs/phase4_synthetic_grading.json`

| Grader | MAE per question | MAE per total | Exact level agreement | Cohen's κ |
|---|---|---|---|---|
| SentinelEdu Agent 2 | 0.00 | 0.00 | 100% (20 criteria) | 1.00 |
| Legacy SBERT threshold + rescue | 0.56 | 1.07 | n/a (no levels) | n/a |

Per submission (S = SentinelEdu, T = placeholder teacher marks, L = legacy):

| Submission | S | T | L | Notes |
|---|---|---|---|---|
| syn_01_clean | 7 | 7 | 6.5 | all criteria at the top level, every quote verified |
| syn_02_messy_illegible | 5 | 5 | 5.2 | Q1 "partial": water missing from the definition; oxygen not mentioned |
| syn_03_blank_out_of_order | 4 | 4 | 3.7 | Q3 blank → `NOT_ATTEMPTED`, not sent to Gemini |
| syn_04_two_pages | 7 | 7 | 3.7 | legacy gave Q3 0/2 for "To make food and to transport nutrients" (similarity below its 0.20 cut-off) |

Exit-gate audit (`check_marks_are_rubric_derived`): 17/17 awarded criteria have `evidence_verified=True`,
0 rubric-consistency problems, every total equals the sum of rubric level marks.

Observations:
- Gemini quoted the student's text with its misspellings intact ("Photosynthesys", "chlorophyl"), so every quote verified at ratio 1.0.
- The legacy grader is rubric-blind: it rewards wording similar to the whole model answer, so a short correct answer
  (syn_04 Q3) gets 0 while a longer, less complete one (syn_02 Q1) gets 2.2/3.
- Latency: 20–26 s per submission for the live calls (3,000–3,600 input tokens, 600–770 output tokens).
  Three of the four script runs were served from the response cache written by the live tests.

### Gold set (real handwriting)
_Pending: task 1.6 with `teacher_marks` filled in. Then run the command above on `tests/fixtures/gold/manifest.json`,
with and without `--from-ocr`, to measure grading with and without OCR errors._

## Phase 5 — Judge & Feedback (2026-11-16)

### Live tests (`tests/integration/test_judge_agent_live.py`, 4 passed)
Agent 2 then Agent 3 on three synthetic submissions, from the transcriptions, with the reference notes indexed.

| Submission | Agent 2 total | Judge recommendation | Judge time | Tokens in/out |
|---|---|---|---|---|
| syn_01_clean | 7/7 | 7/7 (agrees) | 8.3 s | 3,794 / 580 |
| syn_02_messy_illegible | 5/7 | 5/7 (agrees) | 5.4 s | 3,769 / 798 |
| syn_03_blank_out_of_order | 4/7 | 4/7 (agrees; Q3 not attempted) | 5.8 s (after one retry) | 2,960 / 359 |

Every run passed `check_recommendations_within_bounds` and had feedback for all three questions.

### Seeded-error benchmark (task 5.5, completed 2026-11-21)
Command: `python scripts/eval_judge.py tests/fixtures/synthetic/manifest.json --rubric data/rubrics/sample_rubric.json --reference tests/fixtures/reference/photosynthesis_notes.txt --json docs/phase5_seeded_errors.json`
Raw output: `docs/phase5_seeded_errors.json`. Model: `gemini-3.5-flash`.

| # | Injected error | Where | Detected | How |
|---|---|---|---|---|
| 1 | Level one step too high (partial → full) | syn_02 Q1/C1 | yes | Judge proposed `partial`; accepted |
| 2 | Level one step too low (full → partial) | syn_01 Q1/C1 | yes | Judge proposed `full`; accepted |
| 3 | Present concept moved to `missing_concepts`, level unchanged | syn_03 Q1/C1 | **no** | Judge agreed with the (correct) level and did not list the concept as overlooked |
| 4 | Marks edited past the rubric (2 → 3) | syn_04 Q1/C1 | yes | deterministic check `OUT_OF_BOUNDS` |
| 5 | Evidence removed, marks kept | syn_01 Q1/C2 | yes | deterministic check `NO_EVIDENCE` |

**Detection: 4/5** (target ≥ 4/5). False alarms on the 4 unmodified evaluations: **0**.
Recommended marks within rubric bounds: **9/9** runs. Feedback for every answered question: yes.

Bug found by this benchmark and fixed: on the first run (16 Nov, finished 21 Nov after the quota reset), error 4 was
flagged but the Judge's *recommendation* copied Agent 2's inflated 3 marks, breaking the exit gate. `validate_judge`
now derives every recommendation from the rubric level, never from Agent 2's `marks` figure. Regression test:
`test_recommendation_ignores_tampered_marks_and_uses_the_rubric_level`.

The miss (error 3) is low-impact: the marks were right and only the explanation was wrong. It affects the
feedback text, which the teacher sees before approving.

## Phase 6 — Orchestration (2026-11-21)

Live runs through the CLI. Model `gemini-3.5-flash`. Reference notes indexed: 8 chunks. Synthetic pages only, because the gold set (task 1.6) is still pending.

| Run | Path | Final status | Agent 1 | Agent 2 | Agent 3 | Total time | Marks (evaluator / recommended) |
|---|---|---|---|---|---|---|---|
| syn_01_clean, `assess` | happy path | PENDING_TEACHER_REVIEW | 9.6 s | 27.8 s | 15.2 s | 52.6 s | 7 / 7 |
| syn_02_messy_illegible, `assess` | gate stops ("Q2: answer is unreadable") | NEEDS_EXTRACTION_REVIEW | 31.7 s | – | – | 31.8 s | – |
| same, after teacher correction, `resume` | Agent 1 skipped, extraction v2 | PENDING_TEACHER_REVIEW | skipped | 10.6 s | cached | 10.6 s | 5 / 5 |

Each run wrote one `audit_log` row per step it executed (3 rows for the run that stopped at the gate, 7 for full runs), with model, prompt version and duration.
Offline: 292 tests pass with the network blocked, including the full workflow with fake agents (happy path, gate stop, correction + resume with
Agent 1 called exactly once, re-run keeps history, Agent 2 error, Agent 3 outage fallback, contained unexpected exception, finalisation guards).

## Phase 7 — Integrity (2026-11-21)

### Calibration (`scripts/calibrate_integrity.py`, synthetic class)
`tests/fixtures/integrity/calibration.json` is a synthetic class: 10 students, 2 questions with long answers. It gives 90 pairs: 4 planted copies (two paraphrased analogies, one near-verbatim copy with spelling changes, one reordered anecdote) and 2 control pairs where both students copied the textbook notes. Raw output: `docs/phase7_calibration.json`.

| Setting | NEAR_DUP | SIMILAR | MARGIN | Planted found | False positives | Precision | Controls flagged |
|---|---|---|---|---|---|---|---|
| current defaults | 0.92 | 0.85 | 0.10 | 4/4 | 0 | 1.00 | none |

| Pair | Kind | Similarity | Margin over the reference |
|---|---|---|---|
| Q1 s01–s02 | planted paraphrase | 0.942 | 0.167 |
| Q1 s07–s08 | planted near-verbatim | 0.947 | 0.280 |
| Q3 s03–s04 | planted paraphrase | 0.915 | 0.225 |
| Q3 s09–s10 | planted reordered anecdote | 0.959 | 0.364 |
| Q1 s05–s06 | both copied the textbook | 0.993 | 0.031 |
| Q3 s05–s06 | both copied the textbook | 0.997 | −0.003 |
| Q1 s03–s05 (highest other) | independent correct answers | 0.925 | −0.037 |

All 80 grid settings found every planted pair with no control flagged, so this set validates the rule but does not pin the thresholds down. The margin rule does the work: two correct answers can be 0.93 similar, but each is closer still to the reference.

Bug found while building the tests: two students who copied the same two sentences of the notes were flagged. Similarity to a whole 500-character chunk underrated the partial copy (margin 0.33), and sentence windows only reduced it to 0.107. Adding lexical containment against the chunk and its neighbours fixed it (D-11 addendum).

### Live check (2026-11-21)
The clean synthetic page was uploaded again as a different student through `cli assess`.
- Both submissions were flagged `EXACT_DUPLICATE`, and Agent 1 was skipped (the extraction was reused).
- The Judge saw the flag as context, and the marks were unchanged at 7/7.
- Total time was 17.0 s, against 52.6 s for the first upload.
- The Judge echoed the flag back as its own, giving a duplicate in the report. Fixed: repeated flags are now dropped.

### Gold set
_Pending task 1.6: plant 3–4 copied pairs among the real pages and re-run the calibration._

## Phase 8 — Dashboard (2026-11-21)

`streamlit run streamlit_app.py`

| Check | Result |
|---|---|
| AppTest page tests, offline (`tests/integration/test_dashboard.py`) | 11 passed |
| The legacy app as a page | same title, both uploaders and the sidebar score input as `streamlit run app.py`; no `set_page_config` conflict |
| A rubric built in the quick form | equal to `from_simple_model_answer` output, read back through `Storage.get_rubric` |
| Approve & Finalise | disabled until the reviewer is named; disabled again when a mark changes without a reason; finalises with the override recorded |
| Exported CSV | matches `teacher_decisions` row for row, per question and per criterion |
| Refresh | a new session with `?assignment=…&submission=…` reopens the same submission |
| Live run with Gemini (`tests/integration/test_dashboard_live.py`) | Submit → Teacher Review → Approve → Export in 117 s; 7/7, with highlighted evidence |
| Server start | `/_stcore/health` returns `ok` |

Still manual: a person walking through MASTER_PLAN §9 steps 1–8 in a browser, including a refresh mid-review.
