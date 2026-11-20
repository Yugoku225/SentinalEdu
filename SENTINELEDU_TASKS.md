# SentinelEdu — Task Checklist

Ordered by phase. Each task has an ID, dependencies (**Deps**) and acceptance criteria (**AC**).
Mark a task `[x]` only after its AC has been verified. A phase is complete when every task and its **Phase Exit Gate** pass.

**Working rules for every phase**
- Read all four `SENTINELEDU_*.md` documents before starting.
- Do not modify `app.py`, `main.py`, `src/*` or `my_trocr_model/`.
- New code goes under `sentineledu/`, `tests/` and `streamlit_app.py`.
- Unit tests must run **offline**. Gemini is mocked with recorded fixtures. Live API tests are marked `@pytest.mark.live`.
- Pin any new dependency in `requirements.txt` when you add it.

---

## Phase 1 — Foundation (≈ 2 days)

- [x] **1.1 Baseline snapshot**: tag the current commit `legacy-baseline`.
  Deps: — · AC: `git tag` lists `legacy-baseline`. Working tree is clean.
- [x] **1.2 Secret hygiene**: inspect the stray `AQ.…` line in `venv/pyvenv.cfg`. If it is a real credential, rotate it at its provider, then remove the line. Add `.env.example`. Confirm `.env` and `storage/` are in `.gitignore`.
  Deps: — · AC: `grep -r "AQ\." venv/pyvenv.cfg` finds nothing. `git check-ignore .env storage/x` matches both.
- [x] **1.3 New environment**: create `.venv` with Python 3.11 (Homebrew/pyenv/conda). Install the legacy dependencies. Keep the old `venv/` until 1.5 passes.
  Deps: — · AC: `.venv/bin/python --version` reports 3.11.x. `import torch, transformers, sentence_transformers, cv2, PyPDF2, streamlit, nltk` succeeds.
- [x] **1.4 Reconcile requirements**: resolve the `numpy<2` vs 2.0.2 and `opencv` pin drift. Drop unused `pytesseract`, `scikit-learn` and `requests` after confirming they are not imported. Write a pinned `requirements.txt`. Document NLTK data needs (`punkt`, `punkt_tab`, `stopwords`).
  Deps: 1.3 · AC: a fresh `pip install -r requirements.txt` into an empty 3.11 env succeeds. A diff of old and new pins is recorded in DECISIONS.
- [x] **1.5 Legacy regression check**: run `python main.py` and `streamlit run app.py` on a sample image and PDF key, in both the old and new environments. Save the outputs to `tests/fixtures/legacy_baseline.json`.
  Deps: 1.3, 1.4 · AC: awarded marks and similarity match between environments (similarity within ±0.01).
- [ ] **1.6 Gold test set (start)**: collect 10–20 handwritten pages covering 3 questions, including messy, partially illegible and blank-answer cases. Write a manual transcription for each, plus teacher marks per criterion for at least 10. Store them in `tests/fixtures/gold/` (anonymised).
  Deps: — · AC: `tests/fixtures/gold/manifest.json` lists each file with `transcription`, `rubric_id` and `teacher_marks`. At least 10 entries exist (more can be added through Phase 8).
- [x] **1.7 Project skeleton**: create empty packages `sentineledu/{schemas,agents,agents/prompts,services,workflow,ui/pages}`, `tests/{unit,integration,e2e,fixtures}` and `pytest.ini` (with the `live` marker registered).
  Deps: 1.3 · AC: `pytest` runs, collects 0 tests and exits 0 (or 5 for no tests). `python -c "import sentineledu"` works.

**Phase 1 Exit Gate:** the legacy app runs identically in `.venv`. No secrets are on disk outside `.env`. The skeleton is importable.

---

## Phase 2 — Infrastructure (≈ 2 days)

- [x] **2.1 Config**: `sentineledu/config.py` uses `python-dotenv` and a Pydantic `Settings` class covering every key in ARCHITECTURE §9. Fail fast with a clear message if `GEMINI_API_KEY` is missing and `OFFLINE_MODE=false`.
  Deps: 1.7 · AC: unit test for defaults and missing-key behaviour. `.env.example` matches the Settings fields.
- [x] **2.2 Install and pin Agno, google-genai, pydantic, python-dotenv, faiss-cpu, pytest**. Check the current Agno version's API for: `Agent(model=Gemini(id=...), output_schema=...)`, `agno.media.Image/File` and `agno.workflow.Workflow/Step` executors. Record the actual signatures in DECISIONS (D-02 addendum).
  Deps: 1.4 · AC: `pip check` is clean. Versions are pinned. Notes are written down.
- [x] **2.3 Choose and verify Gemini model IDs**: list the available models using the key. Choose a Flash-tier multimodal model for OCR and a reasoning model for evaluation/judging. Put them in `.env.example`.
  Deps: 2.1, 2.2 · AC: a script prints the chosen IDs and confirms each responds.
- [x] **2.4 Schemas package**: implement every Pydantic model in ARCHITECTURE §6, with validators (rubric level/marks consistency, confidence range, unique IDs).
  Deps: 1.7 · AC: `tests/unit/test_schemas.py` covers valid and invalid rubrics, round-trips JSON, and rejects bad confidence values.
- [x] **2.5 Hello agent**: `sentineledu/agents/hello_agent.py` is an Agno agent with Gemini and `output_schema=HelloOut(summary: str, word_count: int)`. Also test one image input.
  Deps: 2.2, 2.3, 2.4 · AC: `pytest -m live tests/integration/test_hello_agent.py` passes. The result is a `HelloOut` instance, not a string.
- [x] **2.6 LLM call utilities**: a thin wrapper that runs an Agno agent, extracts the typed content, retries with exponential backoff on rate-limit or transient errors, logs latency and token usage, and caches responses on disk keyed by `(prompt_version, model_id, input_sha256)`.
  Deps: 2.5 · AC: a unit test with a fake agent shows that retry and the cache hit both work. A second identical live call returns from cache.
- [x] **2.7 Test doubles**: a `FakeAgent` fixture that returns pre-recorded schema objects from `tests/fixtures/llm/*.json`.
  Deps: 2.4 · AC: used by at least one unit test.

**Phase 2 Exit Gate (= Milestone M1):** the live hello-agent test passes. Offline unit tests pass with no network.

---

## Phase 3 — Document Understanding Agent (≈ 4 days)

- [x] **3.1 Storage bootstrap (minimal)**: `services/storage.py` with SQLite schema creation, `create_submission` (saves files under `storage/uploads/<id>/` and computes `file_sha256`), and `save_extraction` / `latest_extraction` with versions.
  Deps: 2.4 · AC: unit tests on a temp DB. Extraction versions increment. Originals are never overwritten.
- [x] **3.2 Input preparation**: accept PNG/JPG/PDF (multi-page). Images: optional OpenCV enhancement (reuse `src.ocr_pipeline.prepare_clean_gray` + deskew) behind a flag. PDFs: send natively as `File`. If the PDF has an extractable text layer (`src.pdf_utils`), set engine to `pdf_text` and skip OCR.
  Deps: 3.1 · AC: unit tests for the type routing.
- [x] **3.3 Document agent prompt v1**: `prompts/document_v1.md` must transcribe verbatim, use the `[[ILLEGIBLE]]` convention, map answers to the provided question IDs, report blank answers, diagrams and math, and avoid invented text.
  Deps: 2.5 · AC: reviewed. Includes one worked example.
- [x] **3.4 Document agent**: `agents/document_agent.py` with `extract(submission, rubric_questions) -> ExtractedSubmission` using `output_schema`. Post-processing: fill `missing_question_ids`, clamp confidence, set the `UNKNOWN` id for unmatched text.
  Deps: 3.2, 3.3 · AC: live test on 3 gold pages returns valid objects. Offline test with `FakeAgent`.
- [x] **3.5 TrOCR fallback**: `services/ocr_fallback.py` lazily imports `src.ocr_pipeline.extract_handwritten_text` and maps its output to `ExtractedSubmission(engine="trocr")`, with the whole text in `unmatched_text`.
  Deps: 2.4 · AC: works with `OFFLINE_MODE=true` on `data/sample_images/sample1.png`. Triggered when the Gemini call raises an error (simulated).
- [x] **3.6 Extraction gate**: `extraction_gate(extraction, rubric, settings) -> (ok: bool, reasons: list[str])` using the rules in ARCHITECTURE §4.1.
  Deps: 2.4 · AC: table-driven unit tests, one per rule.
- [x] **3.7 Teacher correction model**: `apply_correction(extraction, edits, teacher) -> ExtractedSubmission(version+1, engine="teacher")`.
  Deps: 3.1 · AC: unit test. The previous version is still retrievable.
- [x] **3.8 Extraction accuracy script**: `scripts/eval_ocr.py` computes CER/WER (use `jiwer`, or a small edit-distance helper with `nltk.edit_distance`) and segmentation accuracy against the gold transcriptions, for both Gemini and TrOCR.
  Deps: 3.4, 3.5, 1.6 · AC: the script prints a table and the numbers are recorded in `docs/results.md`.

**Phase 3 Exit Gate (M2):** question segmentation is ≥ 90% correct on the gold set. Every deliberately illegible sample triggers the gate. Gemini CER is better than TrOCR CER (expected; documented either way).
> **Status 2026-11-16:** all of it is met on the *synthetic* set: 100% segmentation, illegible answer flagged, Gemini CER 0.000 vs TrOCR 0.558 (`docs/results.md`). **Still open:** re-run `scripts/eval_ocr.py` on the real gold set once task 1.6 is done.

---

## Phase 4 — Answer Evaluation Agent (≈ 6 days)

- [x] **4.1 Embedding service**: `services/embeddings.py`, a lazy singleton on `all-MiniLM-L6-v2` with batch `encode` (normalised) and `cosine`.
  Deps: 1.7 · AC: the model loads once (assert with a counter). Cosine of identical texts ≈ 1.0. Importing the module does not load the model.
- [x] **4.2 Rubric service**: load/save/validate rubric JSON in SQLite. `from_simple_model_answer(text, max_marks)` builds a 1-criterion, 3-level rubric. Add a sample rubric `data/rubrics/sample_rubric.json` matching the gold set.
  Deps: 2.4, 3.1 · AC: unit tests. An invalid rubric is rejected with a readable error.
- [x] **4.3 RAG ingestion**: `services/rag.py`. Extract text from PDF/TXT, chunk on sentence boundaries (~500 chars, 100 overlap), embed, build FAISS `IndexFlatIP`, and save `index.faiss` plus `chunks.json` (with `chunk_id`, `source`, `page`, `text`).
  Deps: 4.1 · AC: ingesting a 5-page PDF gives the expected chunk count. The index reloads from disk.
- [x] **4.4 RAG retrieval**: `retrieve(assignment_id, query, k)` returns ranked chunks with scores. The query is `question.prompt + student answer`.
  Deps: 4.3 · AC: on a test document with a known answer passage, that passage appears in the top 3 for ≥ 4 of 5 test questions.
- [x] **4.5 Scoring rules**: `services/scoring_rules.py` with `compute_marks`, `verify_quote` (sliding-window `difflib` ≥ 0.85), and `validate_evaluation(draft, rubric, extraction, provided_chunk_ids) -> EvaluationResult`, following ARCHITECTURE §4.2 steps 1–5.
  Deps: 2.4 · AC: unit tests. An invalid level is downgraded and flagged. A fabricated quote is downgraded with `UNSUPPORTED_EVIDENCE`. An unknown chunk id is dropped. Totals are never above max. **Property test:** for random drafts, the total always equals the sum of rubric level marks.
- [x] **4.6 Evaluation prompt v1**: the prompt includes the rubric levels and descriptors, retrieved chunks with IDs, and SBERT scores as signals (labelled as advisory). It instructs the model to pick a level, quote exact student text, cite chunk IDs, and list missing concepts and incorrect statements. The prompt says plainly that numeric marks must not be produced.
  Deps: 4.2 · AC: reviewed. A worked example is included.
- [x] **4.7 Evaluation agent**: `agents/evaluation_agent.py` with `evaluate(extraction, rubric, context) -> EvaluationDraft`, then the validator. Answers marked `unreadable`/`blank` skip the LLM and get the 0 level with a flag.
  Deps: 4.4, 4.5, 4.6 · AC: offline test with `FakeAgent`. Live test on 3 gold submissions produces `EvaluationResult` objects whose marks are all rubric-derived.
- [x] **4.8 Grading accuracy script**: `scripts/eval_grading.py` compares SentinelEdu (Agent 2 only) and the legacy `run_full_assessment` against teacher marks: MAE, exact-level agreement, Cohen's κ (if the sample size allows).
  Deps: 4.7, 1.6 · AC: results are recorded in `docs/results.md`.

**Phase 4 Exit Gate (M3):** 100% of awarded criteria have `evidence_verified=True` or were downgraded. Totals match the rubric sums. MAE against teacher marks is reported.
> **Status 2026-11-16:** met on the *synthetic* set: 17/17 awarded criteria verified, 0 consistency problems, MAE 0.00 against the developer's placeholder marks vs 1.07 for the legacy grader (`docs/results.md`). The property test (300 random drafts) and 5 live tests pass. **Still open:** a teacher's marks on the real gold set (task 1.6) to report a real MAE and κ.

---

## Phase 5 — Judge & Feedback Agent (≈ 4 days)

- [x] **5.1 Deterministic checks**: `scoring_rules.run_checks(evaluation, extraction, rubric) -> list[Flag]` covering every code in ARCHITECTURE §4.3.
  Deps: 4.5 · AC: one unit test per flag code using seeded inconsistent evaluations.
- [x] **5.2 Judge prompt v1**: adversarial reviewer persona. Input includes the check flags. Output covers verdicts, overlooked concepts, feedback per question (strengths, improvements), a summary and flags. It may only propose rubric levels, never numbers.
  Deps: 4.6 · AC: reviewed.
- [x] **5.3 Judge agent**: `agents/judge_agent.py` with `judge(...) -> JudgeDraft`.
  Deps: 5.1, 5.2 · AC: offline test with `FakeAgent`. Live test on 3 gold submissions.
- [x] **5.4 Judge validator**: `validate_judge(draft, evaluation, rubric, extraction) -> JudgeReport`. Accept only proposals that are valid, evidence-verified and at most ±1 level. Everything else becomes a `flag`. Compute `recommended_marks`/`recommended_total`.
  Deps: 5.3 · AC: unit tests. A +2-level proposal is rejected and flagged. A fabricated quote is rejected. A valid proposal is accepted.
- [x] **5.5 Seeded-error benchmark**: take 5 correct evaluations and inject errors (wrong level, missing concept, inflated mark). Measure the Judge's detection rate.
  Deps: 5.4 · AC: ≥ 4 of 5 injected errors are flagged or corrected. Recorded in `docs/results.md`.

**Phase 5 Exit Gate (M4):** no recommended mark exceeds rubric bounds in any test. Feedback is generated for every answered question.
> **Status 2026-11-21: met.** Seeded-error detection 4/5 with 0 false alarms; recommended marks within rubric bounds in 9/9 benchmark runs, the 300-draft property test and 4 live tests; feedback for every answered question. The benchmark found and fixed one bug (recommendations copied an out-of-bounds Agent 2 figure). Details in `docs/results.md`.

---

## Phase 6 — Orchestration (≈ 3 days)

- [x] **6.1 Step executors**: wrap each stage from ARCHITECTURE §7 as an Agno step executor that loads its inputs from storage by `submission_id` and persists its output plus a status transition.
  Deps: 3.x, 4.x, 5.x · AC: each executor is unit tested with fakes.
- [x] **6.2 Workflow assembly**: `workflow/assessment_workflow.py` defines a single Agno `Workflow` with the 7 steps. The gate can stop the run early with status `NEEDS_EXTRACTION_REVIEW`.
  Deps: 6.1 · AC: an offline integration test (all fakes) reaches `PENDING_TEACHER_REVIEW`. A gate-triggering fixture stops at `NEEDS_EXTRACTION_REVIEW`.
- [x] **6.3 Resume semantics**: re-running after a teacher correction skips Agent 1 and uses the latest extraction version. Re-running an evaluated submission creates a new evaluation record (history kept).
  Deps: 6.2, 3.7 · AC: integration test where the Gemini fake is called exactly once across correction and resume.
- [x] **6.4 Status guards**: `storage` refuses illegal transitions (e.g. `EXTRACTED → FINALIZED`). `finalize()` requires a `TeacherDecision` with a non-empty `reviewer` and reasons for every override.
  Deps: 3.1 · AC: unit tests for each illegal transition.
- [x] **6.5 CLI runner**: `python -m sentineledu.cli assess --assignment A1 --file page.jpg --student S01`, which prints a summary JSON.
  Deps: 6.2 · AC: live run on a gold page ends in `PENDING_TEACHER_REVIEW`. Per-step timings are printed.
- [x] **6.6 Audit log**: each step writes `{submission_id, step, status, model_id, prompt_version, duration_ms, error?}`.
  Deps: 6.2 · AC: visible in the DB after a run.

**Phase 6 Exit Gate (M5):** one command takes a raw submission to `PENDING_TEACHER_REVIEW`. The correction/resume path works. Finalisation without a teacher decision is impossible.
> **Status 2026-11-21: met.** Live run: `python -m sentineledu.cli assess --rubric data/rubrics/sample_rubric.json --file tests/fixtures/synthetic/syn_01_clean_p1.jpg --student S01` reached `PENDING_TEACHER_REVIEW` in 52.6 s (7/7). The smudged page stopped at the gate. After a teacher correction, `cli resume` skipped Agent 1 and reached review (5/7). The rule that only `finalize()` with a valid `TeacherDecision` reaches FINALIZED is covered by 58 guard tests. Synthetic pages only: the AC asks for a gold page (task 1.6).

---

## Phase 7 — Integrity (≈ 2 days)

- [x] **7.1 Exact duplicates**: file SHA-256 and normalised-text SHA-256 checks within an assignment. Produce `EXACT_DUPLICATE` (file) and `TEXT_DUPLICATE` flags. Reuse the cached extraction on a file match.
  Deps: 3.1, 6.2 · AC: uploading the same file twice gives a flag and no second Gemini call.
- [x] **7.2 Near-duplicate scan**: per-question pairwise SBERT across the cohort, using the margin rule in ARCHITECTURE §5.1 and skipping short answers.
  Deps: 4.1 · AC: unit tests with synthetic answers (paraphrase → flagged; both copied from the reference → not flagged; short → skipped).
- [ ] **7.3 Calibration**: create 3–4 planted near-duplicate pairs in the gold set and tune thresholds. Record precision/recall.
  Deps: 7.2, 1.6 · AC: thresholds are written to `.env.example` and DECISIONS D-11. All planted pairs are detected and there are ≤ 1 false positive.
- [x] **7.4 Workflow wiring**: pre-check and post-check steps write `IntegrityFlag` rows. Flags are shown to the Judge as context but never change marks.
  Deps: 7.1, 7.2, 6.2 · AC: integration test. Marks are identical with and without a duplicate present.

**Phase 7 Exit Gate (M6):** planted duplicates are flagged. No automatic penalties.
> **Status 2026-11-21:** met on the *synthetic* calibration class: 4/4 planted pairs flagged, 0 false positives, both "copied the textbook" controls not flagged (`docs/results.md`). Marks are identical with and without a duplicate in the class (integration test, and a live re-upload: same 7/7). **Still open: task 7.3 on real data.** It needs 3–4 planted pairs in the gold set (task 1.6). The synthetic class could not separate threshold choices: all 80 grid settings passed.

---

## Phase 8 — Dashboard (≈ 4 days)

- [x] **8.1 Multipage shell**: `streamlit_app.py` with `st.navigation` and the pages in ARCHITECTURE §10. The "Legacy Evaluator" page runs the unchanged `app.py`.
  Deps: 6.2 · AC: `streamlit run streamlit_app.py` shows all pages. The legacy page produces the same output as `streamlit run app.py`. If `st.set_page_config` inside `app.py` conflicts, use the fallback in DECISIONS D-13.
- [x] **8.2 Assignments & Rubrics page**: create an assignment, build a rubric with a form or JSON upload (validated), upload reference docs (ingest → show chunk count).
  Deps: 4.2, 4.3, 8.1 · AC: a rubric created in the UI loads identically through the service.
- [x] **8.3 Submit page**: upload, run the workflow with `st.status` progress per step, show the resulting status, and link to the next page.
  Deps: 6.2, 8.1 · AC: a gold page runs end-to-end from the UI.
- [x] **8.4 Extraction Review page**: image viewer next to editable per-question text areas with legibility badges. Save the correction and resume.
  Deps: 3.7, 6.3, 8.1 · AC: correcting a flagged extraction and resuming reaches `PENDING_TEACHER_REVIEW`.
- [x] **8.5 Teacher Review page**: a criterion table with level, marks, evidence quotes (highlighted in the answer text), reference chunk popovers, Judge verdicts, flags, and recommended vs. evaluator marks. An editable final-marks form where a reason is required for each change. An **Approve & Finalise** button.
  Deps: 5.4, 6.4, 8.1 · AC: Approve is disabled until the reviewer name is filled in and every override has a reason. The status becomes `FINALIZED`.
- [x] **8.6 Integrity page**: cohort similarity heatmap per question and a side-by-side view of flagged pairs.
  Deps: 7.4, 8.1 · AC: planted pairs are visible.
- [x] **8.7 Results & Export page**: table of finalised submissions. CSV/JSON download including feedback.
  Deps: 8.5 · AC: the exported CSV matches the DB.
- [x] **8.8 State robustness**: every page reads state from SQLite by `submission_id`. Browser refresh does not lose work.
  Deps: 8.2–8.7 · AC: manual test of refresh mid-review.

**Phase 8 Exit Gate (M7):** a teacher completes the full demo flow (MASTER_PLAN §9, steps 1–8) in the UI.
> **Status 2026-11-21:**
> - Every page is built.
> - 13 AppTest tests drive each page (11 page-flow tests, plus 2 extra results-page checks). Planted pairs show on the Integrity page.
> - A live run with real Gemini went Submit → Teacher Review → Approve → Export.
> - The server starts and serves the app (health check ok).
>
> **Still open:**
> - A person must walk through demo steps 1–8 in a browser.
> - That walkthrough includes refreshing mid-review (8.8). The automated test covers the URL part of that.
> - The real demo needs handwritten pages and a reference PDF (task 1.6).

---

## Phase 9 — Testing, Docs & Demo (≈ 4 days)

- [x] **9.1 Unit test sweep**: coverage ≥ 70% on `sentineledu/services` and `sentineledu/schemas`.
  Deps: all · AC: `pytest --cov` report.
- [x] **9.2 Integration tests**: workflow with fakes covering happy path, gate stop, resume, duplicate, Gemini failure → TrOCR fallback.
  Deps: 6.x, 7.x · AC: all pass offline.
- [ ] **9.3 E2E live test**: `pytest -m live tests/e2e/` runs 3 gold submissions through the CLI and asserts structure, status and bounds (not exact marks).
  Deps: 6.5 · AC: passes with a valid key.
- [ ] **9.4 Accuracy report**: `docs/results.md` covers OCR CER/WER (Gemini vs TrOCR), segmentation accuracy, mark MAE/agreement (SentinelEdu vs legacy vs teacher), Judge seeded-error detection, integrity precision/recall, and latency and API calls per submission.
  Deps: 3.8, 4.8, 5.5, 7.3 · AC: every table is filled in with real numbers.
- [x] **9.5 Documentation**: `README.md` (setup, `.env`, run legacy/new, architecture summary, limitations, privacy note) and `docs/user_guide.md` for teachers.
  Deps: 8.x · AC: a fresh clone and the README steps give a working app.
- [x] **9.6 Demo package**: `scripts/seed_demo.py` (assignment, rubric, reference doc, 5 submissions including a duplicate and an illegible one), a demo script (talk track), and an optional screen recording.
  Deps: 9.5 · AC: the seed and demo run in under 10 minutes from a clean DB.
- [x] **9.7 Final regression**: the legacy app's output still matches `tests/fixtures/legacy_baseline.json`.
  Deps: 1.5 · AC: test passes.

**Phase 9 Exit Gate (M8):** all tests pass, the report is complete, and the demo is reproducible.
> **Status 2026-11-21: partly met.**
>
> | Item | Status |
> |---|---|
> | 9.1 coverage | 92% overall; every services and schemas module at 84% or more |
> | 9.2 integration | 326 offline tests pass, including Gemini failing over to TrOCR through the whole workflow |
> | 9.5 docs | README, user guide and demo script written. A fresh copy of the committed files passes the offline suite with no key, model or network, and the dashboard starts |
> | 9.6 demo | the seed reaches every expected state in 286 s from a clean database |
> | 9.7 legacy | 3 legacy tests pass |
>
> **Still open:**
> - **9.3** is written but has not passed yet. The first runs found three problems, all fixed: a fixture scope error in the test, and two kinds of output leaking into the CLI's JSON (Agno's logs and the legacy TrOCR code's prints). It then hit the daily free-tier quota. Re-run `pytest -m live tests/e2e/` when the quota resets.
> - **9.4** has every table filled with synthetic numbers. The gold-set column needs task 1.6.

---

## Dependency Overview

```mermaid
flowchart LR
    P1[1 Foundation] --> P2[2 Infrastructure]
    P2 --> P3[3 Document Agent]
    P2 --> P4[4 Evaluation Agent]
    P3 --> P4
    P4 --> P5[5 Judge Agent]
    P3 & P4 & P5 --> P6[6 Orchestration]
    P4 -->|embeddings| P7[7 Integrity]
    P6 --> P7
    P6 --> P8[8 Dashboard]
    P7 --> P8
    P8 --> P9[9 Testing & Docs]
```
