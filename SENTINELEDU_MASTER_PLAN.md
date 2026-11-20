# SentinelEdu — Master Plan

**Multi-Agent AI-Based Handwritten Assignment Evaluation and Academic Integrity Platform**

| | |
|---|---|
| Status | Planning, awaiting approval |
| Plan date | 2026-11-15 |
| Target duration | 4–6 weeks (≈ 26 working days plus buffer) |
| Companion docs | [`SENTINELEDU_TASKS.md`](SENTINELEDU_TASKS.md) · [`SENTINELEDU_ARCHITECTURE.md`](SENTINELEDU_ARCHITECTURE.md) · [`SENTINELEDU_DECISIONS.md`](SENTINELEDU_DECISIONS.md) |

---

## 1. Vision

SentinelEdu turns the existing single-question handwritten evaluator into a teacher-supervised, multi-question assessment platform:

1. **Read** handwritten image or PDF submissions accurately (Gemini multimodal OCR, with TrOCR as fallback).
2. **Grade** each answer against a teacher-defined rubric, using teacher-supplied reference material (RAG) and Sentence-BERT similarity as grounding signals.
3. **Verify** every grade with an independent judge that checks consistency, finds missed concepts and writes feedback.
4. **Protect integrity** by detecting duplicate and near-duplicate submissions.
5. **Keep the teacher in charge.** Nothing is final until a teacher approves it.

**Guiding principle: LLMs choose, Python computes.** LLMs select rubric levels and quote evidence. Marks are always calculated in Python from the rubric. A mark that has no rubric level or no verifiable evidence is rejected.

---

## 2. Current State (Repository Inspection, 2026-11-15)

### 2.1 What exists and works

| File | Role | Lines | Reuse in SentinelEdu |
|---|---|---|---|
| `app.py` | Streamlit UI: one PDF key and one image, single score | 98 | **Preserved unchanged.** Exposed as the "Legacy Evaluator" page |
| `main.py` | CLI demo of phases 1–6 | 127 | Preserved unchanged |
| `src/pipeline_runner.py` | `run_full_assessment()` orchestrator | 57 | Preserved. Used as the legacy baseline in comparisons |
| `src/ocr_pipeline.py` | OpenCV preprocessing (CLAHE, Otsu, deskew, line segmentation) and fine-tuned TrOCR | 308 | **Reused:** `extract_handwritten_text()` becomes the OCR fallback. `prepare_clean_gray`/deskew are available for optional image enhancement |
| `src/nlp_processing.py` | Regex clean, NLTK tokenize, stopwords, spell-correct | 83 | **Reused:** `clean_text_basic()` for normalisation and hashing. Spell-correction is **not** used in the new path (it changes domain terms) |
| `src/similarity.py` | SBERT `all-MiniLM-L6-v2` cosine | 28 | **Same model reused**, through a new lazy-loading embedding service that also does batch encoding |
| `src/scoring.py` | Similarity-threshold grade buckets | 36 | Preserved for the legacy path only. Replaced by rubric scoring |
| `src/agent_verifier.py` | DistilBERT-MNLI zero-shot "rescue" (+2.5 marks) | 67 | Preserved for the legacy path only. **Not used**, because it awards marks without evidence |
| `src/feedback.py` | Token-set-difference feedback | 41 | Preserved. Replaced by the Judge agent's feedback |
| `src/pdf_utils.py` | PyPDF2 text extraction | 24 | **Reused** for reference documents and typed PDF submissions |
| `my_trocr_model/` | Fine-tuned TrOCR weights (1.2 GB, git-ignored) | — | Reused by the fallback only. **No retraining** |

### 2.2 Findings to handle in Phase 1 (Foundation)

| # | Finding | Impact | Planned action |
|---|---|---|---|
| F1 | `venv/` uses **Python 3.9.6**, which reached end-of-life in Oct 2025. Current Agno and `google-genai` releases target newer Python versions | Blocks Phase 2 installs | Create a new `.venv` on Python 3.11 and keep the old `venv/` until the new one is verified |
| F2 | `requirements.txt` has drifted. It pins `numpy<2` but numpy 2.0.2 is installed. It pins `opencv-python==4.9.0.80` but 4.13 is installed. `pytesseract`, `scikit-learn` and `requests` are listed but never imported | Environments can't be reproduced | Regenerate pinned requirements from a verified working environment |
| F3 | **`venv/pyvenv.cfg` contains a stray line that looks like an API/OAuth token** (starts with `AQ.`). The file is git-ignored and was never committed | Possible secret exposure | Revoke or rotate the credential if it is real, and delete the line. Store all keys in `.env` from now on |
| F4 | `src/similarity.py` and `src/agent_verifier.py` load models **when imported** | Slow start, and importing either module has side effects | New code must not import `agent_verifier`. Create a lazy embedding singleton |
| F5 | The legacy pipeline runs SBERT on **stopword-stripped, spell-corrected tokens** | Lower semantic fidelity | The new path embeds lightly cleaned natural text |
| F6 | `agent_verifier.rescue_grade_if_needed` adds a flat +2.5 marks | Violates the "no unsupported marks" rule | Keep it in the legacy path only and document why |
| F7 | `app.py` writes uploads to fixed temp filenames, uses a bare `except` and the deprecated `use_column_width` | Concurrent uploads can overwrite each other, and the app shows deprecation warnings | Leave `app.py` alone (preserved). The new UI uses per-submission storage directories |
| F8 | Only one question and one model answer string. No rubric, no persistence, no tests, no `.env` | The core gaps SentinelEdu fills | Phases 3–9 |
| F9 | `data/sample_images/sample1.png` is actually a printed TIFF (`phototest.tif`), not handwriting | No realistic test data | Phase 1 task: assemble a small handwritten test set (gold set) |
| F10 | `answer_deviation` compares OCR text to the *model answer*, so it does not measure OCR quality | The metric is misleading | Phase 9 measures real OCR CER/WER against manual transcriptions |

---

## 3. Target Architecture (Summary)

Full detail and diagrams are in `SENTINELEDU_ARCHITECTURE.md`.

```
Student Submission (image / PDF)
   │
   ▼
[Integrity pre-check: SHA-256]            ← service
   │
   ▼
Agent 1 — Document Understanding (Gemini multimodal → ExtractedSubmission JSON)
   │        └─ fallback: TrOCR (src/ocr_pipeline.py) for images
   ▼
[Extraction gate]  unreadable / low confidence → pause for teacher correction
   │
   ▼
Agent 2 — Answer Evaluation (rubric + FAISS RAG + SBERT → CriterionEvaluation JSON)
   │        └─ Python validator: marks computed from rubric levels; evidence quotes verified
   ▼
Agent 3 — Judge & Feedback (consistency, overlooked concepts, feedback, flags → JudgeReport JSON)
   │        └─ Python validator: adjustments bounded and evidence-backed
   ▼
[Integrity post-check: SBERT near-duplicate across cohort]  ← service
   │
   ▼
Teacher Review (Streamlit) ──approve / override with reason──▶ Final Assessment (SQLite)
```

- **Exactly three primary agents**, built with **Agno** and run as one **Agno Workflow**.
- **Supporting services** are ordinary Python modules, not agents: RAG ingestion/retrieval (FAISS), duplicate detection (SHA-256 + SBERT), rubric management, assessment storage (SQLite), OCR fallback (TrOCR).
- **Status machine:** `UPLOADED → EXTRACTED → [NEEDS_EXTRACTION_REVIEW] → EVALUATED → PENDING_TEACHER_REVIEW → FINALIZED`.

### 3.1 New code layout (additive; legacy files untouched)

```
project_root/
├── app.py, main.py, src/ ...         # LEGACY — unchanged
├── streamlit_app.py                  # NEW multipage entry (st.navigation)
├── sentineledu/
│   ├── config.py                     # .env loading, settings, model IDs
│   ├── schemas/                      # Pydantic v2 models (contracts between agents)
│   ├── agents/                       # document_agent.py, evaluation_agent.py, judge_agent.py, prompts/
│   ├── services/                     # embeddings, rag, duplicate, rubric, storage, ocr_fallback, scoring_rules
│   ├── workflow/assessment_workflow.py
│   └── ui/pages/                     # Streamlit pages
├── tests/ unit/ integration/ e2e/ fixtures/
├── storage/                          # git-ignored: sentinel.db, uploads/, indexes/
├── .env.example                      # committed; .env is git-ignored
└── requirements.txt                  # regenerated and pinned
```

---

## 4. Development Roadmap

Detailed tasks, dependencies and acceptance criteria are in `SENTINELEDU_TASKS.md`.

| Phase | Name | Est. days | Week | Key deliverable | Exit gate |
|---|---|---|---|---|---|
| 1 | Foundation | 2 | W1 | Python 3.11 `.venv`, pinned requirements, legacy app verified, gold test set started, secret cleanup | `streamlit run app.py` grades the sample identically in the new env |
| 2 | Infrastructure | 2 | W1 | `.env` config, Agno and Gemini installed, `HelloAgent` returns a Pydantic object, schemas package | `pytest -m live tests/integration/test_hello_agent.py` passes |
| 3 | Document Agent | 4 | W1–W2 | Gemini OCR to `ExtractedSubmission`, question segmentation, legibility flags, TrOCR fallback, correction model | ≥ 90% correct question segmentation on the gold set; unreadable regions flagged |
| 4 | Evaluation Agent | 6 | W2–W3 | Rubric service, RAG (FAISS) ingestion and retrieval, SBERT service, criterion-wise evaluation, mark validator | Marks always equal the rubric-derived sum; every awarded criterion has a verified quote |
| 5 | Judge Agent | 4 | W3–W4 | Deterministic consistency checks plus an LLM judge, feedback, flags, bounded recommendations | Seeded inconsistencies are detected; no recommendation exceeds rubric bounds |
| 6 | Orchestration | 3 | W4 | One Agno Workflow (A1 → gate → A2 → A3), persistence, resume after correction | One CLI command runs a full submission end-to-end to `PENDING_TEACHER_REVIEW` |
| 7 | Integrity | 2 | W4–W5 | SHA-256 exact and SBERT near-duplicate detection, cohort comparison, flags | Planted duplicates detected; no auto-penalties |
| 8 | Dashboard | 4 | W5 | Multipage Streamlit: assignments/rubrics, upload, extraction correction, review/approve, integrity, legacy page | Teacher can complete the whole flow in the UI; finalisation is impossible without approval |
| 9 | Testing & Docs | 4 | W5–W6 | Unit, integration and E2E suites; accuracy report against the gold set; README, demo script, video | All tests green; demo script runs from a clean checkout |
| — | Buffer | 3–5 | W6 | Rework, API changes, polish | — |

**Critical path:** 1 → 2 → 3 → 4 → 5 → 6 → 8 → 9. Phase 7 depends only on Phases 2 and 4 (embedding service), so it can move earlier if Phase 4 finishes early.

**Rule for future sessions:** implement only the phase requested. Read all four planning docs first. Run that phase's tests before declaring it done. Update the checkboxes in `SENTINELEDU_TASKS.md`.

---

## 5. Milestones

| Milestone | After phase | Demonstrable outcome |
|---|---|---|
| **M1 — "Talking to Gemini"** (first milestone) | 1 + 2 | New environment, legacy app still works, a typed Agno+Gemini agent returns validated JSON from an image |
| M2 — "Reads the paper" | 3 | Upload a handwritten multi-question page and get structured per-question JSON with legibility flags |
| M3 — "Grades with evidence" | 4 | Criterion-wise marks backed by quotes and reference citations, computed from the rubric |
| M4 — "Second opinion" | 5 | Judge report with flags, overlooked concepts and feedback |
| M5 — "One workflow" | 6 | Single command: submission → pending review |
| M6 — "Integrity" | 7 | Duplicate flags across a cohort |
| M7 — "Teacher console" | 8 | Full flow in Streamlit with mandatory approval |
| M8 — "Demo-ready" | 9 | Test suite, accuracy report, docs, demo |

---

## 6. Guardrails (Non-negotiables)

1. **Three primary agents only.** Everything else is a service or a function step.
2. **No unsupported marks.** Marks come from `scoring_rules.compute_marks(rubric, selected_levels)`. The LLM cannot set a number directly. Each criterion awarded above zero needs at least one quote that can be found in the student's extracted text (fuzzy-matched).
3. **Mandatory teacher approval.** `FINALIZED` can only be set by `storage.finalize(decision)`, which requires a `TeacherDecision` with a reviewer name. Overrides require a reason.
4. **Legacy preserved.** No edits to `app.py`, `main.py` or `src/*`. The new code imports from `src/` but never modifies it.
5. **No heavy infrastructure.** No Kubernetes, Redis, Celery or microservices. SQLite and local files only.
6. **No TrOCR retraining.**
7. **Secrets** live only in `.env`, which is git-ignored. `.env.example` is committed.
8. **Demo first.** A working end-to-end path comes before extra polish.

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Gemini free-tier quota/rate limits or model deprecation during the project | High | High | Model ID in `.env`. Retry with backoff. Cache extraction and evaluation results by SHA-256 so files are never re-processed. Fixtures let tests run offline |
| R2 | Agno API changes (v2 renamed several things, e.g. `response_model` → `output_schema`) | Medium | Medium | Pin the exact version in Phase 2. Keep Agno usage in thin `agents/` and `workflow/` modules. Run a spike in Phase 2 |
| R3 | LLM hallucinates marks or evidence | Medium | High | Deterministic mark computation. Quote verification. Judge cross-check. Teacher approval |
| R4 | Poor OCR on messy handwriting, diagrams or equations | High | Medium | Per-answer legibility and confidence fields. Extraction gate. Teacher correction UI. Diagrams are flagged "requires manual grading" |
| R5 | Wrong question segmentation (unnumbered answers, answers spanning pages) | Medium | High | Rubric question list passed to Agent 1 as hints. `unmatched_text` bucket. Correction UI |
| R6 | No ground-truth dataset to measure accuracy | High | Medium | Phase 1 task: build a 10–20 page gold set (transcriptions + teacher marks) |
| R7 | Environment breakage (torch/numpy/opencv on macOS, Python upgrade) | Medium | Medium | New `.venv` built alongside the old one. Pinned requirements. Phase 1 smoke tests |
| R8 | Streamlit reruns lose multi-step state | Medium | Medium | All state lives in SQLite keyed by `submission_id`, not only in `st.session_state` |
| R9 | Latency: about 2 LLM calls per question plus 1 OCR call per submission | Medium | Low | Batch all questions into one evaluator call per submission when small. Progress UI. Single-submission demo |
| R10 | Student data privacy (sending handwriting to an external API) | Medium | Medium | Document it in the README. Use anonymous student IDs. Provide a TrOCR-only offline mode |
| R11 | Scope creep (extra agents, LMS integration, auth) | Medium | Medium | Out-of-scope list (§8). Phase gating |
| R12 | Leaked credential in `venv/pyvenv.cfg` | Unknown | High if real | Rotate it and remove the line in Phase 1 |

---

## 8. Scope

**In scope:** handwritten image (PNG/JPG) and PDF submissions; multi-question assignments; teacher-authored rubrics (JSON or UI form); reference PDFs/TXT for RAG; criterion-wise marks and evidence; judge verification and feedback; duplicate detection within an assignment; teacher correction and approval; SQLite persistence; CSV/JSON export of final marks.

**Out of scope (for this project):** user authentication and roles; LMS integration; multiple concurrent teachers; diagram/equation grading (flagged for manual review instead); non-English scripts; cloud deployment; retraining any model; plagiarism checks against the internet.

**Stretch goals (only if buffer remains):** Gemini-drafted rubric from a model-answer PDF (teacher must approve); batch upload of a whole class; PDF report export per student.

---

## 9. Success Criteria for the Final Demo

1. A teacher creates an assignment, defines a 3-question rubric and uploads one reference PDF.
2. The teacher uploads 3–5 handwritten submissions, including one deliberate near-duplicate.
3. Each submission is extracted. One partially unreadable answer is flagged and corrected in the UI.
4. Each answer receives criterion-wise marks with quotes and reference citations.
5. The Judge flags at least one questionable result and writes feedback.
6. The duplicate pair is flagged in the integrity view.
7. The teacher approves some results and overrides one with a reason. Final marks are exported.
8. The legacy evaluator page still produces its original output.
9. The accuracy report shows OCR CER/WER and mark MAE against teacher marks on the gold set, compared with the legacy pipeline.
