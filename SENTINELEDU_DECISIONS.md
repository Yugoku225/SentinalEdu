# SentinelEdu — Technical Decisions

Architecture Decision Record (ADR-lite). Each entry has: **Decision**, **Why**, **Alternatives considered**, **Consequences**.
Status values: `Accepted` (planned), `To verify` (confirm during the named phase), `Superseded`.
When a decision changes during implementation, add an addendum. Do not rewrite history.

---

## D-01 Additive upgrade; the legacy evaluator is preserved — *Accepted*
**Decision:** All new code goes in `sentineledu/`, `tests/` and `streamlit_app.py`. `app.py`, `main.py` and `src/*` are not modified. The commit is tagged `legacy-baseline`.
**Why:** The project requires preserving the working evaluator. It also serves as the baseline in the accuracy report, which makes the gains measurable.
**Alternatives:** Refactoring `src/` in place. Rejected: it risks breaking the existing app and loses a clean baseline.
**Consequences:** Some duplication (e.g. a new embedding service using the same model as `src/similarity.py`). This is accepted for isolation.

## D-02 Agno for agents and workflow orchestration — *Accepted, API to verify in Phase 2*
**Decision:** Use Agno `Agent` with `agno.models.google.Gemini` and Pydantic `output_schema`. Use `agno.workflow.Workflow` with `Step` executors for orchestration. Pin the exact version.
**Why:** Required by the project. Agno has native Gemini support (through `google-genai`), multimodal inputs (`agno.media.Image/File`), typed structured outputs and deterministic step workflows, with no server needed.
**Alternatives:** LangGraph (more ceremony, larger dependency surface), CrewAI (role-play oriented, less control over deterministic steps), hand-written orchestration (loses the framework requirement).
**Consequences:** Agno changes quickly (v2 renamed `response_model` → `output_schema` and redesigned Workflows). Agno usage is confined to `agents/*` and `workflow/*` so that an API change only touches those files.
**Addendum (Phase 2, 2026-11-15): Agno 3.2.0 pinned** (newer than the 2.x API this plan assumed; the planned shapes still hold). Verified against the installed package:
- `Agent(model=Gemini(...), output_schema=Model, instructions=..., fallback_config=..., telemetry=False)`. `agent.run(input, images=[Image(filepath=...)], files=[File(...)], output_schema=...)` returns `RunOutput` whose `.content` **is the Pydantic instance**.
- **Agno does not raise on provider errors.** It returns `RunOutput(status=RunStatus.error)` with the error text as `content`. `agents/runner.run_structured` checks the status so an error message is never parsed as an answer, and adds retries (429/5xx only), an on-disk cache and logging.
- `FallbackConfig(on_error=[...], on_rate_limit=[...], callback=...)`: `on_error` covers 5xx/network errors only, and 429 needs `on_rate_limit`. Non-retryable errors (400/404) never fall back. `RunOutput.model` still reports the primary model when a fallback answered, so `base.build_agent` records fallback activations through the callback on `agent.fallback_events`.
- `Gemini(id, api_key, temperature, thinking_level ∈ {"low","high"}, timeout)`, where `timeout` is in **seconds** (Agno converts it to ms).
- Workflows: `Workflow(name, steps=[Step(name, agent=... | executor=fn)])`. Executors receive `StepInput` (`get_step_content(name)`, `previous_step_content`) and return `StepOutput(content, success, error, stop)`. `stop=True` ends the run early, which the extraction gate needs. Agno 3.x also has `HumanReview`/`pause` primitives. D-15 stands (approval stays outside the workflow), but they can be reconsidered in Phase 6.
- Telemetry: Agno sends anonymous usage telemetry by default. We set `telemetry=False` and `AGNO_TELEMETRY=false`.

## D-03 Gemini for OCR and reasoning — *Accepted*
**Decision:** Gemini multimodal handles handwriting transcription and question segmentation. Gemini also powers the Evaluation and Judge agents. Model IDs come from `.env` (`GEMINI_OCR_MODEL`, `GEMINI_REASONING_MODEL`) and are chosen in Phase 2 from the models available to the key: Flash-tier for OCR, a stronger reasoning tier for evaluation/judging if the quota allows.
**Why:** The fine-tuned TrOCR is line-level and has no layout understanding. It cannot split answers by question or flag illegible regions. Gemini reads full pages and PDFs natively and can return JSON that matches a schema.
**Alternatives:** TrOCR only (no segmentation), PaddleOCR (models are already in the HF cache, but it is printed-text oriented and has no semantics), Tesseract (poor on handwriting).
**Consequences:** External API dependency, quota limits and privacy considerations (R1, R10). Mitigated by caching, retries, the TrOCR fallback and `OFFLINE_MODE`.
**Addendum (Phase 2, 2026-11-15): models chosen** with `scripts/check_gemini_models.py` against the project key:
| Role | Model | Notes |
|---|---|---|
| OCR + reasoning (primary) | `gemini-3.7-flash` | Agno's default. Works, but returned 503 "high demand" often and took ~30–50 s per small call on this date |
| Fallback (automatic) | `gemini-3.5-flash` | ~5–7 s per small call. Used automatically on 503/429 |
| Unavailable | `gemini-2.5-*` (retired, 404), `gemini-4-*` (no quota on this key, 429), `gemini-3.8-flash` (503 at test time) | — |
`thinking_level=low` is the default, because extraction and rubric selection don't need long reasoning. If 3.7-flash latency stays high, swap the primary and fallback in `.env`. Phase 3 compares their OCR quality on the gold set before that decision.
**Addendum (Phase 3, 2026-11-16): primary switched to `gemini-3.5-flash`, fallback `gemini-3.7-flash`.** On the synthetic pages both scored CER 0.000 with 100% segmentation and correct blank/illegible flags (`docs/results.md`). 3.5-flash averaged 5.9 s per submission. 3.7-flash returned 503 on 3 of 4 requests. Re-check on the real gold set. If 3.7-flash reads real handwriting clearly better, swap back through `.env` (no code change).

## D-04 Exactly three agents; everything else is a service — *Accepted*
**Decision:** Agents: Document Understanding, Answer Evaluation, Judge & Feedback. RAG, duplicate detection, rubric management, storage, OCR fallback, extraction gate and scoring rules are plain Python.
**Why:** This is a project constraint. Work that is deterministic should not involve an LLM. That makes it testable, cheap and reproducible.
**Consequences:** The workflow also contains non-agent function steps. That is allowed and intended.

## D-05 Pydantic v2 schemas as the contracts between agents — *Accepted*
**Decision:** Every agent input and output is a Pydantic model in `sentineledu/schemas/` (see ARCHITECTURE §6). Persisted as JSON in SQLite.
**Why:** The project requires agents to exchange structured outputs. Validation catches malformed LLM output early. The same models drive the UI forms.
**Consequences:** Schema changes need migration care. A `version` field is kept on extraction and rubric objects.

## D-06 "LLMs choose, Python computes" for marks — *Accepted*
**Decision:** The Evaluation agent selects a discrete **rubric level** per criterion and supplies quotes. `scoring_rules.compute_marks` converts levels to marks. The Judge may only propose a different level, at most ±1, with verified evidence. Teacher approval is required to finalise.
**Why:** This directly enforces "do not allow LLMs to assign unsupported marks". It makes grades explainable and stable across runs. It removes arithmetic errors.
**Alternatives:** The LLM outputs numeric marks (unbounded, not reproducible). Pure SBERT threshold scoring (the legacy approach; coarse and rubric-blind).
**Consequences:** Rubrics must define levels. `rubric.from_simple_model_answer()` creates a default 3-level rubric (full/partial/none) for legacy-style single-answer keys.

## D-07 Pre-computed RAG and SBERT context, not agent tool calls — *Accepted*
**Decision:** Before Agent 2 runs, Python retrieves the top-k reference chunks and computes SBERT similarities for each question, then injects them into the prompt with chunk IDs. The agent makes one structured-output call with no tools.
**Why:** (1) Deterministic and auditable: we know exactly which context the model saw, so citations can be validated against it. (2) Fewer round-trips and lower latency and cost. (3) Avoids provider limits on combining function calling with strict structured output.
**Alternatives:** Agentic RAG, where the agent calls a `search_reference` tool. Reconsider only if retrieval quality proves insufficient, because tool use makes citation validation harder.
**Consequences:** The retrieval query is fixed (question prompt + answer). If recall is weak, tune k and chunking first.

## D-08 Sentence-BERT `all-MiniLM-L6-v2` — *Accepted*
**Decision:** Reuse the model already used by `src/similarity.py` (already in the local HF cache), through a new lazy singleton with batch encoding and normalised vectors.
**Why:** No new download. Fast on CPU. 384-dimensional vectors keep FAISS small. Good enough for similarity signals and short-passage retrieval.
**Alternatives:** `all-mpnet-base-v2` (better quality, about 5× slower). Gemini embeddings (more API calls, network dependency). Revisit only if the Phase 4 retrieval test fails.
**Consequences:** SBERT scores are **advisory signals** for Agent 2 and the Judge's disagreement checks. They do not set marks.

## D-09 FAISS `IndexFlatIP`, one index per assignment, stored on disk — *Accepted*
**Decision:** `faiss-cpu`. Exact inner-product search on normalised vectors (equal to cosine). Files are `storage/indexes/<assignment_id>/index.faiss` plus `chunks.json` metadata.
**Why:** The reference corpus per assignment is small (hundreds to low thousands of chunks), so exact search is instant and needs no tuning. FAISS is the required stack and needs no server.
**Alternatives:** Chroma/Qdrant (unnecessary infrastructure), IVF/HNSW indexes (pointless at this scale).
**Consequences:** Rebuild the index when a reference document is added or removed (`rag.rebuild`). That is cheap at this scale.

## D-10 Chunking ~500 characters with 100 overlap on sentence boundaries — *Accepted, tune in Phase 4*
**Why:** MiniLM handles short passages well (it truncates around 256 word pieces). Small chunks make citations precise for evidence display.
**Consequences:** Tune using the Phase 4.4 retrieval test.

## D-11 Integrity: SHA-256 exact matching plus SBERT near-duplicates with a reference margin — *Accepted, thresholds calibrated in Phase 7*
**Decision:** Exact: SHA-256 of file bytes and of normalised text. Near: per-question cosine ≥ 0.92 (high) or ≥ 0.85 (review), and only when the pair is at least 0.10 closer to each other than either answer is to the model answer. Answers under 15 tokens are skipped. Flags only; no automatic penalty.
**Why:** Correct answers naturally resemble the model answer and each other. The margin rule separates "both correct" from "copied from each other". File hashing also prevents paying for OCR twice on resubmissions.
**Alternatives:** MinHash/shingling (lexical only, misses paraphrase), external plagiarism APIs (out of scope).
**Consequences:** O(n²) pairs per question is fine for class sizes up to a few hundred.
**Addendum (Phase 7, 2026-11-21):**
- **Thresholds kept:** `NEAR_DUP_THRESHOLD=0.92`, `SIMILAR_THRESHOLD=0.85`, `INTEGRITY_MARGIN=0.10`, `INTEGRITY_MIN_TOKENS=15`. They are in `.env.example`.
- **Synthetic calibration** (`tests/fixtures/integrity/calibration.json`: 10 students, 90 pairs, 4 planted, 2 textbook-copy controls):
  - recall 4/4, precision 1.00, controls not flagged;
  - smallest planted margin 0.167, largest margin of any innocent pair 0.031;
  - every one of the 80 grid settings passed, so this set validates the rule but cannot choose between thresholds. Re-run `scripts/calibrate_integrity.py` on planted pairs in the gold set.
- **"Reference" in the margin rule** means the larger of three things:
  - SBERT cosine to the model answer;
  - SBERT cosine to the top-k retrieved reference chunks;
  - **lexical containment**: the best sliding-window `difflib` match of the answer inside a retrieved chunk joined with its neighbours.

  Without containment, two students who both copied two sentences of the notes were flagged: cosine to a whole 500-character chunk underrates a partial copy, and the copied passage straddled a chunk boundary. With it, textbook copies score 0.97–1.00 against the reference, while student-to-student paraphrases score 0.35–0.53.
- **Text normalisation** re-implements the regex of the legacy `clean_text_basic` (then lower-cases), because importing `src.nlp_processing` loads NLTK and the spellchecker.
- **No redundant flags:**
  - pairs already flagged `EXACT_DUPLICATE` or `TEXT_DUPLICATE` are not also given per-question near-duplicate flags;
  - Judge-written flags that repeat a context flag are dropped.

## D-12 SQLite (stdlib `sqlite3`) and local files for storage — *Accepted*
**Decision:** One `storage/sentinel.db` with JSON columns, plus upload and index directories. No ORM.
**Why:** Zero infrastructure, survives Streamlit reruns, and is easy to inspect for a demo. It satisfies "no unnecessary infrastructure".
**Alternatives:** PostgreSQL (needs a server), JSON files only (no querying, race-prone), SQLAlchemy (extra abstraction not needed at this size).
**Consequences:** Single writer. Fine for a one-teacher demo.

## D-13 Streamlit multipage via `st.navigation`; legacy app embedded unchanged — *Accepted, verify in Phase 8*
**Decision:** The new entry point `streamlit_app.py` registers SentinelEdu pages and `st.Page("app.py", title="Legacy Evaluator")`.
**Why:** It upgrades the existing Streamlit interface without editing `app.py`. `streamlit run app.py` still works on its own.
**Risk and fallback:** If `st.set_page_config` in `app.py` conflicts with the navigation entry point in the installed Streamlit version, add a thin wrapper page `sentineledu/ui/pages/legacy.py` that reimplements the same UI by calling the unchanged `src.pipeline_runner.run_full_assessment`. `app.py` itself still stays untouched.
**Alternatives:** Next.js (prohibited), the `pages/` directory convention (auto-discovery is less controllable and would make `app.py` the landing page).
**Addendum (Phase 8, verified 2026-11-21):** with Streamlit 1.50, `st.Page("app.py")` under `st.navigation` runs the unchanged legacy app. Its own `st.set_page_config` call does not conflict, and the title, both uploaders and the sidebar score input all render. The fallback wrapper page was not needed.

## D-14 TrOCR kept as an offline fallback only, no retraining — *Accepted*
**Decision:** `services/ocr_fallback.py` wraps `src.ocr_pipeline.extract_handwritten_text()` and is imported lazily. It is used when Gemini fails or when `OFFLINE_MODE=true`, for images only. Fallback output always goes through teacher extraction review, because TrOCR cannot segment by question.
**Why:** Reuses the existing investment, gives resilience to API outages and offers a privacy-preserving mode. Retraining is out of scope.

## D-15 Teacher approval sits outside the Agno workflow — *Accepted*
**Decision:** The workflow ends at `PENDING_TEACHER_REVIEW`. Approval is a UI action that calls `storage.finalize(TeacherDecision)`. The extraction gate stops the workflow early, and re-running resumes from stored state.
**Why:** Review can happen much later. Keeping a workflow suspended across Streamlit reruns is fragile. Persisted state plus an idempotent re-run is simpler and more robust. Finalisation is still structurally impossible without a teacher decision (status guards, D-06).
**Alternatives:** Agno's built-in human-in-the-loop pause/confirmation features. Reconsider if they prove simple to persist across Streamlit sessions.

## D-16 New Python 3.11 virtual environment — *Accepted*
**Decision:** Create `.venv` on Python 3.11. Keep the old `venv/` (3.9.6) until legacy parity is confirmed, then retire it.
**Why:** 3.9 reached end-of-life in Oct 2025. Current Agno, `google-genai` and recent Streamlit/torch releases target newer Pythons. 3.11 is mature and has wheels for torch, faiss-cpu and opencv on macOS.
**Consequences:** Phase 1 must confirm the legacy outputs are unchanged (Task 1.5).
**Addendum (Phase 1, 2026-11-15):** `.venv` was created with Anaconda (`conda create -p ./.venv python=3.11`) because no standalone Python 3.11 was installed. That gave Python 3.11.18 on arm64. Activate it with `conda activate ./.venv`, not `source .venv/bin/activate`. The legacy evaluator (`scripts/legacy_baseline.py` and `main.py`) produced identical OCR text, confidence, similarity, marks and feedback under 3.9.6 and 3.11.18. The baseline is saved in `tests/fixtures/legacy_baseline.json` and checked by `pytest -m legacy`. The old `venv/` can now be deleted.
**Requirements changes (Task 1.4):** pins now match the versions actually installed, not the old `requirements.txt`: `numpy<2` → `numpy==2.0.2`, `opencv-python==4.9.0.80` → `4.13.0.92`. Every other package is pinned to its installed version. Removed `pytesseract` and `requests` (never imported) and `scikit-learn` (never imported; sentence-transformers still installs it as a dependency). Added `pytest==9.2.0`.

## D-17 PDF handling — *Accepted*
**Decision:** Keep **PyPDF2** (through the existing `src.pdf_utils`) for text extraction, as the project's stack requires. Handwritten PDFs go to Gemini natively as files, so no rendering is needed. If page rendering is ever needed (e.g. TrOCR on a PDF), add `pypdfium2` (pip-only, no poppler).
**Note:** PyPDF2 is deprecated upstream in favour of `pypdf`, which has a nearly identical API. Switching is a one-line change if a bug appears. It is not planned.

## D-18 Testing strategy: offline by default, live tests opt-in — *Accepted*
**Decision:** `pytest` with `FakeAgent` fixtures that return recorded schema objects. Live Gemini tests are marked `@pytest.mark.live` and skipped unless `GEMINI_API_KEY` is set. A small gold set provides accuracy metrics.
**Why:** Tests that are deterministic, free and runnable in CI. Live tests catch API drift.

## D-19 Evidence verification with `difflib` fuzzy matching — *Accepted*
**Decision:** A quote is verified if a sliding window over the normalised answer text has `SequenceMatcher.ratio() ≥ 0.85`.
**Why:** It tolerates small whitespace and OCR punctuation differences while rejecting invented quotes. It uses the stdlib only.
**Alternatives:** `rapidfuzz` (faster). Add it only if profiling shows a need.

## D-20 Secrets in `.env` only — *Accepted*
**Decision:** Settings are loaded with `python-dotenv`. `.env.example` is committed. `.env` is already git-ignored. API keys are never logged (redacted in the audit log).
**Note:** Phase 1 removes a credential-like string found in `venv/pyvenv.cfg` (never committed) and rotates it if it is real.

## D-21 New path does not use legacy spell-correction or stopword stripping — *Accepted*
**Decision:** SentinelEdu embeds lightly cleaned natural text (`clean_text_basic`) for SBERT and keeps verbatim text for evidence.
**Why:** `pyspellchecker` changes domain terms (e.g. "mitochondria", "eigenvalue"), and stopword removal hurts sentence-embedding quality. Verbatim text is needed so quotes can be verified.

## D-22 Legacy DistilBERT-MNLI "rescue" is not carried forward — *Accepted*
**Decision:** `src/agent_verifier.py` stays in the legacy path only.
**Why:** It adds a fixed +2.5 marks based on a zero-shot label match against the whole model answer. That is an unsupported mark, so it conflicts with D-06. The Judge agent takes over its "second opinion" role, with evidence.

## D-24 Phase 4 implementation details (verified 2026-11-16) — *Accepted*
**Decision (validator policy, `services/scoring_rules.py`):**
- A question whose answer is blank, unreadable or missing from the extraction is **never sent to the LLM**. It gets the lowest level on every criterion and the flag `NOT_ATTEMPTED`. Any draft the model returns for it is ignored. If every answer is like that, Gemini is not called at all (`model_id="none"`).
- A criterion or question the model leaves out gets the lowest level (`MISSING_CRITERION` / `MISSING_QUESTION`). Ids it invents are dropped (`UNKNOWN_CRITERION`). A duplicated criterion keeps its first entry.
- Citation problems (`UNKNOWN_CITATION`) never change marks; only level validity and quote verification do.
- Quotes the model has silently "corrected" (e.g. `Photosynthesis ... sunlight` for the student's `Photosynthesys ... sunlite`) still pass the 0.85 `difflib` threshold, so a mark is not lost to a spelling fix. Fabricated sentences fail it. Quotes under 3 characters never count.
- Unverified quotes are kept on the result (with `evidence_verified=False` and the flag) so the teacher can see what the model claimed.
- `check_marks_are_rubric_derived()` is the audit used by the exit gate: every mark equals a rubric level, every awarded criterion has verified evidence, totals add up.

**Decision (no offline grading):** with `OFFLINE_MODE=true`, `evaluate()` raises `EvaluationError` instead of producing a result. Unlike OCR, there is no offline substitute that can support marks with evidence, and a silent all-zero result would look like a grade.

**Decision (one call per submission):** all attempted questions go in one structured Gemini call (D-07). Per-question calls can be added later for very long answers; the prompt builder already works per question.

**Decision (chunk ids):** `d<doc>.p<page>.c<n>` (e.g. `d1.p2.c3`), short enough for the model to copy exactly. `chunks.json` records the embedding model; a mismatch forces a rebuild.

**Decision (placeholder marks):** the synthetic manifest carries `teacher_marks` assigned by the developer from the sample rubric, labelled as such in `marks_by`. They exercise `scripts/eval_grading.py`; they are not a measurement of agreement with a teacher. That needs the gold set (task 1.6).

**Decision (no scikit-learn):** Cohen's κ is computed in plain Python in the script, so no new dependency was added.

## D-25 Phase 5 implementation details (verified 2026-11-16) — *Accepted*
**Decision (judge validator, `scoring_rules.validate_judge`):**
- "±1 level" means adjacent in the criterion's levels ordered by marks (`level_distance`). A bigger jump is kept as `JUDGE_PROPOSAL_TOO_LARGE` for the teacher and not applied.
- An **upward** proposal needs a quote that verifies against the answer (same 0.85 `difflib` rule as Agent 2). A **downward** proposal needs either a verifying quote (of the gap or wrong claim) or a justification that cites the rubric: the target level's id/label, ≥ 0.6 fuzzy overlap with its descriptor, or ≥ 4 shared words with the criterion description (`cites_descriptor`). Otherwise `JUDGE_UNSUPPORTED_PROPOSAL`.
- Accepted proposals are also recorded as `JUDGE_ADJUSTMENT` flags, so the teacher's review always shows where the two agents disagreed. `accepted` is True only when a proposal was applied; `agree` and `flag` verdicts keep Agent 2's level.
- Proposals on blank/unreadable questions are rejected (`JUDGE_PROPOSAL_NOT_ATTEMPTED`). Missing verdicts default to `agree` (`JUDGE_VERDICT_MISSING`, info). Unknown ids are dropped (`JUDGE_UNKNOWN_ID`).
- **Feedback is guaranteed for every question.** If the model gives none for an attempted question, it is generated from the evaluation (concepts present/missing, criteria met/not met) and flagged `JUDGE_FEEDBACK_MISSING`. Unattempted questions get a fixed "not graded because …" note. This is what makes the Phase 5 exit gate structural rather than hoped for.
- Judge-written flag codes are normalised to `UPPER_SNAKE_CASE` (anything else becomes `JUDGE_NOTE`); unknown question/criterion ids on them are cleared.

**Decision (deterministic checks, `scoring_rules.run_checks`):** `SBERT_LEVEL_DISAGREEMENT` fires at similarity ≥ 0.80 with the lowest level, or < 0.30 with the top level, on attempted questions only. `VALIDATOR_DOWNGRADE` surfaces Agent 2's validator corrections to the Judge as a flag. Thresholds are constants, not settings, until the gold set shows they need tuning.

**Decision (no offline judge, but a checks-only report):** `judge()` raises `JudgeError` in OFFLINE_MODE, like Agent 2. `checks_only_report()` gives the orchestration (Phase 6) a rule-based fallback: deterministic flags, Agent 2's levels kept, generated feedback, plus a `JUDGE_NOT_RUN` flag so the teacher knows no second opinion was taken.

**Decision (recommendations come from levels, added 2026-11-21):** `validate_judge` starts every recommendation from the rubric level Agent 2 selected (an unknown level becomes the lowest level), never from Agent 2's `marks` number. Found by the seeded-error benchmark, where an edited mark leaked into the recommendation.

**Decision (seeded-error benchmark):** `scripts/eval_judge.py` injects five error kinds (level too high, level too low, present concept listed as missing, marks edited past the rubric, evidence stripped) into copies of correct evaluations. "Detected" means a deterministic check flagged the exact criterion, or the Judge did not agree with it (proposal, flag or overlooked concept). The unmodified evaluations are judged too, to count false alarms.

## D-26 Phase 6 orchestration details (verified 2026-11-21) — *Accepted*
**Agno 3.2.0 workflow behaviour (probed, not assumed):**
- A step whose executor **raises does not stop the workflow**. Later steps still run and the run reports `completed`. Every executor therefore catches its own exceptions, writes an `error` audit row and returns `StepOutput(stop=True, success=False)`.
- Steps are **retried 3 extra times by default** (`max_retries=3`). A retried Gemini stage would multiply calls on top of the runner's own retries, so every step uses `max_retries=0`.
- `StepOutput(stop=True)` ends the run early. The extraction gate uses it.

**Decision (shape):** the seven stages are methods on `workflow.steps.AssessmentSteps`, each taking a `submission_id` and reading and writing storage. `workflow.assessment_workflow` wraps them as Agno steps. The workflow object is built per run, which keeps the run id and the per-run RAG/SBERT context (shared by evaluation and judge) out of global state.

**Decision (status machine):** one transition was added to ARCHITECTURE §3.1: `EVALUATED -> EXTRACTED`, so a run that stopped after evaluation (e.g. the Judge failed with fallback disabled) can be re-run. `set_status` refuses every transition not in `ALLOWED_TRANSITIONS` and never sets FINALIZED. A status change uses compare-and-set on the previous status.

**Decision (where re-runs start):** `integrity_precheck` puts a re-run on the right status (`FAILED -> UPLOADED`; `EVALUATED`/`PENDING_TEACHER_REVIEW -> EXTRACTED`) and stops on `FINALIZED`. `document_understanding` skips Gemini whenever any extraction exists. So "resume after correction" and "re-evaluate" are the same command, and history is kept.

**Decision (failures):**
- Agent 1 + TrOCR both fail: `FAILED`.
- Agent 2 fails: the run stops and the status stays `EXTRACTED`.
- Agent 3 fails: by default the checks-only report is stored (status `fallback` in the audit log, `JUDGE_NOT_RUN` flag), and the submission still reaches teacher review. The teacher just has no second opinion. `WorkflowDeps(judge_fallback=False)` stops instead.

**Decision (finalisation rules, `Storage.finalize`):** besides the reviewer and override reasons that `TeacherDecision` already requires, finalisation is refused unless:
- the submission is `PENDING_TEACHER_REVIEW`;
- final marks cover exactly the rubric's criteria, each within its maximum;
- every mark that differs from the recommendation (the Judge's, else Agent 2's) has an override whose `from`/`to` match the change.

The decision row and the status change are written in one transaction. `workflow.actions.approve()` builds the decision from the recommendation plus the teacher's `(marks, reason)` changes.

**Decision (Phase 7 hooks):** `integrity_precheck` and `integrity_postcheck` exist now. The duplicate checks themselves are Phase 7 (tasks 7.1 and 7.4), so the post-check is logged as `skipped`.

**Decision (audit log):** table `audit_log` holds one row per step per run: `run_id`, `step`, `status` (ok, skipped, stopped, fallback or error), `model_id`, `prompt_version`, `duration_ms`, `detail` and `error`.

## D-27 Phase 7 wiring — *Accepted*
**Decision (step order):** `integrity_postcheck` now runs **after evaluation and before the Judge**. The order is: precheck, document, gate, evaluation, integrity_postcheck, judge, mark_pending_review.

Task 7.4 requires the Judge to see integrity flags as context. Running the scan after Agent 2 also guarantees Agent 2 never sees the flags.

**Decision (flags never change marks):**
- Integrity flags reach the Judge prompt in their own section. That section says the flags are for the teacher only and must not change any verdict, level or feedback.
- The flags are copied into `JudgeReport.flags`, so the teacher sees one list.
- The Judge validator still applies only evidenced ±1 proposals, so a flag cannot move marks by itself.

Two checks confirm this: an integration test where the same submission gets identical marks with and without a paraphrased copy in the class, and a live re-upload that scored the same 7/7.

**Decision (storage):**
- `integrity_flags` stores one row per side of a pair, so both students' review screens show the flag.
- `replace_integrity_flags(submission_id, codes, flags)` deletes earlier rows with those codes on either side before inserting. Re-running a scan never duplicates rows, and a pair that is no longer similar after a teacher correction loses its flag on both sides.

**Decision (exact duplicates):** `integrity_precheck` flags byte-identical uploads in the same assignment. If the new submission has no extraction, it copies the oldest duplicate's latest extraction, with an issue note saying so, so Gemini is not called again.

## D-28 Dashboard implementation details — *Accepted*
**Shape:**
- `streamlit_app.py` registers six pages from `sentineledu/ui/pages/` plus `app.py`.
- `ui/state.py` provides the shared dependency access and selection helpers.
- `ui/components.py` provides badges, quote highlighting and a live progress runner.
- Pages call only services and workflow actions. No grading logic lives in the UI.

**Refresh-safe selection (8.8):**
- The selected assignment and submission are mirrored into the URL query string. A refresh or a shared link reopens the same item.
- `st.session_state` carries the selection across `st.switch_page`, which cannot pass query parameters in 1.50.
- All data is re-read from SQLite on every run.
- Dropdown options are ids (strings) with a label function, not model objects. That is robust across reruns and works with AppTest.

**Progress:** `WorkflowDeps.progress` is an optional callback, called after every step. The Submit, Extraction Review and Teacher Review pages show it in `st.status`. An exception in the callback is logged and never breaks the workflow.

**Rubric quick form:** each question gets one full/partial/none criterion through `from_simple_model_answer`. Rubrics with several criteria per question are imported as JSON, and validation errors are shown verbatim.

**Teacher review:**
- The final-marks form starts from the Judge's recommendation.
- A reason box unlocks when a mark changes. **Approve & Finalise** stays disabled until the reviewer name is entered and every changed mark has a reason.
- `approve()` then builds the overrides, and `Storage.finalize()` re-checks every rule server-side.
- Evidence quotes are highlighted in the HTML-escaped answer text.

**Integrity heatmap:** drawn with Altair, which ships with Streamlit, so no new dependency. Flagged pairs get a red outline.

**Reference uploads** are kept under `storage/references/<assignment_id>/`, which is git-ignored with the rest of `storage/`.

**Testing:** `streamlit.testing.v1.AppTest` with `state.DEPS_FACTORY` pointing at fake agents and a temporary database. AppTest cannot drive `st.file_uploader`, so uploads are created through storage, and the Submit page is tested through its "Run again" path, which runs the same workflow code.

## D-23 Explicitly rejected for this project
| Option | Reason |
|---|---|
| Kubernetes, Docker Compose stacks, microservices | Prohibited, and unnecessary for a single-user demo |
| Redis, Celery, task queues | Prohibited. Synchronous runs with progress UI are enough for single submissions |
| Next.js / React frontend | Prohibited. Streamlit is kept |
| A 4th+ agent (e.g. "Integrity Agent", "Rubric Agent") | Prohibited. These are services |
| Retraining or fine-tuning TrOCR | Prohibited |
| Hosted vector DBs (Pinecone, Qdrant, Chroma server) | FAISS is required and enough |
| Authentication and multi-tenant roles | Out of scope for the demo |
