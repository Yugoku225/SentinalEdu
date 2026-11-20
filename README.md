# SentinelEdu

AI-assisted marking of handwritten assignments, with evidence for every mark and a teacher who approves
every result. SentinelEdu grows out of an existing TrOCR + Sentence-BERT evaluator; that original app
(`app.py`, `main.py`, `src/`) is kept unchanged and still runs.

**What it does**

1. **Reads** the student's pages with Gemini. Each answer is transcribed verbatim, mistakes included, and filed under its question. Unreadable spans are marked, and a page Gemini can't read goes to the teacher to check.
2. **Grades** each answer against the teacher's rubric. The model picks a rubric *level* and must quote the student's own words. Python turns levels into marks and rejects any mark whose quote isn't in the answer.
3. **Reviews** the grading with a second agent that looks for mistakes in both directions. It may only recommend a change of one level, with evidence, and it writes feedback for the student.
4. **Checks integrity** across the class: identical uploads, identical text, and answers that are much closer to each other than to the reference notes. These flags never change marks.
5. **Hands over to the teacher.** Nothing is final until a named teacher approves, and every changed mark needs a reason.

## Setup

Requires macOS or Linux with Python 3.11. The project was developed with a conda-created environment on Apple silicon.

```bash
git clone https://github.com/Yugoku225/SentinalEdu.git
cd SentinalEdu
conda create -p ./.venv python=3.11 -y      # or: python3.11 -m venv .venv
conda activate ./.venv                        # or: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                          # then put your key in .env
```

`.env` needs only `GEMINI_API_KEY`. You can create a key at Google AI Studio. Every other setting has a working default, and `.env.example` lists them all. `.env` is git-ignored: never commit it.

The first run downloads the Sentence-BERT model `all-MiniLM-L6-v2`, about 90 MB.

**The TrOCR model is not in the repository.** It is a 1.2 GB fine-tuned model. If you have it, copy the `my_trocr_model/` folder into the project root. SentinelEdu works without it, but two things need it:
- the legacy evaluator's **Evaluate** button;
- the offline fallback reader, used when Gemini is unavailable or `OFFLINE_MODE=true`.

## Run

| What | Command |
|---|---|
| Dashboard (SentinelEdu + legacy page) | `streamlit run streamlit_app.py` |
| Original app on its own | `streamlit run app.py` |
| Load the demo class | `python scripts/seed_demo.py --reset` (uses Gemini; see `docs/demo_script.md`) |
| Assess from the command line | `python -m sentineledu.cli assess --rubric data/rubrics/sample_rubric.json --file page1.jpg --student S01` |
| Resume after a correction | `python -m sentineledu.cli resume <submission_id>` |
| Status and audit trail | `python -m sentineledu.cli status <submission_id>` |

The teacher's guide is [`docs/user_guide.md`](docs/user_guide.md).

## Tests

| Suite | Command | Needs |
|---|---|---|
| Default: unit, integration and dashboard tests, all offline | `pytest` | nothing (network can be off) |
| With coverage | `pytest --cov=sentineledu` | — |
| Live Gemini tests and the CLI end-to-end test | `pytest -m live` | `GEMINI_API_KEY`; about 30 requests |
| Legacy regression and the real TrOCR fallback | `pytest -m legacy` | `my_trocr_model/` |

At the last full run, the default suite had 326 tests passing with 92% line coverage.

## How it fits together

```
upload ─► duplicate pre-check ─► Agent 1: Document Understanding ─► extraction check ─┬─► teacher fixes transcription ─┐
                                   (Gemini; TrOCR fallback)                            │                                │
                                                                                       └─► Agent 2: Answer Evaluation ◄──┘
                                                                                             (rubric levels + quotes; RAG and SBERT context)
                                                                                                    │  Python computes marks
                                                                         similarity scan ◄──────────┘
                                                                                │
                                                     Agent 3: Judge & Feedback ◄┘  (±1 level, with evidence; feedback)
                                                                │
                                                     PENDING_TEACHER_REVIEW ──► teacher approves (reasons for changes) ──► FINALIZED
```

- **Exactly three agents**, built with Agno and Gemini. Everything else is plain Python: storage (SQLite), RAG (FAISS), embeddings (Sentence-BERT), rubric handling, duplicate detection, the scoring rules and the workflow.
- **LLMs choose, Python computes.** No model ever outputs a number. Marks come from rubric levels, and awarded marks need a quote that matches the student's text.
- **Teacher approval is mandatory.** `FINALIZED` can only be reached through `Storage.finalize()` with a named reviewer, and every changed mark needs a reason.

Design documents:
- [`SENTINELEDU_MASTER_PLAN.md`](SENTINELEDU_MASTER_PLAN.md): goals and roadmap.
- [`SENTINELEDU_ARCHITECTURE.md`](SENTINELEDU_ARCHITECTURE.md): components, data contracts and workflow.
- [`SENTINELEDU_DECISIONS.md`](SENTINELEDU_DECISIONS.md): every design decision and the reason for it.
- [`SENTINELEDU_TASKS.md`](SENTINELEDU_TASKS.md): tasks, with status.
- [`docs/results.md`](docs/results.md): measured results.

```
app.py, main.py, src/           original evaluator (unchanged)
streamlit_app.py                dashboard entry point
sentineledu/agents/             the three agents, their prompts, and the Gemini runner (retries, cache, fallback)
sentineledu/services/           storage, documents, RAG, embeddings, rubric, scoring rules, duplicates, export
sentineledu/workflow/           Agno workflow, steps, teacher actions
sentineledu/ui/                 dashboard pages
sentineledu/cli.py              command line
scripts/                        evaluation, calibration and demo scripts
tests/                          unit, integration, e2e; fixtures (synthetic pages, reference notes)
```

## Limitations

- **Measured on synthetic pages only, so far.** The test pages are rendered from handwriting fonts. Real handwriting accuracy, marking agreement with a teacher, and the duplicate thresholds still need a real gold set, which is pending. `docs/results.md` says which numbers are synthetic.
- **Gemini free tier:** 20 requests per day per model. A submission takes about 3 requests, so a class needs a paid key or several days. Repeated runs are cheaper because responses are cached in `storage/cache/`.
- **Model availability changes.** Gemini model names are set in `.env`. Run `python scripts/check_gemini_models.py` to see what your key can use.
- **What is graded:** diagrams are detected but not graded, and maths is transcribed as plain text. Only English has been tested.
- **The TrOCR fallback is weak.** In testing its character error rate was 0.56, and it can't tell which answer belongs to which question. That is why its output always goes to the teacher.
- **Single user.** It uses a local SQLite database with no logins. It is meant as a demo for one teacher, not a multi-user service.
- **Similarity is not proof of copying.** Integrity flags are prompts for the teacher to look, and they never affect marks.

## Privacy

- **Student pages are sent to Google's Gemini API.** The legacy TrOCR path and `OFFLINE_MODE=true` keep everything local.
- **The free tier is not suitable for real student work.** At the time of writing, Google's Gemini API terms allow content sent through the **free tier** to be used to improve Google's products, and it may be read by human reviewers. Use a paid tier, which has different terms, check the current terms, and get any consent your school requires.
- **Anonymise students.** Use anonymised references such as S01, and keep names and roll numbers off uploaded pages.
- **Everything else stays on your machine.** The uploads, database, indexes, reference documents and the Gemini response cache are all under `storage/`. It is git-ignored; delete it to remove all data.
- **No telemetry.** Agno telemetry is disabled.
- **The API key** lives only in `.env` and is never logged.
