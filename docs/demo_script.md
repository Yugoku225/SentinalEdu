# SentinelEdu — Demo Script

About 10 minutes. It follows the success criteria in MASTER_PLAN §9.

## Before the audience arrives

```bash
python scripts/seed_demo.py --reset     # 3-6 min live; much faster on re-runs (cached)
streamlit run streamlit_app.py
```

The seed creates the assignment **photosynthesis_basics**. It has 3 questions worth 7 marks in total, and a reference PDF of the class notes. It then assesses five students:

| Student | Pages | Expected state |
|---|---|---|
| S01 | clean answers | ready for review, full marks |
| S02 | smudged, Q2 unreadable | **waiting for a transcription check** |
| S03 | answer runs onto a second page | ready for review |
| S04 | original answers with unusual analogies | ready for review |
| S05 | reworded copy of S04 | ready for review, **near-duplicate flag** |

The pages are synthetic, rendered from handwriting fonts. For a real demo, upload real (anonymised) pages on the Submit page instead.

## Talk track

**1. The problem (30 s).** The original app gave one similarity score per page. It had no idea which question an answer belonged to, no rubric, and no evidence. It also added 2.5 "rescue" marks it couldn't justify. Open **Legacy Evaluator** and show it still works unchanged (success criterion 8).

**2. The rubric (1 min).** Go to **Assignments & Rubrics**.
- Show Q1: two criteria, each with levels and descriptors.
- Point out the reference PDF and its chunk count.
- Key message: *the AI never outputs a number; it picks one of these levels.*

**3. Submitting (1 min).** Go to **Submit** and show the submissions table.
- Optionally upload one more page live to show the step-by-step progress.
- S02 stopped before grading: its transcription needs a human.

**4. Fixing a transcription (1.5 min).** Go to **Extraction Review** and open S02.
- The page image sits beside the text, and Q2 has the red *unreadable* badge.
- Type what it says: "The pigment is chlorophyl and it is present inside the chloroplast." Keep the student's spelling.
- Enter your name and press **Save and resume grading**.
- Grading continues without re-reading the page (success criterion 3).

**5. Reviewing marks (3 min).** Go to **Teacher Review** and open S03 or S04.
- Evidence quotes are highlighted in the answer. The criterion table shows level, marks, ✅ verified evidence, the Judge's verdict and the recommendation (success criterion 4).
- Open a 📄 citation to show the reference passage the decision relied on.
- Show the Judge's feedback, plus any adjustment or flag it raised (success criterion 5). Make the point: *it can only move a mark one level, and only with a quote.*
- Approve S01 as is.
- Approve S03 with one override:
  - change a mark and note that **Approve** greys out;
  - type a reason and approve (success criterion 7).

**6. Integrity (1 min).** Go to **Integrity**.
- S04 and S05 are flagged as a pair on Q1 and Q3 (success criterion 6). Show them side by side.
- On the heatmap, note that correct answers are naturally similar. The flag fires because these two are far closer to *each other* than to the notes.
- The marks are untouched; the decision is the teacher's.

**7. Export (30 s).** Go to **Results & Export**. Only approved submissions appear. Download the CSV, and the JSON with feedback and reasons (success criterion 7).

**8. The numbers (1 min).** Open `docs/results.md`. It shows OCR accuracy against TrOCR, grading agreement against the legacy grader, Judge error detection, and integrity calibration (success criterion 9). Be clear which numbers come from the synthetic set and which come from the real gold set.

## If something goes wrong

- **"Gemini could not…" / 429 / 503 errors.** The free tier allows 20 requests per day per model, and the fallback model is often overloaded. Re-run later from **Submit → Retry a stopped submission**. Finished calls are cached.
- **Want a clean slate?** Run `python scripts/seed_demo.py --reset`. It keeps the response cache, so it costs nothing extra.
