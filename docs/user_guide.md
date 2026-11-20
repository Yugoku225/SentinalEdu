# SentinelEdu — Teacher's Guide

SentinelEdu reads your students' handwritten answers, proposes marks against your rubric with evidence for
every mark, gets a second opinion, and then waits for you. **No mark is final until you approve it.**

Start the app with `streamlit run streamlit_app.py`. Then use the pages in the sidebar, roughly in order.

---

## 1. Assignments & Rubrics

**Create the assignment by saving its rubric.**

- **Quick form.** Give an assignment id (for example `bio_unit2`), a title, and for each question its text, marks and a model answer.
  - Each question gets three levels: *full*, *partial* (about half the marks) and *none*.
- **Import JSON.** Use this for questions with several criteria, for example "names the pigment" (1 mark) and "says where it is found" (1 mark).
  - Start from `data/rubrics/sample_rubric.json`, or download an existing rubric from this page and edit it.
  - If something is wrong, for example the criteria don't add up to the question's marks, the page says exactly what and where.

Saving again creates a new rubric version. Earlier gradings keep the version they used.

**Reference documents (optional, recommended).** Upload your notes or the textbook chapter as PDF or text. They help the grader in two ways:
- It can cite the passage that supports each decision.
- The duplicate check can tell students who copied the textbook from students who copied each other.

The page shows how many passages ("chunks") were indexed.

## 2. Submit

Choose the assignment, type an anonymised student reference such as S01, and upload the pages **in order**. Photos, scans and PDFs all work. Then press **Upload and assess**.

The steps run with live progress:

1. **Duplicate pre-check.** If the same file was uploaded before, it is flagged and the earlier transcription is reused.
2. **Reading the pages.** The answers are transcribed word for word, spelling mistakes included, and matched to your questions.
3. **Extraction check.** Grading pauses if anything was unreadable, a question wasn't found, or some writing couldn't be matched to a question.
4. **Grading.** Each criterion gets a rubric level, backed by quotes from the student's answer.
5. **Similarity scan** across the class.
6. **Second review and feedback.**
7. **Ready for you.**

The table at the bottom lists every submission and its status. A submission that stopped, for example because the AI service was busy, can be re-run from **Retry a stopped submission**.

## 3. Extraction Review

You see submissions here whose transcription needs checking (status `NEEDS_EXTRACTION_REVIEW`). The page images are on the left and the transcribed answers on the right. Each answer has a badge for how readable it was.

- **Correct the text** so it says exactly what the student wrote. Keep their spelling mistakes, because grading is about what they wrote.
- Use `[[ILLEGIBLE]]` for words nobody can read.
- If some writing is listed as "not matched to any question", move it into the right answer and clear that box.
- Enter your name and press **Save and resume grading.** Your version is saved as a new version; the original is kept. Grading continues without reading the pages again.

## 4. Teacher Review

This is where marks become final. For each question you see:

- **The student's answer**, with the quoted evidence highlighted.
- **A table of criteria**, showing:
  - the level and marks the grader chose;
  - whether its quotes were found in the answer (✅);
  - what the second reviewer thought;
  - the recommended level.
- **The reasoning**: what was missing or wrong, the reviewer's justification, and 📄 buttons that open the reference passages the grader cited.
- **The feedback** that will go to the student.

At the top:
- **Evaluator** is the first grader's total.
- **Recommended** is the total after the second review. The reviewer may move a criterion by at most one level, and only with evidence.
- **Flags** lists everything worth a look: possible copying, missing evidence, low reading confidence, and disagreements.

**Approving.** The final-marks form starts from the recommendation.

- **To change a mark,** edit it and type a reason. The reason box unlocks when you change the mark.
- **Enter your name.**
- **Approve & Finalise** stays greyed out until your name is entered and every changed mark has a reason. Approved marks can't be changed afterwards.

If the transcription or rubric was wrong, use **Send back for re-evaluation**. The earlier grading is kept for the record.

## 5. Integrity

- **Flagged pairs.** Identical files, identical text, or answers much more alike than either is to your reference material. Pick a pair to read them side by side.
- **Heatmap.** How similar every pair of students' answers to one question is. Flagged pairs are outlined in red.

High similarity alone is not evidence: correct answers naturally look alike. That is why a flag needs the two answers to be clearly closer to each other than to the notes. Flags never change marks. What to do about them is your decision.

## 6. Results & Export

The table shows finalised submissions only, with totals, per-question and per-criterion marks, the reviewer and any overrides. **Download CSV** is for spreadsheets. **Download JSON** also includes the feedback, the override reasons and the integrity flags.

## 7. Legacy Evaluator

This is the original app, unchanged. Use it to compare the two approaches. Its Evaluate button needs the `my_trocr_model/` folder.

---

## Good to know

- **Refreshing is safe.** Everything is saved as you go, and the address bar keeps the selected assignment and submission.
- **What is sent to Google.** The student pages and your rubric are sent to Google's Gemini service. Use anonymised student references; the README explains the privacy terms.
- **Why marks are trustworthy.** The AI never types a number. It picks one of your rubric levels and must quote the student. A quote that isn't really in the answer cancels the marks. You approve every result.
