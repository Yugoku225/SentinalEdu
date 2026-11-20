"""Duplicate detection rules (SENTINELEDU_TASKS.md 7.1, 7.2)."""
from pathlib import Path

import pytest

from sentineledu.config import Settings
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission
from sentineledu.schemas.integrity import IntegrityFlag
from sentineledu.services.duplicate import (
    EXACT_DUPLICATE,
    NEAR_DUPLICATE,
    SIMILAR,
    TEXT_DUPLICATE,
    classify_pair,
    clean_text,
    exact_file_check,
    integrity_scan,
    text_sha256,
    token_count,
)
from sentineledu.services.rag import RagStore
from sentineledu.services.storage import Storage
from wf_helpers import PAGE, RUBRIC

NOTES = Path(__file__).resolve().parents[1] / "fixtures" / "reference" / "photosynthesis_notes.txt"
PAGE2 = PAGE.with_name("syn_03_blank_out_of_order_p1.jpg")

# Q3 answers (>= 15 words). The reference passage is section 6.4 of the notes.
QUIRKY = ("Plants drink water mostly so their stems do not flop over like a wet noodle, and I think the water also "
          "travels up like a lift carrying food salts to every leaf.")
PARAPHRASE = ("Plants take in water mainly so the stem does not flop over like a soggy noodle, and the water also goes up "
              "like an elevator carrying food salts to all the leaves.")
REFERENCE_COPY = ("Water is one of the raw materials of photosynthesis, so a plant cannot make food without it. Water also "
                  "carries dissolved mineral salts such as nitrates from the soil up to the leaves.")
REFERENCE_COPY_2 = ("Water is one of the raw materials of photosynthesis so a plant cannot make food without it. Water also "
                    "carries dissolved mineral salts such as nitrate from the soil up to the leaves.")
INDEPENDENT_1 = "Water is needed for photosynthesis to make glucose, and it carries minerals up from the roots to the leaves of the plant."
INDEPENDENT_2 = "Plants use water as a raw material in photosynthesis and it also keeps their cells turgid so the plant stands up straight."
SHORT = "For photosynthesis and minerals."


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, offline_mode=True, storage_dir=tmp_path / "storage")


@pytest.fixture
def storage(settings):
    s = Storage(settings)
    s.save_rubric(RUBRIC)
    return s


def add(storage, q3: str, page=PAGE, q1="Plants make food from light.", student=None) -> str:
    sid = storage.create_submission(RUBRIC.assignment_id, [page], student_ref=student).submission_id
    answers = [ExtractedAnswer(question_id="Q1", answer_text=q1, legibility="clear", confidence=0.9),
               ExtractedAnswer(question_id="Q2", answer_text="Chlorophyll.", legibility="clear", confidence=0.9),
               ExtractedAnswer(question_id="Q3", answer_text=q3, legibility="clear" if q3 else "blank", confidence=0.9)]
    storage.save_extraction(ExtractedSubmission(submission_id=sid, engine="gemini", answers=answers))
    return sid


def codes(flags):
    return sorted((f.code, f.question_id) for f in flags)


# --- normalisation ---------------------------------------------------------------------------

def test_clean_text_matches_legacy_rules():
    assert clean_text("Hello,\n\tWORLD!! [[ILLEGIBLE]] (x) — ok") == "hello, world!! x ok"
    assert token_count("one two\nthree [[ILLEGIBLE]]") == 3


def test_text_hash_ignores_case_spacing_and_symbols():
    def ext(t):
        return ExtractedSubmission(submission_id="s", engine="gemini", answers=[
            ExtractedAnswer(question_id="Q1", answer_text=t, legibility="clear", confidence=1)])
    assert text_sha256(ext("Plants  make FOOD."), ["Q1"]) == text_sha256(ext("plants make food."), ["Q1"])
    assert text_sha256(ext("Plants make food."), ["Q1"]) != text_sha256(ext("Plants make sugar."), ["Q1"])


# --- the §5.1 rule ---------------------------------------------------------------------------

def test_classify_pair_rule(settings):
    assert classify_pair(0.95, 0.70, 0.72, 20, 20, settings).code == NEAR_DUPLICATE
    assert classify_pair(0.88, 0.70, 0.72, 20, 20, settings).code == SIMILAR
    assert classify_pair(0.80, 0.50, 0.50, 20, 20, settings).code is None  # below both thresholds
    assert classify_pair(0.97, 0.95, 0.90, 20, 20, settings).code is None  # both close to the reference
    r = classify_pair(0.99, 0.10, 0.10, 20, 8, settings)
    assert r.code is None and "shorter" in r.skipped


# --- 7.2 acceptance: real SBERT on synthetic answers ------------------------------------------

def test_paraphrase_is_flagged_on_both_sides(storage, settings):
    a = add(storage, QUIRKY, student="S1")
    b = add(storage, PARAPHRASE, page=PAGE2, student="S2")
    flags = integrity_scan(storage, b, settings=settings, rag=RagStore(settings))
    assert codes(flags) in ([(NEAR_DUPLICATE, "Q3")], [(SIMILAR, "Q3")])
    f = flags[0]
    assert f.other_submission_id == a and f.similarity >= settings.similar_threshold and f.margin >= settings.integrity_margin
    assert [x.other_submission_id for x in storage.integrity_flags(a)] == [b]  # mirrored onto the other submission


def test_both_copied_from_reference_is_not_flagged(storage, settings):
    store = RagStore(settings)
    store.ingest(RUBRIC.assignment_id, [NOTES])
    add(storage, REFERENCE_COPY)
    b = add(storage, REFERENCE_COPY_2, page=PAGE2)
    assert integrity_scan(storage, b, settings=settings, rag=store) == []


def test_independent_correct_answers_are_not_flagged(storage, settings):
    add(storage, INDEPENDENT_1)
    b = add(storage, INDEPENDENT_2, page=PAGE2)
    assert integrity_scan(storage, b, settings=settings, rag=RagStore(settings)) == []


def test_short_answers_are_skipped(storage, settings):
    add(storage, SHORT)
    b = add(storage, SHORT + " ", page=PAGE2)  # identical but only 4 words
    assert integrity_scan(storage, b, settings=settings, rag=RagStore(settings)) == []


def test_blank_answers_are_not_compared(storage, settings):
    add(storage, "")
    b = add(storage, "", page=PAGE2)
    assert integrity_scan(storage, b, settings=settings) == []


# --- 7.1 exact and text duplicates ------------------------------------------------------------

def test_exact_file_duplicate_flags_and_reuses_extraction(storage):
    a = add(storage, QUIRKY, student="S1")
    b = storage.create_submission(RUBRIC.assignment_id, [PAGE], student_ref="S2")
    flags, reuse = exact_file_check(storage, b)
    assert codes(flags) == [(EXACT_DUPLICATE, None)] * 2
    assert {(f.submission_id, f.other_submission_id) for f in flags} == {(b.submission_id, a), (a, b.submission_id)}
    assert reuse.submission_id == b.submission_id and reuse.answer_for("Q3").answer_text == QUIRKY
    assert "Gemini was not called" in reuse.issues[-1]


def test_exact_check_ignores_other_assignments(storage):
    add(storage, QUIRKY)
    other = storage.create_submission("another_assignment", [PAGE])
    assert exact_file_check(storage, other) == ([], None)


def test_text_duplicate_from_different_files(storage, settings):
    a = add(storage, QUIRKY, q1="Plants make their own food from sunlight, water and carbon dioxide in the leaves.")
    b = add(storage, QUIRKY, page=PAGE2, q1="Plants make their own food from sunlight, water and carbon dioxide in the leaves.")
    flags = integrity_scan(storage, b, settings=settings)
    assert codes(flags) == [(TEXT_DUPLICATE, None)]  # no extra per-question flags for the same pair
    assert flags[0].other_submission_id == a


def test_rescan_replaces_flags_instead_of_duplicating(storage, settings):
    a = add(storage, QUIRKY)
    b = add(storage, PARAPHRASE, page=PAGE2)
    integrity_scan(storage, b, settings=settings)
    integrity_scan(storage, b, settings=settings)
    integrity_scan(storage, a, settings=settings)  # scanning the other side replaces the same pair
    assert len(storage.integrity_flags(a)) == 1 and len(storage.integrity_flags(b)) == 1


def test_replace_refuses_codes_outside_the_set(storage):
    f = IntegrityFlag(assignment_id="x", submission_id="a", other_submission_id="b", code=NEAR_DUPLICATE,
                      severity="high", message="m")
    with pytest.raises(ValueError, match="not among"):
        storage.replace_integrity_flags("a", {EXACT_DUPLICATE}, [f])
