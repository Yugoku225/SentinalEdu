import pytest

from sentineledu.schemas import ExtractedAnswer, ExtractedSubmission, SubmissionStatus
from sentineledu.services.storage import NotFoundError, Storage, sha256_files


@pytest.fixture
def store(tmp_path):
    return Storage(storage_dir=tmp_path / "storage")


@pytest.fixture
def pages(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"page-one")
    b.write_bytes(b"page-two")
    return [a, b]


def extraction(sid, text="Plants use light."):
    return ExtractedSubmission(submission_id=sid, engine="gemini", answers=[
        ExtractedAnswer(question_id="Q1", answer_text=text, legibility="clear", confidence=0.9)])


def test_create_submission_copies_files_and_hashes(store, pages):
    sub = store.create_submission("A1", pages, student_ref="S01")
    stored = store.submission_paths(sub)
    assert [p.read_bytes() for p in stored] == [b"page-one", b"page-two"]
    assert all(str(p).startswith(str(store.root / "uploads" / sub.submission_id)) for p in stored)
    assert sub.file_sha256 == sha256_files(pages)
    assert sub.status == SubmissionStatus.UPLOADED
    assert store.get_submission(sub.submission_id) == sub


def test_hash_depends_on_page_order(pages):
    assert sha256_files(pages) != sha256_files(pages[::-1])


def test_original_upload_untouched_by_later_changes(store, pages):
    sub = store.create_submission("A1", pages)
    pages[0].write_bytes(b"changed")
    assert store.submission_paths(sub)[0].read_bytes() == b"page-one"


def test_create_submission_rejects_missing_or_empty(store, tmp_path):
    with pytest.raises(ValueError):
        store.create_submission("A1", [])
    with pytest.raises(FileNotFoundError):
        store.create_submission("A1", [tmp_path / "nope.jpg"])


def test_extraction_versions_increment_and_all_kept(store, pages):
    sid = store.create_submission("A1", pages).submission_id
    v1 = store.save_extraction(extraction(sid, "first"))
    v2 = store.save_extraction(extraction(sid, "second"))
    assert (v1.version, v2.version) == (1, 2)
    assert store.list_extraction_versions(sid) == [1, 2]
    assert store.get_extraction(sid).answers[0].answer_text == "second"
    assert store.get_extraction(sid, 1).answers[0].answer_text == "first"


def test_extraction_for_unknown_submission_rejected(store):
    with pytest.raises(NotFoundError):
        store.save_extraction(extraction("missing"))
    assert store.latest_extraction("missing") is None


def test_set_status(store, pages):
    sid = store.create_submission("A1", pages).submission_id
    store.set_status(sid, SubmissionStatus.EXTRACTED)
    assert store.get_submission(sid).status == SubmissionStatus.EXTRACTED
    with pytest.raises(NotFoundError):
        store.set_status("missing", SubmissionStatus.EXTRACTED)


def test_reopening_storage_keeps_data(tmp_path, pages):
    s1 = Storage(storage_dir=tmp_path / "st")
    sid = s1.create_submission("A1", pages).submission_id
    s1.save_extraction(extraction(sid))
    s2 = Storage(storage_dir=tmp_path / "st")
    assert s2.get_extraction(sid).answers[0].question_id == "Q1"
