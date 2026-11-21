from uuid import uuid4

import pytest

from mouthkeys import file_history
from mouthkeys.file_history import FileHistory

RESULT = {"text": "private file transcript", "duration": 1.0, "processing_seconds": 0.5}


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (["not a result"], "Invalid file transcription result"),
        (RESULT | {"duration": float("nan")}, "Invalid result duration"),
        (RESULT | {"processing_seconds": -1}, "Invalid result processing_seconds"),
        (RESULT | {"segments": {}}, "Invalid file segments"),
        (RESULT | {"segments": [{"text": 1}]}, "Invalid file segment"),
        (
            RESULT | {"segments": [{"text": "example", "start": 1, "end": 0}]},
            "Invalid segment timestamps",
        ),
    ],
)
def test_invalid_file_result_is_not_inserted(invalid, message):
    store = FileHistory()
    store.add("original.wav", RESULT)
    original = store.rows()
    with pytest.raises(ValueError, match=message):
        store.add("replacement.wav", invalid)
    assert store.rows() == original


def test_file_history_rejects_oversized_result_and_invalid_limit(monkeypatch):
    store = FileHistory()
    with pytest.raises(ValueError, match="File history limit must be 1–10000"):
        store.add("recording.wav", RESULT, limit=0)
    monkeypatch.setattr(file_history, "MAX_RESULT", 256)
    with pytest.raises(ValueError, match="File result exceeds 20 MiB"):
        store.add("recording.wav", RESULT | {"text": "x" * 256})
    assert store.rows() == []


@pytest.mark.parametrize(
    ("make_invalid", "message"),
    [
        (lambda entry: {}, "Invalid file history entries"),
        (lambda entry: [entry | {"extra": 1}], "Invalid file history entry"),
        (lambda entry: [entry | {"created": 1e30}], "Invalid file history date"),
        (lambda entry: [entry | {"name": "source/hidden.wav"}], "Invalid file history metadata"),
    ],
)
def test_file_history_restore_validates_all_entries_before_writing(make_invalid, message):
    store = FileHistory()
    store.add("old.wav", RESULT)
    original = store.rows()
    with pytest.raises(ValueError, match=message):
        store.restore(make_invalid(original[0]))
    assert store.rows() == original


def test_file_history_restore_merges_by_id_and_searches_restored_text():
    store = FileHistory()
    item_id = store.add("old.wav", RESULT)
    original = store.rows()[0]
    extra = original | {"id": str(uuid4()), "name": "new.wav", "created": original["created"] + 1}
    store.restore([original | {"result": RESULT | {"text": "corrected"}}, extra])
    assert store.rows(query="CORRECTED")[0]["id"] == item_id
    assert store.rows(query="NEW.WAV") == [extra]
