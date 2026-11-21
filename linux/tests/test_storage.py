from uuid import uuid4

import pytest

from mouthkeys.storage import History


def test_unknown_metadata_does_not_modify_existing_transcript():
    history = History()
    item = str(uuid4())
    history.begin(item)
    history.update(item, "copied", "original")
    before = history.get(item)
    with pytest.raises(ValueError, match="Unknown history metadata"):
        history.update(item, "failed", "replacement", typo="not a column")
    assert history.get(item) == before


def test_latest_transcript_skips_empty_rows_and_preexisting_audio_can_be_cleared():
    history = History()
    first, second = str(uuid4()), str(uuid4())
    history.begin(first)
    with pytest.raises(ValueError, match="No transcript yet"):
        history.latest_text()

    history.update(first, "copied", "keep this text")
    history.begin(second)
    assert history.latest_text() == "keep this text"

    history.recording(first).write_bytes(b"old")
    history.recording(second).write_bytes(b"legacy")
    assert history.audio_usage() == {"bytes": 9, "recordings": 2}
    history.delete_audio(first)
    assert history.audio_usage() == {"bytes": 6, "recordings": 1}
    assert history.get(first)["text"] == "keep this text"
    history.clear()
    assert history.rows() == []
    assert history.audio_usage() == {"bytes": 0, "recordings": 0}
