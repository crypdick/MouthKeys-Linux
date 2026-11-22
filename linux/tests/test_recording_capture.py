from mouthkeys import recording_destination
from mouthkeys.config import Config

from .pipeline_support import finish, make_controller


def test_failed_destination_lookup_still_releases_capture_and_copies_transcript(monkeypatch):
    app = make_controller()
    Config(delivery="x11").save()
    monkeypatch.setattr(recording_destination, "FOCUS_TIMEOUT", 0.25)

    def unavailable_metadata(_adapter):
        raise RuntimeError("Desktop metadata transport unavailable")

    monkeypatch.setattr(
        type(app.delivery), "target_metadata", property(unavailable_metadata), raising=False
    )
    recording = app.command("start")["id"]
    capture = app.recorder
    audio = app.job.audio_path
    app.command("stop")
    finish(app)

    assert capture.stopped
    assert app.capture.stop_done.is_set()
    assert not audio.exists()
    assert app.history.get(recording)["status"] == "copied"
    assert app.delivery.calls[0][2] is None


def test_disappearing_optional_metadata_after_capture_stop_keeps_transcript(monkeypatch):
    app = make_controller()
    lookups = 0

    def metadata(_adapter):
        nonlocal lookups
        lookups += 1
        if lookups == 1:
            return lambda _: {"app": "stale metadata"}
        raise RuntimeError("Desktop metadata transport disappeared")

    monkeypatch.setattr(type(app.delivery), "target_metadata", property(metadata), raising=False)
    recording = app.command("start")["id"]
    capture = app.recorder
    app.command("stop")
    finish(app)

    assert lookups == 2
    assert capture.stopped
    row = app.history.get(recording)
    assert row["status"] == "copied"
    assert row["app"] == ""
    assert row["text"] == app.transcriber.text
