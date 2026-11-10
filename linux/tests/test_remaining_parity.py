import json
import uuid
from dataclasses import asdict

import pytest

from mouthkeys.config import Config
from mouthkeys.file_history import FileHistory
from mouthkeys.profiles import PROFILE_FIELDS, save_profiles
from mouthkeys.routing import save_routes, select_prompt


def result(text="saved result"):
    return {
        "text": text,
        "duration": 5.0,
        "processing_seconds": 1.0,
        "segments": [{"start": 0.0, "end": 5.0, "text": text, "speaker": 0}],
        "speaker_turns": [{"start": 0.0, "end": 5.0, "speaker": 0}],
    }


def profile():
    values = asdict(Config(ai_model="local-model", ai_key_env=""))
    entry = {
        "id": str(uuid.uuid4()),
        "name": "Local cleanup",
        **{key: values[key] for key in PROFILE_FIELDS},
    }
    save_profiles([entry])
    return entry["id"]


def test_file_history_survives_restart_with_speakers_and_bounds():
    store = FileHistory()
    first = store.add("/private/source/first.wav", result("café"), limit=2)
    store.add("second.wav", result("second"), limit=2)
    store.add("third.wav", result("third"), limit=2)
    rows = FileHistory().rows()
    assert len(rows) == 2
    assert all(row["id"] != first for row in rows)
    assert rows[0]["result"]["segments"][0]["speaker"] == 0
    assert rows[0]["name"] == "third.wav"
    assert "private/source" not in json.dumps(rows)
    assert store.path.stat().st_mode & 0o777 == 0o600
    store.delete(rows[0]["id"])
    assert len(FileHistory().rows()) == 1
    store.clear()
    assert FileHistory().rows() == []


def test_file_history_restore_validates_before_mutation():
    store = FileHistory()
    store.add("original.wav", result())
    rows = store.rows()
    rows[0]["created"] = float("nan")
    with pytest.raises(ValueError):
        store.restore(rows)
    assert len(store.rows()) == 1
    rows = store.rows()
    rows[0]["name"] = "../escape.wav"
    with pytest.raises(ValueError):
        store.restore(rows)
    assert len(store.rows()) == 1


def test_app_routing_scope_and_explicit_shortcut_precedence():
    item_id = profile()
    save_routes([{"app": "Editor", "profile": item_id}, {"app": "Terminal", "profile": "off"}])
    base = Config(ai_enabled="on", ai_model="default-model", ai_scope="selected")
    selected = select_prompt(base, "EDITOR")
    assert selected.ai_model == "local-model"
    assert selected.ai_enabled == "on"
    assert selected.ai_profile == ""  # Provider settings now frozen for this job.
    assert select_prompt(base, "unknown").ai_enabled == "off"
    assert select_prompt(base, "Terminal").ai_enabled == "off"
    assert select_prompt(base, "Terminal", override=item_id).ai_enabled == "on"
    assert select_prompt(base, "Editor", override="off").ai_enabled == "off"
    assert select_prompt(Config(), override=item_id).ai_enabled == "on"


def test_routing_rejects_missing_profile_without_erasing_saved_routes():
    save_routes([{"app": "Editor", "profile": "off"}])
    with pytest.raises(ValueError):
        save_routes([{"app": "Editor", "profile": str(uuid.uuid4())}])
    assert select_prompt(Config(ai_enabled="on"), "Editor").ai_enabled == "off"


def test_turn_first_file_transcription_keeps_speaker_change_inside_asr_segment(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import numpy as np
    from faster_whisper import audio

    from mouthkeys import file_transcription

    samples = np.concatenate([np.ones(32000), np.ones(32000) * 2])
    monkeypatch.setattr(audio, "decode_audio", lambda *args, **kwargs: samples)
    monkeypatch.setattr(
        file_transcription,
        "diarize",
        lambda *args: [{"start": 0, "end": 2, "speaker": 0}, {"start": 2, "end": 4, "speaker": 1}],
    )
    calls = []

    def segments(part, config):
        calls.append(len(part))
        return [
            SimpleNamespace(
                start=0, end=len(part) / 16000, text="First." if part[0] == 1 else "Second."
            )
        ]

    result = file_transcription.transcribe_file(
        "speakers.wav",
        replace(Config(), file_speaker_labels="on"),
        transcriber=SimpleNamespace(_segments=segments),
    )
    assert calls == [32000, 32000]
    assert [(s["text"], s["speaker"], s["start"], s["end"]) for s in result["segments"]] == [
        ("First.", 0, 0, 2),
        ("Second.", 1, 2, 4),
    ]
    assert result["text"] == "First. Second."
    assert result["confidence"] is None


def test_recording_freezes_prompt_profile_before_later_edits(monkeypatch):

    from test_pipeline import finish
    from test_pipeline_options import controller

    from mouthkeys.profiles import load_profiles

    item_id = profile()
    Config(ai_enabled="on", ai_profile=item_id).save()
    app = controller("Original words")
    observed = []
    monkeypatch.setattr(
        "mouthkeys.service.process", lambda text, config: observed.append(config.ai_model) or text
    )
    app.command("start")
    edited = load_profiles()
    edited[0]["ai_model"] = "different-model-after-start"
    save_profiles(edited)
    app.command("stop")
    finish(app)
    assert observed == ["local-model"]
    app.command("start")
    app.command("stop")
    finish(app)
    assert observed[-1] == "different-model-after-start"


def test_legacy_backup_preserves_newer_stores_and_referenced_profiles(tmp_path):
    from mouthkeys.dictionary import Dictionary
    from mouthkeys.operations import backup_import
    from mouthkeys.profiles import load_profiles
    from mouthkeys.pronunciation import PronunciationStore
    from mouthkeys.routing import load_routes
    from mouthkeys.storage import History
    from mouthkeys.vocabulary import VocabularyStore

    item_id = profile()
    save_routes([{"app": "Editor", "profile": item_id}])
    VocabularyStore.save_entries([{"text": "MouthKeys", "weight": 3, "aliases": ["mouth keys"]}])
    PronunciationStore.save([])
    before = VocabularyStore.path().read_bytes(), PronunciationStore.path().read_bytes()
    Config(vocabulary_boosting="on").save()
    path = tmp_path / "legacy-v1.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "config": {"model": "small.en"},
                "dictionary": Dictionary.defaults(),
                "profiles": [],
                "history": [],
            }
        )
    )
    backup_import(History(), path)
    assert before == (VocabularyStore.path().read_bytes(), PronunciationStore.path().read_bytes())
    assert load_routes()[0]["profile"] == item_id
    assert load_profiles()[0]["id"] == item_id
    assert Config.load().vocabulary_boosting == "on"
    assert Config.load().model == "small.en"


def test_spoken_send_only_never_claims_copy_or_erases_last_transcript():
    from test_pipeline import finish
    from test_pipeline_options import controller

    Config(delivery="x11", spoken_send="on").save()
    app = controller("Previous dictation")
    app.command("start")
    app.command("stop")
    finish(app)
    previous_calls = len(app.delivery.calls)
    app.transcriber.text = "send it"
    sent = []
    app.delivery.send_key = lambda *args: sent.append(True) or True
    app.command("start")
    app.command("stop")
    finish(app)
    assert sent == [True]
    assert len(app.delivery.calls) == previous_calls
    assert app.history.rows()[0]["status"] == "send_requested"
    assert app.status()["message"].startswith("Send requested")
    app.command("copy")
    assert app.delivery.calls[-1] == "Previous dictation"
