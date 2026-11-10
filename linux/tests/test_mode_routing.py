import json
import threading
import uuid
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_pipeline import FakeDelivery, FakeRecorder, FakeTranscriber, finish

from mouthkeys import assistance
from mouthkeys.config import Config
from mouthkeys.learning import CorrectionLearner
from mouthkeys.profiles import PROFILE_FIELDS, save_profiles
from mouthkeys.routing import save_routes, select_prompt
from mouthkeys.service import Controller
from mouthkeys.storage import History


def profiles():
    values = asdict(Config(ai_enabled="on", ai_key_env="", ai_model="profile-model"))
    entries = [
        {
            "id": str(uuid.uuid4()),
            "name": str(index),
            **{key: values[key] for key in PROFILE_FIELDS},
            "ai_prompt": f"Custom mode prompt {index}",
            "ai_model": f"model-{index}",
        }
        for index in range(3)
    ]
    save_profiles(entries)
    return [entry["id"] for entry in entries]


def controller(text="Instruction"):
    history = History()
    return Controller(history, FakeTranscriber(text), FakeDelivery(history), FakeRecorder)


def test_mode_profiles_providers_prompts_and_override_precedence():
    mode_profile, app_profile, shortcut_profile = profiles()
    config = Config(ai_enabled="on", ai_model="global", edit_profile=mode_profile)
    save_routes(
        [
            {"app": "Editor", "profile": "off"},
            {"app": "Editor", "profile": app_profile, "mode": "edit"},
        ]
    )
    assert select_prompt(config, mode="edit").ai_model == "model-0"
    app = select_prompt(config, "Editor", mode="edit")
    assert app.ai_model == "model-1" and app.ai_prompt == "Custom mode prompt 1"
    shortcut = select_prompt(config, "Editor", override=shortcut_profile, mode="edit")
    assert shortcut.ai_model == "model-2" and shortcut.ai_prompt == "Custom mode prompt 2"
    assert select_prompt(config, "Editor").ai_enabled == "off"
    assert select_prompt(replace(config, edit_profile="off"), mode="edit").ai_enabled == "off"
    assert select_prompt(Config(edit_profile=mode_profile), mode="edit").ai_enabled == "on"


def test_default_mode_uses_global_provider_without_global_dictation_prompt():
    item_id, _, _ = profiles()
    config = Config(ai_enabled="on", ai_profile=item_id, edit_prompt="Explicit edit behavior")
    edit = select_prompt(config, mode="edit", routes=[])
    assert edit.ai_model == "model-0"
    assert edit.ai_prompt == "Explicit edit behavior"
    command = select_prompt(config, mode="command", routes=[])
    assert command.ai_prompt == assistance.COMMAND_PROMPT
    assert command.ai_model == "model-0"


def test_same_application_routes_can_differ_by_mode_and_legacy_is_dictation():
    save_routes(
        [
            {"app": "Editor", "profile": "off"},
            {"app": "Editor", "profile": "default", "mode": "edit"},
        ]
    )
    config = Config(ai_enabled="on", ai_model="global")
    assert select_prompt(config, "Editor", mode="dictation").ai_enabled == "off"
    assert select_prompt(config, "Editor", mode="edit").ai_enabled == "on"
    with pytest.raises(ValueError):
        save_routes(
            [
                {"app": "Editor", "profile": "off"},
                {"app": "EDITOR", "profile": "default", "mode": "dictation"},
            ]
        )


def test_typed_edit_honors_mode_profile_and_passes_stream_cancellation(monkeypatch):
    item_id, _, _ = profiles()
    config = Config(edit_profile=item_id)
    cancelled = threading.Event()
    calls = []

    def complete(text, resolved, **kwargs):
        calls.append((json.loads(text), resolved, kwargs))
        return "Edited"

    monkeypatch.setattr(assistance, "request_completion", complete)
    result = assistance.propose("Improve", config, "edit", "Original", cancel_event=cancelled)
    assert result["text"] == "Edited"
    assert calls[0][1].ai_model == "model-0"
    assert calls[0][1].ai_prompt == "Custom mode prompt 0"
    assert calls[0][2]["cancel_event"] is cancelled


def test_voice_edit_uses_routed_custom_prompt_and_forwards_job_event(monkeypatch):
    item_id, _, _ = profiles()
    Config(edit_profile=item_id).save()
    app = controller()
    seen = []

    def complete(text, config, **kwargs):
        seen.append((config, kwargs))
        return "Edited"

    monkeypatch.setattr(assistance, "request_completion", complete)
    app.command("start", mode="edit", selected_text="Original")
    app.command("stop")
    finish(app)
    assert app.assistant_proposal["text"] == "Edited"
    assert seen[0][0].ai_prompt == "Custom mode prompt 0"
    assert seen[0][1]["cancel_event"] is app.job.cancel_event
    assert not app.delivery.calls


def test_voice_command_cancel_event_interrupts_provider_without_late_proposal(monkeypatch):
    Config(ai_enabled="on", ai_model="model", ai_streaming="on").save()
    app = controller()
    entered = threading.Event()

    def complete(text, config, **kwargs):
        entered.set()
        assert kwargs["cancel_event"].wait(2)
        raise RuntimeError("Cancelled")

    monkeypatch.setattr(assistance, "request_completion", complete)
    app.command("start", mode="command")
    app.command("stop")
    assert entered.wait(2)
    app.command("cancel")
    finish(app)
    assert app.assistant_proposal is None
    assert app.history.rows()[0]["status"] == "cancelled"
    assert not app.delivery.calls


def test_return_to_start_captures_start_even_with_stop_policy():
    Config(delivery="x11", target_policy="stop", return_to_start="on").save()
    app = controller()
    target = {"window": "original", "field": {"field_id": "actual"}}
    calls = []
    app.delivery.capture_target = lambda config: calls.append(True) or target
    app.command("start")
    app.command("stop")
    finish(app)
    assert calls == [True]
    assert app.delivery.calls[0][2] is target


def test_partial_snapshots_never_seed_learning_or_automatic_edit():
    snapshot = {
        "field_id": "bus|path",
        "text": "Original",
        "caret": 8,
        "selection_start": 8,
        "selection_end": 8,
        "focused": True,
        "editable": True,
        "protected": False,
        "complete": False,
    }
    learner = CorrectionLearner(Config(automatic_learning="on"))
    assert not learner.begin("Original", snapshot)
    field = {**snapshot, "selection_start": 0}
    assert not Controller._edit_target_matches(
        SimpleNamespace(start_target={"field": field}, selected_text="Original")
    )


def test_cancel_after_completed_native_preview_releases_cache_for_file_inference():
    app = controller()
    app.transcriber.stream_key = None

    def preview(path, config):
        app.transcriber.stream_key = (path,)
        return "Preview"

    def reset():
        app.transcriber.stream_key = None

    app.transcriber.preview = preview
    app.transcriber.reset_stream = reset
    app.command("start")
    app._preview(app.item_id, app.config)
    assert app.transcriber.stream_key is not None
    app.command("cancel")
    finish(app)
    assert app.transcriber.stream_key is None

    # File inference now sees no active native session, unlike the broken cancellation path.
    def segments(samples, config):
        assert app.transcriber.stream_key is None
        return []

    app.transcriber._segments = segments
    from pathlib import Path

    from mouthkeys.file_transcription import transcribe_file

    result = transcribe_file(
        Path(__file__).parent / "resources" / "dictation_fixture.wav",
        app.config,
        transcriber=app.transcriber,
    )
    assert result["segments"] == []


def test_voice_mode_app_route_resolves_with_history_and_global_ai_disabled(monkeypatch):
    item_id, _, _ = profiles()
    Config(ai_enabled="off", save_history="off").save()
    save_routes([{"app": "Editor", "profile": item_id, "mode": "command"}])
    app = controller()
    app.delivery.target_metadata = lambda target: {"app": "Editor"}
    calls = []

    def complete(text, config, **kwargs):
        calls.append(config)
        return '{"command":"ls", "explanation":"List files"}'

    monkeypatch.setattr(assistance, "request_completion", complete)
    app.command("start", mode="command")
    app.command("stop")
    finish(app)
    assert calls[0].ai_model == "model-0"
    assert app.assistant_proposal["command"] == "ls"
    assert app.history.rows() == []
