import time
from importlib import import_module
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mouthkeys.ai_providers import PROVIDER_ENDPOINTS
from mouthkeys.config import Config, config_path
from mouthkeys.profiles import load_profiles
from mouthkeys.routing import load_routes, save_routes

widgets = pytest.importorskip("PySide6.QtWidgets")
profiles_ui = import_module("mouthkeys.prompt_profiles_ui")
gui = import_module("mouthkeys.gui")


@pytest.fixture
def editor():
    send = Mock()
    tasks = Mock()
    status = Mock()
    changed = Mock()
    keyring = Mock()
    page = profiles_ui.PromptProfilesPage(send, tasks, status, changed, keyring)
    return SimpleNamespace(
        page=page, send=send, tasks=tasks, status=status, changed=changed, keyring=keyring
    )


def click(page, label):
    next(
        button for button in page.findChildren(widgets.QPushButton) if button.text() == label
    ).click()


def select(picker, value):
    index = picker.findData(value)
    assert index >= 0
    picker.setCurrentIndex(index)


def named_draft(page, name="Cleanup"):
    page.profile_name.setText(name)
    page.fields["ai_model"].setText("example-model")
    page.fields["ai_timeout"].setValue(40)
    page.fields["ai_prompt"].setPlainText("Fix grammar\nKeep meaning")
    page.fields["ai_key_env"].setText("EXAMPLE_KEY")
    page.shortcut.setText("super+alt+1")


def confirm_delete(monkeypatch, answer):
    question = Mock(return_value=answer)
    monkeypatch.setattr(
        profiles_ui,
        "QMessageBox",
        SimpleNamespace(question=question, StandardButton=widgets.QMessageBox.StandardButton),
    )
    return question


def test_profile_editors_refresh_all_settings_choices(monkeypatch):
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    window = gui.Window()
    window.timer.stop()
    page = window.prompt_profiles
    click(page, "Save profile")
    assert "name must contain" in window.status.text()
    select(page.fields["ai_provider"], "anthropic")
    click(page, "Use provider endpoint")
    assert page.fields["ai_endpoint"].text() == PROVIDER_ENDPOINTS["anthropic"]
    named_draft(page)
    click(page, "Save profile")
    (profile,) = load_profiles()
    identity = profile["id"]
    assert page.selection == identity
    assert profile["ai_provider"] == "anthropic"
    assert profile["ai_prompt"] == "Fix grammar\nKeep meaning"
    assert profile["ai_timeout"] == 40
    assert profile["shortcut"] == "super+alt+1"
    assert Config.load().ai_profile == ""
    for name in ("ai_profile", "edit_profile", "command_profile"):
        select(window.fields[name], identity)

    page.profile_name.setText("Renamed")
    click(page, "Save profile")
    assert [entry["id"] for entry in load_profiles()] == [identity]
    for name in ("ai_profile", "edit_profile", "command_profile"):
        assert window.fields[name].currentData() == identity
        assert window.fields[name].currentText() == "Renamed"

    select(page.picker, "")
    assert page.profile_name.text() == ""
    assert page.fields["ai_timeout"].value() == Config.load().ai_timeout
    select(page.picker, identity)
    button = widgets.QMessageBox.StandardButton
    question = confirm_delete(monkeypatch, button.Yes)
    click(page, "Delete profile")
    question.assert_called_once()
    assert load_profiles() == []
    assert window.fields["ai_profile"].currentData() == ""
    assert window.fields["edit_profile"].currentData() == "default"
    assert window.fields["command_profile"].currentData() == "default"
    assert not window.isVisible()


@pytest.mark.parametrize("setting", ["ai_profile", "edit_profile", "command_profile"])
def test_delete_refuses_saved_routes_and_active_profiles(editor, monkeypatch, setting):
    page = editor.page
    button = widgets.QMessageBox.StandardButton
    question = confirm_delete(monkeypatch, button.No)
    click(page, "Delete profile")
    question.assert_not_called()
    named_draft(page)
    click(page, "Save profile")
    identity = page.selection
    save_routes([{"app": "Editor", "profile": identity}])
    click(page, "Delete profile")
    editor.status.assert_called_with(
        "Remove application overrides using this profile before deleting it"
    )
    save_routes([])
    Config(**{setting: identity}).save()
    click(page, "Delete profile")
    editor.status.assert_called_with(
        "Choose and save another active profile before deleting this one"
    )
    question.assert_not_called()
    Config().save()
    click(page, "Delete profile")
    question.assert_called_once()
    assert load_profiles()[0]["id"] == identity


def test_route_table_loads_legacy_rows_and_can_use_new_profiles(editor):
    save_routes([{"app": "Old editor", "profile": "off"}])
    page = profiles_ui.PromptProfilesPage(
        editor.send, editor.tasks, editor.status, editor.changed, editor.keyring
    )
    table = page.routes
    assert table.cellWidget(0, 2).currentData() == "dictation"
    click(page, "Remove application")
    assert table.rowCount() == 1
    click(page, "Add application")
    click(page, "Save app overrides")
    editor.status.assert_called_with("Each application needs one unique prompt route")
    assert load_routes() == [{"app": "Old editor", "profile": "off"}]

    named_draft(page)
    click(page, "Save profile")
    assert table.cellWidget(0, 1).currentData() == "off"
    table.item(1, 0).setText("New editor")
    select(table.cellWidget(1, 1), page.selection)
    select(table.cellWidget(1, 2), "edit")
    click(page, "Save app overrides")
    editor.status.assert_called_with("Application prompt overrides saved")
    assert load_routes() == [
        {"app": "Old editor", "profile": "off", "mode": "dictation"},
        {"app": "New editor", "profile": page.selection, "mode": "edit"},
    ]
    table.setCurrentCell(0, 0)
    click(page, "Remove application")
    click(page, "Save app overrides")
    assert [route["app"] for route in load_routes()] == ["New editor"]


def test_prompt_preview_uses_unsaved_values_and_reports_errors(editor, monkeypatch):
    page = editor.page
    click(page, "Test draft prompt")
    assert page.preview_output.toPlainText() == "Provide test text first"
    editor.tasks.assert_not_called()
    named_draft(page)
    page.preview_input.setPlainText("Preview words")
    complete = Mock(return_value="Clean text")
    monkeypatch.setattr(profiles_ui, "process", complete)
    click(page, "Test draft prompt")
    assert page.preview_output.toPlainText() == "Testing draft prompt…"
    function, receive = editor.tasks.call_args.args
    receive({"result": function()})
    assert page.preview_output.toPlainText() == "Clean text"
    text, config = complete.call_args.args
    assert text == "Preview words"
    assert (config.ai_enabled, config.ai_profile, config.ai_model) == ("on", "", "example-model")
    assert config.ai_prompt == "Fix grammar\nKeep meaning"
    receive({"error": "Provider unavailable"})
    assert page.preview_output.toPlainText() == "Provider unavailable"
    assert load_profiles() == []
    assert Config.load().ai_enabled == "off"
    editor.send.assert_not_called()


def test_voice_preview_keeps_config_for_the_successful_recording(editor, monkeypatch):
    page = editor.page
    named_draft(page)
    complete = Mock(return_value="Preview")
    monkeypatch.setattr(profiles_ui, "process", complete)
    click(page, "Record test text")
    command, params, started = editor.send.call_args.args
    assert (command, params) == ("start", {"mode": "training"})
    started({"result": None})
    started({"result": {"id": 9}})
    page.update_status({"id": "test"})
    editor.tasks.assert_not_called()
    started({"result": {"id": "test"}})
    assert page.preview_output.toPlainText() == "Recording test; Stop when finished"
    page.fields["ai_model"].setText("replacement-draft")
    click(page, "Record test text")
    editor.send.call_args.args[2]({"error": "Already recording"})
    click(page, "Stop test recording")
    editor.send.assert_called_with("stop", None, None)

    result = {"mode": "training", "text": "Recorded words"}
    page.update_status({"id": "other", "assistant_proposal": result})
    page.update_status({"id": "test", "assistant_proposal": {"mode": "edit", "text": "wrong"}})
    page.update_status({"id": "test", "assistant_proposal": {"mode": "training", "text": 42}})
    page.update_status({"id": "test", "assistant_proposal": None})
    editor.tasks.assert_not_called()
    for _ in range(2):
        page.update_status({"id": "test", "assistant_proposal": result})
    assert page.preview_input.toPlainText() == "Recorded words"
    editor.tasks.assert_called_once()
    editor.tasks.call_args.args[0]()
    assert complete.call_args.args[1].ai_model == "example-model"

    click(page, "Record test text")
    editor.send.call_args.args[2]({"result": {"id": "cancelled"}})
    page.update_status({"id": "cancelled", "phase": "idle"})
    page.update_status({"id": "cancelled", "assistant_proposal": result})
    editor.tasks.assert_called_once()


def test_invalid_draft_does_not_start_recording_or_task(editor):
    page = editor.page
    page.fields["ai_endpoint"].setText("not-an-endpoint")
    click(page, "Record test text")
    assert "HTTP(S)" in editor.status.call_args.args[0]
    editor.send.assert_not_called()
    click(page, "Test draft prompt")
    assert "HTTP(S)" in page.preview_output.toPlainText()
    editor.tasks.assert_not_called()


def test_profile_credentials_read_multiline_prompt_and_ignore_stale_selection(monkeypatch):
    credentials = import_module("mouthkeys.credentials")
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    monkeypatch.setattr(widgets.QInputDialog, "getText", lambda *_: ("example-test-value", True))
    monkeypatch.setattr(credentials, "store_key", Mock(return_value=None))
    task = Mock()
    monkeypatch.setattr(gui.Window, "run_task", task)
    window = gui.Window()
    window.timer.stop()
    page = window.prompt_profiles
    named_draft(page)
    click(page, "Save profile")
    identity = page.selection
    click(page, "Save keyring API key")
    function, receive = task.call_args.args
    function()
    config, _ = credentials.store_key.call_args.args
    assert config.ai_prompt == "Fix grammar\nKeep meaning"

    select(page.picker, "")
    receive({"result": None})
    assert "earlier endpoint" in window.status.text()
    assert page.fields["ai_credential_source"].currentData() == "environment"
    select(page.picker, identity)
    click(page, "Save keyring API key")
    task.call_args.args[1]({"result": None})
    assert page.fields["ai_credential_source"].currentData() == "secret-service"
    assert load_profiles()[0]["ai_credential_source"] == "environment"

    delete = Mock()
    monkeypatch.setattr(credentials, "delete_key", delete)
    button = widgets.QMessageBox.StandardButton
    question = Mock(return_value=button.No)
    monkeypatch.setattr(widgets.QMessageBox, "question", question)
    task.reset_mock()
    click(page, "Delete keyring API key")
    task.assert_not_called()
    question.return_value = button.Yes
    click(page, "Delete keyring API key")
    function, receive = task.call_args.args
    function()
    assert delete.call_args.args[0].ai_endpoint == config.ai_endpoint
    receive({"error": "Keyring unavailable"})
    assert window.status.text() == "Keyring unavailable"
    receive({"result": None})
    assert window.status.text() == "API key deleted from desktop keyring"


def test_delete_reports_routes_that_cannot_be_read(editor, monkeypatch):
    page = editor.page
    named_draft(page)
    click(page, "Save profile")
    identity = page.selection
    save_routes([])
    config_path().with_name("prompt_routes.json").write_text("{")
    question = confirm_delete(monkeypatch, widgets.QMessageBox.StandardButton.Yes)
    click(page, "Delete profile")
    assert "Expecting property name" in editor.status.call_args.args[0]
    question.assert_not_called()
    assert load_profiles()[0]["id"] == identity


@pytest.mark.parametrize("failure", [False, True])
def test_window_tracks_draft_processing_until_preview_finishes(monkeypatch, failure):
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    gate = Event()

    def complete(text, config):
        assert gate.wait(timeout=2)
        if failure:
            raise RuntimeError("Provider temporarily unavailable")
        return f"{text} ({config.ai_model})"

    monkeypatch.setattr(profiles_ui, "process", complete)
    window = gui.Window()
    window.timer.stop()
    page = window.prompt_profiles
    named_draft(page)
    page.preview_input.setPlainText("Draft")
    try:
        click(page, "Test draft prompt")
        assert window.busy
    finally:
        gate.set()
    app = widgets.QApplication.instance()
    deadline = time.monotonic() + 3
    while window.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert not window.busy
    expected = "Provider temporarily unavailable" if failure else "Draft (example-model)"
    assert page.preview_output.toPlainText() == expected
    assert not window.isVisible()
