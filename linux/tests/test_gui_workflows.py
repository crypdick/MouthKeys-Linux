"""Controller settings workflows with desktop and service effects replaced."""

import stat
from importlib.metadata import version
from pathlib import Path
from unittest.mock import Mock

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QPushButton

from mouthkeys import desktop_entry, gui, hotkeys, shortcut_capture
from mouthkeys.config import Config, state_dir


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(gui.Window, "send", Mock())
    widget = gui.Window()
    widget.timer.stop()
    _reply(widget, {"error": "Service disconnected"})
    widget.send.reset_mock()
    assert not widget.isVisible()
    return widget


def _reply(window, response):
    callback = window.send.call_args.kwargs["callback"]
    callback(response)


def test_poll_does_not_overlap_and_recovers_after_malformed_status(window):
    window.poll()
    window.poll()
    assert window.send.call_count == 1
    _reply(window, {"result": []})
    assert not window.polling
    assert window.service_version.text() == "Unavailable"
    window.poll()
    _reply(
        window,
        {
            "result": {
                "version": version("mouthkeys-linux"),
                "phase": "recording",
                "message": "Listening",
                "seconds": 1.25,
            }
        },
    )
    assert window.status.text() == "RECORDING  1.2s · Listening"
    assert window.toggle.isEnabled()


def test_status_requires_service_labels_and_tolerates_invalid_elapsed_time(window):
    window.poll()
    _reply(window, {"result": {"version": version("mouthkeys-linux"), "phase": "recording"}})
    assert window.service_version.text() == "Unavailable"
    window.poll()
    resident = {
        "engine": "whisper",
        "model": "running",
        "revision": "main",
        "device": "cpu",
        "compute_type": "int8",
    }
    _reply(
        window,
        {
            "result": {
                "version": version("mouthkeys-linux"),
                "phase": "recording",
                "message": "Starting",
                "seconds": 10**400,
                "loaded_model": resident,
            }
        },
    )
    assert window.status.text() == "RECORDING  0.0s · Starting"
    assert window.loaded_model == resident


def test_hotkey_launch_failure_reports_error_without_showing_window(window, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("Shortcut listener unavailable")

    monkeypatch.setattr(hotkeys.subprocess, "Popen", unavailable)
    window.start_hotkeys()
    assert window.status.text() == "Shortcut listener unavailable"
    assert not window.isVisible()


def test_shortcut_listener_reuses_private_detached_launch(window, monkeypatch):
    def launch(args, **kwargs):
        assert args[-3:] == ["-m", "mouthkeys", "hotkeys"]
        output = kwargs["stdout"]
        assert output is kwargs["stderr"]
        assert Path(output.name) == state_dir() / "hotkeys.log"
        assert stat.S_IMODE(Path(output.name).stat().st_mode) == 0o600
        assert kwargs["start_new_session"] is True

    popen = Mock(side_effect=launch)
    monkeypatch.setattr(hotkeys.subprocess, "Popen", popen)
    window.start_hotkeys()
    popen.assert_called_once()
    assert "Shortcut listener requested" in window.status.text()


def test_sound_previews_save_settings_and_skip_audio_on_save_failure(window, monkeypatch):
    play = Mock()
    launch = Mock()
    monkeypatch.setattr(gui, "play_sound", play)
    monkeypatch.setattr(gui, "spawn_hotkeys", launch)
    buttons = {button.text(): button for button in window.findChildren(QPushButton)}
    buttons["Preview recording sound"].click()
    play.assert_called_once_with(Config.load(), stop=False, preview=True)
    buttons["Preview stop sound"].click()
    play.assert_called_with(Config.load(), stop=True, preview=True)

    def unavailable(config):
        raise OSError("Settings not writable")

    monkeypatch.setattr(Config, "save", unavailable)
    buttons["Preview stop sound"].click()
    window.start_hotkeys()
    assert play.call_count == 2
    launch.assert_not_called()
    assert window.status.text() == "Settings not writable"


def test_saving_startup_and_taskbar_changes_applies_them_without_opening_hidden_window(
    window, monkeypatch
):
    startup = Mock()
    show = Mock()
    monkeypatch.setattr(desktop_entry, "apply_startup", startup)
    monkeypatch.setattr(window, "show", show)
    for key in ("launch_at_startup", "hide_from_taskbar"):
        field = window.fields[key]
        field.setCurrentIndex(field.findData("on"))
    assert window.save()
    startup.assert_called_once_with(Config.load())
    show.assert_not_called()

    # Exercise restoration without asking the compositor to show or focus a window.
    with monkeypatch.context() as visible:
        visible.setattr(window, "isVisible", lambda: True)
        field.setCurrentIndex(field.findData("off"))
        assert window.save()
    show.assert_called_once()
    assert not window.isVisible()


def test_shortcut_callback_only_edits_settings_field(window, monkeypatch):
    shortcut_dialog = Mock()
    capture = Mock(return_value=shortcut_dialog)
    monkeypatch.setattr(shortcut_capture, "ShortcutCapture", capture)
    window.capture_shortcut()
    parent, edit_field = capture.call_args.args
    assert parent is window
    edit_field("toggle_shortcut", "alt+space")
    assert window.fields["toggle_shortcut"].text() == "alt+space"
    shortcut_dialog.exec.assert_called_once()
    assert not window.isVisible()


def test_closing_settings_does_not_abandon_an_open_backup_review(window):
    window.backup_page.open_reviews = 1
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert "current operation" in window.status.text()
    window.backup_page.open_reviews = 0
    window.closeEvent(event)
    assert event.isAccepted()
