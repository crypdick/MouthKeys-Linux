"""Exercise tray state with an offscreen menu and fake control responses."""

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QSystemTrayIcon

from mouthkeys import gui
from mouthkeys import tray as tray_module
from mouthkeys.config import Config
from mouthkeys.tray import Tray


@pytest.fixture
def tray():
    icon = Tray()
    icon.timer.stop()
    yield icon
    icon.deleteLater()


def test_disconnection_does_not_replay_completed_recording_alert(tray, monkeypatch):
    Config(show_ai_alerts="on", show_paste_alerts="on", delivery="x11").save()
    notices = []
    monkeypatch.setattr(tray, "showMessage", lambda *args: notices.append(args))
    done = {
        "phase": "idle",
        "id": "completed",
        "preview": "hello",
        "message": "Insertion requested — transcript saved in history",
        "notice": {"kind": "ai", "serial": "unique", "message": "Cleanup unavailable"},
    }
    outcomes = iter(
        (
            {"result": {"phase": "idle", "message": "Ready"}},
            {"result": done},
            {"error": "Service unavailable"},
            {"result": done},
        )
    )

    def send(action, callback):
        assert action == "status"
        callback(next(outcomes))

    monkeypatch.setattr(tray, "send", send)
    tray.poll()
    tray.poll()
    tray.poll()
    assert tray.status_action.text() == "MouthKeys — Service unavailable"
    assert not tray.toggle_action.isEnabled()
    tray.poll()
    assert [title for title, _ in notices] == ["AI processing failed", "Paste requested"]


def test_private_paste_alert_does_not_claim_transcript_saved(tray, monkeypatch):
    Config(show_paste_alerts="on", save_history="off").save()
    notices = []
    monkeypatch.setattr(tray, "showMessage", lambda *args: notices.append(args))
    tray.notify_status({"result": {"phase": "idle"}})
    tray.notify_status(
        {
            "result": {
                "phase": "idle",
                "id": "private-recording",
                "preview": "hello",
                "message": "Insertion requested — history disabled",
            }
        }
    )
    assert notices == [("Paste requested", "Destination acceptance cannot be verified.")]


def test_invalid_status_response_is_displayed_as_unavailable(tray, monkeypatch):
    monkeypatch.setattr(tray, "send", lambda action, callback: callback({"result": None}))
    tray.poll()
    assert tray.status_action.text() == "MouthKeys — Unavailable"
    assert not tray.toggle_action.isEnabled()
    assert not tray.pending


@pytest.mark.parametrize(
    ("payload", "label", "enabled"),
    [
        ({"phase": "idle"}, "Record", True),
        ({"phase": "loading", "can_record": False}, "Record", False),
        ({"phase": "transcribing", "can_record": True}, "Record", True),
        ({"phase": "starting", "can_record": False}, "Cancel microphone start", True),
        ({"phase": "capture_failed", "can_record": False}, "Retry microphone stop", True),
        ({"phase": "switching"}, "Stop recording", True),
        ({"phase": "recording"}, "Stop recording", True),
        ({"phase": "", "can_record": True}, "Record", False),
    ],
)
def test_status_maps_labels_and_availability(tray, payload, label, enabled):
    tray.notify_status({"result": payload})
    assert tray.toggle_action.text() == label
    assert tray.toggle_action.isEnabled() is enabled


def test_poll_does_not_issue_requests_while_awaiting_reply(tray, monkeypatch):
    callbacks = []
    monkeypatch.setattr(tray, "send", lambda action, callback: callbacks.append((action, callback)))
    tray.poll()
    tray.poll()
    assert [action for action, _ in callbacks] == ["status"]
    callbacks[0][1]({"result": {"phase": "idle", "message": "Ready"}})
    tray.poll()
    assert [action for action, _ in callbacks] == ["status", "status"]


def test_microphone_menu_respects_exclusions_and_can_save_default(tray, monkeypatch):
    Config(audio_source="usb", microphone_never="excluded").save()
    monkeypatch.setattr(tray_module, "source_names", lambda: ["usb", "excluded"])
    tray.refresh_microphones()
    default, usb = tray.microphones.actions()
    assert [(action.text(), action.isChecked()) for action in (default, usb)] == [
        ("default", False),
        ("usb", True),
    ]
    default.trigger()
    assert Config.load().audio_source == "default"

    def unavailable():
        raise OSError("PulseAudio unavailable")

    monkeypatch.setattr(tray_module, "source_names", unavailable)
    tray.refresh_microphones()
    (default,) = tray.microphones.actions()
    assert default.text() == "default"
    assert default.isChecked()


def test_only_primary_activation_opens_window_and_reuses_it_without_real_focus(tray, monkeypatch):
    seen = []
    window = SimpleNamespace(
        show=lambda: seen.append("show"),
        raise_=lambda: seen.append("raise"),
        activateWindow=lambda: seen.append("activate"),
    )

    def build_window():
        seen.append("create")
        return window

    monkeypatch.setattr(gui, "Window", build_window)
    tray.activate(QSystemTrayIcon.ActivationReason.Context)
    assert not seen
    tray.activate(QSystemTrayIcon.ActivationReason.Trigger)
    tray.activate(QSystemTrayIcon.ActivationReason.Trigger)
    assert seen == ["create", "show", "raise", "activate", "show", "raise", "activate"]


def test_quit_waits_for_all_control_requests(tray, monkeypatch):
    class Signal:
        def connect(self, handler):
            self.emit = handler

    started, deleted, quit_calls, responses = [], [], [], []

    class Request:
        def __init__(self, action, *, parent):
            assert parent is tray
            self.action = action
            self.done = Signal()
            self.finished = Signal()
            self.deleteLater = lambda: deleted.append(action)

        def start(self):
            started.append(self)

    app = SimpleNamespace(quit=lambda: quit_calls.append("quit"))
    monkeypatch.setattr(tray_module, "QApplication", SimpleNamespace(instance=lambda: app))
    monkeypatch.setattr(tray_module, "Request", Request)
    tray.send("copy", responses.append)
    tray.send("cancel")
    copy, cancel = started
    tray.request_quit()
    assert tray.quitting
    assert not quit_calls
    copy.done.emit({"error": "No transcript yet"})
    assert tray.toolTip() == "MouthKeys — No transcript yet"
    assert responses == [{"error": "No transcript yet"}]
    copy.finished.emit()
    assert not quit_calls
    cancel.done.emit({"result": {}})
    cancel.finished.emit()
    assert deleted == ["copy", "cancel"]
    assert not tray.jobs
    assert quit_calls == ["quit"]


def test_quit_without_pending_work(tray, monkeypatch):
    quits = []
    app = SimpleNamespace(quit=lambda: quits.append(True))
    monkeypatch.setattr(tray_module, "QApplication", SimpleNamespace(instance=lambda: app))
    tray.request_quit()
    assert tray.quitting
    assert quits == [True]


@pytest.mark.parametrize("available", [True, False])
def test_standalone_launch_starts_service_and_offscreen_event_loop(monkeypatch, available):
    seen = []
    app = SimpleNamespace(
        setApplicationName=seen.append,
        setQuitOnLastWindowClosed=seen.append,
        exec=lambda: seen.append("event loop") or 7,
    )
    monkeypatch.setattr(tray_module, "ensure_daemon", lambda: seen.append("daemon"))
    monkeypatch.setattr(tray_module, "ensure_overlay", lambda: seen.append("overlay"))
    monkeypatch.setattr(tray_module, "QApplication", lambda _: app)
    monkeypatch.setattr(tray_module, "create_tray", lambda: object() if available else None)
    if not available:
        with pytest.raises(RuntimeError, match="System tray unavailable"):
            tray_module.launch()
        assert seen == ["daemon", "overlay", "MouthKeys", False]
    else:
        assert tray_module.launch() == 7
        assert seen == ["daemon", "overlay", "MouthKeys", False, "event loop"]


def test_standalone_launch_returns_when_another_tray_holds_the_lock(monkeypatch):
    def already_running(*_):
        raise BlockingIOError

    monkeypatch.setattr(tray_module.fcntl, "flock", already_running)
    monkeypatch.setattr(tray_module, "ensure_daemon", lambda: pytest.fail("Started duplicate tray"))
    assert tray_module.launch() == 0


@pytest.mark.parametrize(
    ("delivery", "message", "notice", "expected"),
    [
        (
            "x11",
            "Copied",
            {},
            "Automatic insertion was not requested. Transcript copied to clipboard.",
        ),
        ("clipboard", "Copied", {}, ""),
        ("x11", "Copied", {"kind": "paste", "message": "Missing focus"}, "Missing focus"),
        ("x11", "No text to deliver", {}, ""),
    ],
)
def test_manual_paste_notifications_follow_delivery_outcome(
    tray, monkeypatch, delivery, message, notice, expected
):
    notices = []
    monkeypatch.setattr(tray, "showMessage", lambda *args: notices.append(args))
    Config(show_paste_alerts="on", delivery=delivery).save()
    tray.notify_status({"result": {"phase": "idle"}})
    status = {
        "phase": "idle",
        "id": "new",
        "preview": "hello",
        "message": message,
        "notice": notice,
    }
    tray.notify_status({"result": status})
    assert notices == ([("Manual paste needed", expected)] if expected else [])


def test_microphone_unavailable_notice_is_shown_at_most_once(tray, monkeypatch):
    notices = []
    monkeypatch.setattr(tray, "showMessage", lambda *args: notices.append(args))
    Config(show_microphone_alerts="on").save()
    fault = {"kind": "error", "serial": "once", "message": "Microphone is no longer available"}
    status = {"phase": "recording", "notice": fault}
    tray.notify_status({"result": status})
    tray.notify_status({"result": status})
    assert notices == [("Microphone unavailable", fault["message"])]
