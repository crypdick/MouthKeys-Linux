import time
from importlib import import_module

import pytest

from mouthkeys.config import Config

Qt = pytest.importorskip("PySide6.QtCore").Qt
QApplication = pytest.importorskip("PySide6.QtWidgets").QApplication
overlay = import_module("mouthkeys.overlay")


def test_overlay_never_takes_focus_and_tracks_recording_lifecycle(monkeypatch):
    app = QApplication.instance() or QApplication([])
    panel = overlay.Overlay()
    panel.timer.stop()
    panel.animation.stop()
    shown = []
    monkeypatch.setattr(panel, "show", lambda: shown.append(True))
    assert panel.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert panel.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert panel.focusPolicy() == Qt.FocusPolicy.NoFocus
    panel.set_status({"phase": "recording", "id": "a", "seconds": 3, "preview": "hello"})
    assert shown == [True]
    assert panel.data["preview"] == "hello"
    assert not panel.buttons["copy"].isEnabled()
    panel.set_status({"phase": "transcribing", "id": "a", "preview": "hello"})
    assert panel.seconds == 3
    panel.set_status(
        {"phase": "idle", "id": "a", "preview": "Hello!", "message": "Copied — paste into your app"}
    )
    assert panel.buttons["copy"].isEnabled()
    assert panel.dismiss_at is not None
    panel.set_status({"phase": "recording", "id": "b", "seconds": 0, "preview": ""})
    assert shown == [True, True]
    assert panel.dismiss_at is None
    panel.set_status({"phase": "idle", "id": "b", "preview": "", "message": "Cancelled"})
    assert not panel.isVisible()
    panel.close()
    app.processEvents()


def test_overlay_deduplicates_meter_frames_and_resets_on_new_recording(monkeypatch):
    app = QApplication.instance() or QApplication([])
    panel = overlay.Overlay()
    panel.timer.stop()
    monkeypatch.setattr(panel, "show", lambda: None)
    now = time.monotonic()
    data = {
        "phase": "recording",
        "id": "a",
        "seconds": 1,
        "meter": {"frame": 16000, "level": 0.8, "at": now},
    }
    panel.set_status(data)
    first_voice = panel.trace.last_voice
    panel.set_status({**data, "meter": {**data["meter"], "at": now + 0.1}})
    assert panel.trace.last_voice == first_voice
    panel.set_status({**data, "id": "b", "meter": {}})
    assert panel.trace.last_voice is None
    assert panel.trace.shown == [2] * 63
    panel.animation.stop()
    panel.close()
    app.processEvents()


def test_panel_dismisses_promptly_after_insertion_request(monkeypatch):
    app = QApplication.instance() or QApplication([])
    panel = overlay.Overlay()
    panel.timer.stop()
    hidden = []
    now = 100.0
    monkeypatch.setattr(overlay.time, "monotonic", lambda: now)
    monkeypatch.setattr(panel, "show", lambda: None)
    monkeypatch.setattr(panel, "hide", lambda: hidden.append(True))
    panel.set_status({"phase": "delivering", "id": "a"})
    done = {
        "phase": "idle",
        "id": "a",
        "message": "Insertion requested — transcript saved in history",
    }
    panel.set_status(done)
    assert not hidden
    now += 1
    panel.set_status(done)
    assert hidden == [True]
    panel.animation.stop()
    panel.close()
    app.processEvents()


def test_panel_shows_fresh_shortcut_error_even_when_idle(monkeypatch):
    app = QApplication.instance() or QApplication([])
    panel = overlay.Overlay()
    panel.timer.stop()
    shown = []
    monkeypatch.setattr(panel, "show", lambda: shown.append(True))
    data = {
        "phase": "idle",
        "message": "Ready",
        "notice": {"serial": 1, "message": "Source unplugged", "at": time.time()},
    }
    panel.set_status(data)
    assert shown == [True]
    assert panel.notice[0] == "Source unplugged"
    panel.set_status(data)
    assert shown == [True]  # Polling must not keep extending the error timeout.
    Config(recording_panel="off").save()
    panel.set_status({**data, "phase": "recording"})
    assert shown == [True]
    for button in panel.buttons.values():
        assert not button.icon().isNull()
        assert button.text() == ""  # No symbol font is needed.
    panel.close()
    app.processEvents()


def test_panel_can_reappear_after_disabled_during_recording(monkeypatch):
    app = QApplication.instance() or QApplication([])
    panel = overlay.Overlay()
    panel.timer.stop()
    shown = []
    monkeypatch.setattr(panel, "show", lambda: shown.append(True))
    data = {"phase": "recording", "id": "same-recording", "seconds": 1}
    panel.set_status(data)
    Config(recording_panel="off").save()
    panel.set_status(data)
    Config(recording_panel="on").save()
    panel.set_status(data)
    assert shown == [True, True]
    panel.animation.stop()
    panel.close()
    app.processEvents()


def test_queued_overlay_cancel_reaches_controller(monkeypatch):
    app = QApplication.instance() or QApplication([])
    panel = overlay.Overlay()
    panel.timer.stop()
    panel.animation.stop()
    calls = []
    monkeypatch.setattr(panel, "show", lambda: None)
    monkeypatch.setattr(panel, "send", lambda action, *args, **kwargs: calls.append(action))
    panel.set_status({"phase": "queued", "id": "pending", "preview": "queued words"})
    assert panel.active
    panel.action("cancel")
    assert calls == ["cancel"]
    panel.close()
    app.processEvents()
