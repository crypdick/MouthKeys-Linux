import time
from importlib import import_module
from types import SimpleNamespace

import pytest

from mouthkeys.config import Config

QtCore = pytest.importorskip("PySide6.QtCore")
Qt = QtCore.Qt
QApplication = pytest.importorskip("PySide6.QtWidgets").QApplication
overlay = import_module("mouthkeys.overlay")


@pytest.fixture
def panel_placement(monkeypatch):
    panel = overlay.Overlay()
    panel.timer.stop()
    shown = []
    monkeypatch.setattr(panel, "show", lambda: shown.append(panel.frameGeometry()))
    screens = []
    monkeypatch.setattr(QApplication, "screens", lambda: screens)
    monkeypatch.setattr(QApplication, "screenAt", lambda _: screens[-1])
    return panel, shown, screens


@pytest.mark.parametrize("edge", ["top", "bottom"])
def test_overlay_clamps_offset_to_screen_work_area(panel_placement, edge):
    panel, shown, screens = panel_placement
    area = QtCore.QRect(-900, 60, 900, 500)
    screens.append(SimpleNamespace(availableGeometry=lambda: area))
    Config(overlay_position=edge, overlay_size=150, overlay_bottom_offset=1000).save()

    panel.set_status({"phase": "recording", "id": "a"})

    (frame,) = shown
    assert area.contains(frame)
    if edge == "top":
        assert frame.bottom() == area.bottom()
    else:
        assert frame.top() == area.top()


def test_overlay_recovers_stale_position_on_next_shortcut(panel_placement):
    panel, shown, screens = panel_placement
    primary = QtCore.QRect(0, 32, 1600, 868)
    external = QtCore.QRect(-1200, 0, 1200, 1200)
    screens.extend(
        [
            SimpleNamespace(availableGeometry=lambda: primary),
            SimpleNamespace(availableGeometry=lambda: external),
        ]
    )
    panel.set_status({"phase": "recording", "id": "a"})
    assert external.contains(shown[-1])
    panel.set_status({"phase": "idle", "id": "a", "message": "Cancelled"})
    screens.pop()

    panel.set_status({"phase": "recording", "id": "b"})

    assert primary.contains(shown[-1])


def test_overlay_scales_to_fit_narrow_display_without_changing_settings(panel_placement):
    panel, shown, screens = panel_placement
    area = QtCore.QRect(-576, 0, 576, 900)
    screens.append(SimpleNamespace(availableGeometry=lambda: area))
    Config(overlay_size=150).save()

    panel.set_status({"phase": "recording", "id": "a"})

    assert area.contains(shown[-1])
    assert panel.rect().contains(panel.buttons["cancel"].geometry())
    assert Config.load().overlay_size == 150
    panel.set_status({"phase": "idle", "id": "a", "message": "Cancelled"})
    area.setWidth(900)
    panel.set_status({"phase": "recording", "id": "b"})
    assert shown[-1].size() == QtCore.QSize(636, 219)


def test_overlay_recovers_when_work_area_changes_mid_recording(panel_placement, monkeypatch):
    panel, _, screens = panel_placement
    area = QtCore.QRect(0, 0, 1600, 1200)
    screens.append(SimpleNamespace(availableGeometry=lambda: area))
    data = {"phase": "recording", "id": "a"}
    panel.set_status(data)
    monkeypatch.setattr(panel, "isVisible", lambda: True)
    area.setHeight(900)
    assert not area.contains(panel.frameGeometry())

    panel.set_status(data)

    assert area.contains(panel.frameGeometry())


def test_overlay_follows_new_recording_but_not_notice_to_pointer_display(
    panel_placement, monkeypatch
):
    panel, shown, screens = panel_placement
    area = QtCore.QRect(0, 0, 1600, 900)
    other = QtCore.QRect(1600, 0, 1600, 900)
    screens.append(SimpleNamespace(availableGeometry=lambda: area))
    data = {"phase": "recording", "id": "a"}
    panel.set_status(data)
    monkeypatch.setattr(panel, "isVisible", lambda: True)
    screens.append(SimpleNamespace(availableGeometry=lambda: other))

    panel.set_status({**data, "notice": {"serial": 1, "message": "Warning", "at": time.time()}})
    assert shown[-1] == shown[0]

    panel.set_status({**data, "id": "b"})
    assert other.contains(shown[-1])


def test_overlay_clamps_drag_and_retains_it_between_shortcuts(panel_placement):
    panel, shown, screens = panel_placement
    area = QtCore.QRect(0, 0, 1600, 900)
    screens.append(SimpleNamespace(availableGeometry=lambda: area))
    panel.set_status({"phase": "recording", "id": "a"})
    panel.mousePressEvent(
        SimpleNamespace(
            button=lambda: Qt.MouseButton.LeftButton,
            globalPosition=lambda: QtCore.QPointF(panel.frameGeometry().center()),
        )
    )
    panel.mouseMoveEvent(SimpleNamespace(globalPosition=lambda: QtCore.QPointF(0, 355)))
    panel.mouseReleaseEvent(None)
    dragged = panel.frameGeometry()
    assert area.contains(dragged)
    assert dragged.left() == area.left()
    panel.set_status({"phase": "idle", "id": "a", "message": "Cancelled"})

    panel.set_status({"phase": "recording", "id": "b"})

    assert shown[-1] == dragged
    panel.mouseDoubleClickEvent(None)
    assert panel.frameGeometry() == shown[0]


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
