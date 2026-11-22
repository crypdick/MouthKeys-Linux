from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSystemTrayIcon, QWidget

from mouthkeys import cli, gui_launch, tray
from mouthkeys.config import Config


@pytest.mark.parametrize(("returning", "has_tray"), [(False, False), (True, True)])
def test_gui_cli_launch_preserves_desktop_lifecycle_without_showing_a_window(
    monkeypatch, returning, has_tray
):
    events = []
    if returning:
        Config().save()

    def launch_app(_argv):
        events.append("app")
        return SimpleNamespace(
            setApplicationName=lambda name: events.append(("name", name)),
            setQuitOnLastWindowClosed=lambda enabled: events.append(("quit on close", enabled)),
            exec=lambda: events.append("event loop") or 7,
        )

    fake_window = SimpleNamespace(
        nav=SimpleNamespace(setCurrentRow=lambda row: events.append(("page", row))),
        setAttribute=events.append,
        show=lambda: events.append("show"),
    )
    monkeypatch.setattr(gui_launch, "ensure_daemon", lambda: events.append("daemon"))
    monkeypatch.setattr(gui_launch, "ensure_overlay", lambda: events.append("overlay"))
    monkeypatch.setattr(gui_launch, "ensure_hotkeys", lambda: events.append("hotkeys"))
    monkeypatch.setattr(gui_launch, "QApplication", launch_app)
    monkeypatch.setattr(gui_launch, "Window", lambda: events.append("window") or fake_window)
    monkeypatch.setattr(
        gui_launch,
        "create_tray",
        lambda window: events.append(("tray", window)) or (object() if has_tray else None),
    )

    assert cli.main(["gui"]) == 7
    expected = ["daemon", "overlay", "hotkeys", "app", ("name", "MouthKeys"), "window"]
    if not returning:
        expected.append(("page", 9))
    expected.extend([("tray", fake_window), ("quit on close", not has_tray)])
    if not has_tray:
        expected.append(Qt.WidgetAttribute.WA_DeleteOnClose)
    assert events == [*expected, "show", "event loop"]


@pytest.mark.parametrize(("enabled", "available"), [("on", True), ("on", False), ("off", True)])
def test_tray_factory_only_shows_when_enabled_and_supported(monkeypatch, enabled, available):
    Config(tray=enabled).save()
    shown = []

    def show(icon):
        shown.append(icon)

    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", lambda: available)
    monkeypatch.setattr(tray.Tray, "show", show)
    window = QWidget()
    result = tray.create_tray(window)
    if enabled == "on" and available:
        assert shown == [result]
        assert result.parent() is window
        assert not result.isVisible()
        result.timer.stop()
    else:
        assert result is None
        assert not shown
    assert not window.isVisible()
