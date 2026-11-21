import subprocess
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMenu

from mouthkeys import overlay, qt_control
from mouthkeys.config import Config


@pytest.fixture
def panel(monkeypatch):
    widget = overlay.Overlay()
    widget.timer.stop()
    monkeypatch.setattr(widget, "show", lambda: None)
    return widget


def test_shortcut_feedback_and_actions_do_not_leave_request_workers(panel, monkeypatch):
    calls = []

    class InlineRequest(qt_control.Request):
        def start(self):
            self.run()
            self.finished.emit()

    def request(action, **data):
        calls.append((action, data))
        if action == "cancel":
            raise RuntimeError("Already delivered")
        return []

    monkeypatch.setattr(overlay, "Request", InlineRequest)
    monkeypatch.setattr(qt_control, "request", request)
    monkeypatch.setattr(QMenu, "popup", lambda *args: None)
    panel.data = {"id": "latest"}
    panel.action("copy")
    panel.send("copy", item_id="selected")
    panel.send("copy", data={"id": "explicit"})
    panel.action("history")
    assert not panel.history_menu.actions()[0].isEnabled()
    panel.active = True
    panel.action("cancel")
    assert panel.display_text() == "Already delivered"
    assert panel.dismiss_at == 0
    assert calls == [
        ("copy", {"id": "latest"}),
        ("copy", {"id": "selected"}),
        ("copy", {"id": "explicit"}),
        ("history", {}),
        ("cancel", {}),
    ]
    assert not panel.jobs
    assert not panel.isVisible()


@pytest.mark.parametrize("unavailable", [False, True])
def test_microphone_menu_filters_sources_and_remains_usable_when_probe_fails(
    panel, monkeypatch, unavailable
):
    Config(microphone_never="excluded").save()

    def sources():
        if unavailable:
            raise OSError("Enumeration failed")
        return ["usb", "excluded"]

    sent = []

    offered = []
    callbacks = []

    class Menu:
        def __init__(self, _parent):
            pass

        def addAction(self, name):  # noqa: N802 - Qt interface
            offered.append(name)
            return SimpleNamespace(triggered=SimpleNamespace(connect=callbacks.append))

        def exec(self, _pos):
            callbacks[-1]()

    monkeypatch.setattr(overlay, "source_names", sources)
    monkeypatch.setattr(overlay, "QMenu", Menu)
    monkeypatch.setattr(panel, "send", lambda action, **data: sent.append((action, data)))
    panel.choose_microphone()
    assert offered == (["default"] if unavailable else ["default", "usb"])
    selected = "default" if unavailable else "usb"
    assert sent == [("select_microphone", {"data": {"source": selected, "persist": False}})]
    assert not panel.isVisible()


def test_history_failure_stays_on_panel_and_settings_opens_only_on_click(panel, monkeypatch):
    monkeypatch.setattr(QMenu, "popup", lambda *args: None)
    panel.show_history({"error": "History unavailable"})
    assert panel.display_text() == "History unavailable"
    calls = []

    def launch(args, **options):
        calls.append((args[1:], options))

    monkeypatch.setattr(subprocess, "Popen", launch)
    panel.show_history({"result": [{"id": "example", "text": "", "error": "Input missing"}]})
    actions = panel.history_menu.actions()
    assert actions[0].text() == "Input missing"
    assert not actions[0].menu().actions()[0].isEnabled()
    assert not calls
    actions[-1].trigger()
    assert calls == [(["-P", "-m", "mouthkeys", "gui"], {"start_new_session": True})]


def test_polling_has_one_inflight_job_and_backs_off_on_disconnect(panel, monkeypatch):
    pending = []

    def send(action, callback):
        assert action == "status"
        pending.append(callback)

    monkeypatch.setattr(panel, "send", send)
    panel.poll()
    panel.poll()
    assert len(pending) == 1
    pending.pop()({"result": {"phase": "recording", "id": "one"}})
    assert panel.active
    assert panel.timer.interval() == 50
    for interval in (400, 800, 1600, 3200, 5000, 5000):
        panel.poll()
        pending.pop()({"error": "Disconnected"})
        assert panel.timer.interval() == interval
        assert not panel.active
        assert not panel.animation.isActive()
    panel.poll()
    pending.pop()({"result": {"phase": "idle"}})
    assert panel.timer.interval() == 200
    assert panel.failures == 0
    assert not panel.isVisible()


def test_compact_panel_shortens_current_paste_notice_and_ignores_stale_serial(panel, monkeypatch):
    clock = SimpleNamespace(time=lambda: 50, monotonic=lambda: 25)
    monkeypatch.setattr(overlay, "time", clock)
    Config(overlay_style="compact").save()
    status = {"phase": "idle", "preview": "Ready to paste"}
    notice = {"kind": "paste", "message": "Focus changed; copied instead", "serial": 1}
    panel.set_status(status | {"notice": notice | {"at": 30}})
    assert panel.notice is None
    panel.set_status(status | {"notice": notice | {"at": 50}})
    assert panel.notice is None  # Repeated notice serial must not reappear.
    panel.set_status(status | {"notice": notice | {"at": 50, "serial": 2}})
    assert panel.notice == ("Auto-paste skipped. Text copied.", 35)
    assert panel.buttons["copy"].isEnabled()


@pytest.mark.parametrize("percent", [None, 73])
def test_microphone_badge_displays_linked_battery_values(panel, monkeypatch, percent):
    result = {"result": "ok", "mic1": {"percent": percent}, "mic2": {"percent": None}}
    monkeypatch.setattr(panel.battery, "poll", lambda *_: result)
    panel.set_status({"phase": "idle", "microphone": "Mic"})
    expected = "Lark A1 battery · " + ("No linked mic" if percent is None else "73%")
    assert panel.mic_button.toolTip() == expected


def test_animation_stops_when_hidden_or_after_idle_waveform_settles(panel, monkeypatch):
    panel.animation.start(16)
    panel.animate()
    assert not panel.animation.isActive()
    clock = [10.0]
    monkeypatch.setattr(overlay, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(panel, "isVisible", lambda: True)
    invalidated = []
    monkeypatch.setattr(panel, "update", invalidated.append)
    panel.trace.begin(clock[0])
    panel.trace.ingest(1, clock[0])
    panel.data = {"phase": "recording"}
    panel.animation.start(16)
    clock[0] += 0.12
    panel.animate()
    assert max(panel.trace.shown) > 2
    assert panel.animation.isActive()
    panel.trace.stop(clock[0])
    panel.data = {"phase": "idle"}
    clock[0] += 0.1
    panel.animate()
    assert not panel.animation.isActive()
    assert panel.trace.shown == [2] * panel.trace.count
    assert len(invalidated) == 2
