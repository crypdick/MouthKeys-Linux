import time
from importlib.metadata import version

import pytest

pytest.importorskip("PySide6")


@pytest.mark.parametrize("success", [True, False])
def test_download_retries_unavailable_model_only_after_success(monkeypatch, tmp_path, success):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication

    from mouthkeys import gui
    from mouthkeys.service import Controller

    app = QApplication.instance() or QApplication([])
    ready = tmp_path / "model-ready"

    def load(config):
        if not ready.exists():
            raise RuntimeError("Model missing")

    controller = Controller(transcriber=SimpleNamespace(load=load))
    controller.command("warmup")
    controller.worker.join(2)
    assert not controller.worker.is_alive()
    assert "Model unavailable" in controller.status()["message"]
    monkeypatch.setattr(gui, "request", controller.command)
    # Exercise real QProcess completion without network or model inference.
    script = (
        f"from pathlib import Path; Path({str(ready)!r}).touch()"
        if success
        else "import sys; print('Download failed'); sys.exit(1)"
    )
    monkeypatch.setattr(gui, "python_command", lambda *args: ["python", "-c", script])
    messages = []
    monkeypatch.setattr(gui.QMessageBox, "information", lambda *args: messages.append(args[-1]))
    window = gui.Window()
    window.timer.stop()
    try:
        window.download_button.click()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            app.processEvents()
            if messages and not window.jobs and controller.status()["phase"] == "idle":
                break
            time.sleep(0.01)
        assert window.download is None
        assert not window.jobs
        assert window.download_button.isEnabled()
        if success:
            assert controller.status()["message"] == "Model ready"
        else:
            assert "Model unavailable" in controller.status()["message"]
            assert "Download failed" in messages[0]
        assert not window.isVisible()
    finally:
        window.close()
        app.processEvents()


def test_settings_show_app_and_running_service_versions(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mouthkeys import gui

    app = QApplication.instance() or QApplication([])
    current = version("mouthkeys-linux")
    status = {"phase": "idle", "message": "Ready", "seconds": 0, "version": current}

    def send(self, action, data=None, callback=None):
        callback({"result": status if action == "status" else []})

    monkeypatch.setattr(gui.Window, "send", send)
    window = gui.Window()
    window.timer.stop()
    window.nav.setCurrentRow(1)
    assert window.app_version.text() == current
    assert window.service_version.text() == current
    status["version"] = "0.0.1"
    window.poll()
    assert window.service_version.text() == "0.0.1 — restart MouthKeys after updating"
    monkeypatch.setattr(
        gui.Window, "send", lambda self, *args, callback: callback({"error": "Disconnected"})
    )
    window.poll()
    assert window.service_version.text() == "Unavailable"
    assert not window.isVisible()
    window.close()
    app.processEvents()


def test_offscreen_controller_renders_history_and_saves_settings(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mouthkeys import gui
    from mouthkeys.config import Config

    app = QApplication.instance() or QApplication([])

    def request(action, **data):
        if action == "status":
            return {
                "phase": "idle",
                "message": "Ready",
                "seconds": 0,
                "version": version("mouthkeys-linux"),
            }
        if action == "history":
            return [
                {
                    "id": "example",
                    "created": 1,
                    "status": "copied",
                    "text": "Hello café",
                    "error": "",
                }
            ]
        raise AssertionError(f"Unexpected GUI request: {action}")

    monkeypatch.setattr(gui, "request", request)
    window = gui.Window()
    window.timer.stop()
    deadline = time.monotonic() + 3
    while (window.jobs or not window.rows) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not window.jobs
    assert window.preview.toPlainText() == "Hello café"
    window.fields["model"].setCurrentText("small.en")
    assert window.save()
    assert Config.load().model == "small.en"
    window.fields["language"].setCurrentIndex(window.fields["language"].findData("es"))
    window.fields["compute_type"].setCurrentIndex(window.fields["compute_type"].findData("auto"))
    window.fields["max_seconds"].setValue(120)
    assert window.save()
    saved = Config.load()
    assert (saved.language, saved.compute_type, saved.max_seconds) == ("es", "auto", 120)
    assert "Spanish" in window.fields["language"].currentText()
    window.fields["model"].setEditText("/custom/local model")
    window.fields["audio_source"].setEditText("my_microphone")
    assert window.save()
    assert Config.load().model == "/custom/local model"
    assert Config.load().audio_source == "my_microphone"
    assert not window.isVisible()  # No visible windows, even on the offscreen platform.
    window.close()


def test_dropdowns_preserve_existing_custom_settings(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mouthkeys import gui
    from mouthkeys.config import Config

    app = QApplication.instance() or QApplication([])
    Config(model="org/custom-model", language="xx", audio_source="custom_source").save()
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    window = gui.Window()
    window.timer.stop()
    assert window.fields["model"].currentText() == "org/custom-model"
    assert window.fields["language"].currentData() == "xx"
    assert window.save()
    assert Config.load().model == "org/custom-model"
    assert Config.load().language == "xx"
    assert Config.load().audio_source == "custom_source"
    assert not window.isVisible()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("model", ["large", "turbo", "large-v3", "large-v3-turbo"])
def test_model_dropdown_uses_specific_names_for_saved_aliases(monkeypatch, model):
    from PySide6.QtWidgets import QApplication

    from mouthkeys import gui
    from mouthkeys.config import Config

    app = QApplication.instance() or QApplication([])
    Config(model=model).save()
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    window = gui.Window()
    window.timer.stop()
    field = window.fields["model"]
    names = [field.itemData(index) for index in range(field.count())]
    assert "large" not in names
    assert "turbo" not in names
    assert names.count("large-v3") == names.count("large-v3-turbo") == 1
    assert names[0] == "large-v3"
    assert names.index("large-v3-turbo") < names.index("medium") < names.index("tiny")
    assert field.currentText() == gui.MODEL_ALIASES.get(model, model)
    assert Config.load().model == model  # Opening settings does not rewrite configuration.
    assert window.save()
    assert Config.load().model == field.currentText()
    assert not window.isVisible()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("model", ["large", "turbo"])
def test_model_dropdown_preserves_local_directory_named_like_alias(monkeypatch, tmp_path, model):
    from PySide6.QtWidgets import QApplication

    from mouthkeys import gui
    from mouthkeys.config import Config

    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    (tmp_path / model).mkdir()
    Config(model=model).save()
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    window = gui.Window()
    window.timer.stop()
    assert window.fields["model"].currentText() == model
    assert window.save()
    assert Config.load().model == model
    assert not window.isVisible()
    window.close()
    app.processEvents()


def test_microphone_choices_use_discovery_without_capture(monkeypatch):
    import subprocess

    from mouthkeys import delivery, gui

    monkeypatch.setattr(delivery, "session", lambda: "x11")

    def run(args, **kwargs):
        assert args == ["pactl", "list", "short", "sources"]
        assert kwargs["timeout"] == 2
        return subprocess.CompletedProcess(args, 0, "12\tmy_microphone\tdriver\tRUNNING\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert gui.audio_sources() == ["default", "my_microphone"]


def test_missing_pactl_keeps_manual_microphone_selection(monkeypatch):
    import subprocess

    from mouthkeys import delivery, gui

    monkeypatch.setattr(delivery, "session", lambda: "x11")

    def run(*args, **kwargs):
        raise FileNotFoundError("pactl")

    monkeypatch.setattr(subprocess, "run", run)
    assert gui.audio_sources() == ["default"]


def test_overlay_never_takes_focus_and_tracks_recording_lifecycle(monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from mouthkeys.overlay import Overlay

    app = QApplication.instance() or QApplication([])
    panel = Overlay()
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
    from PySide6.QtWidgets import QApplication

    from mouthkeys import overlay

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


def test_panel_shows_fresh_shortcut_error_even_when_idle(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from mouthkeys.config import Config
    from mouthkeys.overlay import Overlay

    app = QApplication.instance() or QApplication([])
    panel = Overlay()
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
    from PySide6.QtWidgets import QApplication

    from mouthkeys.config import Config
    from mouthkeys.overlay import Overlay

    app = QApplication.instance() or QApplication([])
    panel = Overlay()
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


def test_every_config_is_editable_in_settings_and_new_options_roundtrip(monkeypatch):
    from dataclasses import asdict

    from PySide6.QtWidgets import QApplication, QScrollArea

    from mouthkeys import gui
    from mouthkeys.config import Config

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    window = gui.Window()
    window.timer.stop()
    assert set(window.fields) == set(asdict(Config()))
    assert window.findChild(QScrollArea) is not None
    window.fields["audio_budget_mb"].setValue(250)
    window.fields["recording_panel"].setCurrentIndex(
        window.fields["recording_panel"].findData("off")
    )
    window.fields["max_seconds"].setValue(30)
    assert window.save()
    saved = Config.load()
    assert (saved.audio_budget_mb, saved.recording_panel, saved.max_seconds) == (250, "off", 30)
    assert window.fields["audio_budget_mb"].suffix() == " MB"
    assert "failures" in window.fields["audio_budget_mb"].toolTip()
    assert not window.isVisible()
    window.close()
    app.processEvents()
