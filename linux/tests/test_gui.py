import time
from importlib import import_module
from importlib.metadata import version

import pytest

QApplication = pytest.importorskip("PySide6.QtWidgets").QApplication
settings_gui = import_module("mouthkeys.gui")


def test_taskbar_hint_is_configurable_and_separate_from_tray(monkeypatch):
    from PySide6.QtCore import Qt

    from mouthkeys.config import Config
    from mouthkeys.gui import Window

    Config(hide_from_taskbar="on", tray="off").save()
    monkeypatch.setattr(Window, "poll", lambda self: None)
    window = Window()
    window.timer.stop()
    assert window.windowType() == Qt.WindowType.Tool
    assert not window.isVisible()
    field = window.fields["hide_from_taskbar"]
    field.setCurrentIndex(field.findData("off"))
    assert window.save()
    assert window.windowType() == Qt.WindowType.Window
    assert Config.load().tray == "off"
    assert not window.isVisible()
    window.close()


def test_model_browser_resident_state_uses_loaded_identity_not_selected_settings(monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QComboBox, QLineEdit, QWidget

    from mouthkeys.models_ui import ModelBrowser

    window = QWidget()
    engine = QComboBox()
    engine.addItem("Whisper", "whisper")
    window.fields = {"engine": engine, "model_revision": QLineEdit("selected-revision")}
    window.loaded_model = {
        "engine": "whisper",
        "model": "resident-model",
        "revision": "loaded-revision",
        "device": "cpu",
        "compute_type": "int8",
    }
    monkeypatch.setattr(
        "mouthkeys.engines.model_catalog",
        lambda *a, **kw: [
            {
                "model": "resident-model",
                "name": "Resident",
                "languages": "en",
                "installed": True,
                "revision": "loaded-revision",
            },
            {
                "model": "selected-model",
                "name": "Selected",
                "languages": "en",
                "installed": True,
                "revision": "selected-revision",
            },
        ],
    )
    browser = ModelBrowser(window)
    browser.table.sortItems(6, Qt.SortOrder.AscendingOrder)
    browser.update_resident()
    states = {
        browser.table.item(row, 0).text(): browser.table.item(row, 6).text()
        for row in range(browser.table.rowCount())
    }
    assert states == {"Resident": "Yes", "Selected": "No"}
    window.loaded_model = None
    browser.update_resident()
    assert all(browser.table.item(row, 6).text() == "No" for row in range(2))
    window.close()


@pytest.mark.parametrize("success", [True, False])
def test_download_retries_unavailable_model_only_after_success(monkeypatch, tmp_path, success):
    from types import SimpleNamespace

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
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    combo = window.fields["delivery"]
    guard = window.fields["destination_guard"]
    assert guard.currentData() == "field_then_window"
    assert guard.currentText() == "Field first, window fallback"
    combo.setCurrentIndex(combo.findData("x11"))
    assert "X11 auto-paste needs an X11 login session" in window.delivery_hint.text()
    combo.setCurrentIndex(combo.findData("clipboard"))
    assert "Dictation copies text" in window.delivery_hint.text()
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
    window.fields["recording_panel"].setCurrentIndex(
        window.fields["recording_panel"].findData("off")
    )
    window.fields["max_seconds"].setValue(30)
    assert window.save()
    saved = Config.load()
    assert (saved.recording_panel, saved.max_seconds) == ("off", 30)
    assert "save_audio" not in window.fields
    assert "audio_budget_mb" not in window.fields
    assert not window.isVisible()
    window.close()
    app.processEvents()


def test_settings_source_change_applies_active_service_and_float_vocabulary(monkeypatch):
    from mouthkeys import gui
    from mouthkeys.config import Config
    from mouthkeys.vocabulary import VocabularyStore

    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    window = gui.Window()
    window.timer.stop()
    calls = []
    monkeypatch.setattr(
        window, "send", lambda action, data=None, **kw: calls.append((action, data))
    )
    try:
        window.fields["audio_source"].setEditText("usb-input")
        assert window.save()
        assert calls == [("select_microphone", {"source": "usb-input", "persist": True})]
        assert Config.load().audio_source == "usb-input"
        page = window.learning_page
        page.add_word(entry={"text": "MouthKeys", "weight": 8.5, "aliases": []})
        page.add_word(entry={"text": "Default", "weight": None, "aliases": []})
        page.save_words()
        assert VocabularyStore.entries() == [
            {"text": "MouthKeys", "weight": 8.5, "aliases": []},
            {"text": "Default", "weight": None, "aliases": []},
        ]
    finally:
        window.close()


def test_dictionary_editor_preserves_imported_rule_id_when_editing(monkeypatch):
    from mouthkeys import gui
    from mouthkeys.dictionary import Dictionary

    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    data = Dictionary.defaults()
    rule_id = "792b9c60-0a77-4e06-8630-8b1046d789ed"
    data["replacements"] = [{"id": rule_id, "triggers": ["heard"], "replacement": "Written"}]
    Dictionary.save(data)
    window = gui.Window()
    window.timer.stop()
    try:
        window.dictionary_tables["replacements"].item(0, 1).setText("Edited")
        assert window.save_dictionary()
        assert Dictionary.load()["replacements"][0]["id"] == rule_id
        assert Dictionary.apply("heard") == "Edited"
    finally:
        window.close()


@pytest.mark.parametrize("accept", [True, False])
def test_backup_import_previews_before_applying_exact_reviewed_file(monkeypatch, accept):
    import json
    import time

    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    from mouthkeys import gui, operations

    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: ("chosen.json", ""))
    report = {
        "format": "upstream",
        "counts": {"history": 2, "profiles": 1, "replacements": 3},
        "warnings": ["Apple microphone IDs not imported"],
        "file_sha256": "a" * 64,
    }
    monkeypatch.setattr(operations, "backup_preview", lambda path: report)
    reviewed = []

    def review(dialog):
        reviewed.append(json.loads(dialog.detailedText()))
        assert not dialog.isVisible()
        return QMessageBox.StandardButton.Yes if accept else QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "exec", review)
    window = gui.Window()
    window.timer.stop()
    calls = []
    monkeypatch.setattr(window, "send", lambda action, data, **kw: calls.append((action, data)))
    try:
        window.backup_action(True)
        deadline = time.monotonic() + 2
        while window.jobs and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.001)
        assert not window.jobs
        assert reviewed == [report]
        assert calls == (
            [("backup_import", {"path": "chosen.json", "expected_sha256": "a" * 64})]
            if accept
            else []
        )
    finally:
        window.close()


def test_stale_search_results_do_not_replace_current_search(monkeypatch):
    app = QApplication.instance()
    monkeypatch.setattr(settings_gui.Window, "poll", lambda self: None)
    window = settings_gui.Window()
    window.timer.stop()
    pending = []
    monkeypatch.setattr(window, "send", lambda action, data, callback: pending.append(callback))
    window.search.setText("A")
    window.search.setText("B")
    window.search.setText("A")
    received = []
    monkeypatch.setattr(window, "update_history", lambda response: received.append(response))
    pending[0]({"result": ["stale A"]})
    pending[1]({"result": ["stale B"]})
    assert not received
    pending[2]({"result": []})
    assert received == [{"result": []}]
    window.close()
    app.processEvents()
