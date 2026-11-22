import sys
import time
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mouthkeys.config import Config

widgets = pytest.importorskip("PySide6.QtWidgets")
qtcore = import_module("PySide6.QtCore")
qtgui = import_module("PySide6.QtGui")
model_downloads_ui = import_module("mouthkeys.model_downloads_ui")
gui = import_module("mouthkeys.gui")
models_ui = import_module("mouthkeys.models_ui")


@pytest.fixture
def downloads(monkeypatch):
    save = Mock(return_value=True)
    selection = Mock(return_value=("example/model", "onnx", "pinned"))
    status = Mock()
    warmup = Mock()
    information = Mock()
    question = Mock(return_value=widgets.QMessageBox.StandardButton.No)
    monkeypatch.setattr(
        model_downloads_ui,
        "QMessageBox",
        SimpleNamespace(
            information=information,
            question=question,
            StandardButton=widgets.QMessageBox.StandardButton,
        ),
    )
    page = model_downloads_ui.ModelDownloads(save, selection, status, warmup)
    return SimpleNamespace(
        page=page,
        save=save,
        selection=selection,
        status=status,
        warmup=warmup,
        information=information,
        question=question,
    )


def wait_until(condition):
    app = widgets.QApplication.instance()
    deadline = time.monotonic() + 4
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert condition()


def child(monkeypatch, script):
    command = Mock(return_value=[sys.executable, "-u", "-c", script])
    monkeypatch.setattr(model_downloads_ui, "python_command", command)
    return command


def test_model_download_guards_actions_and_cancels_without_warming(monkeypatch, downloads):
    command = child(
        monkeypatch,
        """
import time
print('{"download_progress":{"phase":"Preparing","completed":0,"total":null,"unit":"file"}}')
print('{"download_progress":{"phase":"Fetching","completed":3,"total":4,"unit":"file"}}')
time.sleep(6)
""",
    )
    page = downloads.page
    page.cancel_button.click()
    downloads.save.return_value = False
    page.model_button.click()
    assert not page.running
    command.assert_not_called()
    downloads.save.return_value = True
    page.model_button.click()
    assert page.running
    assert page.model_button.text() == "Downloading…"
    for button in (page.model_button, page.speaker_button, page.delete_button):
        assert not button.isEnabled()

    page.download_model()
    page.download_speakers()
    page.delete_model()
    assert downloads.save.call_count == 2
    command.assert_called_once_with("mouthkeys", "download")
    downloads.question.assert_not_called()
    try:
        wait_until(lambda: page.progress.value() == 750)
        assert page.progress.format() == "Fetching · 3/4 file"
    finally:
        page.cancel_button.click()
        wait_until(lambda: not page.running)
    assert downloads.information.call_args.args[-1] == "Download cancelled"
    downloads.warmup.assert_not_called()
    assert page.model_button.isEnabled()
    assert page.delete_button.isEnabled()
    assert page.speaker_button.isEnabled()
    assert page.progress.value() == 0


@pytest.mark.parametrize(
    ("script", "expected", "value"),
    [
        ("print('model cache location')", "Speaker models downloaded", 100),
        ("import sys; print('missing SDK', file=sys.stderr); sys.exit(1)", "missing SDK", 0),
        ("raise SystemExit(1)", "Download failed", 0),
    ],
)
def test_speaker_download_uses_own_command_without_saving_or_warming(
    monkeypatch, downloads, script, expected, value
):
    command = child(monkeypatch, script)
    page = downloads.page
    page.speaker_button.click()
    assert page.running
    assert page.speaker_button.text() == "Downloading speaker models…"
    wait_until(lambda: not page.running)
    command.assert_called_once_with("mouthkeys", "download-speakers")
    downloads.status.assert_called_once_with(expected)
    assert page.progress.value() == value
    downloads.warmup.assert_not_called()
    downloads.save.assert_not_called()
    downloads.information.assert_not_called()


@pytest.mark.parametrize("button", ["model_button", "speaker_button"])
def test_failed_start_reenables_actions_and_allows_retry(monkeypatch, tmp_path, downloads, button):
    monkeypatch.setattr(
        model_downloads_ui, "python_command", lambda *_: [str(tmp_path / "not-a-program")]
    )
    page = downloads.page
    getattr(page, button).click()
    wait_until(lambda: not page.running)
    assert "No such file" in downloads.status.call_args.args[0]
    assert page.progress.value() == 0
    for control in (page.model_button, page.speaker_button, page.delete_button):
        assert control.isEnabled()
    downloads.warmup.assert_not_called()
    downloads.information.assert_not_called()
    child(monkeypatch, "print('successful retry')")
    page.model_button.click()
    wait_until(lambda: not page.running)
    downloads.warmup.assert_called_once_with()
    assert downloads.information.call_args.args[-1] == "Model downloaded"
    assert page.progress.value() == 100


def test_model_crash_reports_failure_without_warming(monkeypatch, downloads):
    child(monkeypatch, "import os, signal; os.kill(os.getpid(), signal.SIGTERM)")
    page = downloads.page
    page.model_button.click()
    wait_until(lambda: not page.running)
    assert downloads.information.call_args.args[-1] == "Download failed"
    downloads.warmup.assert_not_called()


def test_load_model_action_does_not_start_a_download(monkeypatch, downloads):
    command = child(monkeypatch, "raise SystemExit(1)")
    button = next(
        button
        for button in downloads.page.findChildren(widgets.QPushButton)
        if button.text() == "Load model into memory"
    )
    button.click()
    downloads.warmup.assert_called_once_with()
    command.assert_not_called()
    downloads.save.assert_not_called()


def test_cache_removal_requests_confirmation_and_reports_failures(monkeypatch, downloads):
    delete = Mock()
    monkeypatch.setattr(model_downloads_ui, "delete_model", delete)
    downloads.page.delete_button.click()
    delete.assert_not_called()
    assert "example/model" in downloads.question.call_args.args[-1]

    downloads.question.return_value = widgets.QMessageBox.StandardButton.Yes
    downloads.page.delete_button.click()
    delete.assert_called_once_with("example/model", "onnx", revision="pinned")
    downloads.status.assert_called_with("Cached model deleted")
    delete.side_effect = OSError("Cache cannot be written")
    downloads.page.delete_button.click()
    downloads.status.assert_called_with("Cache cannot be written")
    downloads.save.assert_not_called()


def test_late_signals_from_failed_start_cannot_change_retry(monkeypatch, downloads):
    class DeferredProcess(qtcore.QProcess):
        def start(self, *_args):
            pass

    monkeypatch.setattr(model_downloads_ui, "QProcess", DeferredProcess)
    page = downloads.page
    page.speaker_button.click()
    speaker = page.process
    speaker.errorOccurred.emit(qtcore.QProcess.ProcessError.FailedToStart)
    assert not page.running
    page.model_button.click()
    model = page.process
    speaker.readyReadStandardError.emit()
    speaker.finished.emit(0, qtcore.QProcess.ExitStatus.NormalExit)
    speaker.errorOccurred.emit(qtcore.QProcess.ProcessError.FailedToStart)
    assert page.process is model
    downloads.warmup.assert_not_called()
    model.errorOccurred.emit(qtcore.QProcess.ProcessError.FailedToStart)
    model.readyReadStandardOutput.emit()
    model.finished.emit(0, qtcore.QProcess.ExitStatus.NormalExit)
    assert not page.running
    downloads.warmup.assert_not_called()


def test_settings_routes_catalog_and_cache_actions_to_selected_model(monkeypatch, downloads):
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    built_in = "nemo-parakeet-tdt-0.6b-v3"
    Config(engine="onnx", model="custom/model", model_revision="pinned").save()
    window = gui.Window()
    window.timer.stop()
    assert window.fields["model"].currentText() == "custom/model"
    assert window.fields["model"].findData(built_in) >= 0

    engine = window.fields["engine"]
    engine.setCurrentIndex(engine.findData("whisper"))
    assert window.fields["model"].currentText() == "custom/model"
    assert window.fields["model"].findData(built_in) == -1
    assert window.fields["model"].findData("tiny") >= 0
    engine.setCurrentIndex(engine.findData("onnx"))
    assert window.fields["model"].currentText() == "custom/model"

    browser = Mock()
    launch = Mock(return_value=browser)
    monkeypatch.setattr(models_ui, "ModelBrowser", launch)
    button = next(
        button
        for button in window.findChildren(widgets.QPushButton)
        if button.text() == "Inspect model catalog / installed state"
    )
    button.click()
    launch.assert_called_once_with(window)
    browser.exec.assert_called_once_with()
    browser.exec.side_effect = RuntimeError("Catalog unavailable")
    button.click()
    assert window.status.text() == "Catalog unavailable"

    remove = Mock()
    monkeypatch.setattr(model_downloads_ui, "delete_model", remove)
    downloads.question.return_value = widgets.QMessageBox.StandardButton.Yes
    window.delete_model()
    remove.assert_called_once_with("custom/model", "onnx", revision="pinned")
    assert window.status.text() == "Cached model deleted"
    assert not window.isVisible()


def test_settings_keeps_running_download_open_and_groups_controls_under_speech(
    monkeypatch, downloads
):

    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    child(monkeypatch, "import time; time.sleep(6)")
    window = gui.Window()
    window.timer.stop()
    page = window.model_downloads
    window.settings_category.setCurrentText("Appearance")
    assert not window.settings_options.layout.isRowVisible(page)
    window.settings_category.setCurrentText("Speech")
    assert window.settings_options.layout.isRowVisible(page)
    window.download_model()
    try:
        assert window.busy
        event = qtgui.QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()
        assert "current operation" in window.status.text()
    finally:
        page.cancel_button.click()
        wait_until(lambda: not window.busy)
    event = qtgui.QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
    assert downloads.information.call_args.args[-1] == "Download cancelled"
    assert not window.isVisible()
