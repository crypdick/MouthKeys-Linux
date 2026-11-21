import json
import time
from threading import Event
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QWidget,
)

from mouthkeys import benchmark, engines, models_ui
from mouthkeys.config import Config


@pytest.fixture
def browser(monkeypatch):
    host = QWidget()
    engine = QComboBox()
    engine.addItem("Whisper", "whisper")
    model = QComboBox()
    model.setEditable(True)
    host.fields = {"engine": engine, "model": model, "model_revision": QLineEdit("keep")}
    host.jobs = set()
    calls = []
    host.send = calls.append
    host.save = lambda: True
    host.download_model = lambda: calls.append("download")
    host.delete_model = lambda: calls.append("delete")
    catalog = [
        {
            "model": "small.en",
            "name": "Installed",
            "languages": "en",
            "installed": True,
            "revision": "pinned",
            "weight_bytes": 123,
            "measurement": {"decode_seconds": 1},
        },
        {
            "model": "base",
            "name": "Available",
            "languages": "multilingual",
            "installed": False,
        },
    ]
    monkeypatch.setattr(engines, "model_catalog", lambda *args, **kwargs: catalog)
    dialog = models_ui.ModelBrowser(host)
    dialog.timer.stop()
    dialog.table.clearSelection()
    dialog.table.setCurrentCell(-1, -1)
    return dialog, host, calls


def select(dialog, model):
    index = next(
        index
        for index in range(dialog.table.rowCount())
        if dialog.table.item(index, 0).data(Qt.ItemDataRole.UserRole)["model"] == model
    )
    dialog.table.setCurrentCell(index, 0)
    return index


def wait_for(condition):
    deadline = time.monotonic() + 2
    while not condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.001)
    assert condition()


def test_model_actions_require_a_selection_and_respect_unsaved_settings(browser):
    dialog, host, calls = browser
    dialog.activate()
    dialog.download()
    dialog.delete()
    assert not calls

    installed = select(dialog, "small.en")
    assert "decode_seconds" in dialog.table.item(installed, 0).toolTip()
    host.save = lambda: False
    dialog.activate()
    assert not calls
    assert host.fields["model"].currentText() == "small.en"
    assert host.fields["model_revision"].text() == "pinned"

    available = select(dialog, "base")
    dialog.state.setCurrentText("Not downloaded")
    assert dialog.table.isRowHidden(installed)
    assert not dialog.table.isRowHidden(available)
    dialog.state.setCurrentText("All models")
    dialog.activate()
    assert not calls
    assert host.fields["model_revision"].text() == "pinned"

    host.save = lambda: True
    dialog.activate()
    dialog.download()
    dialog.delete()
    assert calls == ["warmup", "download", "delete"]


@pytest.mark.parametrize(
    "stage", ["no selection", "not installed", "no file", "no reference", "invalid settings"]
)
def test_declined_benchmark_does_not_start_or_download_anything(browser, monkeypatch, stage):
    dialog, host, calls = browser
    logs = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: logs.append(args[-1]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: logs.append(args[-1]))
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args: ("" if stage == "no file" else "local.wav", ""),
    )
    monkeypatch.setattr(
        QInputDialog, "getMultiLineText", lambda *args: ("", stage != "no reference")
    )
    monkeypatch.setattr(
        models_ui, "QProgressDialog", lambda *args: pytest.fail("Benchmark started")
    )

    def invalid(_self):
        raise ValueError("Invalid benchmark settings")

    if stage == "invalid settings":
        monkeypatch.setattr(Config, "validate", invalid)
    if stage != "no selection":
        select(dialog, "base" if stage == "not installed" else "small.en")
    dialog.benchmark()
    expected = {
        "not installed": ["Download this model explicitly first."],
        "invalid settings": ["Invalid benchmark settings"],
    }
    assert logs == expected.get(stage, [])
    assert not calls
    assert not host.jobs
    assert dialog.benchmark_task is None


@pytest.mark.parametrize("cancel", [False, True])
def test_model_benchmark_task_tracks_progress_cancellation_and_result(browser, monkeypatch, cancel):
    dialog, host, calls = browser
    select(dialog, "small.en")
    began = Event()
    complete = Event()
    progresses = []

    class Progress:
        def __init__(self, *_):
            self.closed = False
            self.deleted = False
            self.visible = False
            self.cancel_callbacks = []
            self.canceled = SimpleNamespace(connect=self.cancel_callbacks.append)
            progresses.append(self)

        def setWindowModality(self, value):  # noqa: N802 - Qt interface
            assert value == Qt.WindowModality.WindowModal

        def show(self):
            self.visible = True

        def close(self):
            self.closed = True

        def deleteLater(self):  # noqa: N802 - Qt interface
            self.deleted = True

    monkeypatch.setattr(models_ui, "QProgressDialog", Progress)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("local.wav", ""))
    monkeypatch.setattr(
        QInputDialog, "getMultiLineText", lambda *args: (" " if cancel else "private", True)
    )
    warnings = []
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    report = {
        "fixture_sha256": "fixture",
        "accuracy": {"reference_sha256": "reference"},
        "decode_seconds": 0.5,
        "audio_seconds": 1,
        "real_time_factor": 0.5,
        "load_seconds": 2,
        "scope": "Local fixture",
        "memory_scope": "Process tree",
    }

    def show_result(box):
        assert not box.isVisible()
        shown.append(json.loads(box.detailedText()))
        return QMessageBox.StandardButton.Ok

    def run(config, fixture, *, reference, cancel_event):
        assert (config.engine, config.model, config.model_revision) == (
            "whisper",
            "small.en",
            "pinned",
        )
        assert fixture == "local.wav"
        assert reference == (None if cancel else "private")
        began.set()
        assert complete.wait(2)
        assert cancel_event.is_set() is cancel
        if cancel:
            raise RuntimeError("Benchmark cancelled")
        return report

    monkeypatch.setattr(QMessageBox, "exec", show_result)
    monkeypatch.setattr(benchmark, "run_benchmark", run)
    try:
        dialog.benchmark()
        wait_for(began.is_set)
        task = dialog.benchmark_task
        assert task in host.jobs
        dialog.benchmark()
        (progress,) = progresses
        assert progress.visible
        close = QCloseEvent()
        dialog.closeEvent(close)
        assert not close.isAccepted()
        if cancel:
            progress.cancel_callbacks[0]()
        complete.set()
        wait_for(lambda: dialog.benchmark_task is None)
        assert progress.closed
        assert progress.deleted
        assert not host.jobs
        assert not calls
        assert warnings == (["Benchmark cancelled"] if cancel else [])
        assert shown == ([] if cancel else [report])
        assert dialog.measurement_context == (None if cancel else ("fixture", "reference"))
    finally:
        complete.set()
        wait_for(lambda: dialog.benchmark_task is None)
        dialog.close()
