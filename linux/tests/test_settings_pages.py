import json
import time
from threading import Event
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from PySide6.QtCore import QThread
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

from mouthkeys import __version__, backup_ui, gui, help_ui, operations, qt_control
from mouthkeys.config import state_dir
from mouthkeys.stats_view import StatsPage


def click(page, text):
    button = next(button for button in page.findChildren(QPushButton) if button.text() == text)
    button.click()


def wait_for(condition):
    deadline = time.monotonic() + 2
    while not condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.001)
    assert condition()


@pytest.mark.parametrize("accept", [True, False])
def test_settings_waits_for_preview_and_imports_only_the_reviewed_backup(monkeypatch, accept):
    monkeypatch.setattr(gui.Window, "poll", lambda self: None)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: ("chosen.json", ""))
    report = {
        "format": "upstream",
        "counts": {"history": 2, "profiles": 1, "replacements": 3},
        "warnings": ["Apple microphone IDs not imported"],
        "file_sha256": "a" * 64,
    }
    reviewed = []
    calls = []
    finish_preview = Event()
    began_import = Event()
    finish_import = Event()

    def preview(path):
        assert path == "chosen.json"
        assert finish_preview.wait(2)
        return report

    def request(action, **data):
        calls.append((action, data))
        began_import.set()
        assert finish_import.wait(2)
        return "Backup imported"

    monkeypatch.setattr(operations, "backup_preview", preview)
    monkeypatch.setattr(qt_control, "request", request)
    window = gui.Window()
    window.timer.stop()

    def review(dialog):
        assert QThread.currentThread() == QApplication.instance().thread()
        reviewed.append(json.loads(dialog.detailedText()))
        assert not dialog.isVisible()
        assert "2 transcripts" in dialog.text()
        assert "1 migration warnings" in dialog.informativeText()
        assert dialog.defaultButton() == dialog.button(QMessageBox.StandardButton.No)
        # A modal review can process the finished signal before the choice is made.
        wait_for(lambda: not window.backup_page.jobs)
        assert window.busy
        close = QCloseEvent()
        window.closeEvent(close)
        assert not close.isAccepted()
        return QMessageBox.StandardButton.Yes if accept else QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "exec", review)
    try:
        click(window.backup_page, "Import backup")
        assert window.busy
        before_review = QCloseEvent()
        window.closeEvent(before_review)
        assert not before_review.isAccepted()
        assert not calls
        finish_preview.set()
        if accept:
            wait_for(began_import.is_set)
            close_during_import = QCloseEvent()
            window.closeEvent(close_during_import)
            assert not close_during_import.isAccepted()
            finish_import.set()
        wait_for(lambda: not window.busy)
        if accept:
            assert window.status.text() == "Backup imported"
        assert reviewed == [report]
        assert calls == (
            [("backup_import", {"path": "chosen.json", "expected_sha256": "a" * 64})]
            if accept
            else []
        )
    finally:
        finish_preview.set()
        finish_import.set()
        wait_for(lambda: not window.busy)
        window.close()


def test_cancelled_backup_picker_never_starts_work_and_export_returns_feedback(monkeypatch):
    calls = []
    notices = []
    results = []

    def send(action, data, callback):
        calls.append((action, data))
        callback({"result": "Archived"})

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: ("", ""))
    choices = iter([("", ""), ("archive.json", "")])
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: next(choices))
    page = backup_ui.BackupPage(send, notices.append, results.append)
    click(page, "Import backup")
    click(page, "Export backup")
    assert not page.running
    assert not calls
    click(page, "Export backup")
    assert calls == [("backup_export", {"path": "archive.json"})]
    assert results == [{"result": "Archived"}]
    assert not notices


@pytest.mark.parametrize(
    ("response", "notice"),
    [
        ({"error": "Backup was unreadable"}, "Backup was unreadable"),
        ({"result": None}, "Backup preview unavailable"),
        ({"result": {"format": "native"}}, "Backup preview unavailable"),
    ],
)
def test_failed_backup_preview_never_asks_to_apply(monkeypatch, response, notice):
    notices = []
    page = backup_ui.BackupPage(
        lambda *a: pytest.fail("Backup import sent"),
        notices.append,
        lambda _: pytest.fail("Unexpected result"),
    )
    monkeypatch.setattr(QMessageBox, "exec", lambda _: pytest.fail("Unexpected backup review"))
    page.review_import("invalid.json", response)
    assert notices == [notice]
    assert not page.running


def test_older_backup_summary_defaults_missing_optional_counts_and_warnings(monkeypatch):
    texts = []
    page = backup_ui.BackupPage(
        lambda *a: pytest.fail("Backup import sent"),
        lambda _: pytest.fail("Unexpected error"),
        lambda _: pytest.fail("Unexpected result"),
    )

    def review(dialog):
        texts.append((dialog.text(), dialog.informativeText()))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "exec", review)
    page.review_import("older.json", {"result": {"format": "native", "file_sha256": "a" * 64}})
    assert len(texts) == 1
    assert texts[0][0] == "Native backup: 0 transcripts, 0 profiles, 0 replacement rules."
    assert texts[0][1].startswith("0 migration warnings.")


def test_activity_request_updates_local_totals_and_selected_chart_range():
    calls = []
    responses = []

    def send(action, data, callback):
        calls.append((action, data))
        responses.append(callback)

    result = {
        "words": 1234,
        "transcriptions": 21,
        "current_streak": 4,
        "time_saved_seconds": 150,
        "windows": {"7 days": {"words": 567, "transcriptions": 6}},
        "longest_words": 123,
        "most_words_in_day": 444,
        "average_words": 58.8,
        "daily_words": {"2026-11-08": 500, "malformed": "not a count", "negative": -1},
        "top_apps": [["Editor", 10]],
        "peak_hour": [[9, 5]],
        "best_streak": 7,
        "milestones": {
            "Words": [{"target": 1000, "achieved": True}, {"target": 10000, "achieved": False}],
            "Invalid milestone": [None, {"target": "not an integer"}],
            "Invalid list": "not a list",
        },
    }
    page = StatsPage(send)
    assert not calls
    page.chart_range.setCurrentText("Last 30 days")
    click(page, "Refresh stats")
    assert calls == [("stats", None)]
    (receive,) = responses
    receive({"result": result})
    assert page.activity_chart.days == 30
    assert page.activity_chart.values == {"2026-11-08": 500}
    text = page.stats_text.toPlainText()
    for line in (
        "1,234 words",
        "21 dictations",
        "4 day streak",
        "2.5 minutes saved",
        "7 days: 567 words / 6 dictations",
        "Longest dictation: 123 words",
        "Average length: 58.8 words",
        "2026-11-08: 500",
        "Top apps: [['Editor', 10]]",
        "Best streak: 7 days",
        "Words milestones: 1,000 achieved · 10,000",
    ):
        assert line in text
    assert "Invalid" not in text
    page.chart_range.setCurrentText("Last 7 days")
    assert page.activity_chart.days == 7
    assert page.activity_chart.values == {"2026-11-08": 500}
    receive({"error": "Unavailable"})
    assert page.stats_text.toPlainText() == text


def test_activity_page_handles_missing_optional_fields_and_ignores_invalid_response():
    page = StatsPage(lambda *a: pytest.fail("Unexpected request"))
    page.show_stats({"result": []})
    assert not page.stats_text.toPlainText()
    page.show_stats({"result": {"words": 25, "windows": {"7 days": None}, "daily_words": {5: 10}}})
    assert page.activity_chart.values == {}
    assert "25 words  ·  0 dictations" in page.stats_text.toPlainText()
    assert "Best streak: 0 days" in page.stats_text.toPlainText()


def test_diagnostics_reads_only_bounded_log_tail_and_handles_missing_or_unreadable_files(tmp_path):
    page = help_ui.HelpPage()
    path = state_dir() / "daemon.log"
    click(page, "Show diagnostic log")
    assert page.log_text.toPlainText() == "No diagnostic log yet"

    tail = b"." * (help_ui.LOG_TAIL_BYTES - 2) + b"\xff!"
    path.write_bytes(b"old record\n" + tail)
    click(page, "Show diagnostic log")
    text = page.log_text.toPlainText()
    assert len(text) == help_ui.LOG_TAIL_BYTES
    assert text.endswith("\ufffd!")
    assert "old record" not in text

    unreadable = tmp_path / "rotated"
    path.rename(unreadable)
    path.mkdir()
    click(page, "Show diagnostic log")
    assert page.log_text.toPlainText() == "Could not read the diagnostic log"


def test_support_links_open_only_on_explicit_click_without_attaching_log(monkeypatch):
    opened = []
    monkeypatch.setattr(help_ui, "QDesktopServices", SimpleNamespace(openUrl=opened.append))
    monkeypatch.setattr(help_ui.platform, "release", lambda: "example-kernel")
    page = help_ui.HelpPage()
    assert not opened
    click(page, "Report issue")
    click(page, "Open Linux guide")
    report, guide = (urlsplit(url.toString()) for url in opened)
    assert report.netloc == guide.netloc == "github.com"
    assert report.path == "/crypdick/MouthKeys-Linux/issues/new"
    assert parse_qs(report.query) == {
        "body": [f"MouthKeys {__version__}\nLinux example-kernel\n\nDescribe issue:\n"],
    }
    assert guide.path == "/crypdick/MouthKeys-Linux/blob/main/linux/README.md"
