import pytest
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QMessageBox, QPushButton

from mouthkeys.history_ui import HistoryPage


@pytest.fixture
def history():
    calls = []
    callbacks = []

    def send(action, data, callback):
        calls.append((action, data))
        callbacks.append(callback)

    return HistoryPage(send), calls, callbacks


def click(page, text):
    button = next(button for button in page.findChildren(QPushButton) if button.text() == text)
    button.click()


def record(item_id, text, **fields):
    return {
        "id": item_id,
        "created": 1,
        "status": "copied",
        "text": text,
        "error": "",
    } | fields


def test_stale_search_results_do_not_replace_newest_query_even_if_query_repeats(history):
    page, calls, callbacks = history
    for query in ("A", "B", "A"):
        page.search.setText(query)
    page.refresh_history()
    assert calls == [("history", {"query": query, "limit": 100}) for query in ("A", "B", "A")]
    callbacks[0]({"result": [record("old-a", "Stale A")]})
    callbacks[1]({"result": [record("old-b", "Stale B")]})
    assert not page.preview.toPlainText()
    callbacks[2]({"result": [record("new-a", "Fresh A")]})
    assert page.preview.toPlainText() == "Fresh A"
    assert page.selected_id() == "new-a"


def test_history_load_more_respects_server_limit_and_can_retry_failed_query(history):
    page, calls, callbacks = history
    page.limit = 9900
    click(page, "Load more")
    click(page, "Load more")
    assert calls == [("history", {"query": "", "limit": 10_000})]
    callbacks[-1]({"error": "Unavailable"})
    page.refresh_history()
    assert calls[-1] == calls[0]
    assert len(calls) == 2
    callbacks[-1]({"result": [record("first", "Recent")]})
    assert page.preview.toPlainText() == "Recent"


def test_history_keeps_selection_by_id_and_renders_recording_detail(history):
    page, *_ = history
    first = record("first", "Hello")
    selected = record(
        "selected",
        "Edited café",
        error="Recovered",
        duration=3.5,
        app="Editor",
        window="Project",
        raw_text="Original",
    )
    page.apply_history({"result": [first, selected]})
    page.transcripts.setCurrentRow(1)
    assert page.selected_id() == "selected"
    expected = (
        "Edited café\n\nRecovered\n\n3.5s · 2 words · 11 characters"
        "\nEditor · Project\n\nOriginal transcription:\nOriginal"
    )
    assert page.preview.toPlainText() == expected
    page.apply_history({"result": [selected, first]})
    page.apply_history({"result": [selected, first]})
    assert page.selected_id() == "selected"
    assert page.transcripts.currentRow() == 0
    assert page.preview.toPlainText() == expected
    page.apply_history({"result": []})
    assert page.selected_id() is None
    assert not page.preview.toPlainText()


def test_history_ignores_invalid_rows_but_retains_older_entries_and_errors(history):
    page, *_ = history
    rows = [
        [],
        record(None, "Missing ID"),
        record("bad", "Impossible time", created=float("inf")),
        record("legacy", "", status="failed", error="Recovered from failure"),
        record("unchanged", "same", raw_text="same"),
    ]
    page.apply_history({"result": rows})
    assert page.transcripts.count() == 2
    assert page.transcripts.item(0).text().endswith("FAILED\nRecovered from failure")
    assert "Recovered from failure" in page.preview.toPlainText()
    page.transcripts.setCurrentRow(1)
    assert page.preview.toPlainText() == "same"


def test_history_only_sends_actions_for_explicit_choices(history, monkeypatch):
    page, calls, callbacks = history
    approved = QMessageBox.StandardButton.Yes
    declined = QMessageBox.StandardButton.No
    answers = iter([declined, approved, declined, approved])
    titles = []

    def confirm(_page, title, _question):
        titles.append(title)
        return next(answers)

    monkeypatch.setattr(QMessageBox, "question", confirm)
    click(page, "Copy")
    click(page, "Delete selected")
    assert not calls
    assert not titles
    page.apply_history({"result": [record("selected", "Private transcript")]})
    click(page, "Copy")
    click(page, "Delete selected")
    click(page, "Delete selected")
    click(page, "Clear history")
    click(page, "Clear history")
    assert titles == ["Delete transcript", "Delete transcript", "Clear history", "Clear history"]
    assert calls == [("copy", {"id": "selected"}), ("delete", {"id": "selected"}), ("clear", None)]
    callbacks[1]({"result": {"deleted": "selected"}})
    assert calls[-1] == ("history", {"query": "", "limit": 100})


def test_history_response_is_discarded_if_search_changed_without_a_new_request(history):
    page, calls, callbacks = history
    page.search.setText("old filter")
    with QSignalBlocker(page.search):
        page.search.setText("new filter")
    assert len(calls) == 1
    callbacks[0]({"result": [record("outdated", "Not from the displayed search")]})
    assert page.selected_id() is None
    page.refresh_history()
    assert calls[-1] == ("history", {"query": "new filter", "limit": 100})
    callbacks[-1]({"result": [record("fresh", "From displayed search")]})
    assert page.selected_id() == "fresh"
