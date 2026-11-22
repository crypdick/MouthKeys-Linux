"""Exercise the Learning page with an offscreen UI and a fake command sender."""

from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QMessageBox, QPushButton

from mouthkeys.learning_ui import LearningPage
from mouthkeys.pronunciation import FEATURE_VERSION, PronunciationStore
from mouthkeys.vocabulary import VocabularyStore


def click(page, label):
    next(button for button in page.findChildren(QPushButton) if button.text() == label).click()


def test_vocabulary_save_does_not_overwrite_concurrent_learning(monkeypatch):
    VocabularyStore.save(["Original"])
    page = LearningPage(Mock())
    page.vocabulary.item(0, 0).setText("Edited")
    write = VocabularyStore.save_entries

    def write_after_learning(entries, **options):
        write([*VocabularyStore.entries(), {"text": "Just learned", "weight": 8, "aliases": []}])
        return write(entries, **options)

    monkeypatch.setattr(VocabularyStore, "save_entries", write_after_learning)
    click(page, "Save vocabulary")
    assert VocabularyStore.load() == ["Original", "Just learned"]
    assert "Refresh before saving" in page.feedback.text()
    assert page.vocabulary.item(0, 0).text() == "Edited"


def test_take_buttons_never_send_without_owned_enrollment():
    send = Mock()
    page = LearningPage(send)
    click(page, "Stop take")
    click(page, "Cancel")
    send.assert_not_called()


def test_new_vocabulary_rows_save_aliases_and_support_removal():
    page = LearningPage(Mock())
    click(page, "Remove word")
    click(page, "Add word")
    page.vocabulary.item(0, 0).setText("New word")
    page.vocabulary.cellWidget(0, 1).setText("  ")
    page.vocabulary.item(0, 2).setText(" variant, , other ")
    click(page, "Save vocabulary")
    assert VocabularyStore.entries() == [
        {"text": "New word", "weight": None, "aliases": ["variant", "other"]}
    ]
    assert "Vocabulary saved" in page.feedback.text()
    page.vocabulary.setCurrentCell(0, 0)
    click(page, "Remove word")
    click(page, "Save vocabulary")
    assert VocabularyStore.load() == []


def test_vocabulary_reports_draft_errors_without_writing():
    VocabularyStore.save(["Original"])
    page = LearningPage(Mock())
    page.vocabulary.cellWidget(0, 1).setText("not numeric")
    click(page, "Save vocabulary")
    assert "could not convert" in page.feedback.text()
    assert VocabularyStore.load() == ["Original"]
    page.vocabulary.takeItem(0, 0)
    click(page, "Save vocabulary")
    assert page.feedback.text() == "Vocabulary row is incomplete"
    click(page, "Refresh vocabulary")
    assert page.vocabulary.item(0, 0).text() == "Original"


def proposal(id_, corrected):
    return {"id": id_, "heard": "example", "corrected": corrected}


def test_reviews_keep_selection_by_id_and_never_dispatch_a_replacement():
    send = Mock()
    page = LearningPage(send)
    first, second = proposal("a", "First"), proposal("b", "Second")
    page.update_status({"learning_pending": [None, {"heard": "incomplete"}, first, second]})
    assert page.proposals.rowCount() == 2
    assert page.proposals.item(0, 1).text() == "First"
    page.proposals.setCurrentCell(0, 0)
    page.update_status({"learning_pending": [second, first]})
    assert page.proposals.currentRow() == 1
    click(page, "Accept correction")
    send.assert_called_once_with("learning_accept", {"id": "a"}, None)
    send.reset_mock()
    page.update_status({"learning_pending": [second]})
    assert page.proposals.currentRow() == -1
    click(page, "Dismiss correction")
    send.assert_not_called()
    page.proposals.setCurrentCell(0, 0)
    page.update_status({"learning_pending": [second]})
    click(page, "Dismiss correction")
    send.assert_called_once_with("learning_dismiss", {"id": "b"}, None)


def test_take_cancel_waits_for_own_start_ack_and_stop_keeps_identity():
    send = Mock()
    page = LearningPage(send)
    page.label.setText("phrase")
    click(page, "Record pronunciation take")
    click(page, "Record pronunciation take")
    click(page, "Cancel")
    send.assert_called_once()
    action, fields, callback = send.call_args.args
    assert action == "start"
    assert fields == {"mode": "pronunciation", "label": "phrase"}
    callback({"result": {"id": "pronunciation-job"}})
    assert "Listening" in page.feedback.text()
    send.assert_called_with("cancel", {"id": "pronunciation-job"}, None)
    page.update_status({"id": "unrelated", "phase": "idle"})
    page.update_status({"id": "pronunciation-job", "phase": "recording"})
    click(page, "Stop take")
    send.assert_called_with("stop", {"id": "pronunciation-job"}, None)
    page.update_status({"id": "pronunciation-job", "phase": "idle", "message": "Take cancelled"})
    assert page.feedback.text() == "Take cancelled"
    send.reset_mock()
    click(page, "Stop take")
    send.assert_not_called()


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ({"error": "Microphone busy"}, "Microphone busy"),
        ({"result": []}, "Cannot start pronunciation take"),
        ({"result": {"id": ""}}, "Cannot start pronunciation take"),
    ],
)
def test_failed_start_drops_queued_take_action(reply, message):
    send = Mock()
    page = LearningPage(send)
    click(page, "Record pronunciation take")
    click(page, "Stop take")
    send.call_args.args[2](reply)
    assert page.feedback.text() == message
    assert page.enrollment_id is None
    send.assert_called_once()


def test_saved_take_refreshes_profiles_and_delete_requires_confirmation(monkeypatch):
    send = Mock()
    page = LearningPage(send)
    question = Mock(return_value=QMessageBox.StandardButton.No)
    monkeypatch.setattr(QMessageBox, "question", question)
    click(page, "Delete selected pronunciation")
    question.assert_not_called()

    sample = {"duration": 0.5, "features": [[0.0] * 12]}
    PronunciationStore.save(
        [
            {"label": "First", "version": FEATURE_VERSION, "samples": [sample]},
            {"label": "Second", "version": FEATURE_VERSION, "samples": [sample, sample]},
        ]
    )
    click(page, "Record pronunciation take")
    send.call_args.args[2]({"result": {"id": "take"}})
    page.update_status({"id": "take", "assistant_proposal": {"mode": "pronunciation"}})
    assert page.feedback.text().startswith("Take saved")
    assert "needs another take" in page.profiles.itemText(0)
    assert page.profiles.itemText(1).endswith("ready")

    page.profiles.setCurrentIndex(1)
    click(page, "Delete selected pronunciation")
    assert page.profiles.count() == 2
    question.return_value = QMessageBox.StandardButton.Yes
    click(page, "Delete selected pronunciation")
    assert [profile["label"] for profile in PronunciationStore.load()] == ["First"]
    assert page.profiles.count() == 1
    assert "Second" in question.call_args.args[2]
