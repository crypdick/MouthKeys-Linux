from dataclasses import replace

import pytest
from PySide6.QtWidgets import QInputDialog, QLineEdit, QMessageBox, QPushButton

from mouthkeys import credentials, gui
from mouthkeys.config import Config, config_path


def click(parent, label):
    next(b for b in parent.findChildren(QPushButton) if b.text() == label).click()


@pytest.fixture
def pending_tasks(monkeypatch):
    tasks = []
    monkeypatch.setattr(gui.Window, "poll", lambda _: None)
    monkeypatch.setattr(gui.Window, "run_task", lambda _, run, done: tasks.append((run, done)))
    return tasks


@pytest.mark.parametrize("edit", ["endpoint", "provider", "error", "cancelled", "invalid"])
def test_keyring_storage_does_not_change_the_editor_for_another_endpoint_or_failed_request(
    monkeypatch, pending_tasks, edit
):
    initial = Config()
    calls = []
    opened = []

    def dialog(_parent, _title, _label, echo):
        assert echo == QLineEdit.EchoMode.Password
        opened.append(True)
        return ("mock-credential", edit != "cancelled")

    monkeypatch.setattr(QInputDialog, "getText", dialog)
    monkeypatch.setattr(credentials, "store_key", lambda config, _key: calls.append(config))
    window = gui.Window()
    window.timer.stop()
    try:
        endpoint = window.fields["ai_endpoint"]
        provider = window.fields["ai_provider"]
        source = window.fields["ai_credential_source"]
        if edit == "invalid":
            endpoint.setText("")
        click(window, "Save API key in desktop keyring")

        if edit in ("invalid", "cancelled"):
            assert not pending_tasks
            assert opened == ([True] if edit == "cancelled" else [])
            if edit == "invalid":
                assert "ai_endpoint must be a nonempty string" in window.status.text()
        else:
            assert not calls
            [(store, done)] = pending_tasks
            if edit == "endpoint":
                endpoint.setText("http://localhost:23456/v1/chat/completions")
            if edit == "provider":
                provider.setCurrentIndex(provider.findData("ollama"))
            if edit == "error":
                done({"error": "Desktop keyring locked"})
                assert window.status.text() == "Desktop keyring locked"
            else:
                store()
                done({"result": None})
                assert calls == [initial]
                assert window.status.text() == (
                    "API key saved for earlier endpoint; current editor unchanged"
                )
        assert source.currentData() == "environment"
        assert not config_path().exists()
        assert not window.isVisible()
    finally:
        window.close()


def test_delete_keyring_entry_uses_draft_endpoint_without_saving_settings(
    monkeypatch, pending_tasks
):
    config = Config(ai_credential_source="secret-service")
    config.save()
    calls = []
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(credentials, "delete_key", calls.append)
    window = gui.Window()
    window.timer.stop()
    try:
        changed_endpoint = "https://settings.test/v1/chat/completions"
        window.fields["ai_endpoint"].setText(changed_endpoint)
        click(window, "Delete API key from desktop keyring")
        assert not calls
        [(delete, done)] = pending_tasks
        delete()
        done({"result": None})
        assert calls == [replace(config, ai_endpoint=changed_endpoint)]
        assert window.status.text() == "API key deleted from desktop keyring"
        assert window.fields["ai_credential_source"].currentData() == "secret-service"
        assert Config.load() == config
        assert not window.isVisible()
    finally:
        window.close()
