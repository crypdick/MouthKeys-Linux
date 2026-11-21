import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mouthkeys.dictionary import Dictionary
from mouthkeys.vocabulary import VocabularyStore

widgets = pytest.importorskip("PySide6.QtWidgets")
dictionary_ui = import_module("mouthkeys.dictionary_ui")


@pytest.fixture
def editor():
    send = Mock()
    messages = []
    page = dictionary_ui.DictionaryPage(send, messages.append)
    return page, send, messages


def click(page, label):
    next(
        button for button in page.findChildren(widgets.QPushButton) if button.text() == label
    ).click()


def proposal(identity, text, mode="training"):
    return {"id": identity, "assistant_proposal": {"mode": mode, "text": text}}


def test_training_only_stages_matching_result_once(editor):
    page, send, messages = editor
    click(page, "Train by voice")
    send.assert_not_called()
    assert messages[-1].startswith("Enter the correct replacement")

    page.training_replacement.setText("  MouthKeys  ")
    page.tabs.setCurrentIndex(2)
    click(page, "Train by voice")
    send.assert_called_once_with("start", {"mode": "training"}, page.training_started)
    started = send.call_args.args[2]
    started({"error": "not running"})
    started({"result": {"id": 7}})
    page.update_training(proposal("training", "mouth keys."))
    assert page.tables["replacements"].rowCount() == 0

    started({"result": {"id": "training"}})
    click(page, "Stop training")
    send.assert_called_with("stop", None, None)
    page.update_training(proposal("other", "unrelated"))
    page.update_training({"id": "training", "assistant_proposal": None})
    page.update_training(proposal("training", "unrelated", mode="prompt"))
    assert page.tables["replacements"].rowCount() == 0
    for _ in range(2):
        page.update_training(proposal("training", "  mouth keys!?  "))

    table = page.tables["replacements"]
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == "mouth keys"
    assert table.item(0, 1).text() == "MouthKeys"
    assert page.tabs.currentWidget() is table
    assert messages[-1].startswith("Training captured")
    assert Dictionary.load()["replacements"] == []
    click(page, "Save")
    assert Dictionary.apply("say mouth keys") == "say MouthKeys"


@pytest.mark.parametrize("heard", [None, "..."])
def test_empty_training_result_does_not_create_a_rule(editor, heard):
    page, _, messages = editor
    page.training_started({"result": {"id": "training"}})
    page.update_training(proposal("training", heard))
    assert page.tables["replacements"].rowCount() == 0
    assert messages[-1] == "No recognized training phrase; try again"


def test_add_edit_delete_and_save_rules(editor, monkeypatch):
    page, _, messages = editor
    table = page.tables["replacements"]
    edit = Mock()
    monkeypatch.setattr(table, "editItem", edit)
    click(page, "Delete rule")
    click(page, "Add rule")
    edit.assert_called_once_with(table.item(0, 0))
    click(page, "Delete rule")
    assert table.rowCount() == 0
    click(page, "Add rule")
    table.item(0, 1).setText("MouthKeys")
    table.item(0, 2).setText("yes")
    assert not page.save()
    assert messages[-1] == "Enabled must be on or off"

    table.item(0, 2).setText("off")
    table.takeItem(0, 0)
    assert not page.save()
    assert messages[-1] == "Invalid replacements entry"
    table.setItem(0, 0, widgets.QTableWidgetItem(" mouth keys, , mouse keys "))
    actions = page.tables["actions"]
    actions.item(0, 1).setText(r"\\\n\t")
    assert page.save()
    saved = Dictionary.load()
    assert saved["replacements"] == [
        {"triggers": ["mouth keys", "mouse keys"], "replacement": "MouthKeys", "enabled": False}
    ]
    assert saved["actions"][0]["symbol"] == "\\\n\t"
    click(page, "Reload")
    assert actions.item(0, 1).text() == r"\\\n\t"
    assert table.item(0, 2).text() == "off"

    page.tabs.clear()
    click(page, "Add rule")
    click(page, "Delete rule")
    assert table.rowCount() == 1


def test_save_does_not_overwrite_update_arriving_during_save(editor, monkeypatch):
    page, _, messages = editor
    page.tables["actions"].item(0, 1).setText("draft")
    current = Dictionary.load()
    current["replacements"] = [{"triggers": ["wurld"], "replacement": "world"}]
    original_save = Dictionary.save

    def concurrent_update(data, **kwargs):
        original_save(current)
        original_save(data, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Dictionary, "save", concurrent_update)
        assert not page.save()

    assert "Reload before saving" in messages[-1]
    assert page.tables["actions"].item(0, 1).text() == "draft"
    assert Dictionary.load() == current
    click(page, "Reload")
    page.tables["actions"].item(0, 1).setText("draft")
    assert page.save()
    assert Dictionary.apply("Hello wurld") == "Hello world"
    assert Dictionary.load()["actions"][0]["symbol"] == "draft"


def test_reset_restores_only_formatting_and_respects_cancel(editor, monkeypatch):
    page, _, messages = editor
    current = Dictionary.defaults()
    current["actions"] = []
    current["replacements"] = [{"triggers": ["wurld"], "replacement": "world"}]
    Dictionary.save(current)
    answer = Mock(return_value=widgets.QMessageBox.StandardButton.No)
    monkeypatch.setattr(
        dictionary_ui,
        "QMessageBox",
        SimpleNamespace(question=answer, StandardButton=widgets.QMessageBox.StandardButton),
    )

    click(page, "Reset spoken formatting")
    assert Dictionary.load() == current
    answer.return_value = widgets.QMessageBox.StandardButton.Yes
    click(page, "Reset spoken formatting")
    assert Dictionary.load() == {**Dictionary.defaults(), "replacements": current["replacements"]}
    assert page.tables["replacements"].item(0, 1).text() == "world"

    Dictionary.path().write_text("{")
    click(page, "Reload")
    assert "Expecting property name" in messages[-1]
    click(page, "Reset spoken formatting")
    assert "Expecting property name" in messages[-1]
    assert Dictionary.path().read_text() == "{"


@pytest.mark.parametrize(
    ("choice", "words"),
    [("Merge with existing data", ["First", "Second"]), ("Replace supplied sections", ["Second"])],
)
def test_import_updates_rules_and_notifies_vocabulary_page(
    editor, monkeypatch, tmp_path, choice, words
):
    page, _, messages = editor
    refresh = Mock()
    page.vocabulary_changed.connect(refresh)
    VocabularyStore.save(["First"])
    path = tmp_path / "import.json"
    chooser = Mock(return_value=("", ""))
    mode = Mock(return_value=(choice, False))
    monkeypatch.setattr(dictionary_ui, "QFileDialog", SimpleNamespace(getOpenFileName=chooser))
    monkeypatch.setattr(dictionary_ui, "QInputDialog", SimpleNamespace(getItem=mode))
    click(page, "Import")
    mode.assert_not_called()
    chooser.return_value = (str(path), "")
    click(page, "Import")
    assert not messages
    refresh.assert_not_called()

    mode.return_value = (choice, True)
    path.write_text("{}")
    click(page, "Import")
    assert messages[-1].startswith("Expected replacements")
    refresh.assert_not_called()
    path.write_text(
        json.dumps(
            {
                "replacements": [{"triggers": ["wurld"], "replacement": "world"}],
                "customWords": ["Second"],
            }
        )
    )
    click(page, "Import")
    assert messages[-1] == "Dictionary and supplied vocabulary imported"
    assert page.tables["replacements"].item(0, 1).text() == "world"
    assert Dictionary.apply("Hello wurld") == "Hello world"
    assert VocabularyStore.load() == words
    refresh.assert_called_once_with()


@pytest.mark.parametrize(("choice", "includes_actions"), [("Linux", True), ("Upstream", False)])
def test_export_saves_edits_and_writes_selected_format(
    editor, monkeypatch, tmp_path, choice, includes_actions
):
    page, _, messages = editor
    path = tmp_path / "export.json"
    chooser = Mock(return_value=("", ""))
    format_choice = Mock(return_value=(choice, False))
    monkeypatch.setattr(dictionary_ui, "QFileDialog", SimpleNamespace(getSaveFileName=chooser))
    monkeypatch.setattr(dictionary_ui, "QInputDialog", SimpleNamespace(getItem=format_choice))
    VocabularyStore.save(["Example"])
    actions = page.tables["actions"]
    actions.item(0, 1).setText("draft")
    click(page, "Export")
    assert not Dictionary.path().exists()

    chooser.return_value = (str(path), "")
    actions.item(0, 2).setText("invalid")
    click(page, "Export")
    format_choice.assert_not_called()
    assert messages[-1] == "Enabled must be on or off"
    actions.item(0, 2).setText("on")
    click(page, "Export")
    assert not path.exists()
    assert Dictionary.load()["actions"][0]["symbol"] == "draft"

    format_choice.return_value = (choice, True)
    click(page, "Export")
    payload = json.loads(path.read_text())
    assert ("actions" in payload) is includes_actions
    assert payload["customWords"] == [{"text": "Example", "weight": 1, "aliases": []}]
    assert messages[-1] == "Dictionary and vocabulary exported"

    chooser.return_value = (str(tmp_path), "")
    click(page, "Export")
    assert str(tmp_path) in messages[-1]
