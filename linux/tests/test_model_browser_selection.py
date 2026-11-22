"""Model actions only apply to the visible, selected catalog row."""

from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QComboBox, QLineEdit, QWidget

from mouthkeys import engines
from mouthkeys.models_ui import ModelBrowser


@pytest.fixture
def model_window(monkeypatch):
    monkeypatch.setattr(
        engines,
        "model_catalog",
        lambda *args, **kwargs: [
            {"name": "Small model", "model": "small", "languages": "English", "installed": True},
        ],
    )
    window = QWidget()
    engine = QComboBox()
    engine.addItem("Whisper", "whisper")
    model = QComboBox()
    model.setEditable(True)
    model.setEditText("previous")
    window.fields = {"engine": engine, "model": model, "model_revision": QLineEdit("main")}
    window.loaded_model = None
    window.delete_model = Mock()
    window.download_model = Mock()
    return window


@pytest.mark.parametrize("selection", ["cleared", "filtered"])
def test_action_ignores_a_current_row_without_visible_selection(model_window, selection):
    browser = ModelBrowser(model_window)
    browser.timer.stop()
    browser.table.setCurrentCell(0, 0)
    assert browser.table.currentRow() == 0
    if selection == "cleared":
        browser.table.clearSelection()
    else:
        browser.query.setText("different engine")
        assert browser.table.isRowHidden(0)
    browser.delete()
    model_window.delete_model.assert_not_called()
    assert model_window.fields["model"].currentText() == "previous"
    assert not browser.isVisible()


def test_malformed_resident_model_is_not_treated_as_an_active_catalog_row(model_window):
    model_window.loaded_model = {"engine": "whisper"}
    browser = ModelBrowser(model_window)
    browser.timer.stop()
    assert browser.resident.text() == "Resident model: none reported"
    assert browser.table.item(0, 6).text() == "No"
