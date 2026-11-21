"""Read configuration values from Qt settings editors."""

from typing import Any

from PySide6.QtWidgets import QComboBox, QLineEdit, QSpinBox, QTextEdit

SettingsField = QComboBox | QLineEdit | QSpinBox | QTextEdit


def field_value(field: SettingsField) -> Any:  # noqa: ANN401 - Qt item data is a QVariant
    if isinstance(field, QSpinBox):
        return field.value()
    if isinstance(field, QLineEdit):
        return field.text()
    if isinstance(field, QTextEdit):
        return field.toPlainText()
    return field.currentText() if field.isEditable() else field.currentData()
