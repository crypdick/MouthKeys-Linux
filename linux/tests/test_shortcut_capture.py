from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QDialog, QWidget

from mouthkeys.config import config_path
from mouthkeys.shortcut_capture import ShortcutCapture


def test_shortcut_capture_rejects_typing_then_updates_only_selected_editor_field():
    changes = []
    parent = QWidget()
    dialog = ShortcutCapture(parent, lambda key, chord: changes.append((key, chord)))
    dialog.target.setCurrentIndex(dialog.target.findData("cancel_shortcut"))
    dialog.sequence.setKeySequence(QKeySequence("A"))
    dialog.apply()
    assert "require a modifier" in dialog.feedback.text()
    assert not changes
    assert dialog.result() != QDialog.DialogCode.Accepted

    dialog.sequence.setKeySequence(QKeySequence("Ctrl+F8"))
    dialog.apply()
    assert changes == [("cancel_shortcut", "ctrl+f8")]
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert not config_path().exists()
    assert not parent.isVisible()
