import json
from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit

from mouthkeys import desktop, microphone_ui
from mouthkeys.config import Config


def page(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        microphone_ui,
        "discover",
        lambda kind: (
            [("usb", "USB microphone"), ("usb", "duplicate"), ("bt", "Bluetooth")]
            if kind == "sources"
            else [("speakers", "Speakers")]
        ),
    )
    fields = {key: QLineEdit() for key in microphone_ui.GROUPS}
    fields["audio_source"] = QComboBox()
    fields["audio_source"].addItem("default", "default")
    fields["sound_output"] = QLineEdit("default")
    return app, microphone_ui.MicrophonePage(SimpleNamespace(fields=fields)), fields


def test_priority_editor_reorders_moves_and_syncs_app_settings_without_locking_default(monkeypatch):
    Config(microphone_preferred="usb,bt", microphone_fallback="backup").save()
    app, widget, fields = page(monkeypatch)
    preferred = widget.groups["microphone_preferred"]
    preferred.setCurrentRow(1)
    widget.reorder(-1)
    widget.move_selected("microphone_never")
    widget.add_picker.setEditText("custom_source")
    widget.add_source("microphone_preferred")
    widget.sound_output.setCurrentIndex(widget.sound_output.findData("speakers"))
    assert widget.save()
    saved = Config.load()
    assert saved.microphone_preferred == "usb,custom_source"
    assert saved.microphone_never == "bt"
    assert saved.microphone_fallback == "backup"
    assert saved.audio_source == "default"  # Adding priority must not create an explicit override.
    assert saved.sound_output == "speakers"
    assert fields["microphone_preferred"].text() == saved.microphone_preferred
    assert fields["sound_output"].text() == "speakers"
    assert widget.audio_source.findData("usb") >= 0
    assert [widget.audio_source.itemData(i) for i in range(widget.audio_source.count())].count(
        "usb"
    ) == 1
    assert not widget.isVisible()
    widget.close()
    app.processEvents()


def test_priority_editor_refuses_duplicate_drop_without_overwriting_saved_config(monkeypatch):
    Config(microphone_preferred="usb").save()
    app, widget, _ = page(monkeypatch)
    widget.add_item(widget.groups["microphone_never"], "usb")
    assert not widget.save()
    assert Config.load().microphone_never == ""
    assert "only once" in widget.message.text()
    widget.close()
    app.processEvents()


def test_input_meter_uses_active_capture_only_and_locks_priority_controls(monkeypatch):
    app, widget, _ = page(monkeypatch)
    widget.update_status({"phase": "idle", "meter": {"level": 1}})
    assert widget.meter.value() == 0
    widget.update_status({"phase": "recording", "microphone": "usb", "meter": {"level": 0.62}})
    assert widget.meter.value() == 62
    assert widget.current_source.text() == "Active input: usb"
    assert not widget.save_button.isEnabled()
    assert not widget.sound_output.isEnabled()
    widget.update_status({"phase": "recording", "meter": {"level": float("nan")}})
    assert widget.meter.value() == 0
    widget.update_status({"phase": "recording", "meter": {"level": 1.25}})
    assert widget.meter.value() == 100
    widget.update_status({"phase": "recording", "meter": {"level": -0.25}})
    assert widget.meter.value() == 0
    widget.update_status({"phase": "idle"})
    assert widget.save_button.isEnabled()
    widget.close()
    app.processEvents()


def test_headless_device_discovery_never_opens_microphone_or_runs_pactl(monkeypatch):
    monkeypatch.setattr(
        desktop.subprocess,
        "run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Unexpected discovery")),
    )
    assert microphone_ui.discover("sources") == []
    assert microphone_ui.discover("sinks") == []


def test_malformed_meter_and_source_metadata_do_not_break_status_updates(monkeypatch):
    app, widget, _ = page(monkeypatch)
    for status in (
        {"phase": "recording", "microphone": 5, "meter": None},
        {"phase": "recording", "microphone": None, "meter": {"level": 10**400}},
        {"phase": "recording", "meter": {"level": True}},
    ):
        widget.update_status(status)
        assert widget.meter.value() == 0
        assert widget.current_source.text() == "Active input: default"
        assert not widget.save_button.isEnabled()
    widget.close()
    app.processEvents()


def test_priority_actions_ignore_a_current_row_after_selection_is_cleared(monkeypatch):
    Config(microphone_preferred="usb", microphone_fallback="backup").save()
    app, widget, _ = page(monkeypatch)
    preferred = widget.groups["microphone_preferred"]
    preferred.setCurrentRow(0)
    preferred.clearSelection()
    assert preferred.currentRow() == 0  # Current index and selection are independent in Qt.
    widget.reorder(1)
    widget.move_selected("microphone_never")
    widget.remove_selected()
    assert preferred.count() == 1
    assert widget.groups["microphone_never"].count() == 0
    widget.close()
    app.processEvents()


def test_refresh_keeps_manual_choices_when_pactl_is_unavailable(monkeypatch):
    Config(audio_source="offline-mic", sound_output="offline-sink").save()
    discover = microphone_ui.discover
    app, widget, fields = page(monkeypatch)
    monkeypatch.setattr(microphone_ui, "discover", discover)
    widget.add_picker.setEditText("another-mic")
    monkeypatch.setattr(microphone_ui, "session", lambda: "x11")
    sources = [
        None,
        {"name": 12},
        {"name": "usb", "description": None},
        {"name": "bt", "description": "Bluetooth"},
    ]
    monkeypatch.setattr(
        microphone_ui, "command", lambda args: json.dumps(sources if args[-1] == "sources" else {})
    )
    assert microphone_ui.discover("sources") == [("usb", "usb"), ("bt", "Bluetooth")]
    widget.refresh()
    assert "Device discovery unavailable" in widget.message.text()
    assert widget.audio_source.currentText() == "offline-mic"
    assert widget.sound_output.currentText() == "offline-sink"
    assert widget.add_picker.currentText() == "another-mic"
    fields.pop("sound_output")
    assert widget.save()
    assert fields["audio_source"].currentData() == "offline-mic"
    assert Config.load().sound_output == "offline-sink"
    widget.close()
    app.processEvents()


def test_adding_known_input_moves_it_and_only_selected_input_can_be_removed(monkeypatch):
    Config(microphone_preferred="usb,bt").save()
    app, widget, _ = page(monkeypatch)
    preferred = widget.groups["microphone_preferred"]
    fallback = widget.groups["microphone_fallback"]
    for name in ("default", "not,a,source"):
        widget.add_picker.setEditText(name)
        widget.add_source("microphone_fallback")
        assert fallback.count() == 0
        assert "not a priority entry" in widget.message.text()

    widget.add_picker.setCurrentIndex(widget.add_picker.findData("usb"))
    widget.add_source("microphone_fallback")
    assert preferred.count() == 1
    assert preferred.item(0).data(Qt.ItemDataRole.UserRole) == "bt"
    assert fallback.count() == 1
    widget.reorder(-1)
    assert fallback.item(0).data(Qt.ItemDataRole.UserRole) == "usb"
    widget.remove_selected()
    assert fallback.count() == 0
    widget.close()
    app.processEvents()


def test_edited_device_text_is_saved_without_qt_item_data(monkeypatch):
    app, widget, _ = page(monkeypatch)
    widget.audio_source.addItem("custom-mic", 42)
    widget.audio_source.setCurrentIndex(widget.audio_source.count() - 1)
    assert widget.save()
    assert Config.load().audio_source == "custom-mic"
    widget.close()
    app.processEvents()
