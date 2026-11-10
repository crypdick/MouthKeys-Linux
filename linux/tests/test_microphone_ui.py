from types import SimpleNamespace

from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit

from mouthkeys import microphone_ui
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
    app, widget, fields = page(monkeypatch)
    widget.add_item(widget.groups["microphone_never"], "usb")
    assert not widget.save()
    assert Config.load().microphone_never == ""
    assert "only once" in widget.message.text()
    widget.close()
    app.processEvents()


def test_input_meter_uses_active_capture_only_and_locks_priority_controls(monkeypatch):
    app, widget, fields = page(monkeypatch)
    widget.update_status({"phase": "idle", "meter": {"level": 1}})
    assert widget.meter.value() == 0
    widget.update_status({"phase": "recording", "microphone": "usb", "meter": {"level": 0.62}})
    assert widget.meter.value() == 62
    assert widget.current_source.text() == "Active input: usb"
    assert not widget.save_button.isEnabled()
    assert not widget.sound_output.isEnabled()
    widget.update_status({"phase": "recording", "meter": {"level": float("nan")}})
    assert widget.meter.value() == 0
    widget.update_status({"phase": "idle"})
    assert widget.save_button.isEnabled()
    widget.close()
    app.processEvents()


def test_headless_device_discovery_never_opens_microphone_or_runs_pactl(monkeypatch):
    monkeypatch.setattr(
        microphone_ui.subprocess,
        "run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Unexpected discovery")),
    )
    assert microphone_ui.discover("sources") == []
    assert microphone_ui.discover("sinks") == []
