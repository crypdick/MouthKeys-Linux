from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QFormLayout, QInputDialog, QMessageBox, QPushButton

from mouthkeys import desktop_input, gui, provider_tools, settings_choices
from mouthkeys.config import Config
from mouthkeys.regional import UPSTREAM_FILLERS


def test_settings_category_filters_fields_and_keeps_matching_hints(monkeypatch):
    monkeypatch.setattr(gui.Window, "poll", lambda _: None)
    window = gui.Window()
    window.timer.stop()
    fields = {
        "Speech": "native_timeout_seconds",
        "Microphone & media": "mic_battery",
        "Text delivery": "delivery",
        "Spoken Send": "send_after_pause",
        "Text formatting": "filler_words",
        "AI cleanup": "command_confirm",
        "Appearance": "overlay_position",
        "History & stats": "file_history_limit",
        "Desktop": "paste_shortcut",
        "Local API": "api_enabled",
        "Dictionary & learning": "pronunciation_matching",
    }
    form = window.fields["delivery"].parentWidget().layout()
    assert isinstance(form, QFormLayout)
    delivery_row, _ = form.getWidgetPosition(window.fields["delivery"])
    save = next(b for b in window.findChildren(QPushButton) if b.text() == "Save settings")
    try:
        assert window.settings_category.currentText() == "Speech"
        for selected in fields:
            window.settings_category.setCurrentText(selected)
            assert {
                group for group, key in fields.items() if form.isRowVisible(window.fields[key])
            } == {selected}
            assert form.isRowVisible(delivery_row + 1) == (selected == "Text delivery")
            assert form.isRowVisible(save)
        window.settings_category.setCurrentText("All settings")
        assert all(form.isRowVisible(window.fields[key]) for key in fields.values())
        assert form.isRowVisible(delivery_row + 1)
        assert not window.isVisible()
    finally:
        window.close()


def test_settings_engine_refresh_keeps_a_custom_model_in_the_editor(monkeypatch):
    monkeypatch.setattr(gui.Window, "poll", lambda _: None)
    Config(engine="onnx", model="local/custom").save()
    window = gui.Window()
    window.timer.stop()
    try:
        model = window.fields["model"]
        engine = window.fields["engine"]
        assert [model.itemData(i) for i in range(model.count())] == [
            *settings_choices.ONNX_MODELS,
            "local/custom",
        ]
        assert model.currentText() == "local/custom"
        engine.setCurrentIndex(engine.findData("whisper"))
        assert model.findData("base.en") >= 0
        assert all(model.findData(name) == -1 for name in settings_choices.ONNX_MODELS)
        assert model.currentText() == "local/custom"
        model.setCurrentIndex(model.findData("base.en"))
        engine.setCurrentIndex(engine.findData("onnx"))
        assert model.currentText() == "base.en"
        assert window.save()
        assert (Config.load().engine, Config.load().model) == ("onnx", "base.en")
        engine.setCurrentIndex(engine.findData("whisper"))
        assert model.currentText() == "base.en"
        assert not window.isVisible()
    finally:
        window.close()


def test_delivery_hint_changes_when_direct_insertion_is_selected(monkeypatch):
    monkeypatch.setattr(gui.Window, "poll", lambda _: None)
    window = gui.Window()
    window.timer.stop()
    monkeypatch.setattr(desktop_input, "session", lambda: "x11")
    monkeypatch.setattr(desktop_input, "shutil", SimpleNamespace(which=lambda _: None))
    try:
        hint = window.settings_options.delivery_hint
        delivery = window.fields["delivery"]
        insertion = window.fields["insertion_mode"]
        delivery.setCurrentIndex(delivery.findData("x11"))
        assert "Paste backend xdotool not found" in hint.text()
        assert "Direct insertion" not in hint.text()
        insertion.setCurrentIndex(insertion.findData("direct"))
        assert "Direct insertion may still work in supported fields." in hint.text()
        assert "Clipboard fallback stays available." in hint.text()
        assert not window.isVisible()
    finally:
        window.close()


@pytest.mark.parametrize(
    ("action", "models", "choose", "failed", "status"),
    [
        ("verify", [], False, False, "Provider connection verified"),
        ("verify", [], False, True, "Provider unavailable"),
        ("models", [], False, False, "Provider returned no models"),
        ("models", ["available"], False, False, "Checking provider…"),
        ("models", ["available"], True, False, "Model selected; Save settings to apply"),
    ],
)
def test_provider_actions_use_saved_config_and_keep_model_choice_as_draft(
    monkeypatch, action, models, choose, failed, status
):
    calls = []
    tasks = []
    selections = []

    def verify(config):
        calls.append(("verify", config))
        if failed:
            raise OSError("Provider unavailable")

    def list_models(config):
        calls.append(("models", config))
        return models

    def pick_model(parent, title, label, choices, *, editable):
        assert choices == ["available"]
        assert not editable
        selections.append((parent, title, label))
        return choices[0], choose

    monkeypatch.setattr(gui.Window, "poll", lambda _: None)
    monkeypatch.setattr(gui.Window, "run_task", lambda _, run, done: tasks.append((run, done)))
    monkeypatch.setattr(provider_tools, "verify", verify)
    monkeypatch.setattr(provider_tools, "list_models", list_models)
    monkeypatch.setattr(QInputDialog, "getItem", pick_model)
    window = gui.Window()
    window.timer.stop()
    try:
        assert not tasks
        window.fields["ai_model"].setText("configured")
        label = "Verify AI connection" if action == "verify" else "Discover AI models"
        button = next(b for b in window.findChildren(QPushButton) if b.text() == label)
        button.click()
        assert window.status.text() == "Checking provider…"
        assert not calls
        [(run, done)] = tasks
        try:
            result = run()
        except OSError as exc:
            done({"error": str(exc)})
        else:
            done({"result": result})
        assert calls == [(action, Config.load())]
        assert Config.load().ai_model == "configured"
        assert window.fields["ai_model"].text() == ("available" if choose else "configured")
        assert bool(selections) == bool(models)
        assert window.status.text() == status
        assert not window.isVisible()
    finally:
        window.close()


def test_provider_action_does_not_connect_when_settings_cannot_be_saved(monkeypatch):
    monkeypatch.setattr(gui.Window, "poll", lambda _: None)
    monkeypatch.setattr(gui.Window, "save", lambda _: False)
    tasks = []
    monkeypatch.setattr(gui.Window, "run_task", lambda _, *args: tasks.append(args))
    window = gui.Window()
    window.timer.stop()
    try:
        button = next(
            b for b in window.findChildren(QPushButton) if b.text() == "Verify AI connection"
        )
        button.click()
        assert not tasks
    finally:
        window.close()


@pytest.mark.parametrize("answered_elsewhere", [False, True])
def test_regional_dialog_preserves_settings_saved_while_it_is_open(monkeypatch, answered_elsewhere):
    monkeypatch.setenv("TZ", "America/Toronto")
    Config(remove_filler_words="on", filler_words="um,EH").save()
    monkeypatch.setattr(gui.Window, "poll", lambda _: None)

    def answer(*_):
        replace(
            Config.load(),
            audio_source="new-source",
            regional_filler_answered="on" if answered_elsewhere else "off",
        ).save()
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)
    window = gui.Window()
    window.timer.stop()
    try:
        button = next(
            b for b in window.findChildren(QPushButton) if b.text() == "Regional filler suggestion"
        )
        button.click()
        saved = Config.load()
        assert saved.audio_source == "new-source"
        assert saved.filler_words == ("um,EH" if answered_elsewhere else "um")
        assert saved.regional_filler_answered == "on"
        assert window.regional_banner.isHidden()
        assert not window.isVisible()
    finally:
        window.close()


def test_filler_actions_without_active_offer_change_only_the_draft(monkeypatch):
    monkeypatch.setattr(gui.Window, "poll", lambda _: None)

    def unexpected_dialog(*_):
        pytest.fail("An unavailable regional suggestion should not open a dialog")

    monkeypatch.setattr(QMessageBox, "question", unexpected_dialog)
    config = Config(remove_filler_words="off", filler_words="custom")
    config.save()
    window = gui.Window()
    window.timer.stop()
    try:
        buttons = {b.text(): b for b in window.findChildren(QPushButton)}
        buttons["Regional filler suggestion"].click()
        assert "No unanswered regional filler suggestion" in window.status.text()
        buttons["Reset filler words to upstream defaults"].click()
        assert window.fields["filler_words"].text() == UPSTREAM_FILLERS
        assert "Save settings to apply" in window.status.text()
        assert Config.load() == config
        assert not window.isVisible()
    finally:
        window.close()


def test_theme_changes_only_after_saving_preferences(monkeypatch):
    monkeypatch.setattr(gui.Window, "poll", lambda _: None)
    Config(theme="dark").save()
    window = gui.Window()
    window.timer.stop()
    try:
        assert window.styleSheet().startswith("QWidget { background: #111214;")
        field = window.fields["theme"]
        field.setCurrentIndex(field.findData("light"))
        assert window.styleSheet().startswith("QWidget { background: #111214;")
        assert window.save()
        assert window.styleSheet().startswith("QWidget { background: #ffffff;")
        assert not window.isVisible()
    finally:
        window.close()
