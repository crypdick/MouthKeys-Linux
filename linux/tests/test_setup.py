"""Language routes and regional offers stay local and non-delivering."""

import json
import locale
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QPushButton, QWidget

from mouthkeys import onboarding
from mouthkeys.config import Config
from mouthkeys.gui import Window
from mouthkeys.regional import CANADIAN_ZONES, CENTRAL_ZONES, REGIONS, region_offer
from mouthkeys.regional_ui import RegionalFillerBanner
from mouthkeys.setup_languages import COHERE, PARAKEET_V3, languages, routes


@pytest.mark.parametrize("region,word,codes,zones,message", REGIONS)
def test_regional_rules_cover_every_pinned_region_and_time_zone(
    region, word, codes, zones, message
):
    config = Config(remove_filler_words="on", filler_words="eh,AH,um")
    for code in codes:
        assert region_offer(config, region=code, zone="") == {
            "region": region,
            "word": word,
            "message": message,
        }
    for zone in zones:
        assert region_offer(config, region="", zone=zone)["region"] == region
    assert (
        region_offer(replace(config, filler_words="um"), region="", zone=next(iter(zones))) is None
    )


def test_regional_offer_is_once_local_and_ordered_and_never_claims_minnesota_from_timezone():
    config = Config(remove_filler_words="on", filler_words="eh,ah")
    assert "America/Menominee" in REGIONS[1][3]
    assert len(CANADIAN_ZONES) == 37
    for zone in CENTRAL_ZONES:
        possible = region_offer(config, region="US", zone=zone)
        assert "by any chance" in possible["message"]
        heard = region_offer(config, region="US", zone=zone, dictation="Oop, you betcha")
        assert "sounds like Minnesota" in heard["message"]
    assert region_offer(config, region="SG", zone="America/Toronto")["region"] == "canada"
    assert region_offer(config, region="SG", zone="America/Chicago")["region"] == "minnesota"
    assert (
        region_offer(replace(config, filler_words="eh"), region="SG", zone="Asia/Singapore") is None
    )
    assert (
        region_offer(replace(config, regional_filler_answered="on"), region="CA", zone="") is None
    )
    assert region_offer(replace(config, remove_filler_words="off"), region="CA", zone="") is None
    assert region_offer(config, region="US", zone="America/Los_Angeles") is None


def test_all_locked_whisper_languages_have_explicit_routes_and_no_guessed_native_languages():
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    assert set(dict(languages())) == set(_LANGUAGE_CODES)
    for code in _LANGUAGE_CODES:
        choices = routes(code)
        expected = ["large-v3-turbo"] if code == "yue" else ["small", "large-v3-turbo"]
        assert [route["model"] for route in choices if route["engine"] == "whisper"] == expected
        for route in choices:
            if "parakeet-tdt-0.6b-v3" in route["model"]:
                assert code in PARAKEET_V3
            if "cohere" in route["model"]:
                assert code in COHERE
                assert route["language"] == code
                assert route["model_revision"] == "refs/pr/6"
    assert [route["model"] for route in routes("en")][:2] == [
        "nemo-parakeet-tdt-0.6b-v2",
        "nvidia/parakeet_realtime_eou_120m-v1",
    ]
    assert {route["language"] for route in routes("de") if "nemotron" in route["model"]} == {
        "de-DE"
    }
    assert {route["language"] for route in routes("no") if "nemotron" in route["model"]} == {
        "nb-NO"
    }
    assert all(route["engine"] == "whisper" for route in routes("yue"))
    assert routes("not-a-language") == []


def setup_window():
    window = QWidget()
    config = Config.load()
    window.fields = {}
    for key in ("engine", "compute_type", "regional_filler_answered", "onboarding_complete"):
        field = QComboBox()
        for value in {
            "engine": ["whisper", "onnx", "nemo", "transformers"],
            "compute_type": ["default", "float16", "int8"],
            "regional_filler_answered": ["off", "on"],
            "onboarding_complete": ["off", "on"],
        }[key]:
            field.addItem(value, value)
        field.setCurrentIndex(field.findData(getattr(config, key)))
        window.fields[key] = field
    model = QComboBox()
    model.setEditable(True)
    model.addItem(config.model, config.model)
    window.fields["model"] = model
    for key in ("language", "model_revision", "filler_words"):
        window.fields[key] = QLineEdit(getattr(config, key))
    return window


def setup_page(window):
    return onboarding.SetupPage(
        window.fields,
        Mock(),
        open_speech=Mock(),
        open_microphone=Mock(),
        refresh_other_banner=Mock(),
    )


@pytest.mark.parametrize("keep", [True, False])
def test_regional_banner_is_inline_once_and_syncs_settings_without_other_writes(monkeypatch, keep):
    Config(remove_filler_words="on", filler_words="um,EH,eh", audio_source="saved-source").save()
    monkeypatch.setenv("TZ", "America/Toronto")
    window = setup_window()
    changed = []
    window.fields["filler_words"].setText("unsaved-edit")
    banner = RegionalFillerBanner(window, window.fields, on_change=lambda: changed.append(True))
    assert banner.parentWidget() is window
    assert not banner.isWindow()
    assert not banner.isHidden()
    assert not window.isVisible()
    assert QApplication.focusWidget() is None
    banner.answer(keep=keep)
    saved = Config.load()
    assert saved.filler_words == ("um" if keep else "um,EH,eh")
    assert saved.audio_source == "saved-source"
    assert saved.regional_filler_answered == "on"
    assert window.fields["filler_words"].text() == saved.filler_words
    assert window.fields["regional_filler_answered"].currentData() == "on"
    assert banner.isHidden()
    assert changed == [True]
    assert not window.isVisible()
    banner.answer(keep=not keep)
    assert Config.load() == saved


def test_stale_regional_banner_cannot_overwrite_answer_from_another_surface(monkeypatch):
    monkeypatch.setenv("TZ", "America/Toronto")
    Config(remove_filler_words="on", filler_words="um,eh").save()
    setup_parent, settings_parent = setup_window(), setup_window()
    setup = RegionalFillerBanner(setup_parent, setup_parent.fields)
    settings = RegionalFillerBanner(settings_parent, settings_parent.fields)
    assert not setup.isHidden()
    assert not settings.isHidden()
    setup.answer(keep=False)
    settings.answer(keep=True)
    assert settings.isHidden()
    assert Config.load().regional_filler_answered == "on"
    assert Config.load().filler_words == "um,eh"


def test_setup_language_route_is_explicit_local_and_invalidates_stale_tryouts():
    Config(model="base.en", compute_type="float16", audio_source="keep-me").save()
    window = setup_window()
    setup = setup_page(window)
    before = Config.load()
    setup.language_search.setText("Deutsch")
    assert setup.language.currentData() == "de"
    assert Config.load() == before
    setup.send.assert_not_called()
    setup.record()
    action, payload, callback = setup.send.call_args.args
    assert (action, payload) == ("start", {"mode": "training"})
    index = next(
        index
        for index in range(setup.route.count())
        if setup.route.itemData(index)["engine"] == "whisper"
    )
    setup.route.setCurrentIndex(index)
    callback({"result": {"id": "stale-job"}})
    assert setup.tryout_id is None
    setup.apply_route()
    saved = Config.load()
    assert (saved.engine, saved.model, saved.language, saved.model_revision) == (
        "whisper",
        "small",
        "de",
        "main",
    )
    assert saved.compute_type == "default"
    assert saved.audio_source == "keep-me"
    assert window.fields["model"].currentText() == "small"
    assert window.fields["language"].text() == "de"
    for key in ("engine", "compute_type"):
        assert isinstance(window.fields[key], QComboBox)
        assert window.fields[key].currentData() == getattr(saved, key)
    assert setup.send.call_count == 1  # No load/download, navigation, or desktop effects.
    assert not window.isVisible()


def test_setup_tryout_refreshes_regional_offer_using_only_its_current_result(monkeypatch):
    Config(remove_filler_words="on", filler_words="eh,um").save()
    monkeypatch.setenv("TZ", "America/Chicago")
    window = setup_window()
    setup = setup_page(window)
    assert "by any chance" in setup.regional_banner.message.text()
    setup.tryout_id = "current"
    setup.update_status({"id": "stale", "assistant_proposal": {"mode": "training", "text": "Ope"}})
    assert "by any chance" in setup.regional_banner.message.text()
    setup.update_status(
        {"id": "current", "assistant_proposal": {"mode": "training", "text": "Ope, you betcha"}}
    )
    assert "sounds like Minnesota" in setup.regional_banner.message.text()
    assert setup.result.toPlainText() == "Ope, you betcha"
    assert setup.tryout_id is None
    assert not window.isVisible()


def test_setup_does_not_report_listening_if_microphone_open_returns_idle():
    setup = setup_page(setup_window())
    setup.send.side_effect = lambda action, data, callback: callback(
        {"result": {"id": "failed-job", "phase": "idle", "message": "Microphone unavailable"}}
    )
    setup.record()
    assert setup.tryout_id is None
    assert setup.result.toPlainText() == "Microphone unavailable"


def test_setup_ignores_malformed_tryout_proposal_and_keeps_waiting():
    setup = setup_page(setup_window())
    setup.tryout_id = "current"
    setup.update_status(
        {"id": "current", "phase": "transcribing", "assistant_proposal": {"mode": "training"}}
    )
    assert setup.tryout_id == "current"
    setup.update_status({"id": "current", "phase": "idle"})
    assert setup.tryout_id is None
    assert setup.result.toPlainText() == "Tryout finished"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"error": "Service unavailable"}, "Service unavailable"),
        ({"result": {"id": ""}}, "Could not start tryout"),
        ({"result": {"id": 4}}, "Could not start tryout"),
        ({"result": {"id": "current", "phase": "recording"}}, "Listening"),
    ],
)
def test_setup_reports_only_valid_active_recordings_as_listening(response, message):
    setup = setup_page(setup_window())
    setup.send.side_effect = lambda action, data, callback: callback(response)
    setup.record()
    assert setup.result.toPlainText().startswith(message)
    if setup.tryout_id:
        setup.update_status(
            {
                "id": "current",
                "phase": "idle",
                "assistant_proposal": {"mode": "assist"},
                "message": "Tryout cancelled",
            }
        )
        assert setup.result.toPlainText() == "Tryout cancelled"
        assert setup.tryout_id is None


def test_setup_uses_locale_route_and_handles_search_without_matches():
    Config(language="nb-NO", engine="nemo", model="nvidia/nemotron-3.5-asr-streaming-0.6b").save()
    setup = setup_page(setup_window())
    assert setup.language.currentData() == "no"
    assert setup.route.currentData()["language"] == "nb-NO"
    assert setup.route.currentData()["engine"] == "nemo"
    setup.language_search.setText("none matches xyzzy")
    assert setup.language.currentData() == "no"
    setup.language_search.clear()
    before = Config.load()
    setup.language.setCurrentIndex(-1)
    assert setup.route.count() == 0
    setup.apply_route()
    assert Config.load() == before
    setup.send.assert_not_called()


def test_setup_check_and_finish_do_not_start_recording(monkeypatch):
    Config(audio_source="saved-settings").save()
    window = setup_window()
    setup = setup_page(window)
    monkeypatch.setattr(onboarding, "doctor", lambda config: {"input": config.audio_source})
    buttons = {button.text(): button for button in setup.findChildren(QPushButton)}
    buttons["Check setup"].click()
    assert json.loads(setup.result.toPlainText()) == {"input": "saved-settings"}
    buttons["Finish setup"].click()
    assert Config.load().onboarding_complete == "on"
    assert window.fields["onboarding_complete"].currentData() == "on"
    assert "validate" in setup.result.toPlainText()
    setup.send.assert_not_called()
    buttons["Stop tryout"].click()
    setup.send.assert_called_once_with("stop", None, None)
    buttons["Cancel tryout"].click()
    setup.send.assert_called_with("cancel", None, None)
    assert not setup.isVisible()


def test_setup_links_open_settings_inside_same_window(monkeypatch):
    monkeypatch.setattr(Window, "poll", lambda self: None)
    window = Window()
    window.timer.stop()
    window.settings_category.setCurrentText("Appearance")
    buttons = {button.text(): button for button in window.setup_page.findChildren(QPushButton)}
    buttons["Choose speech model"].click()
    assert window.nav.currentItem().text() == "App Settings"
    assert window.settings_category.currentText() == "Speech"
    buttons["Choose microphone and priorities"].click()
    assert window.nav.currentItem().text() == "Microphone"
    assert not window.isVisible()


def test_setup_failed_save_leaves_tryout_and_settings_fields_unchanged(monkeypatch):
    before = Config(model="base.en")
    before.save()
    window = setup_window()
    setup = setup_page(window)
    setup.tryout_id = "active"

    def unavailable(config):
        raise OSError("Config directory unavailable")

    monkeypatch.setattr(Config, "save", unavailable)
    for save in (setup.apply_route, setup.finish):
        save()
        assert setup.result.toPlainText() == "Config directory unavailable"
        assert setup.tryout_id == "active"
        assert Config.load() == before
    assert window.fields["model"].currentText() == "base.en"
    assert window.fields["onboarding_complete"].currentData() == "off"


def test_regional_offer_resolves_localtime_zone_when_tz_is_unset(monkeypatch):
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setattr(locale, "getlocale", lambda: ("en_US", "UTF-8"))
    system_link = Path("/usr/share/zoneinfo/America/Detroit")
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        return system_link if path == Path("/etc/localtime") else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    offer = region_offer(Config(remove_filler_words="on"))
    assert offer is not None
    assert offer["region"] == "michigan"
    assert offer["word"] == "eh"
