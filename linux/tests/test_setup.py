"""Language routes and regional offers stay local and non-delivering."""

from dataclasses import replace

import pytest

from mouthkeys.config import Config
from mouthkeys.regional import CANADIAN_ZONES, CENTRAL_ZONES, REGIONS, region_offer
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
    from PySide6.QtWidgets import QComboBox, QLineEdit, QWidget

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


@pytest.mark.parametrize("keep", [True, False])
def test_regional_banner_is_inline_once_and_syncs_settings_without_other_writes(monkeypatch, keep):
    from PySide6.QtWidgets import QApplication

    from mouthkeys.regional_ui import RegionalFillerBanner

    Config(remove_filler_words="on", filler_words="um,EH,eh", audio_source="saved-source").save()
    monkeypatch.setenv("TZ", "America/Toronto")
    window = setup_window()
    changed = []
    window.fields["filler_words"].setText("unsaved-edit")
    banner = RegionalFillerBanner(window, on_change=lambda: changed.append(True))
    assert banner.parentWidget() is window
    assert not banner.isWindow()
    assert not banner.isHidden()
    assert not window.isVisible()
    assert QApplication.focusWidget() is None
    banner.answer(keep)
    saved = Config.load()
    assert saved.filler_words == ("um" if keep else "um,EH,eh")
    assert saved.audio_source == "saved-source"
    assert saved.regional_filler_answered == "on"
    assert window.fields["filler_words"].text() == saved.filler_words
    assert window.fields["regional_filler_answered"].currentData() == "on"
    assert banner.isHidden()
    assert changed == [True]
    assert not window.isVisible()
    banner.answer(not keep)
    assert Config.load() == saved


def test_setup_language_route_is_explicit_local_and_invalidates_stale_tryouts(monkeypatch):
    from PySide6.QtWidgets import QComboBox

    from mouthkeys.onboarding import SetupPage

    Config(model="base.en", compute_type="float16", audio_source="keep-me").save()
    window = setup_window()
    sent = []
    window.send = lambda *args, **kwargs: sent.append((args, kwargs))
    setup = SetupPage(window)
    before = Config.load()
    setup.language_search.setText("Deutsch")
    assert setup.language.currentData() == "de"
    assert Config.load() == before
    assert not sent
    setup.record()
    callback = sent[-1][1]["callback"]
    assert sent[-1][0] == ("start", {"mode": "training"})
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
    assert len(sent) == 1  # No load/download, navigation, or desktop effects.
    assert not window.isVisible()


def test_setup_tryout_refreshes_regional_offer_using_only_its_current_result(monkeypatch):
    from mouthkeys.onboarding import SetupPage

    Config(remove_filler_words="on", filler_words="eh,um").save()
    monkeypatch.setenv("TZ", "America/Chicago")
    window = setup_window()
    setup = SetupPage(window)
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
