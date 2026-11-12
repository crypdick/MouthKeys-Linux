import wave
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mouthkeys.config import Config
from mouthkeys.dictionary import Dictionary
from mouthkeys.engines import model_catalog
from mouthkeys.native_asr import NEMO_MODELS, NemoBackend, uses_native_streaming
from mouthkeys.transcribe import Transcriber
from mouthkeys.vocabulary import VocabularyStore, context_prompt


def test_upstream_vocabulary_optional_float_weights_and_256_terms():
    entries = [{"text": f"term{index}"} for index in range(256)]
    entries[0].update(weight=8.5, aliases=["heard term"])
    entries[1].update(weight=None, aliases=None)
    normalized = VocabularyStore.save_entries(entries)
    assert len(normalized) == 256
    assert normalized[0] == {"text": "term0", "weight": 8.5, "aliases": ["heard term"]}
    assert normalized[1] == {"text": "term1", "weight": None, "aliases": []}
    assert VocabularyStore.entries() == normalized
    with pytest.raises(ValueError, match="weight"):
        VocabularyStore.validate_entries([{"text": "overflow", "weight": 10**400}])
    with pytest.raises(ValueError, match="256"):
        VocabularyStore.validate_entries([*entries, {"text": "overflow"}])
    assert len(context_prompt(SimpleNamespace(vocabulary_boosting="on")).encode()) <= 4096


def test_decoder_vocabulary_merges_enabled_dictionary_without_mutating_user_store():
    saved = VocabularyStore.save_entries(
        [{"text": "MouthKeys", "weight": 1.5, "aliases": ["mouse keys"]}]
    )
    dictionary = Dictionary.defaults()
    dictionary["replacements"] = [
        {"triggers": ["mouth keys"], "replacement": "MouthKeys"},
        {"triggers": ["disabled"], "replacement": "Excluded", "enabled": False},
        {"triggers": ["paragraph"], "replacement": "\n\n"},
        {"triggers": ["linux"], "replacement": "Linux"},
    ]
    Dictionary.save(dictionary)
    resolved = {entry["text"]: entry for entry in VocabularyStore.effective_entries()}
    assert set(resolved) == {"MouthKeys", "Linux"}
    assert resolved["MouthKeys"]["weight"] == 8
    assert resolved["MouthKeys"]["aliases"] == ["mouse keys", "mouth keys"]
    assert VocabularyStore.entries() == saved
    assert Dictionary.load() == dictionary


def test_dictionary_terms_take_priority_over_low_weight_user_terms():
    VocabularyStore.save_entries([{"text": f"term{index}"} for index in range(256)])
    dictionary = Dictionary.defaults()
    dictionary["replacements"] = [{"triggers": ["heard"], "replacement": "Important"}]
    Dictionary.save(dictionary)
    effective = VocabularyStore.effective_entries()
    assert len(effective) == 256
    assert effective[0]["text"] == "Important"
    assert len(VocabularyStore.entries()) == 256


@pytest.mark.parametrize("model", ["nvidia/parakeet-tdt-0.6b-v2", "nvidia/parakeet-tdt-0.6b-v3"])
def test_native_tdt_catalog_has_real_graphs_word_times_and_offline_encoder(model):
    config = replace(Config(), engine="nemo", model=model, native_streaming="on")
    assert not uses_native_streaming(config)
    row = next(row for row in model_catalog("nemo") if row["model"] == model)
    assert row["vocabulary_method"] == "weighted context graph"
    assert row["word_timing"] is True
    assert row["native_streaming"] is False
    assert row["installed"] is False
    assert row["source"] == f"https://huggingface.co/{model}"
    assert "2.4 GB float32 parameters" in row["memory_note"]
    assert "unmeasured" in row["memory_note"]
    assert row["speed_note"] == "Not benchmarked on this hardware"


def test_native_tdt_live_preview_uses_provisional_segments_not_streaming(monkeypatch, tmp_path):
    path = tmp_path / "dictation.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(16000 * 4 * 2))
    config = replace(
        Config(),
        engine="nemo",
        model="nvidia/parakeet-tdt-0.6b-v3",
        native_streaming="on",
        pronunciation_matching="on",
    )
    transcriber = Transcriber()
    monkeypatch.setattr(transcriber, "load", lambda config: None)
    transcriber.model = SimpleNamespace(
        stream=lambda *a, **kw: pytest.fail("Offline TDT was sent to streaming encoder"),
        segments=lambda *a: [
            SimpleNamespace(
                text="Hello.",
                start=0.5,
                end=1.0,
                words=[SimpleNamespace(word="Hello.", start=0.5, end=1.0)],
            )
        ],
    )
    assert transcriber.preview(path, config) == "Hello."
    assert transcriber.transcribe(path, config) == "Hello."
    assert transcriber.last_alignment["words"] == [{"word": "Hello.", "start": 0.5, "end": 1.0}]


def test_native_tdt_loader_ignores_stream_only_precision_and_latency(monkeypatch, tmp_path):
    from mouthkeys import native_asr

    artifact = tmp_path / NEMO_MODELS["nvidia/parakeet-tdt-0.6b-v2"]["filename"]
    artifact.write_bytes(b"fake archive")
    calls = []

    class Model:
        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

        def eval(self):
            return self

    model = Model()  # No cache-aware encoder methods on this full-context model.
    monkeypatch.setattr(native_asr, "local_artifact", lambda config: artifact)
    monkeypatch.setattr(native_asr, "validate_artifacts", lambda *args: None)
    monkeypatch.setattr(
        native_asr, "torch_runtime", lambda config: (SimpleNamespace(), "cpu", "float16")
    )
    monkeypatch.setattr(
        native_asr.importlib,
        "import_module",
        lambda name: SimpleNamespace(ASRModel=SimpleNamespace(restore_from=lambda *a, **kw: model)),
    )
    config = replace(
        Config(),
        engine="nemo",
        model="nvidia/parakeet-tdt-0.6b-v2",
        compute_type="float16",
        native_streaming="on",
        streaming_latency="80",
    )
    backend = NemoBackend(config)
    assert backend.model is model
    assert calls == [(("cpu",), {}), ((), {"dtype": "float16"})]
    with pytest.raises(ValueError, match="English only"):
        backend._language(replace(config, language="fr"))


def test_model_browser_exposes_real_capabilities_and_unknown_measurements(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QComboBox, QLineEdit, QWidget

    from mouthkeys.models_ui import ModelBrowser

    window = QWidget()
    engine = QComboBox()
    engine.addItem("NeMo", "nemo")
    window.fields = {"engine": engine, "model_revision": QLineEdit("main")}
    window.loaded_model = None
    browser = ModelBrowser(window)
    assert not browser.isVisible()
    columns = {
        browser.table.horizontalHeaderItem(index).text(): index
        for index in range(browser.table.columnCount())
    }
    tdt_row = next(
        index
        for index in range(browser.table.rowCount())
        if browser.table.item(index, 0).text().startswith("Parakeet TDT v2")
    )
    assert "streaming: provisional/offline" in browser.table.item(tdt_row, 4).text()
    assert "weighted context graph" in browser.table.item(tdt_row, 4).text()
    assert browser.table.item(tdt_row, columns["Word timing"]).text() == "Native"
    assert "Not benchmarked" in browser.table.item(tdt_row, columns["Speed (local)"]).text()
    assert "Not evaluated" in browser.table.item(tdt_row, columns["Accuracy (local)"]).text()
    assert "https://huggingface.co/nvidia/" in browser.table.item(tdt_row, 0).toolTip()
    browser.close()
    window.close()


def test_custom_native_model_capability_checks_actual_encoder_context():
    from mouthkeys.native_asr import supports_native_streaming

    def model(context):
        return SimpleNamespace(
            encoder=SimpleNamespace(att_context_size=context),
            conformer_stream_step=lambda **kw: None,
        )

    assert supports_native_streaming(model([8, 1]))
    assert supports_native_streaming(model([-1, 0]))
    assert not supports_native_streaming(model([-1, -1]))
    assert not supports_native_streaming(SimpleNamespace())


def test_custom_full_context_worker_does_not_route_to_stream(monkeypatch):
    transcriber = Transcriber()
    model = SimpleNamespace(native_streaming=False)
    monkeypatch.setattr(transcriber, "load", lambda config: setattr(transcriber, "model", model))
    config = replace(Config(), engine="nemo", model="/custom/full-context.nemo")
    assert not transcriber._uses_native_streaming(config)
    transcriber._loaded(config)
    assert transcriber.loaded_model["native_streaming"] is False
