import fcntl
import json
import os
import uuid

import pytest

from mouthkeys.data_lock import dictionary_lock
from mouthkeys.dictionary import Dictionary
from mouthkeys.dictionary_transfer import decode_document
from mouthkeys.vocabulary import VocabularyStore


def document(tmp_path, data):
    path = tmp_path / "dictionary-transfer.json"
    path.write_text(json.dumps(data))
    return path


def test_import_upstream_dictionary_and_vocabulary_without_resetting_native_rules(tmp_path):
    current = Dictionary.defaults()
    current["punctuation"] = [{"aliases": ["triangle"], "symbol": "△"}]
    Dictionary.save(current)
    path = document(
        tmp_path,
        {
            "replacements": [{"from": ["mouth keys", "mouth kies"], "to": "MouthKeys"}],
            "customWords": ["Linux", {"text": "MouthKeys", "weight": 8.5}],
        },
    )
    result = Dictionary.import_file(path)
    assert Dictionary.apply("mouth keys mouth kies") == "MouthKeys MouthKeys"
    assert result["punctuation"] == current["punctuation"]
    assert VocabularyStore.entries() == [
        {"text": "Linux", "weight": None, "aliases": []},
        {"text": "MouthKeys", "weight": 8.5, "aliases": []},
    ]


def test_merge_preserves_existing_rules_moves_conflicting_triggers_and_keeps_word_weights(tmp_path):
    current = Dictionary.defaults()
    current["replacements"] = [
        {"triggers": ["old", "shared"], "replacement": "First"},
        {"triggers": ["existing"], "replacement": "Second"},
    ]
    Dictionary.save(current)
    VocabularyStore.save_entries([{"text": "Linux", "weight": 3, "aliases": []}])
    path = document(
        tmp_path,
        {
            "replacements": [{"triggers": ["shared", "new"], "replacement": "second"}],
            "customWords": [{"text": "linux", "weight": 10}, "MouthKeys"],
        },
    )
    result = Dictionary.import_file(path, "merge")
    assert result["replacements"] == [
        {"triggers": ["old"], "replacement": "First"},
        {"triggers": ["existing", "shared", "new"], "replacement": "Second"},
    ]
    assert VocabularyStore.entries() == [
        {"text": "Linux", "weight": 3, "aliases": []},
        {"text": "MouthKeys", "weight": None, "aliases": []},
    ]


def test_native_roundtrip_preserves_disabled_rules_formatting_and_vocabulary(tmp_path):
    current = Dictionary.defaults()
    current["replacements"] = [
        {"triggers": ["mouth keys"], "replacement": "MouthKeys", "enabled": False}
    ]
    current["punctuation"][0]["enabled"] = False
    Dictionary.save(current)
    words = [{"text": "MouthKeys", "weight": 8.5, "aliases": ["mouthkies"]}]
    VocabularyStore.save_entries(words)
    path = tmp_path / "export.json"
    Dictionary.export_file(path)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["customWords"] == words
    Dictionary.save(Dictionary.defaults())
    VocabularyStore.save_entries([])
    assert Dictionary.import_file(path) == current
    assert VocabularyStore.entries() == words


@pytest.mark.parametrize("incoming_id", [None, "bb94d8c3-2216-4fe8-893a-9b4875282b93"])
def test_portable_merge_preserves_existing_replacement_id(tmp_path, incoming_id):
    identifier = str(uuid.uuid4())
    current = Dictionary.defaults()
    current["replacements"] = [
        {"id": identifier, "triggers": ["mouth keys"], "replacement": "MouthKeys"}
    ]
    Dictionary.save(current)
    incoming = {"from": ["mouth kies"], "to": "mouthkeys"}
    if incoming_id is not None:
        incoming["id"] = incoming_id
    result = Dictionary.import_file(document(tmp_path, {"replacements": [incoming]}), "merge")
    assert result["replacements"] == [
        {"id": identifier, "triggers": ["mouth keys", "mouth kies"], "replacement": "MouthKeys"}
    ]


def test_upstream_export_uses_portable_schema_and_omits_disabled_rules(tmp_path):
    current = Dictionary.defaults()
    current["replacements"] = [
        {"triggers": ["mouth keys"], "replacement": "MouthKeys"},
        {"triggers": ["disabled"], "replacement": "Inactive", "enabled": False},
    ]
    Dictionary.save(current)
    VocabularyStore.save_entries([{"text": "Linux"}])
    path = tmp_path / "portable.json"
    Dictionary.export_file(path, "upstream")
    assert json.loads(path.read_text()) == {
        "replacements": [{"from": ["mouth keys"], "to": "MouthKeys"}],
        "customWords": [{"text": "Linux", "weight": None, "aliases": []}],
    }


def test_legacy_native_dictionary_import_leaves_vocabulary_intact(tmp_path):
    VocabularyStore.save_entries([{"text": "Linux"}])
    Dictionary.import_file(document(tmp_path, Dictionary.defaults()))
    assert VocabularyStore.entries() == [{"text": "Linux", "weight": None, "aliases": []}]


@pytest.mark.parametrize(
    "data",
    [
        {"replacements": [{"from": [], "to": "bad"}], "customWords": ["new"]},
        {
            "replacements": [{"from": ["valid"], "to": "new"}],
            "customWords": [{"text": "bad", "weight": -1}],
        },
        {"replacements": [], "customWords": ["duplicate", "DUPLICATE"]},
        {"replacements": [], "punctuation": []},
    ],
)
def test_invalid_import_changes_neither_store(tmp_path, data):
    current = Dictionary.defaults()
    current["replacements"] = [{"triggers": ["safe"], "replacement": "old"}]
    Dictionary.save(current)
    VocabularyStore.save_entries([{"text": "Existing"}])
    old_dictionary = Dictionary.path().read_bytes()
    old_words = VocabularyStore.path().read_bytes()
    with pytest.raises(ValueError):
        Dictionary.import_file(document(tmp_path, data))
    assert Dictionary.path().read_bytes() == old_dictionary
    assert VocabularyStore.path().read_bytes() == old_words


def test_second_store_write_failure_rolls_back_vocabulary(tmp_path, monkeypatch):
    current = Dictionary.defaults()
    Dictionary.save(current)
    VocabularyStore.save_entries([{"text": "Existing"}])
    old_words = VocabularyStore.path().read_bytes()
    path = document(tmp_path, {"replacements": [], "customWords": ["New"]})

    def fail(data):
        raise OSError("test disk failure")

    monkeypatch.setattr(Dictionary, "save", fail)
    with pytest.raises(OSError, match="test disk failure"):
        Dictionary.import_file(path)
    assert Dictionary.load() == current
    assert VocabularyStore.path().read_bytes() == old_words


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"terms": ["Linux"]}, "words"),
        ({"items": [{"triggers": ["from"], "replacement": "to"}]}, "replacements"),
        ({"entries": [{"text": "Linux", "weight": 8.0}]}, "words"),
        ({"replacements": [{"from": "from", "to": "to"}]}, "replacements"),
    ],
)
def test_accepts_upstream_transfer_variants(payload, expected):
    sections, words = decode_document(payload)
    assert bool(sections.get("replacements")) if expected == "replacements" else bool(words)


def test_merge_native_formatting_keeps_distinct_same_symbol_alias_groups(tmp_path):
    current = Dictionary.defaults()
    Dictionary.save(current)
    imported = Dictionary.defaults()
    imported["punctuation"] = [{"aliases": ["comma", "comma please"], "symbol": ","}]
    result = Dictionary.import_file(document(tmp_path, imported), "merge")
    assert {
        tuple(entry["aliases"]) for entry in result["punctuation"] if entry["symbol"] == "."
    } == {("period", "full stop"), ("dot",)}
    assert result["punctuation"][-1] == imported["punctuation"][0]


def test_dictionary_lock_is_reentrant_and_cross_process_wait_is_bounded(monkeypatch):
    from mouthkeys import data_lock
    from mouthkeys.config import config_path, private_dir

    monkeypatch.setattr(data_lock, "LOCK_TIMEOUT", 0.02)
    lock_path = private_dir(config_path().parent) / ".dictionary-transfer.lock"
    with lock_path.open("w") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TimeoutError, match="another process"):
            with dictionary_lock():
                pytest.fail("must not bypass the existing file lock")
    with dictionary_lock():
        with dictionary_lock():
            Dictionary.save(Dictionary.defaults())
