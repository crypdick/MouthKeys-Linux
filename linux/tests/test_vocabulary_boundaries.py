import pytest

from mouthkeys.dictionary import Dictionary
from mouthkeys.vocabulary import VocabularyStore


@pytest.mark.parametrize("entry", [None, {"weight": 4}, {"text": "example", "unknown": []}])
def test_bad_entry_shapes_do_not_replace_existing_vocabulary(entry):
    saved = VocabularyStore.save_entries([{"text": "Keep", "weight": 2}])
    with pytest.raises(ValueError, match="Vocabulary entry requires text, weight, and aliases"):
        VocabularyStore.save_entries([entry])
    assert VocabularyStore.entries() == saved


def test_remove_is_case_insensitive_and_preserves_other_weights():
    initial = VocabularyStore.save_entries(
        [
            {"text": "To remove", "aliases": None},
            {"text": "Retained", "weight": 3, "aliases": ["say retained"]},
        ]
    )
    assert initial[0]["aliases"] == []
    with pytest.raises(ValueError, match="Vocabulary must be a list"):
        VocabularyStore.save("not a list")
    assert VocabularyStore.remove("TO REMOVE") == ["Retained"]
    assert VocabularyStore.entries() == initial[1:]


def test_add_normalizes_whitespace_and_keeps_existing_case():
    assert VocabularyStore.add(" new    term ") == ["new term"]
    assert VocabularyStore.add("NEW TERM") == ["new term"]


def test_vocabulary_caps_alias_count_per_word():
    with pytest.raises(ValueError, match="at most 20 aliases"):
        VocabularyStore.save_entries(
            [{"text": "keyword", "aliases": [f"alias {i}" for i in range(21)]}]
        )
    assert VocabularyStore.entries() == []


def test_dictionary_triggers_too_long_for_biasing_are_not_used_as_aliases():
    long_trigger = "a" * 129  # Proposed hint too long, but still a valid dictionary trigger.
    Dictionary.save(
        Dictionary.defaults()
        | {
            "replacements": [
                {"triggers": [long_trigger, "short trigger"], "replacement": "keyword"}
            ]
        }
    )
    assert VocabularyStore.effective_entries() == [
        {"text": "keyword", "weight": 8, "aliases": ["short trigger"]}
    ]
    assert VocabularyStore.apply_aliases(long_trigger) == long_trigger
    assert VocabularyStore.apply_aliases("short trigger") == "short trigger"


def test_vocabulary_rejects_too_many_bytes_before_replacing_store():
    original = VocabularyStore.save_entries([{"text": "Keep"}])
    entries = [
        {"text": f"word {index}", "aliases": [f"{'é' * 108}{index}:{i}" for i in range(20)]}
        for index in range(24)
    ]
    with pytest.raises(ValueError, match="Vocabulary entries exceed 256 KiB"):
        VocabularyStore.save_entries(entries)
    assert VocabularyStore.entries() == original


def test_oversized_vocabulary_file_is_rejected_before_json_parsing():
    path = VocabularyStore.path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b" " * (256 * 1024 + 1))
    with pytest.raises(ValueError, match="Vocabulary file exceeds 256 KiB"):
        VocabularyStore.load()
