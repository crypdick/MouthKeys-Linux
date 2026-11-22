import pytest

from mouthkeys.config import Config
from mouthkeys.dictionary import Dictionary
from mouthkeys.formatting import format_text
from mouthkeys.vocabulary import VocabularyStore


@pytest.mark.parametrize("gap", [" ", "   ", "\t"])
def test_multiword_prefix_accepts_horizontal_asr_spacing(gap):
    config = Config(spoken_formatting="on", literal_prefix="say very")
    assert format_text(f"Before SAY{gap}VERY comma after", config) == "Before, after"


@pytest.mark.parametrize("gap", [", ", ". ", "\n", "\u2028"])
def test_multiword_prefix_cannot_cross_punctuation_or_line_breaks(gap):
    config = Config(spoken_formatting="on", literal_prefix="say very")
    text = f"Before say{gap}very comma after"
    assert format_text(text, config) == text


@pytest.mark.parametrize("enabled", [True, False])
def test_punctuation_rule_claims_same_phrase_before_action_only_when_enabled(enabled):
    config = Config(spoken_formatting="on")
    rules = {
        "replacements": [],
        "punctuation": [{"aliases": ["pause here"], "symbol": ",", "enabled": enabled}],
        "actions": [{"aliases": ["pause here"], "symbol": "\n"}],
    }
    expected = "Before, after" if enabled else "Before\nafter"
    assert format_text("Before literal pause here after", config, rules) == expected


@pytest.mark.parametrize(
    ("text", "expected"), [("literal comma end", ", end"), ("start literal comma", "start,")]
)
def test_generated_comma_at_start_or_end_is_not_removed(text, expected):
    assert format_text(text, Config(spoken_formatting="on")) == expected


def test_vocabulary_aliases_are_resolved_before_dictionary_replacements():
    VocabularyStore.save_entries(
        [{"text": "MouthKeys", "weight": 1, "aliases": ["mouth keys app"]}]
    )
    rules = Dictionary.defaults()
    rules["replacements"].append({"triggers": ["MouthKeys"], "replacement": "MouthKeys Linux"})
    assert format_text("open mouth keys app", Config(vocabulary_boosting="on"), rules) == (
        "open MouthKeys Linux"
    )
    assert format_text("open mouth keys app", Config(), rules) == "open mouth keys app"


def test_empty_filler_list_and_existing_trailing_space_are_preserved():
    config = Config(remove_filler_words="on", filler_words=" , ", space_between_dictations="on")
    assert format_text(" more ", config, context="previous") == " more "


def test_smart_capitalization_leaves_text_with_no_letters_unchanged():
    assert format_text("123!", Config(smart_capitalization="on"), context="Done.") == "123!"
