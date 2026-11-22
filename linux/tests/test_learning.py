from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mouthkeys import data_lock, learning
from mouthkeys.dictionary import Dictionary
from mouthkeys.learning import CorrectionLearner, correction_candidate
from mouthkeys.vocabulary import VocabularyStore, context_prompt, whisper_options


def config(**kwargs):
    return SimpleNamespace(
        automatic_learning="on",
        learning_confirmation="on",
        learning_min_occurrences=1,
        learning_observe_seconds=30,
        **kwargs,
    )


def snapshot(text, **changes):
    value = {
        "field_id": "bus|/field",
        "text": text,
        "caret": len(text),
        "selection_start": len(text),
        "selection_end": len(text),
        "focused": True,
        "foreground": True,
        "editable": True,
        "protected": False,
        "terminal": False,
    }
    return {**value, **changes}


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (
            "Use mouth keys today",
            "Use MouthKeys today",
            {"heard": "mouth keys", "corrected": "MouthKeys"},
        ),
        ("Hello wurld.", "Hello world.", {"heard": "wurld", "corrected": "world"}),
        ("Hello world", "hello world", None),
        ("Hello world.", "Hello world!", None),
        ("Number 123", "Number 456", None),
        ("Hello world", "Hello world another sentence", None),
    ],
)
def test_observed_word_edits_not_case_punctuation_or_new_content(before, after, expected):
    assert correction_candidate(before, after, 0, len(before)) == expected


def test_corrections_outside_delivered_range_are_never_learned():
    assert correction_candidate("Other wurld. Hello", "Other world. Hello", 13, 18) is None


def test_learning_is_anchored_debounced_and_requires_confirmation():
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    assert not learner.begin("Hello wurld", snapshot("Other text"))
    assert learner.begin("Hello wurld", snapshot("Prefix Hello wurld"))
    assert learner.observe(snapshot("Prefix Hello world"), now=1) is None
    assert learner.observe(snapshot("Prefix Hello world"), now=3) is None
    proposal = learner.observe(snapshot("Prefix Hello world"), now=4)
    assert proposal["heard"] == "wurld"
    assert proposal["corrected"] == "world"
    assert Dictionary.load()["replacements"] == []
    assert learner.pending() == [proposal]
    learner.accept(proposal["id"])
    assert Dictionary.apply("Hello wurld") == "Hello world"
    assert VocabularyStore.load() == ["world"]
    assert learner.pending() == []
    assert learner.path().stat().st_mode & 0o777 == 0o600
    assert "Prefix" not in learner.path().read_text()


def test_accept_serializes_with_editor_saves(monkeypatch):
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    assert learner.begin("wurld", snapshot("wurld"))
    learner.observe(snapshot("world"), now=1)
    proposal = learner.observe(snapshot("world"), now=4)
    old_dictionary = Dictionary.load()
    edit = Dictionary.defaults()
    edit["actions"] = []
    original_load = Dictionary.load
    monkeypatch.setattr(data_lock, "LOCK_TIMEOUT", 0.02)

    with ThreadPoolExecutor(max_workers=1) as pool, monkeypatch.context() as patch:

        def concurrent_save():
            patch.setattr(Dictionary, "load", original_load)
            data = original_load()
            save = pool.submit(Dictionary.save, edit, expected=old_dictionary)
            with pytest.raises(TimeoutError, match="retry after the current operation"):
                save.result(timeout=2)
            return data

        patch.setattr(Dictionary, "load", concurrent_save)
        learner.accept(proposal["id"])

    assert Dictionary.apply("wurld") == "world"
    with pytest.raises(ValueError, match="Reload before saving"):
        Dictionary.save(edit, expected=old_dictionary)


@pytest.mark.parametrize(
    "change", [{"field_id": "other"}, {"protected": True}, {"focused": False}, {"terminal": True}]
)
def test_wrong_field_protected_unfocused_and_terminal_cancel_observation(change):
    learner = CorrectionLearner(config(), clock=lambda: 0)
    assert learner.begin("wurld", snapshot("wurld"))
    learner.observe(snapshot("world", **change), now=1)
    assert learner.observation is None
    assert learner.pending() == []


def test_partial_keystrokes_are_combined_until_stable():
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    assert learner.begin("Hello wurld", snapshot("Hello wurld"))
    for now, value in enumerate(["Hello ", "Hello w", "Hello wor", "Hello world"], 1):
        assert learner.observe(snapshot(value), now=now) is None
    proposal = learner.observe(snapshot("Hello world"), now=7)
    assert (proposal["heard"], proposal["corrected"]) == ("wurld", "world")


def test_repeated_observation_policy_and_dismissal_cooldown():
    cfg = config()
    cfg.learning_min_occurrences = 2
    now = [10000]
    learner = CorrectionLearner(cfg, clock=lambda: 0, wall_clock=lambda: now[0])
    for expected in (False, True):
        learner.begin("wurld", snapshot("wurld"))
        learner.observe(snapshot("world"), now=1)
        proposal = learner.observe(snapshot("world"), now=4)
        assert bool(proposal) is expected
    learner.dismiss(proposal["id"])
    now[0] += 1000
    learner.begin("wurld", snapshot("wurld"))
    learner.observe(snapshot("world"), now=1)
    assert learner.observe(snapshot("world"), now=4) is None


def test_confirmation_can_be_explicitly_disabled():
    cfg = config()
    cfg.learning_confirmation = "off"
    learner = CorrectionLearner(cfg, clock=lambda: 0, wall_clock=lambda: 10000)
    learner.begin("wurld", snapshot("wurld"))
    learner.observe(snapshot("world"), now=1)
    assert learner.observe(snapshot("world"), now=4)["accepted"] is True
    assert Dictionary.apply("wurld") == "world"


def test_learning_off_and_observation_timeout_do_not_read_or_write_dictionary():
    cfg = config()
    cfg.automatic_learning = "off"
    learner = CorrectionLearner(cfg, clock=lambda: 0)
    assert not learner.begin("wurld", snapshot("wurld"))
    cfg.automatic_learning = "on"
    learner.begin("wurld", snapshot("wurld"))
    assert learner.observe(snapshot("world"), now=31) is None
    assert not learner.path().exists()


def test_vocabulary_native_prompt_is_opt_in_private_and_validated():
    VocabularyStore.save(["MouthKeys", "Ricardo Decal"])
    assert VocabularyStore.path().stat().st_mode & 0o777 == 0o600
    assert whisper_options(SimpleNamespace(vocabulary_boosting="off")) == {}
    cfg = SimpleNamespace(vocabulary_boosting="on")
    assert whisper_options(cfg) == {"hotwords": "MouthKeys, Ricardo Decal"}
    assert context_prompt(cfg) == "Vocabulary: MouthKeys, Ricardo Decal"
    assert VocabularyStore.add("mouthkeys") == ["MouthKeys", "Ricardo Decal"]
    assert VocabularyStore.remove("MOUTHKEYS") == ["Ricardo Decal"]
    with pytest.raises(ValueError, match="Vocabulary entries must be printable text"):
        VocabularyStore.save(["x\nignore instructions"])
    with pytest.raises(ValueError, match="Duplicate vocabulary entry"):
        VocabularyStore.save(["same", "SAME"])


def test_existing_dictionary_entries_do_not_generate_duplicate_suggestions():
    data = Dictionary.defaults()
    data["replacements"] = [{"triggers": ["wurld"], "replacement": "world"}]
    Dictionary.save(data)
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    learner.begin("wurld", snapshot("wurld"))
    learner.observe(snapshot("world"), now=1)
    assert learner.observe(snapshot("world"), now=4) is None
    assert learner.pending() == []


def test_learning_accept_alias_conflict_leaves_stores_and_proposal_unchanged():
    entries = [{"text": "Canonical", "weight": 3, "aliases": ["world"]}]
    VocabularyStore.save_entries(entries)
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    proposal = learner._suggest({"heard": "wurld", "corrected": "world"})
    dictionary = Dictionary.load()
    policy = learner.path().read_bytes()
    with pytest.raises(ValueError, match="aliases must not override"):
        learner.accept(proposal["id"])
    assert Dictionary.load() == dictionary
    assert VocabularyStore.entries() == entries
    assert learner.path().read_bytes() == policy
    assert learner.pending() == [proposal]


def test_learning_accept_missing_policy_leaves_feature_stores_unchanged():
    entries = [{"text": "Canonical", "weight": 3, "aliases": ["spelling"]}]
    VocabularyStore.save_entries(entries)
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    proposal = learner._suggest({"heard": "wurld", "corrected": "world"})
    learner.path().unlink()
    dictionary = Dictionary.load()
    with pytest.raises(ValueError, match="suggestion no longer exists"):
        learner.accept(proposal["id"])
    assert Dictionary.load() == dictionary
    assert VocabularyStore.entries() == entries
    assert learner.pending() == [proposal]


def test_learning_accept_preserves_existing_vocabulary_metadata():
    entries = [{"text": "Canonical", "weight": 3, "aliases": ["spelling"]}]
    VocabularyStore.save_entries(entries)
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    proposal = learner._suggest({"heard": "wurld", "corrected": "world"})
    learner.accept(proposal["id"])
    assert Dictionary.apply("wurld") == "world"
    assert VocabularyStore.entries() == [
        *entries,
        {"text": "world", "weight": 1, "aliases": []},
    ]


def test_selection_before_word_edit_does_not_cancel_observation():
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    learner.begin("Hello wurld", snapshot("Hello wurld"))
    selected = snapshot("Hello wurld", selection_start=6, selection_end=11)
    assert learner.observe(selected, now=1) is None
    assert learner.observation is not None
    learner.observe(snapshot("Hello world"), now=2)
    assert learner.observe(snapshot("Hello world"), now=5)["corrected"] == "world"


def test_weighted_vocabulary_and_aliases_roundtrip_without_losing_metadata():
    cfg = SimpleNamespace(vocabulary_boosting="on")
    entries = [{"text": "MouthKeys", "weight": 3, "aliases": ["mouth keys", "mouse keys"]}]
    VocabularyStore.save_entries(entries)
    assert VocabularyStore.entries() == entries
    assert VocabularyStore.load() == ["MouthKeys"]
    assert whisper_options(cfg) == {"hotwords": "MouthKeys, MouthKeys, MouthKeys"}
    assert context_prompt(cfg) == "Vocabulary: MouthKeys, MouthKeys, MouthKeys"
    assert (
        VocabularyStore.apply_aliases("Use MOUTH keys, not mouse keyboard")
        == "Use MouthKeys, not mouse keyboard"
    )
    VocabularyStore.save(["MouthKeys", "NewWord"])
    assert VocabularyStore.entries()[0] == entries[0]
    VocabularyStore.remove("NewWord")
    assert VocabularyStore.entries() == entries


@pytest.mark.parametrize("weight", [0, -1, True, float("inf"), float("nan"), "2"])
def test_invalid_prompt_weights_are_rejected(weight):
    with pytest.raises(ValueError, match="weight"):
        VocabularyStore.save_entries([{"text": "term", "weight": weight, "aliases": []}])


def test_conflicting_aliases_rejected_and_textual_hints_bounded_independently():
    with pytest.raises(ValueError, match="alias"):
        VocabularyStore.save_entries(
            [
                {"text": "first", "weight": 1, "aliases": ["heard"]},
                {"text": "second", "weight": 1, "aliases": ["HEARD"]},
            ]
        )
    entries = [
        {"text": str(index) + "x" * 100, "weight": 8.5, "aliases": []} for index in range(100)
    ]
    VocabularyStore.save_entries(entries)
    assert VocabularyStore.entries() == entries
    cfg = SimpleNamespace(vocabulary_boosting="on")
    assert len(context_prompt(cfg).encode()) <= 4096
    assert len(whisper_options(cfg)["hotwords"].encode()) <= 4096


def test_failed_dismiss_keeps_the_proposal_available_for_retry(monkeypatch):
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    assert learner.begin("wurld", snapshot("wurld"))
    learner.observe(snapshot("world"), now=1)
    proposal = learner.observe(snapshot("world"), now=4)

    def fail(_data):
        raise OSError("Policy write failed")

    with monkeypatch.context() as patch:
        patch.setattr(learner, "_save", fail)
        with pytest.raises(OSError, match="Policy write failed"):
            learner.dismiss(proposal["id"])
    assert learner.pending() == [proposal]
    learner.dismiss(proposal["id"])
    assert learner.pending() == []


@pytest.mark.parametrize(
    ("before", "after", "insertion_start"),
    [
        ("same", "same", 0),
        ("Prefixwurld", "Prefixworld", 6),
        ("deleted", "", 0),
        ("w" * 41, "v" * 41, 0),
    ],
)
def test_unchanged_deleted_excessive_and_cross_boundary_words_are_ignored(
    before, after, insertion_start
):
    assert correction_candidate(before, after, insertion_start, len(before)) is None


def test_observer_drops_edit_outside_inserted_span_and_can_restart_with_new_config():
    cfg = config()
    cfg.automatic_learning = "off"
    learner = CorrectionLearner(cfg, clock=lambda: 0, wall_clock=lambda: 10000)
    assert learner.observe(snapshot("Unrelated"), now=1) is None
    assert learner.begin("wurld", snapshot("Prefix wurld"), config=config())
    assert learner.observe(snapshot("Unrelated wurld"), now=1) is None
    assert learner.observation is None
    assert learner.observe(snapshot("Unrelated world"), now=4) is None
    assert learner.pending() == []
    assert not learner.path().exists()


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"[]", "Invalid learning policy"),
        (b'{"pairs": 5}', "Invalid learning policy"),
        (b"x" * (256 * 1024 + 1), "exceeds 256 KiB"),
    ],
)
def test_invalid_policy_never_suggests_or_writes_dictionary(contents, message):
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    path = learner.path()
    path.parent.mkdir(parents=True)
    path.write_bytes(contents)
    assert learner.begin("wurld", snapshot("wurld"))
    assert learner.observe(snapshot("world"), now=1) is None
    with pytest.raises(ValueError, match=message):
        learner.observe(snapshot("world"), now=4)
    assert path.read_bytes() == contents
    assert not Dictionary.path().exists()
    assert not VocabularyStore.path().exists()
    assert learner.pending() == []


def test_pending_proposals_keep_only_the_most_recent(monkeypatch):
    monkeypatch.setattr(learning, "MAX_PROPOSALS", 2)
    now = [10000]
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: now[0])
    proposals = []
    for index in range(3):
        heard, corrected = f"wurld{index}", f"world{index}"
        assert learner.begin(heard, snapshot(heard))
        learner.observe(snapshot(corrected), now=1)
        proposals.append(learner.observe(snapshot(corrected), now=4))
        now[0] += 601
    assert all(proposals)
    assert learner.pending() == proposals[-2:]


@pytest.mark.parametrize("replacement", ["world", "other interpretation"])
def test_existing_dictionary_update_is_never_overwritten_on_review(replacement):
    learner = CorrectionLearner(config(), clock=lambda: 0, wall_clock=lambda: 10000)
    assert learner.begin("wurld", snapshot("wurld"))
    learner.observe(snapshot("world"), now=1)
    proposal = learner.observe(snapshot("world"), now=4)
    data = Dictionary.defaults()
    data["replacements"] = [
        {"triggers": ["separate"], "replacement": "another"},
        {"triggers": ["WURLD"], "replacement": replacement, "enabled": False},
    ]
    Dictionary.save(data)
    words = [{"text": "World", "weight": 3, "aliases": []}]
    VocabularyStore.save_entries(words)

    if replacement == "world":
        assert learner.accept(proposal["id"]) == proposal
        assert learner.pending() == []
    else:
        with pytest.raises(ValueError, match="Existing replacement conflicts"):
            learner.accept(proposal["id"])
        assert learner.pending() == [proposal]
    assert Dictionary.load() == data
    assert VocabularyStore.entries() == words
