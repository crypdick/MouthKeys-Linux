"""Dictation template parity independent of generic Command/Edit transports."""

import threading
import uuid
from dataclasses import asdict, replace

import pytest

from mouthkeys import postprocess, profiles
from mouthkeys.config import Config


@pytest.mark.parametrize(
    "prompt,transcript,expected",
    [
        ("Rewrite ${transcript} please", "Hello", "Rewrite Hello please"),
        ("${transcript}\n${transcript}", "$1\\words", "$1\\words\n$1\\words"),
        ("  Rewrite.\n", "Hello", "  Rewrite.\n\n\nHello"),
        (" \t\n", "Hello", "Hello"),
        ("", "Hello", "Hello"),
        ("${TRANSCRIPT}", "Hello", "${TRANSCRIPT}\n\nHello"),
    ],
)
def test_upstream_dictation_template_examples(prompt, transcript, expected):
    assert postprocess.render_dictation_user_message(prompt, transcript) == expected


def test_profile_template_is_rendered_once_and_keeps_draft_immutable(monkeypatch):
    config = Config(
        ai_enabled="on",
        ai_model="draft",
        ai_prompt="Draft ${transcript}",
        ai_custom_prompt_only="on",
    )
    values = asdict(replace(config, ai_model="profile-model", ai_prompt="Translate: ${transcript}"))
    profile = {
        "id": str(uuid.uuid4()),
        "name": "Translate",
        "shortcut": "",
        **{name: values[name] for name in profiles.PROFILE_FIELDS},
    }
    profiles.save_profiles([profile])
    selected = replace(config, ai_profile=profile["id"])
    calls = []
    cancel = threading.Event()
    delta = lambda text: None  # noqa: E731
    opener = object()

    def request(text, effective, **options):
        calls.append((text, effective, options))
        return "Translated"

    monkeypatch.setattr(postprocess, "request_completion", request)
    assert (
        postprocess.process(
            " \nraw ${transcript}\t", selected, opener=opener, on_delta=delta, cancel_event=cancel
        )
        == "Translated"
    )
    text, effective, options = calls[0]
    assert text == "Translate: raw ${transcript}"
    assert effective.ai_model == "profile-model"
    assert effective.ai_prompt == ""
    assert options == {"opener": opener, "on_delta": delta, "cancel_event": cancel}
    assert selected.ai_prompt == "Draft ${transcript}"


def test_builtin_dictation_prompt_is_configurable_and_never_duplicated(monkeypatch):
    calls = []
    monkeypatch.setattr(
        postprocess,
        "request_completion",
        lambda text, config, **kwargs: calls.append(text) or "Done",
    )
    config = Config(ai_enabled="on", ai_prompt="Translate ${transcript}")
    postprocess.process("Hello", config)
    assert calls[-1] == postprocess.BASE_DICTATION_PROMPT + "\n\nTranslate Hello"
    postprocess.process("Hello", replace(config, ai_custom_prompt_only="on"))
    assert calls[-1] == "Translate Hello"
    composed = replace(
        config, ai_prompt=postprocess.BASE_DICTATION_PROMPT + "\n\nTranslate ${transcript}"
    )
    postprocess.process("Hello", composed)
    assert calls[-1].count(postprocess.BASE_DICTATION_PROMPT) == 1


def test_blank_dictation_and_disabled_cleanup_never_resolve_profile_or_request(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No profile lookup or network request expected")

    monkeypatch.setattr(profiles, "resolve_profile", forbidden)
    monkeypatch.setattr(postprocess, "request_completion", forbidden)
    assert postprocess.process(" \n\t", Config(ai_enabled="on")) == ""
    assert postprocess.process("  unchanged  ", Config()) == "  unchanged  "
