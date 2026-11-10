import json
from dataclasses import replace
from io import BytesIO

import pytest

from mouthkeys.config import Config
from mouthkeys.dictionary import Dictionary
from mouthkeys.formatting import apply_question_mark_shortcut, format_text, parse_send
from mouthkeys.postprocess import process


def test_dictionary_replacements_are_bounded_case_insensitive_and_one_pass(tmp_path):
    data = Dictionary.defaults()
    data["replacements"] = [
        {"triggers": ["mouth keys", "mouthkeys"], "replacement": "MouthKeys"},
        {"triggers": ["MouthKeys" + " Linux"], "replacement": "Linux port"},
    ]
    Dictionary.save(data)
    assert Dictionary.apply("MOUTH   KEYS, mouthkeyss mouthkeys Linux") == (
        "MouthKeys, mouthkeyss Linux port"
    )
    exported = tmp_path / "dictionary.json"
    Dictionary.export_file(exported)
    assert exported.stat().st_mode & 0o777 == 0o600
    Dictionary.save(Dictionary.defaults())
    Dictionary.import_file(exported)
    assert Dictionary.load() == data


def test_dictionary_validation_rejects_duplicate_and_invalid_entries():
    data = Dictionary.defaults()
    data["replacements"] = [{"triggers": ["same", " SAME "], "replacement": "x"}]
    with pytest.raises(ValueError, match="Duplicate"):
        Dictionary.save(data)
    data["replacements"] = [{"triggers": "same", "replacement": "x"}]
    with pytest.raises(ValueError, match="Invalid"):
        Dictionary.save(data)


def test_formatting_is_opt_in_and_handles_literals_and_qq():
    text = "Um, Hello literal comma world literal new line slash compact Q Q Next Q."
    assert format_text(text, Config()) == text
    config = Config(
        remove_filler_words="on",
        spoken_formatting="on",
        literal_formatting="on",
        question_mark_shortcut="on",
        lowercase_first="on",
        remove_trailing_period="on",
    )
    assert format_text(text, config) == "hello, world\n/compact? Next?"
    assert format_text("at work mention ricardo", config) == "at work @ricardo"
    assert format_text("cd ..", replace(config, lowercase_first="off")) == "cd .."


@pytest.mark.parametrize(
    "text,expected,send",
    [
        ("Fix the typo send it.", "Fix the typo", True),
        ("Fix the typo and send it", "Fix the typo", True),
        ("Okay, send it", "Okay", True),
        ("Can you send it?", "Can you send it?", False),
        ("Don't send it", "Don't send it", False),
        ("I'll just send it", "I'll just send it", False),
        ("I already send it", "I already send it", False),
        ("Thank you send it", "Thank you", True),
        ("Literal, send it.", "send it", False),
        ("Um, send it", "", True),
        ("I'll send it, send it", "I'll send it", True),
        ("send it tomorrow", "send it tomorrow", False),
        ("Just send it", "", True),
    ],
)
def test_spoken_send_does_not_mistake_sentences_for_commands(text, expected, send):
    plan = parse_send(text, active=True)
    assert (plan.text, plan.should_send) == (expected, send)
    assert parse_send(text, active=False).text == text


def test_custom_send_phrase_and_literal_prefix():
    assert parse_send("hello Send, send.", "send send", True).should_send
    assert not parse_send("verbatim send send", "send send", True, "verbatim").should_send


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Is that right Q", "Is that right?"),
        ("Is that right Q.", "Is that right?"),
        ("Is that right, Q?", "Is that right?"),
        ("Is that right? Q.", "Is that right?"),
        ("Is that right cue", "Is that right?"),
        ("Q", "?"),
        ("Is this OK Q Q and then deploy it", "Is this OK? And then deploy it"),
        ("Is this OK QQ. Then deploy", "Is this OK? Then deploy"),
        ("Is this OK queue, queue", "Is this OK?"),
        ("Is this OK cue cue.", "Is this OK?"),
        ("Was it fast Q. Q.", "Was it fast?"),
        ("Does it work Q Q iPhone first", "Does it work? iPhone first"),
        ("Is it fast Q Q the API is", "Is it fast? The API is"),
        ("Works Q Q API stays uppercase", "Works? API stays uppercase"),
        ("Is this OK Q Q\nNext item", "Is this OK?\nNext item"),
        ("Plan literal Q", "Plan Q"),
        ("Call it literal Q Q", "Call it Q Q"),
        ("Literal Q.", "Q."),
    ],
)
def test_question_shortcut_matches_upstream_spoken_send_examples(text, expected):
    assert format_text(text, Config(question_mark_shortcut="on")) == expected
    assert format_text(text, Config()) == text


@pytest.mark.parametrize(
    "text",
    [
        "Add it to the queue",
        "Add it to the queue.",
        "The Q3 numbers and IQ tests",
        "Star Trek's Q is back",
        "Q&A",
        "uv run pytest -q",
        "apt-get install -qq",
        "cd ~/q",
        "Make a Q-Q plot",
        "path/q",
        r"path\q",
        "flag=q",
        "test@qq",
        "variable_q",
        "éq",
        "q’",
        "#q",
        "$q",
        "file.q",
    ],
)
def test_question_shortcut_preserves_words_flags_paths_and_identifiers(text):
    assert format_text(text, Config(question_mark_shortcut="on")) == text


def test_question_shortcut_preserves_newline_spacing_and_custom_literal_prefix_once():
    assert apply_question_mark_shortcut("Works Q Q\n\t  next") == "Works?\n\t  Next"
    assert apply_question_mark_shortcut("Plan leave alone Q", "leave alone") == "Plan Q"
    assert apply_question_mark_shortcut("Call it literal Q Q.") == "Call it Q Q."
    plan = parse_send("Is that right Q send send", "send send", True)
    assert plan.should_send
    assert format_text(plan.text, Config(question_mark_shortcut="on")) == "Is that right?"


def test_ai_disabled_does_not_make_any_request():
    def forbidden(*args, **kwargs):
        raise AssertionError("Network called while disabled")

    assert process("private words", Config(), opener=forbidden) == "private words"


def test_ai_cleanup_explicit_payload_env_secret_and_bounded_timeout(monkeypatch):
    monkeypatch.setenv("TEST_AI_KEY", "secret-token")
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return BytesIO(json.dumps({"choices": [{"message": {"content": "Clean text"}}]}).encode())

    config = Config(ai_enabled="on", ai_model="local-model", ai_key_env="TEST_AI_KEY")
    assert process("raw text", config, opener=opener) == "Clean text"
    request, timeout = calls[0]
    assert timeout == 30
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert json.loads(request.data)["messages"][-1]["content"] == config.ai_prompt + "\n\nraw text"
    assert "secret-token" not in json.dumps(json.loads(request.data))


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com/v1/chat/completions",
        "file:///etc/passwd",
        "https://u:p@example.com",
    ],
)
def test_ai_rejects_unsafe_endpoints_before_network(endpoint):
    with pytest.raises(ValueError, match="HTTP|HTTPS"):
        process("raw", Config(ai_enabled="on", ai_model="model", ai_endpoint=endpoint))


def test_ai_rejects_invalid_empty_and_oversized_completions():
    config = Config(ai_enabled="on", ai_model="model")
    for body in (b"{}", b"not-json", b"x" * (1024 * 1024 + 1)):
        with pytest.raises(ValueError):
            process("raw", config, opener=lambda *a, body=body, **kw: BytesIO(body))


def test_context_formatting_requires_actual_context_and_preserves_explicit_overrides():
    config = Config(smart_capitalization="on", space_between_dictations="on")
    assert format_text("World", config, context=None) == "World"
    assert format_text("World", config, context="Hello") == " world"
    assert format_text("world", config, context="Done. ") == "World"
    assert format_text("world", config, context="") == "World"
    assert format_text("literal new line", Config(spoken_formatting="on")) == "\n"
    assert format_text("hello NASA", Config(lowercase_first="on")) == "hello NASA"
    assert format_text("École", Config(lowercase_first="on")) == "école"
