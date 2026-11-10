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
    assert format_text(text, config) == "hello, world\nslash compact? Next?"
    assert format_text("at work mention ricardo", config) == "at work @ricardo"
    assert format_text("cd ..", replace(config, lowercase_first="off")) == "cd ."


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

    config = Config(
        ai_enabled="on",
        ai_model="local-model",
        ai_key_env="TEST_AI_KEY",
        ai_custom_prompt_only="on",
    )
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
    assert format_text("World", config, context="Hello") == " world "
    assert format_text("world", config, context="Done. ") == "World "
    assert format_text("world", config, context="") == "World "
    assert format_text("literal new line", Config(spoken_formatting="on")) == "\n"
    assert format_text("hello NASA", Config(lowercase_first="on")) == "hello NASA"
    assert format_text("École", Config(lowercase_first="on")) == "école"


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Hello literal comma world literal question mark literal open paren yes "
            "literal close paren literal quote done literal quote",
            'Hello, world? (yes) "done"',
        ),
        ("Hello comma world question mark", "Hello comma world question mark"),
        (
            "email literal at the rate example literal dot com literal slash help "
            "literal underscore me",
            "email@example.com/help_me",
        ),
        ("email literal at sign example literal dot com", "email@example.com"),
        ("x literal hyphen ray costs 50 literal percent", "x-ray costs 50%"),
        ("a literal plus b literal equals c", "a + b = c"),
        ("literal plus literal equal 50 literal percent", "+ = 50%"),
        ("example literal dot com", "example.com"),
        ("version 1 literal dot 2", "version 1.2"),
        ("literal hyphen literal comma literal hyphen literal comma literal hyphen", "---"),
        ("50 literal comma literal percent", "50%"),
        ("literal open bracket literal comma literal close bracket", "[]"),
        ("literal open paren literal comma literal close paren", "()"),
        ("literal question mark literal comma literal exclamation mark", "?!"),
        ("Thanks, @Sam", "Thanks, @Sam"),
        ("Use C++, now", "Use C++, now"),
        ("-,-,-", "-,-,-"),
        ("50, %", "50, %"),
        ("First literal next line second", "First\nsecond"),
        ("First literal next paragraph second", "First\n\nsecond"),
        ("one literal tab two", "one\ttwo"),
        ("one   literal space   two", "one two"),
        ("First next line second", "First next line second"),
        ("First. literal new line. Second", "First\nSecond"),
        ("First. literal new paragraph. Second", "First\n\nSecond"),
        ("one. literal tab. two", "one\ttwo"),
        ("one. literal space. two", "one two"),
        ("First literal period literal new line Second", "First.\nSecond"),
        ("First literal new line, Second", "First\nSecond"),
        ("First literal new paragraph, Second", "First\n\nSecond"),
        ("First literal new line literal comma Second", "First\n, Second"),
        ("one literal tab, two", "one\t, two"),
        ("Wait... literal new line. Continue", "Wait...\nContinue"),
        ("one literal space.. two", "one .. two"),
        ("literal, comma", "literal, comma"),
        ("literal\ncomma", "literal\ncomma"),
    ],
)
def test_spoken_punctuation_upstream_rendering_contract(text, expected):
    # Adapted from upstream DictationE2ETests, not inferred from Linux implementation.
    assert format_text(text, Config(spoken_formatting="on")) == expected


def test_full_upstream_symbol_alias_inventory_and_custom_rows_are_independent():
    data = Dictionary.defaults()
    expected = {
        "next line": "\n",
        "next paragraph": "\n\n",
        "dot": ".",
        "questionmark": "?",
        "bang": "!",
        "semi colon": ";",
        "three dots": "...",
        "forwardslash": "/",
        "back slash": "\\",
        "hyphen": "-",
        "long dash": "—",
        "en dash": "–",
        "left paren": "(",
        "right paren": ")",
        "left square bracket": "[",
        "right square bracket": "]",
        "open curly bracket": "{",
        "close curly bracket": "}",
        "less than sign": "<",
        "greater than sign": ">",
        "double quote": '"',
        "opening double quote": '"',
        "closing double quote": '"',
        "single quote": "'",
        "apostrophe": "'",
        "commercial at": "@",
        "and sign": "&",
        "plus": "+",
        "equal": "=",
        "percentage sign": "%",
        "dollar": "$",
        "number sign": "#",
        "star symbol": "*",
        "vertical bar": "|",
        "tilde": "~",
        "caret": "^",
        "back tick": "`",
    }
    for alias, symbol in expected.items():
        assert (
            format_text("literal " + alias, Config(spoken_formatting="on"), dictionary=data)
            == symbol
        )
    data["punctuation"][0]["aliases"].append("custom comma")
    assert "custom comma" not in Dictionary.defaults()["punctuation"][0]["aliases"]


def test_saved_dictionary_is_not_replaced_and_rule_edits_take_effect():
    data = {
        "replacements": [],
        "punctuation": [{"aliases": ["right arrow", "arrow"], "symbol": "->"}],
        "actions": [],
    }
    Dictionary.save(data)
    assert Dictionary.load() == data
    config = Config(spoken_formatting="on", literal_prefix="type")
    assert format_text("one type right arrow two", config) == "one->two"
    assert format_text("type comma", config) == "type comma"
    assert format_text("literal right arrow", config) == "literal right arrow"
    data["actions"] = [
        {"aliases": ["drop down"], "symbol": "\n"},
        {"aliases": ["little gap"], "symbol": " ", "enabled": False},
    ]
    assert format_text("First type drop down second", config, dictionary=data) == "First\nsecond"
    assert format_text("type little gap", config, dictionary=data) == "type little gap"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("slash compact", "/compact"),
        ("Use slash Compact", "Use /compact"),
        ("/ PLAN", "/plan"),
        ("Run / STATUS please", "Run /status please"),
        ("Do forward slash model", "Do /model"),
        (
            "Use 1/2 and and/or. Open src slash services. Go to https slash slash example dot com. "
            "Slash and burn.",
            "Use 1/2 and and/or. Open src slash services. Go to https slash slash example dot com. "
            "Slash and burn.",
        ),
        ("I saw slash compact", "I saw slash compact"),
        ("at the rate Paul", "@Paul"),
        ("Mention Paul Smith", "@Paul Smith"),
        ("tag ricardo", "@ricardo"),
        ("at sign work", "at sign work"),
        ("mention Paul's work", "mention Paul's work"),
        ("at Paul", "at Paul"),
    ],
)
def test_literal_commands_and_explicit_mentions_match_upstream(text, expected):
    assert format_text(text, Config(literal_formatting="on")) == expected


@pytest.mark.parametrize(
    "metadata",
    [
        {"app_name": "Slack"},
        {"app_id": "com.discordapp.Discord"},
        {"window_title": "Microsoft Teams"},
    ],
)
def test_relaxed_mentions_need_real_chat_context(metadata):
    config = Config(literal_formatting="on")
    assert format_text("Ping at Paul Smith", config, **metadata) == "Ping @Paul Smith"
    prose = "I am at the store. Meet me at lunch. I am at Paul. Look at Paul's message."
    assert format_text(prose, config, **metadata) == prose
    assert format_text("at paul", config, **metadata) == "at paul"
    assert format_text("at Paul", config) == "at Paul"


@pytest.mark.parametrize(
    "text,context,metadata,expected",
    [
        ("slash model", "", {"app_name": "Codex"}, "/Model"),
        ("slash model", "", {"app_name": "Notes"}, "/Model "),
        ("Run slash status please", "", {"window_title": "Codex"}, "Run /status please "),
        ("mention Paul", "", {"app_id": "Slack"}, "@Paul"),
        ("mention Paul can you check this", "", {"app_name": "Slack"}, "@Paul can you check this "),
        ("world", 'Done.") ', {}, "World "),
        ("world", "hello\n", {}, "World "),
        ("World", 'hello") ', {}, "world "),
        ("world", "Hello", {}, " world "),
        ("world", None, {}, "world"),
    ],
)
def test_continuous_formatting_and_autocomplete_contract(text, context, metadata, expected):
    config = Config(
        literal_formatting="on", space_between_dictations="on", smart_capitalization="on"
    )
    assert format_text(text, config, context=context, **metadata) == expected


def test_disabled_formatting_preserves_whitespace_and_explicit_gaav_matches_upstream():
    assert format_text("  Keep   spacing  ", Config()) == "  Keep   spacing  "
    assert format_text('"Hello."', Config(lowercase_first="on")) == '"Hello."'
    assert format_text("cd ..", Config(remove_trailing_period="on")) == "cd ."
