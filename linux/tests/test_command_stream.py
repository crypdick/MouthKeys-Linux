import json

import pytest

from mouthkeys import assistance
from mouthkeys.command_preview import MAX_RESULT_BYTES, CommandExplanationPreview
from mouthkeys.config import Config


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 2, 7, 32])
def test_only_root_explanation_streams_with_split_unicode_and_escapes(
    monkeypatch, fenced, chunk_size
):
    expected = 'Answer: "hello"\npath\\name / 日本語 😀'
    output = json.dumps(
        {
            "metadata": {"explanation": "NESTED MUST NOT LEAK", "array": [1, {"x": "secret"}]},
            "command": "SHELL COMMAND MUST NOT LEAK",
            "explanation": expected,
        }
    )
    if fenced:
        output = "```json\n" + output + "\n```"
    deltas = []

    def completion(text, config, **kwargs):
        for index in range(0, len(output), chunk_size):
            kwargs["on_delta"](output[index : index + chunk_size])
            assert expected.startswith("".join(deltas))
        return output

    monkeypatch.setattr(assistance, "request_completion", completion)
    proposal = assistance.propose(
        "Help", Config(ai_enabled="on"), "command", on_delta=deltas.append
    )
    assert "".join(deltas) == expected
    assert proposal["text"] == expected
    assert proposal["command"] == "SHELL COMMAND MUST NOT LEAK"


def test_incomplete_surrogate_is_held_until_complete_pair():
    deltas = []
    stream = CommandExplanationPreview(deltas.append)
    stream.feed('{"explanation":"First \\uD83D')
    assert deltas == ["First "]
    stream.feed("\\uDE")
    assert deltas == ["First "]
    stream.feed('00 last","command":""}')
    assert "".join(deltas) == "First 😀 last"


@pytest.mark.parametrize(
    "output",
    [
        '{"metadata":{"explanation":"nested"},"command":"ls"}',
        '{"explanation":"Visible answer","command":42}',
        '{"explanation":"Partial answer',
        '{"explanation":"Bad \\u 123","command":"ls"}',
    ],
)
def test_malformed_command_result_never_returns_actionable_proposal(monkeypatch, output):
    deltas = []

    def completion(text, config, **kwargs):
        for char in output:
            kwargs["on_delta"](char)
        return output

    monkeypatch.setattr(assistance, "request_completion", completion)
    with pytest.raises(ValueError, match="no valid command proposal"):
        assistance.propose("Help", Config(ai_enabled="on"), "command", on_delta=deltas.append)
    assert "nested" not in "".join(deltas)
    assert "ls" not in "".join(deltas)


def test_long_incomplete_preceding_value_has_bounded_streaming_work():
    stream = CommandExplanationPreview(lambda _: pytest.fail("Non-explanation leaked"))
    stream.feed('{"metadata":"')
    for _ in range(2000):
        stream.feed("x" * 64)
    assert stream.phase == "done"
    assert stream.work <= 1024 * 1024 + 128 * 1024
    assert len(stream.buffer.encode()) <= 128 * 1024


@pytest.mark.parametrize("repeated_fence", [False, True])
def test_preview_stops_at_invalid_envelope_or_wire_text(monkeypatch, repeated_fence):
    output = (
        '```json\n```json\n{"explanation":"Do not display","command":""}\n```'
        if repeated_fence
        else '{"explanation":"Invalid \ud800","command":""}'
    )
    deltas = []

    def completion(_text, _config, **kwargs):
        kwargs["on_delta"](output)
        return output

    monkeypatch.setattr(assistance, "request_completion", completion)
    message = "no valid command proposal" if repeated_fence else "invalid Unicode"
    with pytest.raises(ValueError, match=message):
        assistance.propose("Help", Config(ai_enabled="on"), "command", on_delta=deltas.append)
    assert deltas == []


@pytest.mark.parametrize("field", ["command", "explanation", "working_directory"])
def test_escaped_unpaired_surrogate_never_returns_an_actionable_proposal(
    monkeypatch, tmp_path, field
):
    fields = {"command": "printf demo", "explanation": "Explain", "working_directory": ""}
    fields[field] += "\ud800"
    if field == "working_directory":
        fields[field] = f"{tmp_path}/{fields[field]}"
    monkeypatch.setattr(
        assistance, "request_completion", lambda *_args, **_options: json.dumps(fields)
    )
    with pytest.raises(ValueError, match="invalid Unicode"):
        assistance.propose("Help", Config(ai_enabled="on"), "command")


def test_deep_preceding_value_fails_as_invalid_proposal_without_emitting(monkeypatch):
    output = (
        '{"metadata":' + "[" * 1500 + "0" + "]" * 1500 + ',"explanation":"Hidden","command":""}'
    )

    raw_decode = json.JSONDecoder.raw_decode

    def recursion_limited_decoder(decoder, text, idx=0):
        # Exercise the decoder's recursion limit without depending on interpreter depth.
        if text.startswith(('{"metadata":', "["), idx):
            raise RecursionError
        return raw_decode(decoder, text, idx)

    def completion(_text, _config, **kwargs):
        kwargs["on_delta"](output)
        return output

    monkeypatch.setattr(json.JSONDecoder, "raw_decode", recursion_limited_decoder)
    monkeypatch.setattr(assistance, "request_completion", completion)
    with pytest.raises(ValueError, match="no valid command proposal"):
        assistance.propose(
            "Help",
            Config(ai_enabled="on"),
            "command",
            on_delta=lambda _: pytest.fail("Invalid envelope leaked explanation"),
        )


@pytest.mark.parametrize(
    "prefix",
    [
        "plain text",
        "```bash\n",
        '{42: "bad",',
        '{"command" "ignored",',
        '{"command":123x,',
        '{"command":0\n}',
        '{"explanation":0,',
    ],
)
def test_preview_does_not_resynchronize_after_invalid_or_closed_root(prefix):
    pieces = []
    preview = CommandExplanationPreview(pieces.append)
    preview.feed(prefix)
    preview.feed(' "explanation":"Not valid" }')
    assert pieces == []
    assert preview.phase == "done"


@pytest.mark.parametrize(
    "bad_suffix",
    [
        "\x01raw",
        "\\q",
        "\\u000G",
        "\\uDE00",
        "\\uD83Dabcdef",
        "\\uD83D\\u0041",
        "\\uD83D\\u000G",
    ],
)
def test_preview_keeps_only_text_before_malformed_string(bad_suffix):
    pieces = []
    preview = CommandExplanationPreview(pieces.append)
    preview.feed('{"explanation":"Visible' + bad_suffix)
    preview.feed('hidden"}')
    assert "".join(pieces) == "Visible"
    assert preview.phase == "done"


def test_preview_accepts_json_whitespace_delimiters_and_escapes():
    pieces = []
    preview = CommandExplanationPreview(pieces.append)
    for text in [
        '  {"command":-1',
        '23 \r \n  , "explanation"  :  "tab\\tbackspace\\bnewline\\r',
        'return\\fslash\\/escape\\u0061end"}',
    ]:
        preview.feed(text)
    assert "".join(pieces) == "tab\tbackspace\bnewline\rreturn\fslash/escapeaend"


def test_preview_stops_before_buffering_oversized_utf8_stream():
    pieces = []
    preview = CommandExplanationPreview(pieces.append)
    prefix = '{"explanation":"'
    preview.feed(prefix)
    preview.feed("é" * (MAX_RESULT_BYTES // 2))
    preview.feed('late"}')
    assert not pieces
    assert preview.buffer == prefix


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"command": "a" * 16385}, "invalid shell command"),
        ({"command": "broken\0value"}, "invalid shell command"),
        ({"working_directory": 12}, "invalid working directory"),
        ({"working_directory": "/" + "a" * 4096}, "invalid working directory"),
        ({"working_directory": "/invalid\0path"}, "invalid working directory"),
        ({"working_directory": "relative/path"}, "working directory must be absolute"),
    ],
)
def test_final_command_validation_is_independent_of_preview(monkeypatch, fields, message):
    output = json.dumps({"command": "printf demo", "explanation": "Explain", **fields})
    monkeypatch.setattr(assistance, "request_completion", lambda *_args, **_options: output)
    with pytest.raises(ValueError, match=message):
        assistance.propose("Help", Config(ai_enabled="on"), "command")


def test_final_limit_measures_encoded_bytes(monkeypatch):
    output = "é" * (MAX_RESULT_BYTES // 2 + 1)
    monkeypatch.setattr(assistance, "request_completion", lambda *_args, **_options: output)
    with pytest.raises(ValueError, match="exceeds 128 KiB"):
        assistance.propose("Help", Config(ai_enabled="on"), "edit")


@pytest.mark.parametrize(
    ("instruction", "mode", "selected_text", "error"),
    [
        ("Help", "missing", "", "Assistant mode must be command or edit"),
        ("  ", "command", "", "Provide an instruction first"),
        (None, "edit", "", "Provide an instruction first"),
        ("Help", "edit", 4, "Selected text must be text"),
    ],
)
def test_invalid_request_never_contacts_provider(
    monkeypatch, instruction, mode, selected_text, error
):
    monkeypatch.setattr(
        assistance, "request_completion", lambda *_args, **_options: pytest.fail("Provider called")
    )
    with pytest.raises(ValueError, match=error):
        assistance.propose(instruction, Config(ai_enabled="on"), mode, selected_text)


def test_completed_command_uses_trimmed_command_when_explanation_is_empty(monkeypatch, tmp_path):
    response = json.dumps(
        {"command": "  printf demo  ", "explanation": "  ", "working_directory": str(tmp_path)}
    )
    monkeypatch.setattr(assistance, "request_completion", lambda *_args, **_options: response)
    proposal = assistance.propose("Help", Config(ai_enabled="on"), "command")
    assert proposal["command"] == proposal["text"] == "printf demo"
    assert proposal["working_directory"] == str(tmp_path)
