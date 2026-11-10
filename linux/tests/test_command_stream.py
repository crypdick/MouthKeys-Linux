import json

import pytest

from mouthkeys import assistance
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
    stream = assistance._CommandExplanationStream(deltas.append)
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
    stream = assistance._CommandExplanationStream(lambda _: pytest.fail("Non-explanation leaked"))
    stream.feed('{"metadata":"')
    for _ in range(2000):
        stream.feed("x" * 64)
    assert stream.phase == "done"
    assert stream.work <= 1024 * 1024 + 128 * 1024
    assert len(stream.buffer.encode()) <= 128 * 1024
