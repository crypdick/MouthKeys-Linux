import json

import pytest

from mouthkeys.model_progress import MAX_OUTPUT_BYTES, DownloadOutput


def line(payload):
    return (json.dumps({"download_progress": payload}, ensure_ascii=False) + "\n").encode()


def test_output_reassembles_partial_utf8_and_emits_only_complete_progress():
    output = DownloadOutput()
    event = {"phase": "Préparation", "completed": 1.5, "total": 2, "unit": "stage"}
    encoded = line(event)
    cut = encoded.index("é".encode()) + 1
    assert list(output.feed(encoded[:cut])) == []
    assert list(output.feed(encoded[cut:] + b"ordinary diagnostic\n" + line({}))) == [
        event,
        {"phase": "Downloading", "completed": 0, "total": None, "unit": ""},
    ]
    assert "Préparation" in output.tail.decode()


@pytest.mark.parametrize(
    "payload",
    [
        b"null\n",
        b'{"other_event": 42}\n',
        line([]),
        line({"completed": True}),
        line({"completed": -1}),
        line({"completed": float("inf")}),
        line({"completed": 10**400}),
        line({"total": "invalid"}),
        line({"phase": []}),
        line({"unit": 3}),
        b"[" * 2000 + b"]" * 2000 + b"\n",
    ],
)
def test_malformed_download_output_cannot_interrupt_valid_progress(payload):
    output = DownloadOutput()
    event = {"phase": "Ready", "completed": 2, "total": 2, "unit": "stage"}
    assert list(output.feed(payload + line(event))) == [event]


def test_oversize_line_is_discarded_until_newline_and_diagnostics_keep_only_tail():
    output = DownloadOutput()
    event = {"phase": "Ready", "completed": 2, "total": 2, "unit": "stage"}
    assert list(output.feed(b"x" * MAX_OUTPUT_BYTES)) == []
    assert list(output.feed(b"y")) == []
    assert list(output.feed(b"still too long")) == []
    assert list(output.feed(line(event) + line(event))) == [event]
    assert len(output.tail) == MAX_OUTPUT_BYTES
    assert output.tail.endswith(line(event) + line(event))
    assert list(output.feed((b"x" * (MAX_OUTPUT_BYTES + 1)) + b"\n" + line(event))) == [event]
