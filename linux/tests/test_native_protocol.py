"""Exercise native framing with deterministic in-memory and fake pipe I/O."""

import io
import json
import struct
from functools import partial
from types import SimpleNamespace

import pytest

from mouthkeys import native_protocol


def encoded(header):
    raw = json.dumps(header).encode()
    return io.BytesIO(struct.pack("!I", len(raw)) + raw)


def test_protocol_rejects_unexpected_audio_before_reading_body():
    stream = encoded({"audio_samples": 2})
    with pytest.raises(ValueError, match="Unexpected audio"):
        native_protocol.read_message(stream)
    assert stream.read() == b""


@pytest.mark.parametrize("length", [0, native_protocol.MAX_HEADER + 1])
def test_protocol_rejects_invalid_header_length(length):
    with pytest.raises(ValueError, match="message size"):
        native_protocol.read_message(io.BytesIO(struct.pack("!I", length)))


@pytest.mark.parametrize("count", [True, -1, "3", 2**31 + 1])
def test_protocol_rejects_invalid_sample_count(count):
    with pytest.raises(ValueError, match="audio size"):
        native_protocol.read_message(encoded({"audio_samples": count}), allow_audio=True)


def test_protocol_rejects_non_object_and_large_outgoing_headers(monkeypatch):
    with pytest.raises(ValueError, match="Invalid native speech message"):
        native_protocol.read_message(encoded([]))
    monkeypatch.setattr(native_protocol, "MAX_HEADER", 5)
    stream = io.BytesIO()
    with pytest.raises(ValueError, match="exceeds"):
        native_protocol.write_message(stream, {"result": "large"})
    assert stream.getvalue() == b""


def test_protocol_is_binary_and_handles_partial_writes(monkeypatch):
    class Pipe(io.BytesIO):
        def write(self, data):
            return super().write(data[:2])

    monkeypatch.setattr(native_protocol.select, "select", lambda read, write, *_: (read, write, []))
    stream = Pipe()
    native_protocol.write_message(stream, {"method": "stream"}, [1, -2, 3.5], timeout=10)
    stream.seek(0)
    header, samples = native_protocol.read_message(stream, timeout=10, allow_audio=True)
    assert header == {"method": "stream", "audio_samples": 3}
    assert list(samples) == [1, -2, 3.5]
    assert stream.read() == b""


def test_protocol_raises_when_writer_stops_consuming():
    class ClosedPipe(io.BytesIO):
        def write(self, _data):
            return 0

    with pytest.raises(BrokenPipeError, match="pipe closed"):
        native_protocol.write_message(ClosedPipe(), {"result": "ready"})


@pytest.mark.parametrize("writing", [True, False])
def test_protocol_checks_expired_deadline_before_polling(monkeypatch, writing):
    ticks = iter([0, 1])
    monkeypatch.setattr(native_protocol, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(native_protocol.select, "select", lambda *_: pytest.fail("expired"))
    operation = (
        partial(native_protocol.write_message, io.BytesIO(), {}, timeout=1)
        if writing
        else partial(native_protocol.read_message, encoded({}), timeout=1)
    )
    with pytest.raises(TimeoutError, match="timed out"):
        operation()
