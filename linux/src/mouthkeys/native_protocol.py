"""Bounded framing for native speech requests and replies."""

import json
import select
import struct
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

    AudioInput = ArrayLike
    AudioSamples = NDArray[np.float32]
else:
    AudioInput = AudioSamples = object

MAX_HEADER = 8 * 1024 * 1024


@runtime_checkable
class ReadablePipe(Protocol):
    def fileno(self) -> int: ...

    def read(self, size: int, /) -> bytes | None: ...


@runtime_checkable
class WritablePipe(Protocol):
    def fileno(self) -> int: ...

    def write(self, data: memoryview, /) -> int | None: ...

    def flush(self) -> None: ...


def pcm_samples(samples: AudioInput) -> AudioSamples:
    import numpy as np  # noqa: PLC0415 - worker applies offline isolation first

    return np.asarray(samples, dtype="<f4").reshape(-1)


def write_message(
    stream: WritablePipe,
    value: Mapping[str, object],
    samples: AudioInput | None = None,
    *,
    timeout: float | None = None,
) -> None:
    deadline = None if timeout is None else time.monotonic() + timeout
    audio = None if samples is None else pcm_samples(samples)
    if audio is not None:
        value = {**value, "audio_samples": len(audio)}
    header = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    if len(header) > MAX_HEADER:
        raise ValueError("Native speech message exceeds 8 MiB")
    chunks = [memoryview(struct.pack("!I", len(header))), memoryview(header)]
    if audio is not None:
        chunks.append(memoryview(audio).cast("B"))
    for chunk in chunks:
        view = chunk
        while view:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [stream], [], remaining)[1]:
                    raise TimeoutError("Native speech request timed out")
                # PIPE_BUF-sized writes after readiness cannot block waiting for
                # the worker to consume a large audio message.
                written = stream.write(view[:4096])
            else:
                written = stream.write(view)
            if not written:
                raise BrokenPipeError("Native speech pipe closed")
            view = view[written:]
    stream.flush()


def _read_exact(stream: ReadablePipe, length: int, deadline: float | None) -> bytearray:
    value = bytearray()
    while len(value) < length:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
                raise TimeoutError("Native speech response timed out")
        chunk = stream.read(min(65536, length - len(value)))
        if not chunk:
            raise EOFError("Native speech pipe closed")
        value.extend(chunk)
    return value


def read_message(
    stream: ReadablePipe, timeout: float | None = None, *, allow_audio: bool = False
) -> tuple[dict[str, Any], AudioSamples | None]:
    deadline = None if timeout is None else time.monotonic() + timeout
    length = struct.unpack("!I", _read_exact(stream, 4, deadline))[0]
    if not 0 < length <= MAX_HEADER:
        raise ValueError("Invalid native speech message size")
    value: object = json.loads(_read_exact(stream, length, deadline))
    if not isinstance(value, dict):
        raise ValueError("Invalid native speech message")  # noqa: TRY004 - malformed frame
    samples = None
    if "audio_samples" in value:
        if not allow_audio:
            raise ValueError("Unexpected audio in native speech response")
        count = value["audio_samples"]
        if type(count) is not int or not 0 <= count <= 2**31:
            raise ValueError("Invalid native speech audio size")
        import numpy as np  # noqa: PLC0415 - worker applies offline isolation first

        samples = np.frombuffer(_read_exact(stream, count * 4, deadline), dtype="<f4")
    return value, samples
