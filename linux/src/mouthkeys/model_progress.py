"""Encode Hub progress and read the download subprocess output."""

import json
from collections.abc import Callable, Iterator
from math import isfinite
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from tqdm.auto import tqdm

if TYPE_CHECKING:
    _ProgressBar = tqdm[object]
else:
    _ProgressBar = tqdm


class DownloadProgress(TypedDict):
    phase: str
    completed: float
    total: float | None
    unit: str
    description: NotRequired[str]


# Keep diagnostic tails and incomplete progress lines bounded independently.
MAX_OUTPUT_BYTES = 100_000


def _counter(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if value >= 0 and isfinite(value) else None
    except OverflowError:
        return None


def _decode_progress(line: bytes) -> DownloadProgress | None:
    try:
        message = json.loads(line)
    except (ValueError, RecursionError):
        return None
    event = message.get("download_progress") if isinstance(message, dict) else None
    if not isinstance(event, dict):
        return None
    phase = event.get("phase", "Downloading")
    completed = _counter(event.get("completed", 0))
    raw_total = event.get("total")
    total = _counter(raw_total)
    unit = event.get("unit", "")
    if (
        completed is None
        or (total is None and raw_total is not None)
        or not isinstance(phase, str)
        or not isinstance(unit, str)
    ):
        return None
    return {"phase": phase, "completed": completed, "total": total, "unit": unit}


class DownloadOutput:
    def __init__(self) -> None:
        self.tail = b""
        self._partial = b""
        self._discarding_line = False

    def append_tail(self, chunk: bytes) -> None:
        self.tail = (self.tail + chunk)[-MAX_OUTPUT_BYTES:]

    def feed(self, chunk: bytes) -> Iterator[DownloadProgress]:
        self.append_tail(chunk)
        *lines, partial = chunk.split(b"\n")
        for line in lines:
            if (
                not self._discarding_line
                and len(self._partial) + len(line) <= MAX_OUTPUT_BYTES
                and (event := _decode_progress(self._partial + line))
            ):
                yield event
            self._partial = b""
            self._discarding_line = False
        if self._discarding_line or len(self._partial) + len(partial) > MAX_OUTPUT_BYTES:
            self._partial = b""
            self._discarding_line = True
        else:
            self._partial += partial


def progress_tqdm(callback: Callable[[DownloadProgress], None]) -> type[_ProgressBar]:
    class Progress(_ProgressBar):
        def __init__(self, *args: object, **kwargs: object) -> None:
            # Forward the Hub's options unchanged except that counters must stay enabled.
            kwargs["disable"] = False
            initialize = cast("Callable[..., None]", super().__init__)
            initialize(*args, **kwargs)
            self._report()

        def display(self, *_args: object, **_kwargs: object) -> None:
            """Use structured events instead of writing terminal escape sequences."""

        def _report(self) -> None:
            callback(
                {
                    "phase": "download",
                    "completed": self.n,
                    "total": self.total,
                    "unit": self.unit,
                    "description": self.desc,
                }
            )

        def update(self, n: float | None = 1) -> bool | None:
            result = super().update(n)
            self._report()
            return result

    return Progress
