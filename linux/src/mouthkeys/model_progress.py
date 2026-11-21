"""Machine-readable progress from the Hub's supported tqdm callback interface."""

from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict, cast

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
    description: str


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
