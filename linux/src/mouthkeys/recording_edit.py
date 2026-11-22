"""Validate captured selection evidence and consume an Edit replacement at most once."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import Config
from .recording_job import MAX_SELECTION_LENGTH, DictationJob


def _safe_field(target: object) -> dict[str, object] | None:
    if not isinstance(target, dict):
        return None
    field = target.get("field")
    if (
        not isinstance(field, dict)
        or field.get("protected") is not False
        or field.get("complete") is False
    ):
        return None
    return field


def _selection(field: dict[str, object]) -> str:
    text = field.get("text")
    start, end = field.get("selection_start"), field.get("selection_end")
    if (
        not isinstance(text, str)
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(text)
        or end - start > MAX_SELECTION_LENGTH
    ):
        return ""
    return text[start:end]


def captured_selection(target: object) -> str:
    """Read a bounded selection from a complete, non-protected desktop snapshot."""
    field = _safe_field(target)
    return _selection(field) if field is not None else ""


def matches_edit_selection(target: object, selected_text: str) -> bool:
    field = _safe_field(target)
    return bool(
        field
        and field.get("field_id")
        and field.get("editable") is True
        and not field.get("terminal")
        and selected_text
        and _selection(field) == selected_text
    )


def can_replace_edit(job: DictationJob) -> bool:
    return matches_edit_selection(job.start_target, job.selected_text)


def reviewed_return_target(data: Mapping[str, object]) -> bool:
    value = data.get("return_to_target", False)
    if type(value) is not bool:
        raise ValueError("Return to target must be an explicit boolean")
    return value


@dataclass(frozen=True)
class EditReplacement:
    job: DictationJob
    text: str
    config: Config
    target: object

    def apply(self, adapter: object) -> str:
        """Attempt replacement without owning the service state lock."""
        if self.job.cancel_event.is_set():
            return "cancelled"
        replace_selection = getattr(adapter, "replace_selection", None)
        if not callable(replace_selection):
            raise TypeError("Desktop adapter cannot replace a selection")
        result: str = replace_selection(
            self.text, self.config, self.target, cancel_event=self.job.cancel_event
        )
        return result


def claim_edit(
    job: DictationJob, proposal: dict[str, Any] | None, config: Config
) -> EditReplacement | None:
    """Call under the controller lock after checking that the job still owns the UI."""
    if (
        job.mode != "edit"
        or job.cancelled
        or job.cancel_event.is_set()
        or job.edit_consumed
        or not proposal
        or proposal.get("edit_id") != job.item_id
        or not can_replace_edit(job)
    ):
        return None
    text = proposal.get("text")
    if not isinstance(text, str):
        return None
    # A reported failure can follow a real insertion, so never offer this edit again.
    job.edit_consumed = True
    proposal["can_accept_edit"] = False
    return EditReplacement(job, text, config, job.start_target)


def edit_message(result: str) -> str:
    if result == "pasted":
        return "Edit inserted"
    if result == "cancelled":
        return "Edit cancelled"
    if result in ("paste_requested", "insert_requested"):
        return "Edit insertion requested; do not retry automatically"
    return "Edit copied; captured selection changed or unavailable"
