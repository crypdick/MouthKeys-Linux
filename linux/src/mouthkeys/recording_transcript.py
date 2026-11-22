"""Prepare locally formatted dictation before optional cleanup and desktop delivery."""

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .formatting import enabled, format_text
from .pronunciation import PronunciationStore
from .recording_job import DictationJob
from .spoken_send import parse_send


@dataclass(frozen=True)
class FormattedDictation:
    text: str
    should_send: bool
    pronunciation_warning: str


def _correct_pronunciation(
    text: str, audio: Path, alignment: object, config: Config
) -> tuple[str, str]:
    if not enabled(config, "pronunciation_matching"):
        return text, ""
    alignment = alignment if isinstance(alignment, dict) else {}
    alignment_warning = alignment.get("warning", "")
    if not alignment.get("words"):
        return PronunciationStore.correct(text, audio), alignment_warning or ""
    correction = PronunciationStore.correct_spans(
        text, audio, alignment, budget=config.pronunciation_span_budget
    )
    warning = ""
    if correction.get("truncated"):
        warning = "Pronunciation matching budget reached; unmatched ASR text retained"
    elif reason := correction.get("skipped_reason"):
        warning = f"Pronunciation spans skipped: {reason}"
    return correction["text"], ". ".join(filter(None, (warning, alignment_warning)))


def format_dictation(
    job: DictationJob, config: Config, text: str, alignment: object, *, has_target: bool
) -> FormattedDictation:
    text, warning = _correct_pronunciation(text, job.audio_path, alignment, config)
    plan = parse_send(
        text,
        config.spoken_send_phrase,
        active=enabled(config, "spoken_send") and has_target,
        literal_prefix=config.literal_prefix,
    )
    formatted = format_text(
        plan.text,
        config,
        context=job.context,
        app={
            "app_name": job.metadata.get("app", ""),
            "app_id": job.metadata.get("app_id", job.metadata.get("app", "")),
            "window_title": job.metadata.get("window", ""),
        },
    )
    return FormattedDictation(formatted, plan.should_send, warning)
