"""Prepare a recording without taking ownership of the controller queue."""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace

from . import routing
from .config import Config, private_dir, runtime_dir
from .microphone import MicrophoneSelection
from .profiles import load_profiles
from .recording_assistant import InlineContext, inline_context
from .recording_destination import snapshot_for_start
from .recording_edit import captured_selection
from .recording_job import DictationJob, recording_input

ASSISTED_MODES = ("dictation", "edit", "command")


@dataclass(frozen=True)
class RecordingSetup:
    job: DictationJob
    capture_config: Config
    context: InlineContext | None


def prepare_recording(
    data: Mapping[str, object],
    config: Config,
    pending: list[DictationJob],
    microphone: MicrophoneSelection,
    delivery: object,
) -> RecordingSetup:
    mode, selected, label = recording_input(data)
    context = inline_context(data, mode, config)
    if len(pending) >= config.queue_limit:
        raise ValueError("Dictation queue is full; wait for earlier recordings")

    profiles = load_profiles()
    routes = routing.load_routes(profiles)
    base_config = microphone.startup_config(config)
    override = data.get("profile")
    job_config = base_config
    if override is not None:
        job_config = routing.select_prompt(
            base_config,
            override=override,
            mode=mode if mode in ASSISTED_MODES else "dictation",
            profiles=profiles,
        )

    target = snapshot_for_start(delivery, base_config, mode)
    if mode == "edit" and not selected:
        selected = captured_selection(target)

    item_id = str(uuid.uuid4())
    audio_path = private_dir(runtime_dir() / "private-recordings") / f"{item_id}.wav"
    job = DictationJob(
        item_id,
        job_config,
        tuple(previous.done for previous in pending),
        audio_path,
        mode=mode,
        selected_text=selected,
        label=label,
        held_activation=data.get("held") is True,
        phase="starting",
        prompt_override=override if isinstance(override, str) else None,
        profiles=profiles,
        routes=routes,
        start_target=target,
    )
    return RecordingSetup(job, base_config, context)


def routed_config(job: DictationJob) -> tuple[Config, str]:
    if job.prompt_override is not None or job.mode not in ASSISTED_MODES:
        return job.config, ""
    try:
        return routing.select_prompt(
            job.config,
            job.metadata.get("app", ""),
            mode=job.mode,
            profiles=job.profiles,
            routes=job.routes,
        ), ""
    # Invalid optional routing must not disable local dictation or leak data into logs.
    except Exception as exc:  # allow: exception-handling  # noqa: BLE001
        warning = f"Prompt routing failed ({exc}); using original transcription"
        return replace(job.config, ai_enabled="off"), warning
