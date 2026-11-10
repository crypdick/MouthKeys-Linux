import copy
import fcntl
import json
import os
import shutil
import signal
import socket
import socketserver
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import __version__
from .accessibility import read_snapshot
from .audio import Recorder, check_audio, live_meter
from .config import Config, private_dir, runtime_dir, state_dir
from .delivery import Delivery
from .desktop import MediaSession, choose_source, play_sound
from .formatting import enabled, format_text, parse_send
from .learning import CorrectionLearner
from .postprocess import process
from .processes import python_command
from .pronunciation import PronunciationStore
from .storage import History
from .transcribe import Transcriber

MAX_MESSAGE = 1024 * 1024


@dataclass
class DictationJob:
    item_id: str
    config: Config
    predecessors: tuple
    audio_path: Path
    mode: str = "dictation"
    selected_text: str = ""
    label: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    held_activation: bool = False
    cancelled: bool = False
    duration: float = 0
    phase: str = "recording"
    send_pending: bool = False
    send_dispatched: bool = False
    context: str | None = None
    suppress_send: bool = False
    metadata: dict = field(default_factory=dict)
    prompt_override: str | None = None
    profiles: list = field(default_factory=list)
    routes: list = field(default_factory=list)
    start_target: object = None
    assistant_session_id: str | None = None
    assistant_memory_session: dict | None = None


class Controller:
    def __init__(self, history=None, transcriber=None, delivery=None, recorder=Recorder):
        self.history = history or History()
        self.history.recover()
        self.transcriber = transcriber or Transcriber()
        self.delivery = delivery or Delivery()
        self.recorder_factory = recorder
        self.lock = threading.RLock()
        self.phase = "idle"
        self.message = "Ready"
        self.item_id = None
        self.recorder = None
        self.cancelled = False
        self.started = None
        self.worker = None
        self.model_lock = threading.Lock()
        self.preview_worker = None
        self.preview_text = ""
        self.preview_at = 0
        self.meter = {}
        self.interrupted = False
        self.dictation_delivery = False
        self.stop_done = threading.Event()
        self.stop_done.set()
        self.notice = None
        self.notice_serial = 0
        self.notice_token = uuid.uuid4().hex
        self.pruned_at = float("-inf")
        self.media = MediaSession()
        self.recorded_seconds = 0
        self.send_armed_at = None
        self.send_disarmed_text = None
        self.last_voice_at = 0
        self.ai_warning = ""
        self.last_text = ""
        self.send_pending = False
        self.job = None
        self.jobs = []
        self.assistant_proposal = None
        self.assistant_memory = {}
        self.learning_stop = threading.Event()
        self.learning_worker = None
        self.learning_generation = 0
        self.learning_proposals = {}

    def report_error(self, message):
        with self.lock:
            self.notice_serial += 1
            self.notice = {
                "serial": f"{self.notice_token}:{self.notice_serial}",
                "message": message,
                "at": time.time(),
            }

    def status(self):
        with self.lock:
            return {
                "version": __version__,
                "pending_jobs": sum(not job.done.is_set() for job in self.jobs),
                "can_record": self.phase
                in ("idle", "recording", "transcribing", "queued", "delivering"),
                "assistant_proposal": self.assistant_proposal,
                "loaded_model": copy.deepcopy(getattr(self.transcriber, "loaded_model", None)),
                "learning_pending": [proposal for _, proposal in self.learning_proposals.values()],
                "notice": self.notice,
                "preview": self.preview_text,
                "meter": self.meter if self.phase == "recording" else {},
                "microphone": self.config.audio_source if hasattr(self, "config") else "default",
                "phase": self.phase,
                "message": self.message,
                "id": self.item_id,
                "seconds": round(time.monotonic() - self.started, 1) if self.started else 0,
            }

    def command(self, action, **data):
        # A stop acknowledgment must not precede its focus snapshot. The worker
        # resolves focus outside the state lock; a delayed lookup falls back to copy.
        gate = {"ready": threading.Event(), "valid": True}
        result = self._command(action, _focus_gate=gate, **data)
        if action in ("stop", "toggle") and result.get("phase") == "stopping":
            if not gate["ready"].wait(timeout=4):
                gate["valid"] = False
        return result

    def _command(self, action, _focus_gate, **data):
        if action in ("copy", "paste"):
            return self._manual_delivery(action, data)
        if action in ("learning_pending", "learning_accept", "learning_dismiss"):
            return self._learning_command(action, data)
        with self.lock:
            if action == "status":
                return self.status()
            if action == "release_hold":
                if self.job and self.phase == "recording" and data.get("id") == self.job.item_id:
                    self.job.held_activation = False
                return self.status()
            from .operations import handle_action

            result = handle_action(
                self.history,
                action,
                data,
                allow_mutation=self.phase == "idle" and not self._pending_jobs(),
            )
            if result is not NotImplemented:
                return result
            if action == "toggle":
                action = "stop" if self.phase == "recording" else "start"
            if self.interrupted:
                raise ValueError("Service is shutting down")
            if action == "cancel":
                self._cancel_learning()
                if self.phase == "delivering" and self.job and self.job.send_pending:
                    if self.job.send_dispatched:
                        raise ValueError("Send already dispatched; cannot withdraw the key")
                    self.cancelled = True
                    self.job.cancelled = True
                    self.job.cancel_event.set()
                    self.message = "Send cancelled; text already inserted"
                    return self.status()
                if self.phase == "delivering":
                    raise ValueError(
                        "Delivery already started; waiting for clipboard/paste to finish"
                    )
                self.preview_text = ""
                self.cancelled = True
                if self.job:
                    self.job.cancelled = True
                    self.job.cancel_event.set()
                if self.phase == "recording":
                    self._stop(None)
                elif self.phase in ("stopping", "transcribing", "queued", "loading"):
                    self.message = "Cancelling; waiting for capture/model to finish"
                return self.status()
            if action == "stop":
                if data.get("id") is not None and (not self.job or data["id"] != self.job.item_id):
                    return self.status()
                if self.phase != "recording":
                    raise ValueError("Not recording")
                target = (
                    "capture-focus"
                    if self.config.delivery in ("x11", "wayland")
                    and (not data.get("automatic") or data.get("spoken"))
                    else None
                )
                self._stop(target, _focus_gate)
                return self.status()
            can_start = action == "start" and self.phase in ("transcribing", "queued", "delivering")
            if self.phase != "idle" and not can_start:
                raise ValueError(f"Busy ({self.phase}); wait or cancel before {action}")
            if action != "start" and self._pending_jobs():
                raise ValueError(f"Busy (queued); wait or cancel before {action}")
            if action == "start":
                from .profiles import load_profiles
                from .routing import load_routes

                mode = data.get("mode", "dictation")
                selected_text = data.get("selected_text", "")
                if mode not in ("dictation", "command", "edit", "training", "pronunciation"):
                    raise ValueError("Unknown recording mode")
                if not isinstance(selected_text, str) or len(selected_text) > 100000:
                    raise ValueError("Selected text exceeds 100000 characters")
                label = data.get("label", "")
                if mode == "pronunciation" and (
                    not isinstance(label, str)
                    or not label.strip()
                    or len(label) > 128
                    or len(label.split()) > 5
                    or not all(character.isprintable() for character in label)
                ):
                    raise ValueError(
                        "Pronunciation label must contain 1–5 words, up to 128 characters"
                    )
                config = Config.load()
                assistant_context = None
                assistant_executions = None
                if "assistant_history" in data:
                    from .ai_providers import validate_history

                    assistant_context = validate_history(data["assistant_history"])
                    if mode not in ("command", "edit") or config.save_history != "off":
                        raise ValueError(
                            "Inline assistant history requires private Command/Edit mode"
                        )
                    from .assistant_history import validate_executions

                    assistant_executions = validate_executions(
                        data.get("assistant_executions", []), len(assistant_context)
                    )
                elif "assistant_executions" in data:
                    raise ValueError("Execution context requires inline assistant history")
                if len(self._pending_jobs()) >= config.queue_limit:
                    raise ValueError("Dictation queue is full; wait for earlier recordings")
                config = replace(config, audio_source=choose_source(config))
                profiles = load_profiles()
                routes = load_routes(profiles)
                self.meter = {}
                self.preview_text = ""
                self.preview_at = time.monotonic()
                self.config = config
                self.send_armed_at = None
                self.send_disarmed_text = None
                self.last_voice_at = time.monotonic()
                self._cancel_learning()
                for previous in self._pending_jobs():
                    previous.suppress_send = True
                self.item_id = str(uuid.uuid4())
                self._new_job(
                    mode,
                    selected_text,
                    profiles=profiles,
                    routes=routes,
                    assistant_context=assistant_context,
                    assistant_executions=assistant_executions,
                )
                self.job.held_activation = data.get("held") is True
                self.job.label = label.strip() if mode == "pronunciation" else ""
                if (
                    self.config.target_policy == "start"
                    or self.config.return_to_start == "on"
                    or mode == "edit"
                ) and hasattr(self.delivery, "capture_target"):
                    try:
                        self.job.start_target = self.delivery.capture_target(self.config)
                    except Exception:
                        self.job.start_target = None  # Optional desktop probing fails closed.
                    if (
                        mode == "edit"
                        and not selected_text
                        and isinstance(self.job.start_target, dict)
                    ):
                        captured = self.job.start_target.get("field", {})
                        begin, end = (
                            captured.get("selection_start", 0),
                            captured.get("selection_end", 0),
                        )
                        if captured.get("complete") is not False:
                            self.job.selected_text = captured.get("text", "")[begin:end]
                self.job.prompt_override = data.get("profile")
                if self.job.prompt_override is not None:
                    from .routing import select_prompt

                    try:
                        self.job.config = select_prompt(
                            self.config,
                            override=self.job.prompt_override,
                            mode=mode if mode in ("dictation", "edit", "command") else "dictation",
                            profiles=self.job.profiles,
                        )
                    except Exception:
                        self.job.done.set()
                        raise
                self.assistant_proposal = None
                self.cancelled = False
                if self.config.save_history == "on":
                    self.history.begin(self.item_id)
                try:
                    self.media.pause(self.config)
                    self.recorder = self.recorder_factory(self.config, self.job.audio_path)
                except Exception as exc:
                    self.media.resume()
                    self._update(self.job, "failed", error=str(exc))
                    self.message = str(exc)
                    if self.config.save_history == "off":
                        self.job.audio_path.unlink(missing_ok=True)
                    self.job.done.set()
                    self.phase = "idle"
                    raise
                self.notice = None
                self.phase, self.message = "recording", "Listening — toggle again to finish"
                self.started = time.monotonic()
                play_sound(self.config)
                return self.status()
            if action == "reprocess":
                item_id = data.get("id") or self.item_id
                if not item_id:
                    rows = self.history.rows()
                    item_id = rows[0]["id"] if rows else None
                if not item_id or not self.history.recording(item_id).exists():
                    raise ValueError("No retained audio for this entry")
                self.config = Config.load()
                # A retry is a new attempt: failure/cancellation must not erase the original.
                self.preview_text = ""
                retry_id = str(uuid.uuid4())
                self.item_id = retry_id
                job = self._new_job()
                try:
                    shutil.copyfile(self.history.recording(item_id), job.audio_path)
                except OSError:
                    job.done.set()
                    job.audio_path.unlink(missing_ok=True)
                    raise
                if self.config.save_history == "on":
                    self.history.begin(retry_id)
                self.cancelled = False
                self._transcribe(None)  # Reprocessing never pastes into an old destination.
                return self.status()
            if action == "warmup":
                self.config = Config.load()
                self.cancelled = False
                self.phase, self.message = "loading", "Loading local model"
                self.worker = threading.Thread(target=self._warmup, daemon=True)
                self.worker.start()
                return self.status()
            if action == "clear":
                self._cancel_learning()
                self.learning_proposals.clear()
                self.preview_text = ""
                self.last_text = ""
                self.assistant_proposal = None
                self.history.clear()
                self.item_id = None
                self.message = "History and recordings cleared"
                return self.status()
            raise ValueError(f"Unknown command: {action}")

    def _cancel_learning(self):
        self.learning_stop.set()
        self.learning_generation += 1

    def _learning_command(self, action, data):
        with self.lock:
            if action == "learning_pending":
                return [proposal for _, proposal in self.learning_proposals.values()]
            if self.interrupted:
                raise ValueError("Service is shutting down")
            proposal_id = data.get("id")
            if proposal_id not in self.learning_proposals:
                raise ValueError("Learning suggestion no longer exists")
            learner, proposal = self.learning_proposals[proposal_id]
        # Dictionary writes never hold the controller state lock.
        if action == "learning_accept":
            learner.accept(proposal_id)
        else:
            learner.dismiss(proposal_id)
        with self.lock:
            self.learning_proposals.pop(proposal_id, None)
        return proposal

    def _start_learning(self, job, text):
        initial = getattr(self.delivery, "last_verified", None)
        if not isinstance(initial, dict):
            return
        with self.lock:
            if self.interrupted or job.cancelled or self.job is not job:
                return
            self._cancel_learning()
            stop = self.learning_stop = threading.Event()
            generation = self.learning_generation
            # Keep the final mutation behind the generation check even in automatic mode.
            learner = CorrectionLearner(replace(job.config, learning_confirmation="on"))
            if not learner.begin(text, initial):
                return
            self.learning_worker = threading.Thread(
                target=self._observe_learning,
                args=(job, learner, dict(initial), generation, stop),
                daemon=True,
            )
            self.learning_worker.start()

    def _observe_learning(self, job, learner, target, generation, stop):
        deadline = time.monotonic() + job.config.learning_observe_seconds
        try:
            while not stop.wait(0.5) and time.monotonic() < deadline:
                if Config.load().automatic_learning != "on":
                    return
                # The helper itself has a bounded timeout; no desktop calls under state lock.
                snapshot = read_snapshot(target)
                if stop.is_set():
                    return
                proposal = learner.observe(snapshot)
                if proposal:
                    with self.lock:
                        if (
                            stop.is_set()
                            or self.interrupted
                            or job.cancelled
                            or self.job is not job
                            or generation != self.learning_generation
                        ):
                            return
                        if job.config.learning_confirmation == "off":
                            learner.accept(proposal["id"])
                        else:
                            self.learning_proposals[proposal["id"]] = (learner, proposal)
                            while len(self.learning_proposals) > 200:
                                self.learning_proposals.pop(next(iter(self.learning_proposals)))
                    return
                if learner.observation is None:
                    return
        except Exception as exc:
            if not stop.is_set():
                self.report_error(f"Correction learning unavailable: {exc}")
        finally:
            learner.cancel()

    @staticmethod
    def _edit_target_matches(job):
        if not isinstance(job.start_target, dict) or not job.selected_text:
            return False
        captured = job.start_target.get("field")
        if (
            not isinstance(captured, dict)
            or not captured.get("field_id")
            or captured.get("protected") is not False
            or captured.get("complete") is False
            or captured.get("editable") is not True
            or captured.get("terminal")
        ):
            return False
        start, end = captured.get("selection_start"), captured.get("selection_end")
        text = captured.get("text")
        return (
            isinstance(text, str)
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(text)
            and text[start:end] == job.selected_text
        )

    def _pending_jobs(self):
        return [job for job in self.jobs if not job.done.is_set()]

    def _new_job(
        self,
        mode="dictation",
        selected_text="",
        profiles=None,
        routes=None,
        assistant_context=None,
        assistant_executions=None,
    ):
        from .profiles import load_profiles
        from .routing import load_routes

        profiles = load_profiles() if profiles is None else profiles
        routes = load_routes(profiles) if routes is None else routes
        assistant_session = None
        if mode in ("command", "edit"):
            if self.config.save_history == "on":
                from .assistant_history import AssistantHistory

                assistant_session = AssistantHistory().current(mode)
            else:
                previous = self.assistant_memory.get(mode)
                if assistant_context is not None and (
                    previous is None
                    or previous["messages"] != assistant_context
                    or previous.get("executions", []) != assistant_executions
                ):
                    self.assistant_memory[mode] = {
                        "id": uuid.uuid4().hex,
                        "messages": assistant_context,
                        "executions": assistant_executions or [],
                    }
                assistant_session = self.assistant_memory.setdefault(
                    mode, {"id": uuid.uuid4().hex, "messages": []}
                )

        pending = self._pending_jobs()
        audio_path = self.history.recording(self.item_id)
        if self.config.save_history == "off":
            audio_path = private_dir(runtime_dir() / "private-recordings") / f"{self.item_id}.wav"
        self.job = DictationJob(
            self.item_id,
            self.config,
            tuple(job.done for job in pending),
            audio_path,
            mode=mode,
            selected_text=selected_text,
        )
        self.jobs = [*pending, self.job]
        self.job.profiles = profiles
        self.job.routes = routes
        if assistant_session is not None:
            self.job.assistant_session_id = assistant_session["id"]
            if self.config.save_history == "off":
                self.job.assistant_memory_session = assistant_session
        return self.job

    def _update(self, job, status, text="", error="", **metadata):
        # Explicit privacy opt-out overrides durable transcript-before-delivery.
        if job.config.save_history == "on":
            if status == "cancelled":
                metadata["raw_text"] = ""
            self.history.update(job.item_id, status, text, error, **{**job.metadata, **metadata})

    def _job_status(self, job, phase, message):
        with self.lock:
            job.phase = phase
            if self.job is job and not self.interrupted:
                self.phase, self.message = phase, message

    def _wait_for_predecessors(self, job):
        for previous in job.predecessors:
            while not previous.wait(0.1):
                if self.interrupted:
                    return False
        return not self.interrupted

    def interrupt(self):
        """Stop on logout without interpreting it as the user's discard command."""
        with self.lock:
            self.interrupted = True
            self._cancel_learning()
            self.cancelled = True
            recorder, self.recorder = self.recorder, None
            for job in self._pending_jobs():
                job.cancelled = True
                job.cancel_event.set()
                rows = self.history.rows(limit=10000)
                row = next((r for r in rows if r["id"] == job.item_id), {})
                if job.config.save_history == "off":
                    job.audio_path.unlink(missing_ok=True)
                else:
                    self._update(
                        job,
                        "failed",
                        row.get("text", ""),
                        "Interrupted; reprocess audio",
                    )
            self.phase, self.message = "interrupted", "Interrupted; audio retained for reprocessing"
        if recorder is not None:
            try:
                recorder.stop()
            except Exception:
                pass  # The partial file remains available to recovery.
        # An already-stopping worker owns its recorder. Let it finalize the WAV, without
        # waiting for model inference or making a second concurrent stop call.
        self.stop_done.wait(timeout=11)
        self.media.resume()
        if hasattr(self, "config") and self.config.save_history == "off" and self.item_id:
            self.job.audio_path.unlink(missing_ok=True)

    def _stop(self, target, focus_gate=None):
        recorder, self.recorder = self.recorder, None
        self.recorded_seconds = time.monotonic() - self.started if self.started else 0
        self.job.duration = self.recorded_seconds
        self.job.phase = "stopping"
        self.stop_done.clear()
        self.notice = None
        self.phase, self.message, self.started = "stopping", "Stopping microphone", None
        self.worker = threading.Thread(
            target=self._stopped, args=(self.job, recorder, target, focus_gate), daemon=True
        )
        self.worker.start()

    def _stopped(self, job, recorder, target, focus_gate):
        try:
            if target == "capture-focus":
                try:
                    if job.config.target_policy == "start" or job.config.return_to_start == "on":
                        target = job.start_target
                    elif hasattr(self.delivery, "capture_target"):
                        target = self.delivery.capture_target(job.config)
                    else:
                        target = self.delivery.focus()
                except Exception:
                    target = None
            metadata_target = target
            needs_metadata = (
                job.config.save_history == "on"
                or job.config.ai_enabled == "on"
                or job.mode in ("edit", "command")
            )
            if (
                needs_metadata
                and metadata_target is None
                and hasattr(self.delivery, "target_metadata")
            ):
                try:
                    metadata_target = (
                        self.delivery.capture_target(job.config)
                        if hasattr(self.delivery, "capture_target")
                        else self.delivery.focus()
                    )
                except Exception:
                    metadata_target = None
            if enabled(job.config, "smart_capitalization") or enabled(
                job.config, "space_between_dictations"
            ):
                try:
                    from .accessibility import read_context

                    job.context = read_context()
                except Exception:
                    job.context = None  # Optional context must never prevent microphone shutdown.
            if focus_gate:
                focus_gate["ready"].set()
            try:
                recorder.stop()
            finally:
                self.stop_done.set()
                self.media.resume()
                play_sound(job.config, stop=True)
            if needs_metadata and hasattr(self.delivery, "target_metadata"):
                try:
                    job.metadata = self.delivery.target_metadata(metadata_target)
                except Exception:
                    pass  # Destination metadata is optional; never block transcription.
        except Exception as exc:
            with self.lock:
                if not self.interrupted:
                    self._update(job, "failed", error=str(exc))
                    self._job_status(job, "idle", str(exc))
                    if job.config.save_history == "off":
                        job.audio_path.unlink(missing_ok=True)
                job.done.set()
            return
        with self.lock:
            if self.interrupted:
                job.done.set()
                return
            if job.cancelled:
                self._job_status(job, "queued", "Cancelling; waiting for model cleanup")
            else:
                self._job_status(job, "queued", "Waiting for earlier dictation")
                self._update(job, "transcribing")
        if focus_gate and not focus_gate["valid"]:
            target = None
        self._finish(job, target)

    def _manual_delivery(self, action, data):
        with self.lock:
            if self.interrupted:
                raise ValueError("Service is shutting down")
            if self.phase != "idle" or self._pending_jobs():
                raise ValueError(f"Busy ({self.phase}); wait or cancel before {action}")
            text = self.last_text or self.history.latest_text()
            if data.get("id"):
                text = self.history.get(data["id"])["text"]
                if not text:
                    raise ValueError("No transcript for this entry")
            config = Config.load()
            self.phase, self.message = "delivering", "Delivering transcript"
        try:
            self.delivery.copy(text)
            if action == "paste" and not self.interrupted:
                time.sleep(0.2)
                if not self.interrupted:
                    self.delivery.paste_key(config)
            with self.lock:
                if not self.interrupted:
                    self.message = "Copied" if action == "copy" else "Paste requested"
        except Exception as exc:
            with self.lock:
                if not self.interrupted:
                    self.message = str(exc)
            raise
        finally:
            with self.lock:
                if not self.interrupted:
                    self.phase = "idle"
        return self.status()

    def _warmup(self):
        try:
            with self.model_lock:
                self.transcriber.load(self.config)
            message = "Model ready"
        except Exception as exc:
            message = f"Model unavailable: {exc}. Run mouthkeys download first."
        with self.lock:
            if not self.interrupted:
                self.phase, self.message = "idle", "Cancelled" if self.cancelled else message

    def _transcribe(self, target):
        job = self.job
        if job is None or job.item_id != self.item_id or job.done.is_set():
            job = self._new_job()
        self.phase, self.message = "queued", "Waiting for earlier dictation"
        self.started = None
        self._update(job, "transcribing")
        self.worker = threading.Thread(target=self._finish, args=(job, target), daemon=True)
        self.worker.start()

    def _finish(self, job, target):
        text = ""
        alignment = None
        pronunciation_warning = ""
        ai_warning = ""
        send_warning = ""
        config = job.config
        path = job.audio_path
        try:
            if job.cancelled:
                self._update(job, "cancelled")
                path.unlink(missing_ok=True)
                self._job_status(job, "idle", "Cancelled")
                return
            if not self._wait_for_predecessors(job):
                return
            with self.lock:
                if not job.cancelled:
                    self._job_status(job, "transcribing", "Transcribing")
            check_audio(path)
            if enabled(config, "skip_silent_recordings") and _short_silence(path):
                self._update(job, "cancelled")
                path.unlink(missing_ok=True)
                self._job_status(job, "idle", "Silent recording skipped")
                return
            with self.model_lock:
                if not job.cancelled and not self.interrupted:
                    text = self.transcriber.transcribe(path, config)
                    if config.pronunciation_matching == "on":
                        alignment = copy.deepcopy(getattr(self.transcriber, "last_alignment", None))
            with self.lock:
                if self.interrupted:
                    return
                if job.cancelled:
                    self._update(job, "cancelled")
                    path.unlink(missing_ok=True)
                    self._job_status(job, "idle", "Cancelled")
                    return
                if not text:
                    if self.job is job:
                        self.preview_text = ""
                    self._update(job, "failed", error="No speech detected")
                    retained = (
                        "audio retained for reprocessing"
                        if config.save_history == "on"
                        else "history disabled; audio discarded"
                    )
                    self._job_status(job, "idle", f"No speech detected; {retained}")
                    return
                # Raw text survives a crash or timeout during optional cleanup.
                self._update(
                    job,
                    "transcribing",
                    text,
                    raw_text=text,
                    duration=job.duration,
                    model=config.model,
                    language=config.language,
                )
            if job.prompt_override is None and job.mode in ("dictation", "edit", "command"):
                try:
                    from .routing import select_prompt

                    config = select_prompt(
                        config,
                        job.metadata.get("app", ""),
                        mode=job.mode,
                        profiles=job.profiles,
                        routes=job.routes,
                    )
                except Exception as exc:
                    ai_warning = f"Prompt routing failed ({exc}); using original transcription"
                    config = replace(config, ai_enabled="off")
                    if self.job is job:
                        self.report_error(ai_warning)
            if job.mode != "dictation":
                if job.mode == "training":
                    proposal = {"mode": "training", "text": text}
                elif job.mode == "pronunciation":
                    with self.lock:
                        if self.interrupted or job.cancelled:
                            return
                        # Serialize this small feature-store update with cancellation.
                        enrollment = PronunciationStore.enroll(job.label, path)
                    proposal = {"mode": "pronunciation", "text": job.label, **enrollment}
                else:
                    from .assistance import propose
                    from .assistant_history import (
                        AssistantHistory,
                        append_private_messages,
                        context_messages,
                        pending_executions,
                    )

                    if config.save_history == "on":
                        sessions = AssistantHistory()
                        session = sessions.get(job.assistant_session_id)
                    else:
                        sessions = None
                        session = job.assistant_memory_session
                    conversation = context_messages(session["messages"])
                    observed = pending_executions(session)
                    instruction = (
                        json.dumps(
                            {"instruction": text, "observed_command_results": observed},
                            ensure_ascii=False,
                        )
                        if observed
                        else text
                    )
                    thoughts = []
                    thought_bytes = 0

                    def thinking(piece):
                        nonlocal thought_bytes
                        with self.lock:
                            if not job.cancelled and not self.interrupted:
                                # External-provider display only; never enters
                                # transcript text or retained conversation turns.
                                bounded = piece.encode("utf-8")[: 65536 - thought_bytes].decode(
                                    "utf-8", errors="ignore"
                                )
                                thought_bytes += len(bounded.encode("utf-8"))
                                if bounded:
                                    thoughts.append(bounded)

                    def delta(piece):
                        with self.lock:
                            if self.job is job and not job.cancelled and not self.interrupted:
                                self.preview_text += piece

                    options = {"resolved_config": True, "cancel_event": job.cancel_event}
                    if conversation:
                        options["history"] = conversation
                    if getattr(config, "ai_show_thinking", "off") == "on":
                        options["on_thinking"] = thinking
                    if config.ai_streaming == "on" and job.mode == "edit":
                        options["on_delta"] = delta
                    proposal = propose(instruction, config, job.mode, job.selected_text, **options)
                    request = proposal.pop(
                        "request",
                        json.dumps(
                            {"instruction": text, "selected_text": job.selected_text},
                            ensure_ascii=False,
                        ),
                    )
                    response = proposal.pop("response", proposal.get("command", proposal["text"]))
                    with self.lock:
                        if self.interrupted:
                            return
                        if job.cancelled:
                            self._update(job, "cancelled")
                            path.unlink(missing_ok=True)
                            self._job_status(job, "idle", "Cancelled")
                            return
                        turns = [
                            {"role": "user", "content": request},
                            {"role": "assistant", "content": response},
                        ]
                        if sessions is not None:
                            sessions.append(session["id"], turns)
                        else:
                            append_private_messages(session, turns)
                        proposal["session_id"] = session["id"]
                        if thoughts:
                            proposal["thinking"] = "".join(thoughts)
                        if sessions is None:
                            proposal["private_messages"] = context_messages(session["messages"])
                            removed = len(session["messages"]) - len(proposal["private_messages"])
                            proposal["private_executions"] = [
                                {
                                    "message_count": event["message_count"] - removed,
                                    "result": dict(event["result"]),
                                }
                                for event in session.get("executions", [])
                                if event["message_count"] >= removed
                            ]

                        # The status response repeats preview text and escapes
                        # Unicode. Bound its actual serialized bytes, not only
                        # characters, reserving room for the other status fields.
                        def proposal_size():
                            return len(
                                json.dumps(
                                    {"assistant_proposal": proposal, "preview": proposal["text"]}
                                ).encode()
                            )

                        while (
                            proposal.get("private_executions")
                            and proposal_size() > MAX_MESSAGE - 131072
                        ):
                            del proposal["private_executions"][0]
                        while (
                            proposal.get("private_messages")
                            and proposal_size() > MAX_MESSAGE - 131072
                        ):
                            del proposal["private_messages"][:2]
                            proposal["private_executions"] = [
                                {
                                    "message_count": event["message_count"] - 2,
                                    "result": event["result"],
                                }
                                for event in proposal.get("private_executions", [])
                                if event["message_count"] >= 2
                            ]
                        if proposal_size() > MAX_MESSAGE - 131072:
                            raise ValueError(
                                "Assistant proposal exceeds status limit; "
                                "request a shorter response"
                            )
                with self.lock:
                    if self.interrupted:
                        return
                    if job.cancelled:
                        self._update(job, "cancelled")
                        path.unlink(missing_ok=True)
                        self._job_status(job, "idle", "Cancelled")
                        return
                    text = proposal["text"]
                    self._update(job, "transcribed", text)
                    if config.save_audio == "off" and config.save_history == "on":
                        self.history.prune_audio(job.item_id)
                    if self.job is job:
                        self.assistant_proposal = proposal
                        self.preview_text = text
                    self._job_status(job, "idle", "Recording ready for review; nothing inserted")
                if (
                    job.mode == "edit"
                    and config.edit_delivery == "replace"
                    and self._edit_target_matches(job)
                    and not job.cancelled
                    and not self.interrupted
                ):
                    with self.lock:
                        if job.cancelled or self.interrupted:
                            return
                        self._job_status(job, "delivering", "Replacing captured selection")
                    result = self.delivery.replace_selection(text, config, job.start_target)
                    self._update(job, result, text)
                    self._job_status(
                        job,
                        "idle",
                        "Edit inserted"
                        if result == "pasted"
                        else "Edit copied; selection changed or unavailable",
                    )
                return
            if enabled(config, "pronunciation_matching"):
                if isinstance(alignment, dict) and alignment.get("words"):
                    correction = PronunciationStore.correct_spans(
                        text, path, alignment, budget=config.pronunciation_span_budget
                    )
                    text = correction["text"]
                    if correction.get("truncated"):
                        pronunciation_warning = (
                            "Pronunciation matching budget reached; unmatched ASR text retained"
                        )
                    elif correction.get("skipped_reason"):
                        pronunciation_warning = (
                            f"Pronunciation spans skipped: {correction['skipped_reason']}"
                        )
                else:
                    text = PronunciationStore.correct(text, path)
            plan = parse_send(
                text,
                config.spoken_send_phrase,
                active=enabled(config, "spoken_send") and target is not None,
                literal_prefix=config.literal_prefix,
            )
            text = format_text(plan.text, config, context=job.context)
            if enabled(config, "ai_enabled") and text:
                try:
                    if config.ai_streaming == "on":
                        streamed = []

                        def delta(piece):
                            with self.lock:
                                if self.job is job and not job.cancelled and not self.interrupted:
                                    streamed.append(piece)
                                    self.preview_text = "".join(streamed)

                        text = process(text, config, on_delta=delta, cancel_event=job.cancel_event)
                    else:
                        text = process(text, config)
                except Exception as exc:
                    ai_warning = f"AI cleanup failed ({exc}); using original transcription"
                    if self.job is job:
                        self.report_error(ai_warning)
            with self.lock:
                if self.interrupted:
                    return
                if job.cancelled:
                    self._update(job, "cancelled")
                    path.unlink(missing_ok=True)
                    self._job_status(job, "idle", "Cancelled")
                    return
                if self.job is job:
                    self.preview_text = text
                    self.dictation_delivery = True
                if text:
                    self.last_text = text
                self._update(job, "transcribed", text)
                self._job_status(job, "delivering", "Delivering transcript")
            result = self.delivery.deliver(text, config, target) if text else "transcribed"
            if result == "pasted" and enabled(config, "automatic_learning"):
                self._start_learning(job, text)
            if plan.should_send and (
                not text or result in ("paste_requested", "insert_requested", "pasted")
            ):
                job.send_pending = True
                if self.job is job:
                    self.send_pending = True
                try:
                    time.sleep(0.5)

                    def reserve_send():
                        with self.lock:
                            if self.interrupted or job.cancelled or job.suppress_send:
                                return False
                            job.send_dispatched = True
                            return True

                    with self.lock:
                        send = not self.interrupted and not job.cancelled and not job.suppress_send
                    if send:
                        if getattr(self.delivery, "send_key_guarded", False):
                            sent = self.delivery.send_key(
                                config,
                                target,
                                before_send=reserve_send,
                                allow_without_insert=not bool(text),
                            )
                        else:
                            sent = (
                                self.delivery.send_key(config, target) if reserve_send() else False
                            )
                        if not sent:
                            send_warning = "Spoken Send skipped: destination could not be verified"
                        elif not text:
                            result = "send_requested"
                    elif job.suppress_send:
                        send_warning = "Spoken Send skipped: a newer recording started"
                finally:
                    job.send_pending = False
                    if self.job is job:
                        self.send_pending = False
            with self.lock:
                if self.interrupted:
                    return
                self._update(job, result, text, error=send_warning)
                if config.save_audio == "off" and config.save_history == "on":
                    self.history.prune_audio(job.item_id)
                message = (
                    "Send requested — no text inserted"
                    if result == "send_requested"
                    else "No text to deliver"
                    if not text
                    else "Copied — paste into your app"
                    if result == "copied"
                    else (
                        "Insertion requested — transcript saved in history"
                        if config.save_history == "on"
                        else "Insertion requested — history disabled"
                    )
                )
                if ai_warning:
                    message += f". {ai_warning}"
                if pronunciation_warning:
                    message += f". {pronunciation_warning}"
                if send_warning:
                    message += f". {send_warning}"
                self._job_status(job, "idle", message)
        except Exception as exc:
            with self.lock:
                if self.interrupted:
                    return
                if job.cancelled:
                    self._update(job, "cancelled")
                    path.unlink(missing_ok=True)
                    self._job_status(job, "idle", "Cancelled")
                else:
                    if self.job is job:
                        self.preview_text = text
                    self._update(job, "failed", text, str(exc))
                    retained = (
                        "Saved in history; audio retained for reprocessing."
                        if config.save_history == "on"
                        else "History disabled; audio discarded."
                    )
                    self._job_status(job, "idle", f"{exc}. {retained}")
        finally:
            key = getattr(self.transcriber, "stream_key", None)
            if job.cancelled and key and key[0] == job.audio_path:
                with self.model_lock:
                    self._reset_stream_if_owned(job)
            with self.lock:
                if not self.interrupted:
                    if config.save_history == "off":
                        job.audio_path.unlink(missing_ok=True)
                    if self.job is job:
                        self.dictation_delivery = False
                        self.phase = "idle"
                    self._prune_budget()
                job.done.set()

    def _preview(self, item_id, config):
        try:
            with self.model_lock:
                with self.lock:
                    if self.phase != "recording" or self.item_id != item_id:
                        return
                    if any(not done.is_set() for done in self.job.predecessors):
                        return  # Preserve earlier native stream caches until ordered finalization.
                    path = self.job.audio_path
                    preview_job = self.job
                text = self.transcriber.preview(path, config)
                if preview_job.cancelled or self.interrupted:
                    self._reset_stream_if_owned(preview_job)
            with self.lock:
                if self.phase != "recording" or self.item_id != item_id or self.cancelled:
                    return
                self.preview_text = text
                plan = parse_send(
                    text,
                    config.spoken_send_phrase,
                    active=enabled(config, "spoken_send"),
                    literal_prefix=config.literal_prefix,
                )
                if plan.should_send and text != self.send_disarmed_text:
                    if self.send_armed_at is None:
                        self.send_armed_at = time.monotonic()
                else:
                    self.send_armed_at = None
        except Exception:
            pass  # Preview is provisional; final inference reports actionable errors.
        finally:
            with self.lock:
                if self.item_id == item_id:
                    self.preview_at = time.monotonic()

    def _reset_stream_if_owned(self, job):
        """Caller owns model_lock; never clear a newer recording's native caches."""
        key = getattr(self.transcriber, "stream_key", None)
        if key and key[0] == job.audio_path and hasattr(self.transcriber, "reset_stream"):
            self.transcriber.reset_stream()

    def _prune_budget(self):
        try:
            budget = Config.load().audio_budget_mb * 1000000
        except (OSError, ValueError, TypeError):
            return  # Invalid settings must not stop the daemon or guess a deletion budget.
        self.history.prune_budget(budget)

    def tick(self):
        with self.lock:
            if self.phase == "idle" and time.monotonic() - self.pruned_at >= 60:
                self._prune_budget()
                self.pruned_at = time.monotonic()
            if self.phase == "recording":
                try:
                    meter = live_meter(self.job.audio_path)
                    if meter["frame"] != self.meter.get("frame"):
                        self.meter = {**meter, "at": time.monotonic()}
                        if meter["level"] >= 0.12:
                            self.last_voice_at = time.monotonic()
                            if (
                                self.send_armed_at is not None
                                and self.last_voice_at - self.send_armed_at >= 0.35
                            ):
                                self.send_disarmed_text = self.preview_text
                                self.send_armed_at = None
                except (OSError, ValueError, EOFError, wave.Error):
                    self.meter = {}
            if (
                self.phase == "recording"
                and (self.config.live_preview == "on" or self.config.send_after_pause == "on")
                and time.monotonic() - self.preview_at >= self.config.preview_interval_ms / 1000
                and (self.preview_worker is None or not self.preview_worker.is_alive())
                and hasattr(self.transcriber, "preview")
            ):
                self.preview_worker = threading.Thread(
                    target=self._preview, args=(self.item_id, self.config), daemon=True
                )
                self.preview_worker.start()
            if (
                self.phase == "recording"
                and self.send_armed_at is not None
                and enabled(self.config, "send_after_pause")
                and not self.job.held_activation
                and enabled(self.config, "spoken_send")
                and time.monotonic() - self.send_armed_at >= 0.5
                and time.monotonic() - self.last_voice_at >= 0.5
            ):
                self.command("stop", automatic=True, spoken=True)
            if self.phase == "recording" and (
                self.recorder.finished()
                or time.monotonic() - self.started >= self.config.max_seconds
            ):
                try:
                    self.command("stop", automatic=True)
                except Exception:
                    pass  # The stop path has already persisted the error for the UI.


def _short_silence(path):
    """Only skip up to four seconds of near-zero PCM, never ordinary quiet speech."""
    import numpy as np

    with wave.open(str(path), "rb") as source:
        if source.getnframes() > source.getframerate() * 4 or source.getsampwidth() != 2:
            return False
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    return bool(len(samples)) and int(np.max(np.abs(samples.astype(np.int32)))) <= 1


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(5)
        try:
            line = self.rfile.readline(MAX_MESSAGE + 1)
            if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
                raise ValueError("Request too large or incomplete")
            data = json.loads(line)
            action = data.pop("action")
            if action == "shutdown":
                self.server.controller.interrupt()
                result = self.server.controller.status()
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                result = self.server.controller.command(action, **data)
            response = {"ok": True, "result": result}
        except Exception as exc:
            if locals().get("action") in ("start", "stop", "toggle", "reprocess", "warmup"):
                self.server.controller.report_error(str(exc))
            response = {"ok": False, "error": str(exc)}
        try:
            self.wfile.write(json.dumps(response).encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def service_actions(self):
        self.controller.tick()


def serve():
    os.umask(0o077)
    root = runtime_dir()
    with (root / "daemon.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # Another launcher won the startup race.
        path = root / "control.sock"
        path.unlink(missing_ok=True)
        for stale in (root / "private-recordings").glob("*.wav"):
            stale.unlink(missing_ok=True)
        controller = Controller()
        api = None
        try:
            from .local_api import start

            api = start(controller)
        except Exception as exc:
            controller.report_error(f"Local API unavailable: {exc}")
        with Server(str(path), Handler) as server:
            server.controller = controller

            def shutdown(*_):
                controller.interrupt()
                threading.Thread(target=server.shutdown, daemon=True).start()

            signal.signal(signal.SIGTERM, shutdown)
            signal.signal(signal.SIGINT, shutdown)
            try:
                server.serve_forever(poll_interval=0.025)
            finally:
                path.unlink(missing_ok=True)
                if api is not None:
                    api.shutdown()
                    api.server_close()


def request(action, **data):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(20 if action == "shutdown" else 10)
        sock.connect(str(runtime_dir() / "control.sock"))
        sock.sendall(json.dumps({"action": action, **data}).encode() + b"\n")
        with sock.makefile("rb") as stream:
            line = stream.readline(MAX_MESSAGE + 1)
        if len(line) > MAX_MESSAGE:
            raise RuntimeError("Response too large")
        result = json.loads(line)
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result["result"]


def ensure_daemon():
    try:
        request("status")
        return
    except (FileNotFoundError, ConnectionRefusedError):
        pass
    log = state_dir() / "daemon.log"
    with log.open("a") as output:
        log.chmod(0o600)
        subprocess.Popen(
            python_command("mouthkeys", "daemon"),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            request("status")
            return
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.05)
    raise RuntimeError(f"Daemon did not start; see {log}")


def ensure_overlay():
    """Optional desktop feedback; CLI-only installations remain usable."""
    import importlib.util

    if Config.load().recording_panel == "off":
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    if importlib.util.find_spec("PySide6") is None:
        return
    with (runtime_dir() / "overlay.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as wake:
                wake.settimeout(0.1)
                try:
                    wake.sendto(b"wake", str(runtime_dir() / "overlay-wake.sock"))
                except OSError:
                    pass  # An initializing panel does its own immediate first poll.
            return
    with (state_dir() / "overlay.log").open("a") as output:
        os.chmod(output.name, 0o600)
        subprocess.Popen(
            python_command("mouthkeys", "overlay"),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
