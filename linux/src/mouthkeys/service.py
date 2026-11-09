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
from .audio import Recorder, check_audio, live_meter
from .config import Config, private_dir, runtime_dir, state_dir
from .delivery import Delivery
from .desktop import MediaSession, choose_source, play_sound
from .formatting import enabled, format_text, parse_send
from .postprocess import process
from .processes import python_command
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
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False
    duration: float = 0
    phase: str = "recording"
    send_pending: bool = False
    context: str | None = None
    suppress_send: bool = False
    metadata: dict = field(default_factory=dict)


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
        with self.lock:
            if action == "status":
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
                if self.phase == "delivering" and self.job and self.job.send_pending:
                    self.cancelled = True
                    self.job.cancelled = True
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
                if self.phase == "recording":
                    self._stop(None)
                elif self.phase in ("stopping", "transcribing", "queued", "loading"):
                    self.message = "Cancelling; waiting for capture/model to finish"
                return self.status()
            if action == "stop":
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
                mode = data.get("mode", "dictation")
                selected_text = data.get("selected_text", "")
                if mode not in ("dictation", "command", "edit", "training"):
                    raise ValueError("Unknown recording mode")
                if not isinstance(selected_text, str) or len(selected_text) > 100000:
                    raise ValueError("Selected text exceeds 100000 characters")
                self.meter = {}
                self.preview_text = ""
                self.preview_at = time.monotonic()
                self.config = Config.load()
                self.config = replace(self.config, audio_source=choose_source(self.config))
                self.send_armed_at = None
                self.send_disarmed_text = None
                self.last_voice_at = time.monotonic()
                if len(self._pending_jobs()) >= getattr(self.config, "queue_limit", 8):
                    raise ValueError("Dictation queue is full; wait for earlier recordings")
                for previous in self._pending_jobs():
                    previous.suppress_send = True
                self.item_id = str(uuid.uuid4())
                self._new_job(mode, selected_text)
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
                self.preview_text = ""
                self.last_text = ""
                self.assistant_proposal = None
                self.history.clear()
                self.item_id = None
                self.message = "History and recordings cleared"
                return self.status()
            raise ValueError(f"Unknown command: {action}")

    def _pending_jobs(self):
        return [job for job in self.jobs if not job.done.is_set()]

    def _new_job(self, mode="dictation", selected_text=""):
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
            self.cancelled = True
            recorder, self.recorder = self.recorder, None
            for job in self._pending_jobs():
                job.cancelled = True
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
                    target = self.delivery.focus()
                except Exception:
                    target = None
            metadata_target = target
            if (
                job.config.save_history == "on"
                and metadata_target is None
                and hasattr(self.delivery, "target_metadata")
            ):
                try:
                    metadata_target = self.delivery.focus()
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
            if job.config.save_history == "on" and hasattr(self.delivery, "target_metadata"):
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
                job.audio_path.unlink(missing_ok=True)
                self._update(job, "cancelled")
                self._job_status(job, "idle", "Cancelled")
                if job.config.save_history == "off":
                    job.audio_path.unlink(missing_ok=True)
                job.done.set()
                return
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
        ai_warning = ""
        send_warning = ""
        config = job.config
        path = job.audio_path
        try:
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
            if job.mode != "dictation":
                if job.mode == "training":
                    proposal = {"mode": "training", "text": text}
                else:
                    from .assistance import propose

                    proposal = propose(text, config, job.mode, job.selected_text)
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
                return
            plan = parse_send(
                text,
                config.spoken_send_phrase,
                active=enabled(config, "spoken_send") and target is not None,
                literal_prefix=config.literal_prefix,
            )
            text = format_text(plan.text, config, context=job.context)
            if enabled(config, "ai_enabled") and text:
                try:
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
                self.last_text = text
                self._update(job, "transcribed", text)
                self._job_status(job, "delivering", "Delivering transcript")
            result = self.delivery.deliver(text, config, target) if text else "copied"
            if plan.should_send and (
                not text or result in ("paste_requested", "insert_requested", "pasted")
            ):
                job.send_pending = True
                if self.job is job:
                    self.send_pending = True
                try:
                    time.sleep(0.5)
                    with self.lock:
                        send = not self.interrupted and not job.cancelled and not job.suppress_send
                    if send:
                        if not self.delivery.send_key(config, target):
                            send_warning = "Spoken Send skipped: destination could not be verified"
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
                    "Copied — paste into your app"
                    if result == "copied"
                    else (
                        "Insertion requested — transcript saved in history"
                        if config.save_history == "on"
                        else "Insertion requested — history disabled"
                    )
                )
                if ai_warning:
                    message += f". {ai_warning}"
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
                    path = self.job.audio_path
                text = self.transcriber.preview(path, config)
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
                and time.monotonic() - self.preview_at >= 2
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
