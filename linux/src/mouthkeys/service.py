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

from . import __version__
from .audio import Recorder, check_audio, live_meter
from .config import Config, runtime_dir, state_dir
from .delivery import Delivery
from .processes import python_command
from .storage import History
from .transcribe import Transcriber

MAX_MESSAGE = 1024 * 1024


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
            if action == "history":
                return self.history.rows()
            if action == "toggle":
                action = "stop" if self.phase == "recording" else "start"
            if self.interrupted:
                raise ValueError("Service is shutting down")
            if action == "cancel":
                if self.phase == "delivering":
                    raise ValueError(
                        "Delivery already started; waiting for clipboard/paste to finish"
                    )
                self.preview_text = ""
                self.cancelled = True
                if self.phase == "recording":
                    self._stop(None)
                elif self.phase in ("stopping", "transcribing", "loading"):
                    self.message = "Cancelling; waiting for capture/model to finish"
                return self.status()
            if action == "stop":
                if self.phase != "recording":
                    raise ValueError("Not recording")
                target = (
                    "capture-focus"
                    if self.config.delivery == "x11" and not data.get("automatic")
                    else None
                )
                self._stop(target, _focus_gate)
                return self.status()
            if self.phase != "idle":
                raise ValueError(f"Busy ({self.phase}); wait or cancel before {action}")
            if action == "start":
                self.meter = {}
                self.preview_text = ""
                self.preview_at = time.monotonic()
                self.config = Config.load()
                self.item_id = str(uuid.uuid4())
                self.cancelled = False
                self.history.begin(self.item_id)
                try:
                    self.recorder = self.recorder_factory(
                        self.config, self.history.recording(self.item_id)
                    )
                except Exception as exc:
                    self.history.update(self.item_id, "failed", error=str(exc))
                    self.message = str(exc)
                    raise
                self.notice = None
                self.phase, self.message = "recording", "Listening — toggle again to finish"
                self.started = time.monotonic()
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
                shutil.copyfile(self.history.recording(item_id), self.history.recording(retry_id))
                self.history.begin(retry_id)
                self.item_id = retry_id
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
                self.history.clear()
                self.item_id = None
                self.message = "History and recordings cleared"
                return self.status()
            raise ValueError(f"Unknown command: {action}")

    def interrupt(self):
        """Stop on logout without interpreting it as the user's discard command."""
        with self.lock:
            self.interrupted = True
            self.cancelled = True
            recorder, self.recorder = self.recorder, None
            if self.item_id and (
                self.phase in ("recording", "stopping", "transcribing")
                or (self.phase == "delivering" and self.dictation_delivery)
            ):
                # Preserve a transcript already persisted before delivery, if any.
                row = next((r for r in self.history.rows() if r["id"] == self.item_id), {})
                self.history.update(
                    self.item_id, "failed", row.get("text", ""), "Interrupted; reprocess audio"
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

    def _stop(self, target, focus_gate=None):
        recorder, self.recorder = self.recorder, None
        self.stop_done.clear()
        self.notice = None
        self.phase, self.message, self.started = "stopping", "Stopping microphone", None
        self.worker = threading.Thread(
            target=self._stopped, args=(recorder, target, focus_gate), daemon=True
        )
        self.worker.start()

    def _stopped(self, recorder, target, focus_gate):
        try:
            if target == "capture-focus":
                try:
                    target = self.delivery.focus()
                except Exception:
                    target = None
            if focus_gate:
                focus_gate["ready"].set()
            try:
                recorder.stop()
            finally:
                self.stop_done.set()
        except Exception as exc:
            with self.lock:
                if not self.interrupted:
                    self.history.update(self.item_id, "failed", error=str(exc))
                    self.phase, self.message = "idle", str(exc)
            return
        with self.lock:
            if self.interrupted:
                return
            if self.cancelled:
                self.history.recording(self.item_id).unlink(missing_ok=True)
                self.history.update(self.item_id, "cancelled")
                self.phase, self.message = "idle", "Cancelled"
                return
            self.phase, self.message = "transcribing", "Transcribing locally"
            self.history.update(self.item_id, "transcribing")
        if focus_gate and not focus_gate["valid"]:
            target = None
        self._finish(target)

    def _manual_delivery(self, action, data):
        with self.lock:
            if self.interrupted:
                raise ValueError("Service is shutting down")
            if self.phase != "idle":
                raise ValueError(f"Busy ({self.phase}); wait or cancel before {action}")
            text = self.history.latest_text()
            if data.get("id"):
                text = next(
                    (row["text"] for row in self.history.rows() if row["id"] == data["id"]), ""
                )
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
        self.phase, self.message = "transcribing", "Transcribing locally"
        self.started = None
        self.history.update(self.item_id, "transcribing")
        self.worker = threading.Thread(target=self._finish, args=(target,), daemon=True)
        self.worker.start()

    def _finish(self, target):
        text = ""
        try:
            path = self.history.recording(self.item_id)
            check_audio(path)
            with self.model_lock:
                with self.lock:
                    skip = self.cancelled or self.interrupted
                if not skip:
                    text = self.transcriber.transcribe(path, self.config)
            with self.lock:
                if self.interrupted:
                    return
                if self.cancelled:
                    self.history.update(self.item_id, "cancelled")
                    path.unlink(missing_ok=True)
                    self.message = "Cancelled"
                    return
                if not text:
                    self.preview_text = ""
                    self.history.update(self.item_id, "failed", error="No speech detected")
                    self.message = "No speech detected; audio retained for reprocessing"
                    return
                self.preview_text = text
                self.history.update(self.item_id, "transcribed", text)
                self.dictation_delivery = True
                self.phase, self.message = "delivering", "Delivering transcript"
            result = self.delivery.deliver(text, self.config, target)
            with self.lock:
                if self.interrupted:
                    return
                self.history.update(self.item_id, result, text)
                self.history.prune_audio(self.item_id)
                self.message = (
                    "Copied — paste into your app"
                    if result == "copied"
                    else "Paste requested — transcript also on clipboard"
                )
        except Exception as exc:
            with self.lock:
                if self.interrupted:
                    return
                if self.cancelled:
                    self.history.update(self.item_id, "cancelled")
                    self.history.recording(self.item_id).unlink(missing_ok=True)
                    self.message = "Cancelled"
                else:
                    self.preview_text = text
                    self.history.update(self.item_id, "failed", text, str(exc))
                    self.message = f"{exc}. Saved in history; audio retained for reprocessing."
        finally:
            with self.lock:
                if not self.interrupted:
                    self.dictation_delivery = False
                    self.phase = "idle"
                    self._prune_budget()

    def _preview(self, item_id, config):
        try:
            with self.model_lock:
                with self.lock:
                    if self.phase != "recording" or self.item_id != item_id:
                        return
                text = self.transcriber.preview(self.history.recording(item_id), config)
            with self.lock:
                if self.phase == "recording" and self.item_id == item_id and not self.cancelled:
                    self.preview_text = text
        except Exception:
            pass  # Preview is provisional; final inference reports actionable errors.
        finally:
            with self.lock:
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
                    meter = live_meter(self.history.recording(self.item_id))
                    if meter["frame"] != self.meter.get("frame"):
                        self.meter = {**meter, "at": time.monotonic()}
                except (OSError, ValueError, EOFError, wave.Error):
                    self.meter = {}
            if (
                self.phase == "recording"
                and time.monotonic() - self.preview_at >= 2
                and (self.preview_worker is None or not self.preview_worker.is_alive())
                and hasattr(self.transcriber, "preview")
            ):
                self.preview_worker = threading.Thread(
                    target=self._preview, args=(self.item_id, self.config), daemon=True
                )
                self.preview_worker.start()
            if self.phase == "recording" and (
                self.recorder.finished()
                or time.monotonic() - self.started >= self.config.max_seconds
            ):
                try:
                    self.command("stop", automatic=True)
                except Exception:
                    pass  # The stop path has already persisted the error for the UI.


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
        controller = Controller()
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
