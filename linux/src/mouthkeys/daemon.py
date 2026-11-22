"""Host the singleton control socket and optional local HTTP API."""

import fcntl
import json
import os
import signal
import socketserver
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Protocol, runtime_checkable

from .config import runtime_dir
from .control import MAX_MESSAGE
from .local_api import start as start_local_api
from .service import Controller


@runtime_checkable
class ControlCommands(Protocol):
    def command(self, action: str, **data: object) -> object: ...
    def status(self) -> object: ...
    def interrupt(self) -> None: ...
    def tick(self) -> None: ...
    def report_error(self, message: str) -> None: ...


class Handler(socketserver.StreamRequestHandler):
    server: "Server"

    def handle(self) -> None:
        self.connection.settimeout(5)
        action = None
        shutdown = False
        try:
            line = self.rfile.readline(MAX_MESSAGE + 1)
            if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
                raise ValueError("Request too large or incomplete")
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError("Request must be a JSON object")
            action = data.pop("action", None)
            if not isinstance(action, str) or not action:
                raise ValueError("Request must name an action")
            controller = self.server.controller
            if action == "shutdown":
                controller.interrupt()
                result = controller.status()
                shutdown = True
            else:
                result = controller.command(action, **data)
            response = {"ok": True, "result": result}
        # Return command failures over IPC; do not duplicate private data in logs.
        except Exception as exc:  # allow: exception-handling  # noqa: BLE001
            if action in ("start", "stop", "toggle", "warmup"):
                self.server.controller.report_error(str(exc))
            response = {"ok": False, "error": str(exc)}
        try:
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(json.dumps(response).encode() + b"\n")
        finally:
            # The process can exit as soon as serve_forever stops; acknowledge first.
            if shutdown:
                self.server.schedule_shutdown()


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, controller: ControlCommands) -> None:
        self.controller = controller
        super().__init__(path, Handler)

    def service_actions(self) -> None:
        self.controller.tick()

    def schedule_shutdown(self) -> None:
        threading.Thread(target=self.shutdown, daemon=True).start()


def serve(make_controller: Callable[[], Controller] = Controller) -> None:
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
        controller = make_controller()
        with Server(str(path), controller) as server:
            api = None
            try:
                api = start_local_api(controller)
            # The control socket remains available if the optional API fails.
            except Exception as exc:  # allow: exception-handling  # noqa: BLE001
                server.controller.report_error(f"Local API unavailable: {exc}")

            def shutdown(_signal: int, _frame: object) -> None:
                server.controller.interrupt()
                server.schedule_shutdown()

            try:
                signal.signal(signal.SIGTERM, shutdown)
                signal.signal(signal.SIGINT, shutdown)
                server.serve_forever(poll_interval=0.025)
            finally:
                path.unlink(missing_ok=True)
                if api is not None:
                    api.shutdown()
                    api.server_close()
