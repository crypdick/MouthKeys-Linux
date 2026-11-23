"""Unix socket client and launch helpers; importing this does not load the controller."""

import fcntl
import importlib.util
import json
import logging
import os
import socket
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from .config import Config, runtime_dir, state_dir
from .processes import python_command

MAX_MESSAGE = 1024 * 1024


def request(action: str, **data: object) -> object:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(
            20 if action in ("shutdown", "start", "toggle", "select_microphone") else 10
        )
        client.connect(str(runtime_dir() / "control.sock"))
        client.sendall(json.dumps({"action": action, **data}).encode() + b"\n")
        with client.makefile("rb") as stream:
            line = stream.readline(MAX_MESSAGE + 1)
    if len(line) > MAX_MESSAGE:
        raise RuntimeError("Response too large")
    response = json.loads(line)
    if isinstance(response, dict):
        if response.get("ok") is True and "result" in response:
            return response["result"]
        error = response.get("error")
        if response.get("ok") is False and isinstance(error, str):
            raise RuntimeError(error)
    raise RuntimeError("Invalid daemon response")


def _launch(
    action: str, *arguments: str, pass_fds: tuple[int, ...] = ()
) -> tuple[subprocess.Popen[bytes], Path]:
    log = state_dir() / f"{action}.log"
    with log.open("a") as output:
        log.chmod(0o600)
        process = subprocess.Popen(  # noqa: S603
            python_command("mouthkeys", action, *arguments),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
            pass_fds=pass_fds,
        )
    return process, log


def ensure_daemon() -> None:
    with suppress(FileNotFoundError, ConnectionRefusedError):
        request("status")
        return
    _, log = _launch("daemon")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            request("status")
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.05)
        else:
            return
    raise RuntimeError(f"Daemon did not start; see {log}")


def launch_tray() -> tuple[socket.socket, subprocess.Popen[bytes]] | None:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    if Config.load().tray != "on" or importlib.util.find_spec("PySide6") is None:
        return None
    service, tray = socket.socketpair()
    try:
        with tray:
            fd = tray.fileno()
            process, _ = _launch("tray", "--service-fd", str(fd), pass_fds=(fd,))
    except OSError as exc:
        service.close()
        logging.getLogger(__name__).warning("MouthKeys tray: %s", exc)
        return None
    return service, process


def ensure_overlay() -> None:
    """Optional desktop feedback; CLI-only installations remain usable."""
    if Config.load().recording_panel == "off":
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    if importlib.util.find_spec("PySide6") is None:
        return
    root = runtime_dir()
    with (root / "overlay.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as wake:
                wake.settimeout(0.1)
                # An initializing panel does its own immediate first poll.
                with suppress(OSError):
                    wake.sendto(b"wake", str(root / "overlay-wake.sock"))
            return
    _launch("overlay")
