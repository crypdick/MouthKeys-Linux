"""Opt-in authenticated loopback API. Never accepts arbitrary filesystem paths."""

import base64
import binascii
import hmac
import json
import os
import socket
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from .api_dictionary import custom_words, replacements
from .config import Config
from .service import Controller
from .statistics import statistics

MAX_BODY = 25 * 1024 * 1024


_ROUTES = {
    "/v1/health": {"GET"},
    "/v1/history": {"GET"},
    "/v1/stats": {"GET"},
    "/v1/dictionary/replacements": {"GET", "POST"},
    "/v1/dictionary/custom-words": {"GET", "POST"},
    "/v1/postprocess": {"POST"},
    "/v1/transcribe": {"POST"},
}


class RequestRejectedError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ApiRequest:
    path: str
    query: dict[str, list[str]]
    native: bool
    data: dict[str, Any]
    raw: bytes
    is_json: bool


def history_item(row: dict[str, Any]) -> dict[str, Any]:
    # Legacy history lacks AI metadata; formatting differences do not establish AI use.
    raw, text = row.get("raw_text", ""), row["text"]
    return {
        **row,
        "timestamp": datetime.fromtimestamp(row["created"], UTC).isoformat(),
        "originalText": raw,
        "finalText": text,
        "rawText": raw,
        "processedText": text,
        "appName": row.get("app", ""),
        "windowTitle": row.get("window", ""),
        "characterCount": len(text),
        "wasAIProcessed": (
            bool(row["was_ai_processed"]) if row.get("was_ai_processed") is not None else None
        ),
        "aiProcessingError": row.get("ai_processing_error"),
        "processingModel": row.get("ai_processing_model"),
    }


class Handler(BaseHTTPRequestHandler):
    server: "Server"
    server_version = "MouthKeys"

    def log_message(self, _format: str, *args: object) -> None:
        pass  # Requests can contain transcript data; never log request bodies or headers.

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)
        self.read_deadline = threading.Timer(15, self.expire_read)
        self.read_deadline.daemon = True
        self.read_deadline.start()

    def expire_read(self) -> None:
        with suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)

    def finish(self) -> None:
        self.read_deadline.cancel()
        super().finish()

    def answer(self, status: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    def dispatch(self) -> None:
        with suppress(BrokenPipeError, ConnectionResetError):
            try:
                self.answer(200, self._handle(self._read_request()))
            except RequestRejectedError as exc:
                self.answer(exc.status, {"error": str(exc)})
            except (ValueError, TypeError, KeyError, binascii.Error, UnicodeError, AttributeError):
                self.answer(400, {"error": "Invalid request"})
            except (BrokenPipeError, ConnectionResetError):
                raise
            # Do not expose private request data when an operation fails.
            except Exception:  # allow: exception-handling  # noqa: BLE001
                self.answer(500, {"error": "Operation failed; verify configuration and models"})

    def _authorize(self) -> None:
        host = self.headers.get("Host", "").split(":", 1)[0]
        if host not in ("127.0.0.1", "localhost") or self.headers.get("Origin"):
            raise RequestRejectedError(403, "Only direct loopback clients are allowed")
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {self.server.token}"):
            raise RequestRejectedError(401, "Bearer token required")

    def _read_request(self) -> ApiRequest:
        self._authorize()
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked requests are not supported")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= MAX_BODY:
            raise RequestRejectedError(413, "Request exceeds 25 MB")
        parsed = urlsplit(self.path)
        path, query = parsed.path, parse_qs(parsed.query)
        format_name = query.get("format", ["upstream"])[0]
        if format_name not in ("native", "upstream"):
            raise ValueError("Unknown response format")
        if path not in _ROUTES:
            raise RequestRejectedError(404, "Unknown route")
        if self.command not in _ROUTES[path]:
            raise RequestRejectedError(405, "Method not allowed")

        data, raw = {}, b""
        is_json = self.headers.get_content_type() == "application/json"
        if self.command == "POST":
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request body")
            if is_json:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")
            elif path not in ("/v1/transcribe", "/v1/postprocess"):
                raise ValueError("Content-Type must be application/json")
        self.read_deadline.cancel()
        return ApiRequest(path, query, format_name == "native", data, raw, is_json)

    def _handle(self, request: ApiRequest) -> object:
        controller = self.server.controller
        path = request.path
        if path == "/v1/health":
            return {"status": "ok", **controller.status()}
        if path == "/v1/history":
            query = request.query
            with controller.lock:
                rows = controller.history.rows(
                    query=query.get("query", [""])[0],
                    limit=min(1000, int(query.get("limit", ["100"])[0])),
                    offset=int(query.get("offset", ["0"])[0]),
                )
            return (
                rows
                if request.native
                else {
                    "count": len(rows),
                    "items": [history_item(row) for row in rows],
                }
            )
        if path == "/v1/stats":
            return statistics(controller.history)
        update = request.data if self.command == "POST" else None
        if path.startswith("/v1/dictionary/"):
            return (
                custom_words(update)
                if path == "/v1/dictionary/custom-words"
                else replacements(update, native=request.native)
            )
        if path == "/v1/postprocess":
            return self._postprocess(request)
        return self._transcribe(_uploaded_audio(request))

    def _postprocess(self, request: ApiRequest) -> dict[str, str]:
        from .postprocess import process  # noqa: PLC0415 - load inference and profile helpers only on the matching route
        from .profiles import resolve_profile  # noqa: PLC0415 - load inference and profile helpers only on the matching route

        text = request.data.get("text") if request.is_json else request.raw.decode("utf-8")
        if not isinstance(text, str):
            raise ValueError("text must be a string")  # noqa: TRY004 - client validation
        with self.server.claim_inference():
            config = Config.load()
            if config.ai_enabled == "on":
                config = replace(resolve_profile(config), ai_profile="")
            return {
                "text": process(text, config),
                "provider": config.ai_provider if config.ai_enabled == "on" else "",
                "model": config.ai_model if config.ai_enabled == "on" else "",
            }

    def _transcribe(self, audio: bytes) -> dict[str, object]:
        from .file_transcription import SegmentDecoder, transcribe_file  # noqa: PLC0415 - load inference and profile helpers only on the matching route

        controller = self.server.controller
        _require_recording_idle(controller)
        with self.server.claim_inference(), tempfile.NamedTemporaryFile(suffix=".audio") as file:
            file.write(audio)
            file.flush()
            with controller.model_lock:
                _require_recording_idle(controller)
                config = Config.load()
                result = transcribe_file(
                    Path(file.name),
                    config,
                    transcriber=cast("SegmentDecoder", controller.transcriber),
                    max_seconds=600,
                )
        return {
            **result,
            "sampleCount": round(result["duration"] * 16000),
            "provider": result.get("model", config.model),
        }


def _uploaded_audio(request: ApiRequest) -> bytes:
    data = request.data
    if data.get("path"):
        raise ValueError("File paths are not accepted; upload audio instead")
    audio = request.raw
    if request.is_json:
        if (
            "audio_base64" in data
            and "audioBase64" in data
            and data["audio_base64"] != data["audioBase64"]
        ):
            raise ValueError("Conflicting audio fields")
        encoded = data.get("audioBase64", data.get("audio_base64", ""))
        if not isinstance(encoded, str):
            raise ValueError("Audio must be base64 text")
        audio = base64.b64decode(encoded, validate=True)
    if not audio:
        raise ValueError("Empty audio")
    return audio


def _require_recording_idle(controller: Controller) -> None:
    with controller.lock:
        if controller.pending_jobs():
            raise RequestRejectedError(409, "Recording or dictation inference is active")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    token: str
    controller: Controller

    def __init__(
        self,
        address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        bind_and_activate: bool = True,
    ) -> None:
        self.inference_lock = threading.Lock()
        self.request_slots = threading.BoundedSemaphore(8)
        super().__init__(address, handler_class, bind_and_activate=bind_and_activate)

    @contextmanager
    def claim_inference(self) -> Iterator[None]:
        if not self.inference_lock.acquire(blocking=False):
            raise RequestRejectedError(409, "API inference is busy")
        try:
            yield
        finally:
            self.inference_lock.release()

    def process_request(
        self, request: socket.socket | tuple[bytes, socket.socket], address: tuple[str, int]
    ) -> None:
        if not self.request_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.request_slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket | tuple[bytes, socket.socket], address: tuple[str, int]
    ) -> None:
        try:
            super().process_request_thread(request, address)
        finally:
            self.request_slots.release()


def start(controller: Controller) -> Server | None:
    config = Config.load()
    if config.api_enabled != "on":
        return None
    token = os.environ.get(config.api_token_env, "")
    if len(token) < 24:
        raise ValueError("Local API requires an environment token of at least 24 characters")
    server = Server(("127.0.0.1", config.api_port), Handler)
    server.token = token
    server.controller = controller
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
