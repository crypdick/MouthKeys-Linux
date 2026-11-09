"""Opt-in authenticated loopback API. Never accepts arbitrary filesystem paths."""

import base64
import binascii
import hmac
import json
import os
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .config import Config
from .dictionary import Dictionary
from .operations import statistics

MAX_BODY = 25 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "MouthKeys"

    def log_message(self, *_):
        pass  # Requests can contain transcript data; never log request bodies or headers.

    def setup(self):
        super().setup()
        self.connection.settimeout(10)
        self.read_deadline = threading.Timer(15, self.expire_read)
        self.read_deadline.daemon = True
        self.read_deadline.start()

    def expire_read(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def finish(self):
        self.read_deadline.cancel()
        super().finish()

    def answer(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch()

    def dispatch(self):
        try:
            host = self.headers.get("Host", "").split(":", 1)[0]
            if host not in ("127.0.0.1", "localhost") or self.headers.get("Origin"):
                self.answer(403, {"error": "Only direct loopback clients are allowed"})
                return
            token = self.server.token
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, f"Bearer {token}"):
                self.answer(401, {"error": "Bearer token required"})
                return
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Chunked requests are not supported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= MAX_BODY:
                self.answer(413, {"error": "Request exceeds 25 MB"})
                return
            data = {}
            if self.command == "POST":
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")
            self.read_deadline.cancel()
            path = urlsplit(self.path)
            query = parse_qs(path.query)
            controller = self.server.controller
            if path.path == "/v1/health" and self.command == "GET":
                self.answer(200, controller.status())
            elif path.path == "/v1/history" and self.command == "GET":
                self.answer(
                    200,
                    controller.command(
                        "history",
                        query=query.get("query", [""])[0],
                        limit=min(1000, int(query.get("limit", [100])[0])),
                        offset=int(query.get("offset", [0])[0]),
                    ),
                )
            elif path.path == "/v1/stats" and self.command == "GET":
                self.answer(200, statistics(controller.history))
            elif path.path in ("/v1/dictionary/replacements", "/v1/dictionary/custom-words"):
                if path.path.endswith("custom-words"):
                    self.answer(501, {"error": "Engine vocabulary boosting is not available"})
                    return
                dictionary = Dictionary.load()
                if self.command == "POST":
                    dictionary["replacements"] = data["replacements"]
                    Dictionary.save(dictionary)
                self.answer(200, dictionary["replacements"])
            elif path.path == "/v1/postprocess" and self.command == "POST":
                from .postprocess import process

                text = data.get("text")
                if not isinstance(text, str):
                    raise ValueError("text must be a string")
                # Explicit endpoint call still honors configured opt-in/provider policy.
                if not self.server.inference_lock.acquire(blocking=False):
                    self.answer(409, {"error": "API inference is busy"})
                    return
                try:
                    self.answer(200, {"text": process(text, Config.load())})
                finally:
                    self.server.inference_lock.release()
            elif path.path == "/v1/transcribe" and self.command == "POST":
                from .file_transcription import transcribe_file

                audio = base64.b64decode(data["audio_base64"], validate=True)
                if not audio:
                    raise ValueError("Empty audio")
                if not self.server.inference_lock.acquire(blocking=False):
                    self.answer(409, {"error": "API inference is busy"})
                    return
                try:
                    with tempfile.NamedTemporaryFile(suffix=".audio") as file:
                        file.write(audio)
                        file.flush()
                        with controller.model_lock:
                            result = transcribe_file(
                                Path(file.name),
                                Config.load(),
                                transcriber=controller.transcriber,
                                max_seconds=600,
                            )
                    self.answer(200, result)
                finally:
                    self.server.inference_lock.release()
            else:
                self.answer(404, {"error": "Unknown route"})
        except (ValueError, TypeError, KeyError, binascii.Error):
            self.answer(400, {"error": "Invalid request"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.answer(500, {"error": "Operation failed; verify configuration and models"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        self.request_slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def process_request(self, request, address):
        if not self.request_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.request_slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.request_slots.release()


def start(controller):
    config = Config.load()
    if config.api_enabled != "on":
        return None
    token = os.environ.get(config.api_token_env, "")
    if len(token) < 24:
        raise ValueError("Local API requires an environment token of at least 24 characters")
    server = Server(("127.0.0.1", config.api_port), Handler)
    server.token = token
    server.controller = controller
    server.inference_lock = threading.Lock()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
