"""Opt-in authenticated loopback API. Never accepts arbitrary filesystem paths."""

import base64
import binascii
import hmac
import json
import os
import socket
import tempfile
import threading
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .config import Config
from .data_lock import dictionary_lock
from .dictionary import Dictionary
from .operations import statistics

MAX_BODY = 25 * 1024 * 1024


def replacement_id(entry):
    if "id" in entry:
        return str(uuid.UUID(entry["id"]))
    value = json.dumps([entry["triggers"], entry["replacement"]], ensure_ascii=False)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "mouthkeys:replacement:" + value))


def history_item(row):
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
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, f"Bearer {self.server.token}"):
                self.answer(401, {"error": "Bearer token required"})
                return
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Chunked requests are not supported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= MAX_BODY:
                self.answer(413, {"error": "Request exceeds 25 MB"})
                return
            path = urlsplit(self.path)
            query = parse_qs(path.query)
            native = query.get("format", ["upstream"])[0] == "native"
            if query.get("format", ["upstream"])[0] not in ("native", "upstream"):
                raise ValueError("Unknown response format")
            routes = {
                "/v1/health": {"GET"},
                "/v1/history": {"GET"},
                "/v1/stats": {"GET"},
                "/v1/dictionary/replacements": {"GET", "POST"},
                "/v1/dictionary/custom-words": {"GET", "POST"},
                "/v1/postprocess": {"POST"},
                "/v1/transcribe": {"POST"},
            }
            if path.path not in routes:
                self.answer(404, {"error": "Unknown route"})
                return
            if self.command not in routes[path.path]:
                self.answer(405, {"error": "Method not allowed"})
                return
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
                elif path.path not in ("/v1/transcribe", "/v1/postprocess"):
                    raise ValueError("Content-Type must be application/json")
            self.read_deadline.cancel()
            controller = self.server.controller
            if path.path == "/v1/health":
                self.answer(200, {"status": "ok", **controller.status()})
            elif path.path == "/v1/history":
                rows = controller.command(
                    "history",
                    query=query.get("query", [""])[0],
                    limit=min(1000, int(query.get("limit", [100])[0])),
                    offset=int(query.get("offset", [0])[0]),
                )
                items = [history_item(row) for row in rows]
                self.answer(200, rows if native else {"count": len(items), "items": items})
            elif path.path == "/v1/stats":
                self.answer(200, statistics(controller.history))
            elif path.path.startswith("/v1/dictionary/"):
                self.dictionary(path.path, data, native=native)
            elif path.path == "/v1/postprocess":
                from .postprocess import process
                from .profiles import resolve_profile

                text = data.get("text") if is_json else raw.decode("utf-8")
                if not isinstance(text, str):
                    raise ValueError("text must be a string")
                if not self.server.inference_lock.acquire(blocking=False):
                    self.answer(409, {"error": "API inference is busy"})
                    return
                try:
                    config = Config.load()
                    if config.ai_enabled == "on":
                        config = replace(resolve_profile(config), ai_profile="")
                    self.answer(
                        200,
                        {
                            "text": process(text, config),
                            "provider": config.ai_provider if config.ai_enabled == "on" else "",
                            "model": config.ai_model if config.ai_enabled == "on" else "",
                        },
                    )
                finally:
                    self.server.inference_lock.release()
            else:
                if data.get("path"):
                    raise ValueError("File paths are not accepted; upload audio instead")
                if is_json:
                    if "audio_base64" in data and "audioBase64" in data:
                        if data["audio_base64"] != data["audioBase64"]:
                            raise ValueError("Conflicting audio fields")
                    audio = base64.b64decode(
                        data.get("audioBase64", data.get("audio_base64", "")), validate=True
                    )
                else:
                    audio = raw
                if not audio:
                    raise ValueError("Empty audio")
                self.transcribe(audio)
        except (ValueError, TypeError, KeyError, binascii.Error, UnicodeError, AttributeError):
            self.answer(400, {"error": "Invalid request"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.answer(500, {"error": "Operation failed; verify configuration and models"})

    def dictionary(self, path, data, *, native):
        from .dictionary_transfer import merge_replacements, phrase_key, replacement_entries
        from .vocabulary import VocabularyStore

        with dictionary_lock():
            if path.endswith("custom-words"):
                entries = VocabularyStore.entries()
                if self.command == "POST":
                    mode = data.get("mode", "append")
                    if mode not in ("append", "replace"):
                        raise ValueError("Invalid vocabulary write mode")
                    incoming = data.get("entries", [data] if "text" in data else None)
                    if incoming is None:
                        raise ValueError("Expected custom word entries")
                    if "entries" not in data:
                        incoming = [{key: value for key, value in data.items() if key != "mode"}]
                    incoming = VocabularyStore.validate_entries(incoming)
                    keys = {item["text"].casefold() for item in incoming}
                    entries = incoming + (
                        []
                        if mode == "replace"
                        else [item for item in entries if item["text"].casefold() not in keys]
                    )
                    entries = VocabularyStore.save_entries(entries)
                self.answer(200, {"count": len(entries), "items": entries})
                return
            dictionary = Dictionary.load()
            if self.command == "POST":
                mode = data.get("mode", "append")
                if mode not in ("append", "replace"):
                    raise ValueError("Invalid replacement write mode")
                if "replacements" in data:
                    incoming = replacement_entries(data["replacements"])
                    mode = "replace"
                else:
                    values = data.get("entries")
                    if values is None and "triggers" in data:
                        values = [{key: value for key, value in data.items() if key != "mode"}]
                    if values is None:
                        raise ValueError("Expected replacement entries")
                    incoming = replacement_entries(values)
                identifiers = {entry["id"] for entry in incoming if "id" in entry}
                replacements = {phrase_key(entry["replacement"]) for entry in incoming}
                existing = (
                    []
                    if mode == "replace"
                    else [
                        entry
                        for entry in dictionary["replacements"]
                        if replacement_id(entry) not in identifiers
                        and phrase_key(entry["replacement"]) not in replacements
                    ]
                )
                dictionary["replacements"] = merge_replacements(existing, incoming)
                Dictionary.save(dictionary)
            entries = dictionary["replacements"]
            items = [{"id": replacement_id(entry), **entry} for entry in entries]
            self.answer(200, entries if native else {"count": len(items), "items": items})

    def transcribe(self, audio):
        from .file_transcription import transcribe_file

        controller = self.server.controller
        with controller.lock:
            busy = bool(controller._pending_jobs())
        if busy:
            self.answer(409, {"error": "Recording or dictation inference is active"})
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self.answer(409, {"error": "API inference is busy"})
            return
        try:
            with tempfile.NamedTemporaryFile(suffix=".audio") as file:
                file.write(audio)
                file.flush()
                with controller.model_lock:
                    with controller.lock:
                        busy = bool(controller._pending_jobs())
                    if busy:
                        self.answer(409, {"error": "Recording or dictation inference is active"})
                        return
                    config = Config.load()
                    result = transcribe_file(
                        Path(file.name), config, transcriber=controller.transcriber, max_seconds=600
                    )
            self.answer(
                200,
                {
                    **result,
                    "sampleCount": round(result["duration"] * 16000),
                    "provider": result.get("model", config.model),
                },
            )
        finally:
            self.server.inference_lock.release()


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
