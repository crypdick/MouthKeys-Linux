import base64
import io
import json
import threading
import urllib.error
import urllib.request
import uuid
from email.message import Message
from types import SimpleNamespace

import pytest

from mouthkeys import local_api
from mouthkeys.config import Config
from mouthkeys.dictionary import Dictionary
from mouthkeys.local_api import Handler, Server, history_item
from mouthkeys.service import Controller


@pytest.fixture
def api_server():
    controller = Controller()
    server = Server(("127.0.0.1", 0), Handler)
    server.controller = controller
    server.token = "a-long-test-token-for-loopback-only"  # noqa: S105
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    assert not thread.is_alive()


@pytest.fixture
def api(api_server):
    return f"http://127.0.0.1:{api_server.server_port}", api_server.token


def call(api, path, data=None, *, authenticated=True, headers=None):
    address, token = api
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    if authenticated:
        request_headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(  # noqa: S310
        address + path,
        headers=request_headers,
        data=json.dumps(data).encode() if data is not None else None,
    )
    with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310
        return json.load(response)


def test_authenticated_health_history_and_dictionary(api):
    assert call(api, "/v1/health")["phase"] == "idle"
    assert call(api, "/v1/health")["status"] == "ok"
    assert call(api, "/v1/history") == {"count": 0, "items": []}
    assert call(api, "/v1/history?format=native") == []
    replacements = [{"triggers": ["mouth keys"], "replacement": "MouthKeys"}]
    result = call(api, "/v1/dictionary/replacements", {"replacements": replacements})
    assert result["count"] == 1
    assert result["items"][0]["replacement"] == "MouthKeys"
    uuid.UUID(result["items"][0]["id"])
    assert call(api, "/v1/dictionary/replacements") == result
    assert call(api, "/v1/dictionary/replacements?format=native") == replacements
    assert call(api, "/v1/stats")["words"] == 0


@pytest.mark.parametrize(
    "options",
    [
        {"authenticated": False},
        {"headers": {"Origin": "https://attacker.example"}},
        {"headers": {"Host": "attacker.example"}},
    ],
)
def test_rejects_unauthenticated_browser_and_rebinding_requests(api, options):
    with pytest.raises(urllib.error.HTTPError) as error:
        call(api, "/v1/health", **options)
    assert error.value.code in (401, 403)


def test_api_does_not_read_arbitrary_file_paths(api):
    with pytest.raises(urllib.error.HTTPError) as error:
        call(api, "/v1/transcribe", {"path": "/etc/passwd"})
    assert error.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as error:
        call(api, "/v1/history?limit=-1")
    assert error.value.code == 400


def raw_call(api, path, body, content_type):
    address, token = api
    request = urllib.request.Request(  # noqa: S310
        address + path,
        data=body,
        headers={"Content-Type": content_type, "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310
        return json.load(response)


def test_upstream_replacement_entries_and_id_update(api):
    result = call(
        api,
        "/v1/dictionary/replacements",
        {"entries": [{"triggers": ["mouth keys"], "replacement": "MouthKeys"}]},
    )
    entry = result["items"][0]
    changed = call(
        api,
        "/v1/dictionary/replacements",
        {"entries": [{**entry, "replacement": "MouthKeys Linux"}]},
    )
    assert changed == {"count": 1, "items": [{**entry, "replacement": "MouthKeys Linux"}]}
    assert Dictionary.apply("mouth keys") == "MouthKeys Linux"


def test_old_api_id_still_updates_replacement_after_portable_merge(api, tmp_path):
    identifier = str(uuid.uuid4())
    call(
        api,
        "/v1/dictionary/replacements",
        {"id": identifier, "triggers": ["mouth keys"], "replacement": "MouthKeys"},
    )
    path = tmp_path / "portable.json"
    path.write_text(json.dumps({"replacements": [{"from": ["mouth kies"], "to": "MouthKeys"}]}))
    Dictionary.import_file(path, "merge")
    result = call(
        api,
        "/v1/dictionary/replacements",
        {"id": identifier, "triggers": ["updated"], "replacement": "MouthKeys Linux"},
    )
    assert result == {
        "count": 1,
        "items": [{"id": identifier, "triggers": ["updated"], "replacement": "MouthKeys Linux"}],
    }


def test_upstream_custom_words_optional_fields_and_float_weights(api):
    result = call(api, "/v1/dictionary/custom-words", {"text": "MouthKeys", "weight": 8.5})
    assert result == {"count": 1, "items": [{"text": "MouthKeys", "weight": 8.5, "aliases": []}]}
    assert call(
        api, "/v1/dictionary/custom-words", {"mode": "replace", "entries": [{"text": "Linux"}]}
    ) == {"count": 1, "items": [{"text": "Linux", "weight": None, "aliases": []}]}


@pytest.mark.parametrize("payload", [{"audioBase64": "YWJj"}, {"audio_base64": "YWJj"}, None])
def test_upstream_and_native_audio_uploads(api, monkeypatch, payload):
    observed = []

    def fake_transcribe(path, config, **options):
        observed.append((path.read_bytes(), options))
        return {"text": "hello", "duration": 1.25, "model": "fixture", "confidence": None}

    monkeypatch.setattr("mouthkeys.file_transcription.transcribe_file", fake_transcribe)
    result = (
        call(api, "/v1/transcribe", payload)
        if payload is not None
        else raw_call(api, "/v1/transcribe", b"abc", "audio/wav")
    )
    assert result == {
        "text": "hello",
        "duration": 1.25,
        "model": "fixture",
        "confidence": None,
        "sampleCount": 20000,
        "provider": "fixture",
    }
    assert observed[0][0] == b"abc"
    assert observed[0][1]["max_seconds"] == 600


def test_plain_text_postprocess_and_response_metadata(api, monkeypatch):
    monkeypatch.setattr("mouthkeys.postprocess.process", lambda text, config: text.upper())
    Config(ai_enabled="on", ai_model="test-model").save()
    assert raw_call(api, "/v1/postprocess", "hello λ".encode(), "text/plain; charset=utf-8") == {
        "text": "HELLO Λ",
        "provider": "compatible",
        "model": "test-model",
    }


def test_history_envelope_metadata_and_legacy_rows():
    row = {
        "id": "test",
        "created": 0,
        "text": "Hello.",
        "raw_text": "hello",
        "app": "Editor",
        "window": "notes",
    }
    result = history_item(row)
    assert result["timestamp"] == "1970-01-01T00:00:00+00:00"
    assert result["originalText"] == "hello"
    assert result["finalText"] == "Hello."
    assert result["appName"] == "Editor"
    assert result["windowTitle"] == "notes"
    assert result["characterCount"] == 6
    assert result["wasAIProcessed"] is None
    assert result["aiProcessingError"] is None
    assert all(result[key] == value for key, value in row.items())


@pytest.mark.parametrize(
    "payload",
    [
        {"audioBase64": "%%%"},
        {"audioBase64": ""},
        {"audioBase64": "YWJj", "audio_base64": base64.b64encode(b"different").decode()},
        {"path": "/etc/passwd", "audioBase64": "YWJj"},
    ],
)
def test_rejects_ambiguous_or_invalid_audio(api, payload):
    with pytest.raises(urllib.error.HTTPError) as error:
        call(api, "/v1/transcribe", payload)
    assert error.value.code == 400


def test_api_replacement_failure_preserves_dictionary(api):
    call(api, "/v1/dictionary/replacements", {"triggers": ["old"], "replacement": "safe"})
    with pytest.raises(urllib.error.HTTPError) as error:
        call(
            api,
            "/v1/dictionary/replacements",
            {"entries": [{"triggers": ["new"], "replacement": "new"}, {"triggers": []}]},
        )
    assert error.value.code == 400
    assert Dictionary.apply("old") == "safe"


def test_api_upsert_replaces_target_triggers_instead_of_transfer_merge(api):
    call(api, "/v1/dictionary/replacements", {"triggers": ["old"], "replacement": "same"})
    result = call(api, "/v1/dictionary/replacements", {"triggers": ["new"], "replacement": "same"})
    assert result["count"] == 1
    assert result["items"][0]["triggers"] == ["new"]
    assert Dictionary.apply("old new") == "old same"


@pytest.mark.parametrize(
    ("path", "data", "code"),
    [
        ("/v1/history", {}, 405),
        ("/v1/missing", None, 404),
        ("/v1/history?format=garbage", None, 400),
    ],
)
def test_route_and_format_contract(api, path, data, code):
    with pytest.raises(urllib.error.HTTPError) as error:
        call(api, path, data)
    assert error.value.code == code


@pytest.mark.parametrize("route", ["postprocess", "transcribe"])
def test_completed_inference_does_not_block_while_response_is_written(api, monkeypatch, route):
    writing, release = threading.Event(), threading.Event()
    responses = []
    respond = Handler.answer

    def answer(handler, status, value):
        if status == 200 and value.get("text") == "first":
            writing.set()
            assert release.wait(3)
        return respond(handler, status, value)

    monkeypatch.setattr(Handler, "answer", answer)
    if route == "postprocess":
        monkeypatch.setattr("mouthkeys.postprocess.process", lambda text, _config: text)
    else:
        monkeypatch.setattr(
            "mouthkeys.file_transcription.transcribe_file",
            lambda path, _config, **_options: {"text": path.read_text(), "duration": 1.0},
        )

    def submit(text):
        data = (
            {"text": text}
            if route == "postprocess"
            else {"audioBase64": base64.b64encode(text.encode()).decode()}
        )
        return call(api, f"/v1/{route}", data)["text"]

    worker = threading.Thread(target=lambda: responses.append(submit("first")))
    worker.start()
    try:
        assert writing.wait(2)
        assert submit("second") == "second"
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert responses == ["first"]


@pytest.mark.parametrize(
    ("headers", "raw", "response"),
    [
        ({"Transfer-Encoding": "chunked"}, b"", 400),
        ({"Content-Length": str(local_api.MAX_BODY + 1)}, b"", 413),
        ({"Content-Length": "-1"}, b"", 413),
        ({"Content-Length": "2"}, b"x", 400),
        ({"Content-Type": "application/json", "Content-Length": "2"}, b"[]", 400),
        ({"Content-Type": "text/plain", "Content-Length": "3"}, b"abc", 400),
    ],
)
def test_dictionary_rejects_invalid_body_before_mutation(monkeypatch, headers, raw, response):
    request_headers = Message()
    request_headers["Host"] = "localhost"
    request_headers["Authorization"] = "Bearer test-token"
    for key, value in headers.items():
        request_headers[key] = value
    replies = []
    handler = object.__new__(Handler)
    handler.headers = request_headers
    handler.server = SimpleNamespace(token="test-token")  # noqa: S106
    handler.command = "POST"
    handler.path = "/v1/dictionary/replacements"
    handler.rfile = io.BytesIO(raw)
    handler.answer = lambda status, body: replies.append((status, body))
    monkeypatch.setattr(Dictionary, "save", lambda _: pytest.fail("Invalid request wrote data"))
    handler.dispatch()
    assert replies[0][0] == response


@pytest.mark.parametrize("route", ["postprocess", "transcribe"])
def test_inference_rejects_request_while_slot_is_claimed(api_server, api, route):
    data = {"text": "hello"} if route == "postprocess" else {"audioBase64": "YWJj"}
    with api_server.inference_lock, pytest.raises(urllib.error.HTTPError) as caught:
        call(api, f"/v1/{route}", data)
    assert caught.value.code == 409
    assert json.load(caught.value)["error"] == "API inference is busy"


@pytest.mark.parametrize("during_lock", [False, True])
def test_upload_never_transcribes_while_recording_is_pending(
    api_server, api, monkeypatch, during_lock
):
    checked = []

    def pending():
        held = api_server.controller.model_lock.locked()
        checked.append(held)
        return [object()] if held is during_lock else []

    monkeypatch.setattr(api_server.controller, "pending_jobs", pending)
    monkeypatch.setattr(
        "mouthkeys.file_transcription.transcribe_file", lambda *a, **kw: pytest.fail("Model ran")
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        call(api, "/v1/transcribe", {"audioBase64": "YWJj"})
    assert caught.value.code == 409
    assert json.load(caught.value)["error"] == "Recording or dictation inference is active"
    assert checked == ([False, True] if during_lock else [False])
    assert not api_server.inference_lock.locked()


def test_failed_postprocessing_is_generic_and_releases_inference_slot(api, monkeypatch):
    def fail(_text, _config):
        raise RuntimeError("Private detail")

    monkeypatch.setattr("mouthkeys.postprocess.process", fail)
    with pytest.raises(urllib.error.HTTPError) as caught:
        call(api, "/v1/postprocess", {"text": "hello"})
    assert caught.value.code == 500
    assert json.load(caught.value) == {"error": "Operation failed; verify configuration and models"}
    monkeypatch.setattr("mouthkeys.postprocess.process", lambda text, _config: text)
    assert call(api, "/v1/postprocess", {"text": "hello"}) == {
        "text": "hello",
        "provider": "",
        "model": "",
    }


@pytest.mark.parametrize(
    ("route", "payload"),
    [
        ("postprocess", {"text": 1}),
        ("transcribe", {"audioBase64": 1}),
        ("transcribe", {"audio_base64": [], "audioBase64": []}),
        ("dictionary/replacements", {"mode": "unknown"}),
        ("dictionary/replacements", {}),
        ("dictionary/custom-words", {"mode": "unknown"}),
        ("dictionary/custom-words", {}),
    ],
)
def test_invalid_route_payload_is_rejected(api, route, payload):
    with pytest.raises(urllib.error.HTTPError) as caught:
        call(api, f"/v1/{route}", payload)
    assert caught.value.code == 400


def test_replacement_batch_replaces_old_rules_and_custom_words_can_be_read(api):
    call(api, "/v1/dictionary/replacements", {"triggers": ["old"], "replacement": "gone"})
    rules = [{"triggers": ["new"], "replacement": "kept"}]
    assert (
        call(
            api,
            "/v1/dictionary/replacements?format=native",
            {
                "mode": "replace",
                "entries": rules,
            },
        )
        == rules
    )
    assert Dictionary.apply("old new") == "old kept"

    call(api, "/v1/dictionary/custom-words", {"text": "Linux", "weight": 1})
    call(api, "/v1/dictionary/custom-words", {"text": "GTK"})
    result = call(api, "/v1/dictionary/custom-words", {"text": "LINUX", "weight": 9})
    assert call(api, "/v1/dictionary/custom-words") == result
    assert {e["text"]: e["weight"] for e in result["items"]} == {"LINUX": 9, "GTK": None}


def test_start_is_opt_in_and_requires_token(monkeypatch):
    monkeypatch.delenv(Config().api_token_env, raising=False)
    controller = Controller()
    assert local_api.start(controller) is None
    Config(api_enabled="on").save()
    with pytest.raises(ValueError, match="at least 24 characters"):
        local_api.start(controller)


def test_start_serves_only_loopback_and_sets_configured_token(monkeypatch):
    env_name = "MOUTHKEYS_TEST_TOKEN"
    config = SimpleNamespace(api_enabled="on", api_token_env=env_name, api_port=0)
    token = "a-long-test-token-for-loopback"  # noqa: S105
    monkeypatch.setenv(config.api_token_env, token)
    monkeypatch.setattr(Config, "load", lambda: config)
    server = local_api.start(Controller())
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert (
            call((f"http://127.0.0.1:{server.server_port}", token), "/v1/health")["status"] == "ok"
        )
    finally:
        server.shutdown()
        server.server_close()
