import base64
import json
import threading
import urllib.error
import urllib.request
import uuid

import pytest

from mouthkeys.config import Config
from mouthkeys.dictionary import Dictionary
from mouthkeys.local_api import Handler, Server
from mouthkeys.service import Controller


@pytest.fixture
def api():
    controller = Controller()
    server = Server(("127.0.0.1", 0), Handler)
    server.controller = controller
    server.token = "a-long-test-token-for-loopback-only"
    server.inference_lock = threading.Lock()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", server.token
    server.shutdown()
    server.server_close()
    thread.join(2)
    assert not thread.is_alive()


def call(api, path, data=None, *, authenticated=True, headers=None):
    address, token = api
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    if authenticated:
        request_headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        address + path,
        headers=request_headers,
        data=json.dumps(data).encode() if data is not None else None,
    )
    with urllib.request.urlopen(request, timeout=2) as response:
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
    request = urllib.request.Request(
        address + path,
        data=body,
        headers={"Content-Type": content_type, "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
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
    from mouthkeys.local_api import history_item

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
    "path,data,code",
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
