import json
import threading
import urllib.error
import urllib.request

import pytest

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
    assert call(api, "/v1/history") == []
    replacements = [{"triggers": ["mouth keys"], "replacement": "MouthKeys"}]
    assert call(api, "/v1/dictionary/replacements", {"replacements": replacements}) == replacements
    assert call(api, "/v1/dictionary/replacements") == replacements
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
