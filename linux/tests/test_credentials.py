import io
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mouthkeys import credentials
from mouthkeys.ai_providers import request_headers
from mouthkeys.config import Config, config_path


def test_environment_remains_default_and_keyring_never_probed(monkeypatch):
    monkeypatch.setenv("MOUTHKEYS_AI_API_KEY", "environment-secret")
    monkeypatch.setattr(credentials, "_run", lambda *a, **kw: pytest.fail("Keyring probe"))
    assert request_headers(Config(), "compatible")["Authorization"] == "Bearer environment-secret"


def test_keyring_key_scoped_to_endpoint_and_never_in_arguments_or_settings(monkeypatch):
    cfg = Config(
        ai_credential_source="secret-service",
        ai_endpoint="https://provider.example/v1/chat/completions",
    )
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"saved-secret\n")

    monkeypatch.setattr(credentials.subprocess, "run", run)
    credentials.store_key(cfg, "saved-secret")
    assert calls[0][1]["input"] == b"saved-secret"
    assert "saved-secret" not in " ".join(calls[0][0])
    assert calls[0][0][-2:] == ["endpoint", cfg.ai_endpoint]
    assert not config_path().exists()
    assert request_headers(cfg, "anthropic")["x-api-key"] == "saved-secret"
    assert calls[-1][1]["timeout"] == 3
    assert calls[-1][1]["stderr"] == subprocess.DEVNULL
    credentials.delete_key(cfg)
    assert calls[-1][0][1] == "clear"
    assert calls[-1][1]["stdout"] == subprocess.DEVNULL
    assert calls[-1][0][-1] == cfg.ai_endpoint


@pytest.mark.parametrize(
    "failure", [FileNotFoundError(), subprocess.TimeoutExpired("secret-tool", 3)]
)
def test_keyring_failure_never_falls_back_to_environment_or_leaks_diagnostics(monkeypatch, failure):
    monkeypatch.setenv("MOUTHKEYS_AI_API_KEY", "wrong-endpoint-secret")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(credentials.subprocess, "run", fail)
    with pytest.raises(RuntimeError) as error:
        credentials.api_key(Config(ai_credential_source="secret-service"))
    assert "wrong-endpoint-secret" not in str(error.value)


@pytest.mark.parametrize("raw", [b"bad\nheader\n", b"x" * 4097, b"\xff", b"\n"])
def test_keyring_validates_returned_secrets(monkeypatch, raw):
    monkeypatch.setattr(credentials, "_run", lambda *a, **kw: raw)
    with pytest.raises((ValueError, RuntimeError)):
        request_headers(Config(ai_credential_source="secret-service"), "compatible")


def test_cli_credential_stdin_never_echoes_or_saves_secret(monkeypatch, capsys):
    from mouthkeys.cli import main

    calls = []
    monkeypatch.setattr("sys.stdin", io.StringIO("stdin-secret\n"))
    monkeypatch.setattr(credentials, "store_key", lambda cfg, key: calls.append((cfg, key)))
    assert main(["credential", "set", "--stdin"]) == 0
    assert calls[0][1] == "stdin-secret"
    assert "stdin-secret" not in capsys.readouterr().out
    assert not config_path().exists()


def test_legacy_profiles_gain_environment_default():
    from dataclasses import asdict

    from mouthkeys.profiles import PROFILE_FIELDS, validate_profiles

    values = asdict(Config(ai_model="example"))
    profile = {key: values[key] for key in PROFILE_FIELDS if key != "ai_credential_source"}
    profile.update(id="8dbe8247-7c5f-4f19-b08e-bbb8c397e879", name="legacy")
    assert validate_profiles([profile])[0]["ai_credential_source"] == "environment"
    profile["ai_credential_source"] = "plaintext"
    with pytest.raises(ValueError, match="credential source"):
        validate_profiles([profile])


def test_config_rejects_unknown_destination_guard_and_credential_source():
    for key in ("destination_guard", "ai_credential_source"):
        with pytest.raises(ValueError):
            replace(Config(), **{key: "anything"}).validate()


def test_settings_keyring_storage_is_explicit_and_enables_only_editor_source(monkeypatch):
    from PySide6.QtWidgets import QApplication, QInputDialog

    from mouthkeys import gui

    monkeypatch.setattr(gui.Window, "send", lambda *a, **kw: None)
    monkeypatch.setattr(gui, "audio_sources", lambda: ["default"])
    calls = []
    monkeypatch.setattr(credentials, "store_key", lambda config, key: calls.append((config, key)))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("typed-secret", True))
    window = gui.Window()
    try:
        assert not calls
        window.keyring_action("store")
        # Only the fake storage runs; window is never shown and no keyring exists.
        import time

        deadline = time.monotonic() + 2
        while window.jobs and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.001)
        assert not window.jobs
        assert calls[0][1] == "typed-secret"
        assert window.fields["ai_credential_source"].currentData() == "secret-service"
        assert not config_path().exists()
        assert "typed-secret" not in window.status.text()
    finally:
        window.close()
