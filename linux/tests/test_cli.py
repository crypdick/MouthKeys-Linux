import json
import os
import socket
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import pytest

from mouthkeys.cli import main
from mouthkeys.config import runtime_dir
from mouthkeys.service import request


def test_microphone_cli_preserves_session_and_persistent_selection(monkeypatch, capsys):
    from mouthkeys import service

    calls = []
    monkeypatch.setattr(service, "ensure_daemon", lambda: None)
    monkeypatch.setattr(
        service, "request", lambda action, **data: calls.append((action, data)) or {}
    )
    assert main(["microphone", "usb"]) == 0
    assert main(["microphone", "default", "--persist"]) == 0
    assert calls == [
        ("select_microphone", {"source": "usb", "persist": False}),
        ("select_microphone", {"source": "default", "persist": True}),
    ]


def test_dictionary_cli_portable_roundtrip_and_merge(tmp_path, capsys):
    from mouthkeys.dictionary import Dictionary
    from mouthkeys.vocabulary import VocabularyStore

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "replacements": [{"from": ["heard"], "to": "Written"}],
                "customWords": [{"text": "Written", "weight": 8.5}],
            }
        )
    )
    assert main(["dictionary-import", str(source), "--mode", "merge", "--yes"]) == 0
    assert Dictionary.apply("heard") == "Written"
    assert VocabularyStore.entries()[0]["weight"] == 8.5
    output = tmp_path / "portable.json"
    assert main(["dictionary-export", str(output), "--format", "upstream"]) == 0
    saved = json.loads(output.read_text())
    assert saved["replacements"] == [{"from": ["heard"], "to": "Written"}]
    assert saved["customWords"][0]["weight"] == 8.5


def test_benchmark_cli_uses_overrides_without_daemon_or_saved_config(monkeypatch, tmp_path, capsys):
    from mouthkeys import benchmark, service
    from mouthkeys.config import config_path

    calls = []
    reference = tmp_path / "reference.txt"
    reference.write_text("hello fluid voice")
    monkeypatch.setattr(service, "ensure_daemon", lambda: pytest.fail("Benchmark started daemon"))
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda config, fixture, **kwargs: (
            calls.append((config, fixture, kwargs)) or {"measured": True}
        ),
    )
    assert (
        main(
            [
                "benchmark",
                "fixture.wav",
                "--model",
                "/cached/snapshot",
                "--engine",
                "whisper",
                "--device",
                "cpu",
                "--compute-type",
                "int8",
                "--revision",
                "pinned",
                "--reference-file",
                str(reference),
                "--timeout",
                "90",
            ]
        )
        == 0
    )
    config, fixture, options = calls[0]
    assert (
        config.model,
        config.engine,
        config.device,
        config.compute_type,
        config.model_revision,
    ) == ("/cached/snapshot", "whisper", "cpu", "int8", "pinned")
    assert str(fixture) == "fixture.wav"
    assert options == {"reference": "hello fluid voice", "timeout": 90.0}
    assert json.loads(capsys.readouterr().out) == {"measured": True}
    assert not config_path().exists()


def test_backup_preview_cli_never_constructs_history_or_starts_daemon(
    monkeypatch, tmp_path, capsys
):
    from dataclasses import asdict

    from mouthkeys import service, storage
    from mouthkeys.config import Config, config_path
    from mouthkeys.dictionary import Dictionary

    path = tmp_path / "backup.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "config": asdict(Config()),
                "dictionary": Dictionary.defaults(),
                "history": [],
                "profiles": [],
            }
        )
    )
    monkeypatch.setattr(service, "ensure_daemon", lambda: pytest.fail("Preview started daemon"))
    monkeypatch.setattr(storage, "History", lambda *a, **kw: pytest.fail("Preview opened history"))
    assert main(["backup-preview", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["format"] == "native"
    assert result["counts"]["history"] == 0
    assert len(result["file_sha256"]) == 64
    assert not config_path().exists()


def test_cli_reports_version_without_starting_service(monkeypatch, capsys):
    from mouthkeys import service

    def unexpected():
        raise AssertionError("Version must not start a service")

    monkeypatch.setattr(service, "ensure_daemon", unexpected)
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == f"MouthKeys {version('mouthkeys-linux')}"


def test_config_cli_and_doctor(capsys):
    assert main(["config", "language", "auto"]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["language"] == "auto"
    assert config["compute_type"] == "default"
    assert main(["config", "compute_type", "int8"]) == 0
    assert json.loads(capsys.readouterr().out)["compute_type"] == "int8"
    assert main(["config", "language", "en"]) == 0
    assert json.loads(capsys.readouterr().out)["compute_type"] == "int8"
    assert main(["doctor"]) == 1  # No real desktop is exposed to tests.
    assert json.loads(capsys.readouterr().out)["session"] == "headless"
    assert main(["config", "max_seconds", "zero"]) == 1


def test_desktop_installer_uses_absolute_executable_and_no_autostart_by_default():
    assert main(["install-desktop"]) == 0
    entry = Path(os.environ["XDG_DATA_HOME"]) / "applications/mouthkeys.desktop"
    assert 'Exec="/' in entry.read_text()
    assert not (Path(os.environ["XDG_CONFIG_HOME"]) / "autostart").exists()


def test_real_daemon_ipc_duplicate_start_malformed_request_and_shutdown():
    process = subprocess.Popen(
        [sys.executable, "-m", "mouthkeys", "daemon"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            try:
                result = request("status")
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.03)
        else:
            raise AssertionError("Daemon failed to start")
        assert result["phase"] == "idle"
        assert result["version"] == version("mouthkeys-linux")
        duplicate = subprocess.run([sys.executable, "-m", "mouthkeys", "daemon"], timeout=3)
        assert duplicate.returncode == 0
        assert request("status")["phase"] == "idle"
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(2)
            client.connect(str(runtime_dir() / "control.sock"))
            client.sendall(b"not-json\n")
            assert json.loads(client.recv(4096))["ok"] is False
        assert request("history") == []
    finally:
        process.terminate()
        try:
            _, errors = process.communicate(timeout=4)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=2)
            raise
    assert process.returncode == 0, errors.decode()
    assert not (runtime_dir() / "control.sock").exists()


def test_overlay_launcher_skips_headless_and_existing_panel(monkeypatch):
    import fcntl

    from mouthkeys.service import ensure_overlay

    def unexpected(*args, **kwargs):
        raise AssertionError("Must not launch a desktop process")

    monkeypatch.setattr(subprocess, "Popen", unexpected)
    ensure_overlay()
    monkeypatch.setenv("DISPLAY", ":fake")
    with (runtime_dir() / "overlay.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as wake:
            wake.bind(str(runtime_dir() / "overlay-wake.sock"))
            wake.settimeout(1)
            ensure_overlay()
            assert wake.recv(64) == b"wake"


def test_installed_command_from_untrusted_directory_starts_safe_daemon(tmp_path):
    package = tmp_path / "mouthkeys"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('untrusted package imported')")
    (tmp_path / "json.py").write_text("raise RuntimeError('untrusted dependency imported')")
    # Use the actual installed console entry point; its child must not import cwd.
    executable = Path(sys.executable).parent / "mouthkeys"
    try:
        result = subprocess.run(
            [str(executable), "status"], cwd=tmp_path, capture_output=True, timeout=8
        )
        assert result.returncode == 0, result.stderr.decode()
        assert json.loads(result.stdout)["phase"] == "idle"
    finally:
        try:
            request("shutdown")
        except (FileNotFoundError, ConnectionRefusedError):
            pass
        deadline = time.monotonic() + 3
        while (runtime_dir() / "control.sock").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
    assert not (runtime_dir() / "control.sock").exists()


def test_failed_shortcut_still_starts_feedback_panel(monkeypatch, capsys):
    from mouthkeys import service

    launched = []
    monkeypatch.setattr(service, "ensure_daemon", lambda: None)
    monkeypatch.setattr(service, "ensure_overlay", lambda: launched.append(True))

    def fail(*args, **kwargs):
        raise RuntimeError("Busy (loading)")

    monkeypatch.setattr(service, "request", fail)
    assert main(["toggle"]) == 1
    assert launched == [True]
    assert "Busy (loading)" in capsys.readouterr().err


def test_reviewed_edit_cli_preserves_explicit_target_and_focus_choice(monkeypatch, capsys):
    from mouthkeys import service

    calls = []
    monkeypatch.setattr(service, "ensure_daemon", lambda: None)
    monkeypatch.setattr(
        service, "request", lambda action, **data: calls.append((action, data)) or {}
    )
    assert main(["accept-edit", "--id", "captured-job", "--return-to-target"]) == 0
    assert calls == [("accept_edit", {"id": "captured-job", "return_to_target": True})]


def test_socket_errors_persist_feedback_for_panel():
    import threading

    from mouthkeys.service import Controller, Handler, Server

    controller = Controller()
    controller.phase = "loading"
    with Server(str(runtime_dir() / "control.sock"), Handler) as server:
        server.controller = controller
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            import pytest

            with pytest.raises(RuntimeError, match="Busy"):
                request("toggle")
            status = request("status")
            assert status["phase"] == "loading"
            assert "Busy (loading)" in status["notice"]["message"]
        finally:
            server.shutdown()
            thread.join(2)
        assert not thread.is_alive()
