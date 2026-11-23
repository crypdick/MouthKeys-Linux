import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from mouthkeys import (
    benchmark,
    cli,
    control,
    credentials,
    desktop_input,
    engines,
    file_transcription,
    operations,
    transcribe,
)
from mouthkeys.config import Config, config_path
from mouthkeys.file_history import FileHistory
from mouthkeys.profiles import PROFILE_FIELDS, save_profiles


@pytest.fixture(autouse=True)
def fake_daemon(monkeypatch):
    ensure = Mock()
    request = Mock(return_value={"ok": True})
    overlay = Mock()
    monkeypatch.setattr(control, "ensure_daemon", ensure)
    monkeypatch.setattr(control, "request", request)
    monkeypatch.setattr(control, "ensure_overlay", overlay)
    return SimpleNamespace(ensure=ensure, request=request, overlay=overlay)


@pytest.mark.parametrize(
    ("arguments", "action", "payload", "overlay"),
    [
        (
            ["history", "--query", "words", "--limit", "5", "--offset", "2"],
            "history",
            {"query": "words", "limit": 5, "offset": 2},
            False,
        ),
        (
            [
                "start",
                "--mode",
                "edit",
                "--label",
                "Editor",
                "--selected-text",
                "Draft",
                "--profile",
                "off",
            ],
            "start",
            {"mode": "edit", "label": "Editor", "selected_text": "Draft", "profile": "off"},
            True,
        ),
        (["toggle"], "toggle", {"mode": "dictation", "selected_text": ""}, True),
        (["copy"], "copy", {}, False),
        (["delete", "--id", "entry"], "delete", {"id": "entry"}, False),
        (["clear-history", "--yes"], "clear", {}, False),
        (["shutdown"], "shutdown", {}, False),
    ],
)
def test_ipc_dispatch_matches_cli_grammar(fake_daemon, capsys, arguments, action, payload, overlay):
    assert cli.main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    fake_daemon.request.assert_called_once_with(action, **payload)
    assert fake_daemon.ensure.call_count == (action != "shutdown")
    assert fake_daemon.overlay.call_count == overlay


def test_overlay_failure_does_not_change_command_result(fake_daemon, capsys):
    fake_daemon.overlay.side_effect = OSError("panel unavailable")
    assert cli.main(["warmup"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {"ok": True}
    assert output.err == "MouthKeys panel: panel unavailable\n"


def test_keyboard_interrupt_is_not_a_cli_error(fake_daemon, capsys):
    fake_daemon.request.side_effect = KeyboardInterrupt
    assert cli.main(["stop"]) == 130
    fake_daemon.overlay.assert_called_once_with()
    assert capsys.readouterr() == ("", "")


def test_backup_arguments_use_absolute_paths_and_explicit_mapping(
    fake_daemon, monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    mapping = tmp_path / "map.json"
    mapping.write_text('{"mac.editor": "linux-editor"}')
    assert cli.main(["backup-import", "incoming.json", "--yes", "--app-map", "map.json"]) == 0
    fake_daemon.request.assert_called_once_with(
        "backup_import",
        path=str(tmp_path / "incoming.json"),
        app_map={"mac.editor": "linux-editor"},
    )
    fake_daemon.request.reset_mock()
    assert cli.main(["backup-export", "outgoing.json"]) == 0
    fake_daemon.request.assert_called_once_with(
        "backup_export", path=str(tmp_path / "outgoing.json")
    )
    fake_daemon.request.reset_mock()
    assert cli.main(["backup-import", "incoming.json", "--yes"]) == 0
    fake_daemon.request.assert_called_once_with(
        "backup_import", path=str(tmp_path / "incoming.json")
    )
    assert not capsys.readouterr().err


@pytest.mark.parametrize(
    ("mapping", "message"),
    [(b"[]", "must be a JSON object"), (b"x" * (256 * 1024 + 1), "exceeds 256 KiB")],
)
def test_preview_rejects_malformed_mapping_without_work(
    fake_daemon, monkeypatch, tmp_path, capsys, mapping, message
):
    path = tmp_path / "map.json"
    path.write_bytes(mapping)
    preview = Mock()
    monkeypatch.setattr(operations, "backup_preview", preview)
    assert cli.main(["backup-preview", "backup.json", "--app-map", str(path)]) == 1
    assert message in capsys.readouterr().err
    preview.assert_not_called()
    fake_daemon.ensure.assert_not_called()


def test_benchmark_default_and_oversized_reference(monkeypatch, tmp_path, capsys):
    run = Mock(return_value={"score": 1})
    monkeypatch.setattr(benchmark, "run_benchmark", run)
    assert cli.main(["benchmark", "test.wav"]) == 0
    run.assert_called_once_with(Config(), Path("test.wav"), reference=None, timeout=600)
    capsys.readouterr()
    reference = tmp_path / "reference.txt"
    reference.write_bytes(b"a" * (256 * 1024 + 1))
    assert cli.main(["benchmark", "test.wav", "--reference-file", str(reference)]) == 1
    assert "Benchmark reference exceeds 256 KiB" in capsys.readouterr().err
    assert run.call_count == 1


@pytest.mark.parametrize("override", [None, "chosen/model"])
def test_explicit_model_download_uses_selected_engine_and_separates_progress(
    monkeypatch, capsys, override, fake_daemon
):
    Config(
        engine="onnx", model="default/model", compute_type="int8", model_revision="pinned"
    ).save()
    download = Mock()
    progress = {"phase": "download", "unit": "file", "completed": 1, "total": 1}

    def run(*args, **options):
        options["progress"](progress)
        return "/cached/model"

    download.side_effect = run
    monkeypatch.setattr(transcribe, "download", download)
    assert cli.main(["download", *([override] if override else [])]) == 0
    assert download.call_args.args == (override or "default/model", "onnx", "int8")
    assert download.call_args.kwargs["revision"] == "pinned"
    output = capsys.readouterr()
    assert output.out == "/cached/model\n"
    assert json.loads(output.err) == {"download_progress": progress}
    fake_daemon.ensure.assert_not_called()


def test_speaker_download_models_and_cache_deletion_keep_command_contracts(monkeypatch, capsys):
    Config(engine="onnx", model_revision="pinned").save()
    speakers = Mock(return_value="/cached/speakers")
    catalog = Mock(return_value=[{"model": "example"}])
    remove = Mock()
    monkeypatch.setattr(file_transcription, "download_diarization", speakers)
    monkeypatch.setattr(engines, "model_catalog", catalog)
    monkeypatch.setattr(engines, "delete_model", remove)
    assert cli.main(["download-speakers"]) == 0
    assert capsys.readouterr().out == "/cached/speakers\n"
    speakers.assert_called_once_with()
    assert cli.main(["models"]) == 0
    assert json.loads(capsys.readouterr().out) == [{"model": "example"}]
    catalog.assert_called_once_with("onnx", revision="pinned")
    assert cli.main(["delete-model", "example"]) == 0
    assert capsys.readouterr().out == "Cached model deleted\n"
    remove.assert_called_once_with("example", "onnx", revision="pinned")


@pytest.mark.parametrize("disabled", ["save_history", "save_file_history"])
def test_file_history_opt_out_keeps_text_output_and_prevents_saving(monkeypatch, capsys, disabled):
    Config(**{disabled: "off"}).save()
    monkeypatch.setattr(
        file_transcription,
        "transcribe_file",
        Mock(return_value={"text": "Résumé", "duration": 1, "processing_seconds": 1}),
    )
    assert cli.main(["transcribe", "test.wav", "--save-history"]) == 0
    assert capsys.readouterr() == ("Résumé\n", "")
    assert FileHistory().rows() == []


def test_file_history_filters_and_explicit_deletion(capsys):
    result = {"text": "Transcript", "duration": 1, "processing_seconds": 1}
    history = FileHistory()
    first = history.add("first.wav", result)
    second = history.add("second.wav", result)
    assert cli.main(["file-history", "--query", "first"]) == 0
    assert [entry["id"] for entry in json.loads(capsys.readouterr().out)] == [first]
    assert cli.main(["delete-file-history", "--id", first, "--yes"]) == 0
    assert [entry["id"] for entry in history.rows()] == [second]
    assert cli.main(["delete-file-history", "--yes"]) == 0
    assert history.rows() == []
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--new", "--select"], "Choose one chat action"),
        (["--new"], "--new requires --mode"),
        (["--delete"], "Deleting a chat requires --id and --yes"),
        (["--rename", "Label"], "--rename requires --id"),
        (["--select"], "--select requires --id"),
    ],
)
def test_saved_chat_usage_errors_keep_stdout_clean(capsys, flags, message):
    assert cli.main(["assistant-history", *flags]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert message in output.err


def test_saved_chat_select_and_delete_use_explicit_id(capsys):
    assert cli.main(["assistant-history", "--new", "--mode", "edit"]) == 0
    item = json.loads(capsys.readouterr().out)
    assert cli.main(["assistant-history", "--mode", "command"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    for flags in ([], ["--select"]):
        assert cli.main(["assistant-history", "--id", item["id"], *flags]) == 0
        assert json.loads(capsys.readouterr().out) == item
    assert cli.main(["assistant-history", "--delete", "--yes", "--id", item["id"]]) == 0
    assert json.loads(capsys.readouterr().out) == {"deleted": item["id"]}
    assert cli.main(["assistant-history"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_profile_credential_commands_use_hidden_prompt_and_resolved_endpoint(monkeypatch, capsys):
    profile_id = str(uuid4())
    values = asdict(Config(ai_model="example-model", ai_endpoint="https://example.invalid/v1"))
    save_profiles(
        [
            {
                **{field: values[field] for field in PROFILE_FIELDS},
                "id": profile_id,
                "name": "Preview",
            }
        ]
    )
    prompt = Mock(return_value="key-for-test")
    store, delete = Mock(), Mock()
    monkeypatch.setattr(cli.getpass, "getpass", prompt)
    monkeypatch.setattr(credentials, "store_key", store)
    monkeypatch.setattr(credentials, "delete_key", delete)
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    assert cli.main(["credential", "set"]) == 1
    assert "Use --stdin" in capsys.readouterr().err
    prompt.assert_not_called()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    assert cli.main(["credential", "set", "--profile", profile_id]) == 0
    prompt.assert_called_once_with("API key: ")
    resolved, key = store.call_args.args
    assert (resolved.ai_model, resolved.ai_endpoint, key) == (
        "example-model",
        "https://example.invalid/v1",
        "key-for-test",
    )
    assert "key-for-test" not in capsys.readouterr().out
    assert cli.main(["credential", "delete", "--stdin"]) == 1
    delete.assert_not_called()
    assert "only valid" in capsys.readouterr().err
    assert cli.main(["credential", "delete", "--profile", profile_id]) == 0
    delete.assert_called_once_with(resolved)
    assert capsys.readouterr().out == "API key deleted\n"
    assert not config_path().exists()


@pytest.mark.parametrize("missing", [False, True])
def test_doctor_exit_status_is_based_on_report_without_running_desktop(
    monkeypatch, capsys, missing
):
    report = {
        "session": "x11",
        "delivery_warning": None,
        "commands": {"tool": None if missing else "/bin/tool"},
    }
    monkeypatch.setattr(desktop_input, "doctor", Mock(return_value=report))
    assert cli.main(["doctor"]) == int(missing)
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize(
    ("command", "module"),
    [
        ("gui", "gui_launch"),
        ("tray", "tray"),
        ("overlay", "overlay_process"),
        ("hotkeys", "hotkeys"),
    ],
)
def test_desktop_commands_load_only_requested_launcher(monkeypatch, command, module):
    launch = Mock(return_value=7)
    loader = Mock(return_value=SimpleNamespace(launch=launch))
    monkeypatch.setattr(cli, "import_module", loader)
    assert cli.main([command]) == 7
    loader.assert_called_once_with("." + module, "mouthkeys")
    launch.assert_called_once_with()


def test_tray_command_passes_daemon_lifeline(monkeypatch):
    launch = Mock(return_value=0)
    monkeypatch.setattr(cli, "import_module", Mock(return_value=SimpleNamespace(launch=launch)))
    assert cli.main(["tray", "--service-fd", "7"]) == 0
    launch.assert_called_once_with(7)


def test_config_missing_value_reports_parser_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["config", "language"])
    assert exc.value.code == 2
    assert "config KEY requires VALUE" in capsys.readouterr().err
    assert not config_path().exists()
