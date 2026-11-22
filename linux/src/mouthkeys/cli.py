"""Parse CLI options and dispatch commands, keeping stdout available for results."""

import argparse
import getpass
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from importlib import import_module
from pathlib import Path
from typing import cast

from . import control, desktop_entry, desktop_input
from .cli_options import build_parser
from .config import RANGES, Config

MAX_TEXT_BYTES = 256 * 1024


def _read_bounded(path: Path, label: str) -> bytes:
    with path.open("rb") as source:
        raw = source.read(MAX_TEXT_BYTES + 1)
    if len(raw) > MAX_TEXT_BYTES:
        raise ValueError(f"{label} exceeds 256 KiB")
    return raw


def _read_app_map(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    mapping = json.loads(_read_bounded(path, "Application mapping"))
    if not isinstance(mapping, dict):
        raise TypeError("Application mapping must be a JSON object of macOS IDs to Linux IDs")
    return mapping


def _configure(args: argparse.Namespace) -> int:
    values = asdict(Config.load())
    if args.key is not None:
        values[args.key] = int(args.value) if args.key in RANGES else args.value
        updated = Config.from_mapping(values)
        updated.save()
        values = asdict(updated)
        if args.key in ("delivery", "paste_backend", "insertion_mode") and (
            warning := desktop_input.automatic_delivery_warning(updated)
        ):
            print(f"Warning: {warning}", file=sys.stderr)
        if args.key in ("launch_at_startup", "show_window_at_login", "tray"):
            desktop_entry.apply_startup(updated)
    print(json.dumps(values, indent=2))
    return 0


def _key_from_input(*, stdin: bool) -> str:
    if stdin:
        return sys.stdin.read(4098).removesuffix("\n")
    if not sys.stdin.isatty():
        raise ValueError("Use --stdin to supply a key without an interactive terminal")
    return getpass.getpass("API key: ")


def _credential(args: argparse.Namespace) -> int:
    from .credentials import delete_key, store_key  # noqa: PLC0415 - keep unrelated CLI commands lightweight
    from .profiles import resolve_profile  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    config = Config.load()
    if args.profile is not None:
        config = replace(config, ai_profile=args.profile)
    config = resolve_profile(config)
    if args.action == "set":
        store_key(config, _key_from_input(stdin=args.stdin))
        print("API key saved; select secret-service in target settings/profile to use it")
    else:
        if args.stdin:
            raise ValueError("--stdin is only valid with credential set")
        delete_key(config)
        print("API key deleted")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    changes = {
        name: getattr(args, name)
        for name in ("model", "engine", "device", "compute_type", "model_revision")
        if getattr(args, name) is not None
    }
    config = replace(Config.load(), **changes).validate()
    reference = (
        _read_bounded(args.reference_file, "Benchmark reference").decode("utf-8")
        if args.reference_file is not None
        else None
    )
    result = run_benchmark(config, args.file, reference=reference, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _download_model(args: argparse.Namespace) -> int:
    from .transcribe import download  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    config = Config.load()

    def progress(event: Mapping[str, object]) -> None:
        print(json.dumps({"download_progress": event}), file=sys.stderr, flush=True)

    print(
        download(
            args.model or config.model,
            config.engine,
            config.compute_type,
            revision=config.model_revision,
            progress=progress,
        )
    )
    return 0


def _transcribe_file(args: argparse.Namespace) -> int:
    from .file_transcription import export_result, result_text, transcribe_file  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    config = Config.load()

    def file_progress(event: dict[str, object]) -> None:
        if args.progress:
            print("transcription_progress " + json.dumps(event), file=sys.stderr, flush=True)

    result = (
        transcribe_file(args.file, config, on_progress=file_progress)
        if args.progress
        else transcribe_file(args.file, config)
    )
    if args.save_history and config.save_history == "on" and config.save_file_history == "on":
        from .file_history import FileHistory  # noqa: PLC0415 - keep unrelated CLI commands lightweight

        file_progress({"stage": "save"})
        FileHistory().add(args.file.name, result, config.file_history_limit)
    if args.output:
        file_progress({"stage": "export"})
        export_result(result, args.output, "json" if args.json else "txt")
    else:
        print(json.dumps(result, ensure_ascii=False) if args.json else result_text(result))
    file_progress({"stage": "complete"})
    return 0


def _file_history(args: argparse.Namespace) -> int:
    from .file_history import FileHistory  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    history = FileHistory()
    if args.command == "file-history":
        print(json.dumps(history.rows(args.query), ensure_ascii=False, indent=2))
    elif args.id:
        history.delete(args.id)
    else:
        history.clear()
    return 0


def _assistant_history(args: argparse.Namespace) -> int:
    from .assistant_history import AssistantHistory  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    history = AssistantHistory()
    result: object
    if sum((args.new, args.select, args.rename is not None, args.delete)) > 1:
        raise ValueError("Choose one chat action")
    if args.new:
        if not args.mode:
            raise ValueError("--new requires --mode")
        if Config.load().save_history != "on":
            raise ValueError("Saved chats are disabled by save_history")
        result = history.new(args.mode)
    elif args.delete:
        if not args.id or not args.yes:
            raise ValueError("Deleting a chat requires --id and --yes")
        history.delete(args.id)
        result = {"deleted": args.id}
    elif args.rename is not None:
        if not args.id:
            raise ValueError("--rename requires --id")
        history.rename(args.id, args.rename)
        result = history.get(args.id)
    elif args.select:
        if not args.id:
            raise ValueError("--select requires --id")
        result = history.select(args.id)
    else:
        result = history.get(args.id) if args.id else history.rows(args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _download_speakers(_args: argparse.Namespace) -> int:
    from .file_transcription import download_diarization  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    print(download_diarization())
    return 0


def _models(args: argparse.Namespace) -> int:
    from .engines import delete_model, model_catalog  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    config = Config.load()
    if args.command == "models":
        print(json.dumps(model_catalog(config.engine, revision=config.model_revision), indent=2))
    else:
        delete_model(args.model, config.engine, revision=config.model_revision)
        print("Cached model deleted")
    return 0


def _backup_preview(args: argparse.Namespace) -> int:
    from .operations import backup_preview  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    result = backup_preview(args.path, app_map=_read_app_map(args.app_map))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _dictionary(args: argparse.Namespace) -> int:
    from .dictionary import Dictionary  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    if args.command == "dictionary-import":
        Dictionary.import_file(args.path, mode=args.mode)
    else:
        Dictionary.export_file(args.path, format=args.format)
    print(str(args.path))
    return 0


def _doctor(_args: argparse.Namespace) -> int:
    result = desktop_input.doctor(Config.load())
    print(json.dumps(result, indent=2))
    return int(
        result["session"] == "headless"
        or bool(result["delivery_warning"])
        or not all(result["commands"].values())
    )


def _install_desktop(args: argparse.Namespace) -> int:
    if args.autostart:
        replace(Config.load(), launch_at_startup="on").save()
    print(desktop_entry.install_desktop(autostart=args.autostart))
    return 0


def _launch_desktop(args: argparse.Namespace) -> int:
    module_name = {"gui": "gui_launch", "overlay": "overlay_process"}.get(
        args.command, args.command
    )
    module = import_module("." + module_name, __package__)
    return cast("int", module.launch())


def _serve(_args: argparse.Namespace) -> int:
    from .daemon import serve  # noqa: PLC0415 - keep unrelated CLI commands lightweight

    serve()
    return 0


def _request_data(args: argparse.Namespace) -> dict[str, object]:
    data: dict[str, object] = {}
    if identifier := getattr(args, "id", None):
        data["id"] = identifier
    command = args.command
    if command == "microphone":
        data.update(source=args.source, persist=args.persist)
    if command in ("backup-import", "backup-export"):
        data["path"] = str(args.path.resolve())
    if command == "backup-import" and args.app_map is not None:
        data["app_map"] = _read_app_map(args.app_map)
    if command == "accept-edit":
        data["return_to_target"] = args.return_to_target
    if command in ("start", "toggle"):
        data.update(mode=args.mode, selected_text=args.selected_text)
        if args.label:
            data["label"] = args.label
        if args.profile is not None:
            data["profile"] = args.profile
    if command == "history":
        data.update(query=args.query, limit=args.limit, offset=args.offset)
    return data


def _show_overlay() -> None:
    try:
        control.ensure_overlay()
    # The CLI reports panel failures without masking the command result.
    except Exception as exc:  # allow: exception-handling  # noqa: BLE001
        print(f"MouthKeys panel: {exc}", file=sys.stderr)


def _daemon_command(args: argparse.Namespace) -> int:
    command = args.command
    if command != "shutdown":
        control.ensure_daemon()
    action = {"clear-history": "clear", "microphone": "select_microphone"}.get(
        command, command.replace("-", "_")
    )
    try:
        result = control.request(action, **_request_data(args))
    finally:
        if command in ("start", "stop", "toggle", "warmup"):
            _show_overlay()
    print(json.dumps(result, indent=2))
    return 0


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "credential": _credential,
    "benchmark": _benchmark,
    "config": _configure,
    "download": _download_model,
    "transcribe": _transcribe_file,
    "file-history": _file_history,
    "delete-file-history": _file_history,
    "assistant-history": _assistant_history,
    "download-speakers": _download_speakers,
    "models": _models,
    "delete-model": _models,
    "backup-preview": _backup_preview,
    "dictionary-import": _dictionary,
    "dictionary-export": _dictionary,
    "doctor": _doctor,
    "install-desktop": _install_desktop,
    "overlay": _launch_desktop,
    "gui": _launch_desktop,
    "tray": _launch_desktop,
    "hotkeys": _launch_desktop,
    "daemon": _serve,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "config" and args.key is not None and args.value is None:
        parser.error("config KEY requires VALUE")
    try:
        return _HANDLERS.get(args.command, _daemon_command)(args)
    except KeyboardInterrupt:
        return 130
    # The CLI reports command failures on stderr.
    except Exception as exc:  # allow: exception-handling  # noqa: BLE001
        print(f"MouthKeys: {exc}", file=sys.stderr)
        return 1
