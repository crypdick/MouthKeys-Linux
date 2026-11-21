import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

from .cli_options import build_parser
from .config import RANGES, Config


def install_desktop(autostart=False):
    executable = shutil.which("mouthkeys") or sys.argv[0]
    executable = str(Path(executable).resolve())
    # Desktop Entry Exec is not a shell; escape its reserved characters and field codes.
    quoted = (
        '"'
        + executable.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("%", "%%")
        + '"'
    )
    data = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    applications = data / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    entry = (
        "[Desktop Entry]\nType=Application\nName=MouthKeys\n"
        "Comment=Local voice to text\nIcon=audio-input-microphone\n"
        f"Exec={quoted} gui\nTerminal=false\nCategories=AudioVideo;Audio;Utility;\n"
    )
    (applications / "mouthkeys.desktop").write_text(entry)
    if autostart:
        import importlib.util

        config = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        directory = config / "autostart"
        directory.mkdir(parents=True, exist_ok=True)
        settings = Config.load()
        command = (
            "gui"
            if settings.show_window_at_login == "on"
            else (
                "tray"
                if settings.tray == "on" and importlib.util.find_spec("PySide6")
                else "daemon"
            )
        )
        (directory / "mouthkeys.desktop").write_text(entry.replace(" gui\n", f" {command}\n"))
    return str(applications / "mouthkeys.desktop")


def apply_startup(config):
    if config.launch_at_startup == "on":
        install_desktop(autostart=True)
    else:
        directory = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        (directory / "autostart" / "mouthkeys.desktop").unlink(missing_ok=True)


def read_app_map(path):
    if path is None:
        return None
    with path.open("rb") as source:
        raw = source.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError("Application mapping exceeds 256 KiB")
    mapping = json.loads(raw)
    if not isinstance(mapping, dict):
        raise ValueError("Application mapping must be a JSON object of macOS IDs to Linux IDs")
    return mapping


def _configure(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    config = asdict(Config.load())
    if args.key:
        if args.value is None:
            parser.error("config KEY requires VALUE")
        config[args.key] = int(args.value) if args.key in RANGES else args.value
        updated = Config.from_mapping(config)
        updated.save()
        config = asdict(updated)
        if args.key in ("delivery", "paste_backend", "insertion_mode"):
            from .delivery import automatic_delivery_warning

            if warning := automatic_delivery_warning(updated):
                print(f"Warning: {warning}", file=sys.stderr)
        if args.key in ("launch_at_startup", "show_window_at_login", "tray"):
            apply_startup(Config.load())
    elif args.value:
        parser.error("config requires KEY VALUE")
    print(json.dumps(config, indent=2))


def main(argv: list[str] | None = None) -> int | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "credential":
            from .credentials import delete_key, store_key
            from .profiles import resolve_profile

            config = Config.load()
            if args.profile is not None:
                config = replace(config, ai_profile=args.profile)
            config = resolve_profile(config)
            if args.action == "set":
                if args.stdin:
                    key = sys.stdin.read(4098).removesuffix("\n")
                else:
                    import getpass

                    if not sys.stdin.isatty():
                        raise ValueError(
                            "Use --stdin to supply a key without an interactive terminal"
                        )
                    key = getpass.getpass("API key: ")
                store_key(config, key)
                print("API key saved; select secret-service in target settings/profile to use it")
            else:
                if args.stdin:
                    raise ValueError("--stdin is only valid with credential set")
                delete_key(config)
                print("API key deleted")
        elif args.command == "benchmark":
            from .benchmark import run_benchmark

            changes = {
                name: getattr(args, name)
                for name in ("model", "engine", "device", "compute_type", "model_revision")
                if getattr(args, name) is not None
            }
            config = replace(Config.load(), **changes).validate()
            reference = None
            if args.reference_file is not None:
                with args.reference_file.open("rb") as source:
                    raw = source.read(256 * 1024 + 1)
                if len(raw) > 256 * 1024:
                    raise ValueError("Benchmark reference exceeds 256 KiB")
                reference = raw.decode("utf-8")
            result = run_benchmark(config, args.file, reference=reference, timeout=args.timeout)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "config":
            _configure(args, parser)
        elif args.command == "download":
            from .transcribe import download

            config = Config.load()

            def progress(event):
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
        elif args.command == "transcribe":
            from .file_transcription import export_result, result_text, transcribe_file

            config = Config.load()

            def file_progress(event):
                if args.progress:
                    print(
                        "transcription_progress " + json.dumps(event), file=sys.stderr, flush=True
                    )

            options = {"on_progress": file_progress} if args.progress else {}
            result = transcribe_file(args.file, config, **options)
            if (
                args.save_history
                and config.save_history == "on"
                and config.save_file_history == "on"
            ):
                from .file_history import FileHistory

                file_progress({"stage": "save"})
                FileHistory().add(args.file.name, result, config.file_history_limit)
            if args.output:
                file_progress({"stage": "export"})
                export_result(result, args.output, "json" if args.json else "txt")
            else:
                print(json.dumps(result, ensure_ascii=False) if args.json else result_text(result))
            file_progress({"stage": "complete"})
        elif args.command in ("file-history", "delete-file-history"):
            from .file_history import FileHistory

            history = FileHistory()
            if args.command == "file-history":
                print(json.dumps(history.rows(args.query), ensure_ascii=False, indent=2))
            elif args.id:
                history.delete(args.id)
            else:
                history.clear()
        elif args.command == "assistant-history":
            from .assistant_history import AssistantHistory

            history = AssistantHistory()
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
        elif args.command == "download-speakers":
            from .file_transcription import download_diarization

            print(download_diarization())
        elif args.command in ("models", "delete-model"):
            from .engines import delete_model, model_catalog

            if args.command == "models":
                print(json.dumps(model_catalog(Config.load().engine), indent=2))
            else:
                config = Config.load()
                delete_model(args.model, config.engine, revision=config.model_revision)
                print("Cached model deleted")
        elif args.command == "backup-preview":
            from .operations import backup_preview

            result = backup_preview(args.path, app_map=read_app_map(args.app_map))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command.startswith("dictionary-"):
            from .dictionary import Dictionary

            if args.command == "dictionary-import":
                Dictionary.import_file(args.path, mode=args.mode)
            else:
                Dictionary.export_file(args.path, format=args.format)
            print(str(args.path))
        elif args.command == "doctor":
            from .delivery import doctor

            result = doctor(Config.load())
            print(json.dumps(result, indent=2))
            return int(
                result["session"] == "headless"
                or bool(result["delivery_warning"])
                or not all(result["commands"].values())
            )
        elif args.command == "install-desktop":
            if args.autostart:
                replace(Config.load(), launch_at_startup="on").save()
            print(install_desktop(args.autostart))
        elif args.command == "overlay":
            from .overlay import launch

            return launch()
        elif args.command == "gui":
            from .gui import launch

            return launch()
        elif args.command == "tray":
            from .tray import launch

            return launch()
        elif args.command == "hotkeys":
            from .hotkeys import launch

            return launch()
        elif args.command == "daemon":
            from .service import serve

            serve()
        else:
            from .service import ensure_daemon, request

            if args.command != "shutdown":
                ensure_daemon()
            action = "clear" if args.command == "clear-history" else args.command.replace("-", "_")
            try:
                data = {"id": args.id} if hasattr(args, "id") and args.id else {}
                if args.command == "microphone":
                    action = "select_microphone"
                    data.update(source=args.source, persist=args.persist)
                if hasattr(args, "path"):
                    data["path"] = str(args.path.resolve())
                if args.command == "backup-import" and args.app_map is not None:
                    data["app_map"] = read_app_map(args.app_map)
                if args.command == "accept-edit":
                    data["return_to_target"] = args.return_to_target
                if hasattr(args, "mode"):
                    data.update(mode=args.mode, selected_text=args.selected_text)
                    if args.label:
                        data["label"] = args.label
                    if args.profile is not None:
                        data["profile"] = args.profile
                if args.command == "history":
                    data.update(query=args.query, limit=args.limit, offset=args.offset)
                result = request(action, **data)
            finally:
                if args.command in ("start", "stop", "toggle", "warmup"):
                    from .service import ensure_overlay

                    try:
                        ensure_overlay()
                    # The CLI reports the failure on stderr.
                    except Exception as exc:  # allow: exception-handling  # noqa: BLE001
                        print(f"MouthKeys panel: {exc}", file=sys.stderr)
            print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    # The CLI reports the failure on stderr.
    except Exception as exc:  # allow: exception-handling  # noqa: BLE001
        print(f"MouthKeys: {exc}", file=sys.stderr)
        return 1
