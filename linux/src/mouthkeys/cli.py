import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import __version__
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="MouthKeys — local Linux dictation")
    parser.add_argument("--version", action="version", version=f"MouthKeys {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for action in (
        "gui",
        "tray",
        "overlay",
        "daemon",
        "start",
        "stop",
        "toggle",
        "cancel",
        "status",
        "history",
        "stats",
        "audio-usage",
        "warmup",
        "doctor",
        "shutdown",
    ):
        command = commands.add_parser(action)
        if action in ("start", "toggle"):
            command.add_argument(
                "--mode", choices=["dictation", "command", "edit", "training"], default="dictation"
            )
            command.add_argument("--selected-text", default="")
        if action == "history":
            command.add_argument("--query", default="")
            command.add_argument("--limit", type=int, default=100)
            command.add_argument("--offset", type=int, default=0)
    for action in ("copy", "paste", "reprocess", "delete"):
        commands.add_parser(action).add_argument(
            "--id", required=action == "delete", help="History entry UUID (required for delete)"
        )
    download_parser = commands.add_parser("download", help="Explicitly download a local model")
    download_parser.add_argument("model", nargs="?")
    transcribe_parser = commands.add_parser("transcribe", help="Transcribe a file to stdout only")
    transcribe_parser.add_argument("file", type=Path)
    transcribe_parser.add_argument("--json", action="store_true")
    transcribe_parser.add_argument("--output", type=Path)
    commands.add_parser("download-speakers", help="Explicitly download diarization models")
    commands.add_parser("models", help="List locally known models and cache status")
    commands.add_parser("delete-model").add_argument("model")
    for action in ("backup-export", "backup-import", "export-audio"):
        command = commands.add_parser(action)
        command.add_argument("path", type=Path)
        if action == "backup-import":
            command.add_argument("--yes", action="store_true", required=True)
        if action == "export-audio":
            command.add_argument("--id")
    commands.add_parser("delete-audio").add_argument("--yes", action="store_true", required=True)
    for action in ("dictionary-import", "dictionary-export"):
        command = commands.add_parser(action)
        command.add_argument("path", type=Path)
        if action == "dictionary-import":
            command.add_argument("--yes", action="store_true", required=True)
    config_parser = commands.add_parser("config", help="Show config, or set one value")
    config_parser.add_argument("key", nargs="?", choices=list(asdict(Config())))
    config_parser.add_argument("value", nargs="?")
    commands.add_parser("clear-history").add_argument("--yes", action="store_true", required=True)
    commands.add_parser("install-desktop").add_argument("--autostart", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            config = asdict(Config.load())
            if args.key:
                if args.value is None:
                    parser.error("config KEY requires VALUE")
                config[args.key] = int(args.value) if args.key in RANGES else args.value
                Config(**config).save()
                if args.key in ("launch_at_startup", "show_window_at_login", "tray"):
                    apply_startup(Config.load())
            elif args.value:
                parser.error("config requires KEY VALUE")
            print(json.dumps(config, indent=2))
        elif args.command == "download":
            from .transcribe import download

            config = Config.load()
            print(download(args.model or config.model, config.engine, config.compute_type))
        elif args.command == "transcribe":
            from .file_transcription import export_result, result_text, transcribe_file

            result = transcribe_file(args.file, Config.load())
            if args.output:
                export_result(result, args.output, "json" if args.json else "txt")
            else:
                print(json.dumps(result, ensure_ascii=False) if args.json else result_text(result))
        elif args.command == "download-speakers":
            from .file_transcription import download_diarization

            print(download_diarization())
        elif args.command in ("models", "delete-model"):
            from .engines import delete_model, model_catalog

            if args.command == "models":
                print(json.dumps(model_catalog(Config.load().engine), indent=2))
            else:
                delete_model(args.model, Config.load().engine)
                print("Cached model deleted")
        elif args.command.startswith("dictionary-"):
            from .dictionary import Dictionary

            if args.command == "dictionary-import":
                Dictionary.import_file(args.path)
            else:
                Dictionary.export_file(args.path)
            print(str(args.path))
        elif args.command == "doctor":
            from .delivery import doctor

            result = doctor(Config.load())
            print(json.dumps(result, indent=2))
            return int(result["session"] == "headless" or not all(result["commands"].values()))
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
                if hasattr(args, "path"):
                    data["path"] = str(args.path.resolve())
                if hasattr(args, "mode"):
                    data.update(mode=args.mode, selected_text=args.selected_text)
                if args.command == "history":
                    data.update(query=args.query, limit=args.limit, offset=args.offset)
                result = request(action, **data)
            finally:
                if args.command in ("start", "stop", "toggle", "reprocess", "warmup"):
                    from .service import ensure_overlay

                    try:
                        ensure_overlay()
                    except Exception as exc:
                        print(f"MouthKeys panel: {exc}", file=sys.stderr)
            print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"MouthKeys: {exc}", file=sys.stderr)
        return 1
