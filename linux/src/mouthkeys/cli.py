import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .config import Config


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
        config = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        directory = config / "autostart"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "mouthkeys.desktop").write_text(entry.replace(" gui\n", " daemon\n"))
    return str(applications / "mouthkeys.desktop")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MouthKeys — local Linux dictation")
    parser.add_argument("--version", action="version", version=f"MouthKeys {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for action in (
        "gui",
        "overlay",
        "daemon",
        "start",
        "stop",
        "toggle",
        "cancel",
        "status",
        "history",
        "warmup",
        "doctor",
        "shutdown",
    ):
        commands.add_parser(action)
    for action in ("copy", "paste", "reprocess"):
        commands.add_parser(action).add_argument(
            "--id", help="History entry UUID (default: latest)"
        )
    download_parser = commands.add_parser("download", help="Explicitly download a local model")
    download_parser.add_argument("model", nargs="?")
    transcribe_parser = commands.add_parser("transcribe", help="Transcribe a file to stdout only")
    transcribe_parser.add_argument("file", type=Path)
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
                config[args.key] = (
                    int(args.value)
                    if args.key in ("max_seconds", "audio_budget_mb")
                    else args.value
                )
                Config(**config).save()
            elif args.value:
                parser.error("config requires KEY VALUE")
            print(json.dumps(config, indent=2))
        elif args.command == "download":
            from .transcribe import download

            print(download(args.model or Config.load().model))
        elif args.command == "transcribe":
            from .transcribe import Transcriber

            print(Transcriber().transcribe(args.file, Config.load()))
        elif args.command == "doctor":
            from .delivery import doctor

            result = doctor(Config.load())
            print(json.dumps(result, indent=2))
            return int(result["session"] == "headless" or not all(result["commands"].values()))
        elif args.command == "install-desktop":
            print(install_desktop(args.autostart))
        elif args.command == "overlay":
            from .overlay import launch

            return launch()
        elif args.command == "gui":
            from .gui import launch

            return launch()
        elif args.command == "daemon":
            from .service import serve

            serve()
        else:
            from .service import ensure_daemon, request

            if args.command != "shutdown":
                ensure_daemon()
            action = "clear" if args.command == "clear-history" else args.command
            try:
                result = request(action, **({"id": args.id} if hasattr(args, "id") else {}))
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
