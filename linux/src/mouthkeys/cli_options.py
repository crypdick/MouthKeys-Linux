"""Argument grammar for the installed CLI."""

import argparse
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .config import Config


# Keep the declarative command grammar together. Execution belongs in cli.py.
def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="MouthKeys — local Linux dictation")
    parser.add_argument("--version", action="version", version=f"MouthKeys {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    microphone = commands.add_parser(
        "microphone", help="Select microphone, including during capture"
    )
    microphone.add_argument("source", help="PulseAudio source name, or default")
    microphone.add_argument(
        "--persist", action="store_true", help="Also save source for future sessions"
    )
    credential = commands.add_parser("credential", help="Store/delete an AI key in desktop keyring")
    credential.add_argument("action", choices=["set", "delete"])
    credential.add_argument("--profile", help="Use endpoint from this saved profile UUID")
    credential.add_argument(
        "--stdin", action="store_true", help="Read key from stdin instead of hidden prompt"
    )
    for action in (
        "gui",
        "tray",
        "hotkeys",
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
        if action == "tray":
            command.add_argument("--service-fd", type=int, help=argparse.SUPPRESS)
        if action in ("start", "toggle"):
            command.add_argument("--profile", help="Prompt profile UUID, default, or off")
            command.add_argument(
                "--mode",
                choices=["dictation", "command", "edit", "training", "pronunciation"],
                default="dictation",
            )
            command.add_argument("--label", default="")
            command.add_argument("--selected-text", default="")
        if action == "history":
            command.add_argument("--query", default="")
            command.add_argument("--limit", type=int, default=100)
            command.add_argument("--offset", type=int, default=0)
    for action in ("copy", "paste", "delete"):
        commands.add_parser(action).add_argument(
            "--id", required=action == "delete", help="History entry UUID (required for delete)"
        )
    accept = commands.add_parser(
        "accept-edit", help="Replace the unchanged captured Edit selection"
    )
    accept.add_argument("--id", required=True, help="Edit recording UUID shown in status")
    accept.add_argument(
        "--return-to-target",
        action="store_true",
        help="Explicitly return focus to the captured field before replacement",
    )
    download_parser = commands.add_parser("download", help="Explicitly download a local model")
    download_parser.add_argument("model", nargs="?")
    transcribe_parser = commands.add_parser("transcribe", help="Transcribe a file to stdout only")
    transcribe_parser.add_argument("file", type=Path)
    transcribe_parser.add_argument("--json", action="store_true")
    transcribe_parser.add_argument("--output", type=Path)
    transcribe_parser.add_argument("--save-history", action="store_true")
    transcribe_parser.add_argument(
        "--progress", action="store_true", help="Write structured stage progress to stderr"
    )
    benchmark = commands.add_parser(
        "benchmark", help="Measure an already-installed model offline without desktop delivery"
    )
    benchmark.add_argument("file", type=Path, help="Existing audio fixture (at most ten minutes)")
    benchmark.add_argument("--reference-file", type=Path, help="Exact UTF-8 reference for WER/CER")
    benchmark.add_argument("--timeout", type=float, default=600)
    benchmark.add_argument("--model", help="Override model without changing saved configuration")
    benchmark.add_argument("--engine", choices=["whisper", "onnx", "transformers", "nemo"])
    benchmark.add_argument("--device", choices=["cpu", "cuda", "auto"])
    benchmark.add_argument("--compute-type")
    benchmark.add_argument("--revision", dest="model_revision")
    files = commands.add_parser("file-history")
    files.add_argument("--query", default="")
    files = commands.add_parser("delete-file-history")
    files.add_argument("--id")
    files.add_argument("--yes", action="store_true", required=True)
    chats = commands.add_parser("assistant-history", help="Manage saved Command/Edit chats")
    chats.add_argument("--mode", choices=["command", "edit"])
    chats.add_argument("--id")
    chats.add_argument("--new", action="store_true")
    chats.add_argument("--select", action="store_true")
    chats.add_argument("--rename")
    chats.add_argument("--delete", action="store_true")
    chats.add_argument("--yes", action="store_true")
    commands.add_parser("download-speakers", help="Explicitly download diarization models")
    commands.add_parser("models", help="List locally known models and cache status")
    commands.add_parser("delete-model").add_argument("model")
    for action in ("backup-export", "backup-import", "backup-preview"):
        command = commands.add_parser(action)
        command.add_argument("path", type=Path)
        if action == "backup-import":
            command.add_argument("--yes", action="store_true", required=True)
        if action in ("backup-import", "backup-preview"):
            command.add_argument(
                "--app-map", type=Path, help="JSON object mapping macOS app IDs to Linux identities"
            )
    commands.add_parser("delete-audio").add_argument("--yes", action="store_true", required=True)
    for action in ("dictionary-import", "dictionary-export"):
        command = commands.add_parser(action)
        command.add_argument("path", type=Path)
        if action == "dictionary-import":
            command.add_argument("--yes", action="store_true", required=True)
            command.add_argument("--mode", choices=["merge", "replace"], default="replace")
        else:
            command.add_argument("--format", choices=["native", "upstream"], default="native")
    config_parser = commands.add_parser("config", help="Show config, or set one value")
    config_parser.add_argument("key", nargs="?", choices=list(asdict(Config())))
    config_parser.add_argument("value", nargs="?")
    commands.add_parser("clear-history").add_argument("--yes", action="store_true", required=True)
    startup = commands.add_parser("install-desktop").add_mutually_exclusive_group()
    startup.add_argument("--autostart", action="store_true", help="Enable startup at login")
    startup.add_argument(
        "--sync-autostart", action="store_true", help="Apply the saved login startup setting"
    )
    return parser
