"""Opt-in real Whisper acceptance using existing local weights, never a download.

Examples after an explicit model download::

    uv run --project linux python linux/tests/smoke_transcription.py --model tiny.en
    uv run --project linux python linux/tests/smoke_transcription.py \
        --model /absolute/path/to/complete/snapshot --word-times --stream \
        --report /tmp/whisper-acceptance.json

Only the redistributable repository fixture is decoded. No microphone, desktop,
clipboard, or installed-app state is used. Internet sockets are denied in this
process. A short greeting is a regression check, not a general accuracy benchmark.
"""

import argparse
import hashlib
import json
import os
import tempfile
import time
import wave
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from mouthkeys.config import Config
from mouthkeys.engines import resolve_whisper
from mouthkeys.native_worker import isolate_network
from mouthkeys.transcribe import Transcriber
from mouthkeys.word_alignment import align_words


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny.en", help="Existing local model name or snapshot")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--word-times", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--file-workflow", action="store_true")
    parser.add_argument(
        "--force-align",
        action="store_true",
        help="Also verify real cached-Whisper DTW fallback on exact recognized fixture text",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    isolate_network()
    model = Path(resolve_whisper(Config(model=args.model))).resolve()
    fixture = Path(__file__).resolve().parent / "resources/dictation_fixture.wav"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="mouthkeys-whisper-acceptance-") as directory:
        for kind in ("CONFIG", "STATE", "CACHE", "DATA", "RUNTIME"):
            path = Path(directory) / kind.lower()
            path.mkdir(mode=0o700)
            os.environ[f"XDG_{kind}_HOME" if kind != "RUNTIME" else "XDG_RUNTIME_DIR"] = str(path)
        config = Config(
            model=str(model),
            compute_type=args.compute_type,
            pronunciation_matching="on" if args.word_times or args.force_align else "off",
        )
        transcriber = Transcriber()
        text = transcriber.transcribe(fixture, config)
        assert "hello" in text.casefold(), "Expected greeting was not recognized"
        with wave.open(str(fixture)) as audio:
            rate = audio.getframerate()
            frames = audio.readframes(audio.getnframes())
            duration = len(frames) / (audio.getsampwidth() * audio.getnchannels() * rate)
        words = align_words(text, transcriber.words, duration=duration) if args.word_times else []
        if args.word_times:
            assert words, "No real word timing returned"
        forced_words = []
        if args.force_align:
            from faster_whisper.audio import decode_audio

            from mouthkeys.forced_alignment import WhisperForcedAligner

            aligner = WhisperForcedAligner()
            try:
                aligned, warning = aligner.align(
                    decode_audio(str(fixture), sampling_rate=16000),
                    [SimpleNamespace(start=0, end=duration, text=text)],
                    replace(config, pronunciation_aligner_model=str(model)),
                )
                assert not warning, warning
                forced_words = [vars(word) for word in aligned[0].words]
                assert align_words(text, forced_words, duration=duration), (
                    "No exact forced alignment"
                )
                assert all(
                    word["alignment_source"].startswith("whisper-forced:") for word in forced_words
                )
            finally:
                aligner.clear()
        committed = 0
        if args.stream:
            recording = Path(directory) / "repeated-fixture.wav"
            # Public fixture with silence between three utterances; never record a device.
            with wave.open(str(recording), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(rate)
                audio.writeframes((frames + b"\0\0" * (rate // 2)) * 3)
            transcriber.preview(recording, config)
            committed = len(transcriber.committed)
            assert committed, "Preview did not exercise committed-segment continuation"
            final = transcriber.transcribe(recording, config)
            assert final.casefold().count("hello") >= 2, "Repeated utterances were lost"
            if args.word_times:
                align_words(final, transcriber.words, duration=3 * (duration + 0.5))
        stages = []
        if args.file_workflow:
            from mouthkeys.file_history import FileHistory
            from mouthkeys.file_transcription import export_result, transcribe_file

            result = transcribe_file(
                fixture, config, transcriber=transcriber, on_progress=stages.append
            )
            assert "hello" in result["text"].casefold()
            assert result["segments"] and result["duration"] > 0
            assert [event["stage"] for event in stages] == ["decode", "transcribe", "finalize"]
            item_id = FileHistory().add(fixture.name, result)
            saved = next(row for row in FileHistory().rows() if row["id"] == item_id)
            assert saved["result"] == result
            for kind in ("json", "txt"):
                target = Path(directory) / f"export.{kind}"
                export_result(result, target, kind)
                assert target.stat().st_mode & 0o777 == 0o600
                assert "hello" in target.read_text().casefold()
        report = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "result": "passed",
            "engine": "whisper",
            "model": model.parent.parent.name if model.parent.name == "snapshots" else model.name,
            "revision": model.name if model.parent.name == "snapshots" else "local",
            "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "expected_greeting": True,
            "actual_word_count": len(words),
            "forced_alignment_word_count": len(forced_words),
            "committed_stream_segments": committed,
            "file_workflow_stages": [event["stage"] for event in stages],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "versions": {name: version(name) for name in ("faster-whisper", "ctranslate2")},
            "network": "AF_INET/AF_INET6 denied by seccomp; existing weights only",
            "scope": (
                "Pretrained fixture recognition/timing/optional committed-preview regression; "
                "not general accuracy, latency or language coverage"
            ),
        }
        if args.report:
            from mouthkeys.operations import write_private

            write_private(args.report, json.dumps(report, indent=2).encode())
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
