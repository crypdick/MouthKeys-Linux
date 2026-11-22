"""Explicit speaker model setup and speaker attribution for local audio files."""

import math
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .config import cache_dir, private_dir
from .file_history import FileSegment, SpeakerTurn

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = "sherpa-onnx-pyannote-segmentation-3-0"
EMBEDDING = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
MAX_MODEL_BYTES = 512 * 1024 * 1024
SAMPLE_RATE = 16000


def speaker_paths() -> tuple[Path, Path]:
    root = cache_dir() / "models" / "speakers"
    return root / "segmentation.onnx", root / "embedding.onnx"


def fetch_model(url: str, destination: Path, max_bytes: int = MAX_MODEL_BYTES) -> None:
    """Download into a staging directory with a bounded size and private permissions."""
    if not url.startswith("https://"):
        raise ValueError("Speaker model downloads require HTTPS")
    with tempfile.NamedTemporaryFile(dir=destination.parent) as stream:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - HTTPS only
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Speaker model download exceeds size limit")
                stream.write(chunk)
        stream.flush()
        shutil.copyfile(stream.name, destination)
    destination.chmod(0o600)


def _unpack_segmentation(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:bz2") as package:
        try:
            member = package.getmember(f"{SEGMENTATION}/model.onnx")
        except KeyError:
            raise ValueError("Segmentation model missing from archive") from None
        if not member.isfile() or not 0 < member.size <= MAX_MODEL_BYTES:
            raise ValueError("Invalid segmentation model archive")
        # Only read the expected member. Never extract archive paths or links.
        source = package.extractfile(member)
        if source is None:
            raise ValueError("Cannot read segmentation model archive")
        with source, destination.open("wb") as out:
            shutil.copyfileobj(source, out)


def download_diarization() -> str:
    """Explicitly fetch public sherpa-onnx segmentation and embedding artifacts."""
    segmentation, embedding = speaker_paths()
    private_dir(segmentation.parent)
    with tempfile.TemporaryDirectory(dir=segmentation.parent) as directory:
        root = Path(directory)
        archive = root / "segmentation.tar.bz2"
        model = root / "segmentation.onnx"
        other = root / "embedding.onnx"
        fetch_model(f"{RELEASES}/speaker-segmentation-models/{SEGMENTATION}.tar.bz2", archive)
        _unpack_segmentation(archive, model)
        fetch_model(f"{RELEASES}/speaker-recongition-models/{EMBEDDING}", other)
        model.chmod(0o600)
        model.replace(segmentation)
        other.replace(embedding)
    return str(segmentation.parent)


def diarize(samples: object, speaker_count: object = 0) -> list[SpeakerTurn]:
    if type(speaker_count) is not int or not 0 <= speaker_count <= 20:
        raise ValueError("Speaker count must be between 0 (automatic) and 20")
    segmentation, embedding = speaker_paths()
    if not segmentation.is_file() or not embedding.is_file():
        raise RuntimeError("Speaker models missing; run mouthkeys download-speakers first")
    try:
        import sherpa_onnx  # noqa: PLC0415 - optional inference backend
    except ImportError:
        raise RuntimeError("Speaker labels require the diarization optional extra") from None
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation)
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding)),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=speaker_count or -1, threshold=0.5
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError("Invalid or incomplete speaker model files")
    model = sherpa_onnx.OfflineSpeakerDiarization(config)
    if model.sample_rate != SAMPLE_RATE:
        raise ValueError("Speaker model must accept 16 kHz audio")
    return [
        {"start": float(turn.start), "end": float(turn.end), "speaker": int(turn.speaker)}
        for turn in model.process(samples).sort_by_start_time()
    ]


def clip_turns(turns: list[SpeakerTurn], duration: float) -> list[SpeakerTurn]:
    valid: list[SpeakerTurn] = []
    for turn in turns:
        if not (math.isfinite(turn["start"]) and math.isfinite(turn["end"])):
            continue
        start, end = max(0.0, turn["start"]), min(duration, turn["end"])
        if start < end:
            valid.append({"start": start, "end": end, "speaker": turn["speaker"]})
    return valid


def turn_chunks(turns: list[SpeakerTurn], duration: float) -> list[tuple[float, float, int | None]]:
    """Decode uncovered intervals as well as speaker turns so no words disappear."""
    chunks: list[tuple[float, float, int | None]] = []
    covered = 0.0
    for turn in sorted(turns, key=lambda item: item["start"]):
        start, end = turn["start"], turn["end"]
        if start > covered:
            chunks.append((covered, start, None))
        chunks.append((start, end, turn["speaker"]))
        covered = max(covered, end)
    if covered < duration:
        chunks.append((covered, duration, None))
    return chunks


def label_segments(segments: list[FileSegment], turns: list[SpeakerTurn]) -> None:
    """Assign each segment to its longest overlap; leave unmatched segments unlabelled."""
    for segment in segments:
        longest = 0.0
        speaker = None
        for turn in turns:
            overlap = min(segment["end"], turn["end"]) - max(segment["start"], turn["start"])
            if overlap > longest:
                longest, speaker = overlap, turn["speaker"]
        segment["speaker"] = speaker
