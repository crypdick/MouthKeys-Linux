"""Cepstral features and bounded alignment for local pronunciation learning."""

import wave
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt


class AcousticFeatures(TypedDict):
    duration: float
    features: list[list[float]]


def _samples(audio: object) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    if isinstance(audio, str | Path):
        with wave.open(str(audio), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (
                1,
                2,
                16000,
            ):
                raise ValueError("Pronunciation audio must be mono 16-bit PCM at 16 kHz")
            if source.getnframes() > 160000:
                raise ValueError("Pronunciation phrase exceeds ten seconds")
            samples = (
                np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                / 32768
            )
    else:
        samples = np.asarray(cast("npt.ArrayLike", audio), dtype=np.float32)
    if samples.ndim != 1 or not 4000 <= len(samples) <= 160000 or not np.all(np.isfinite(samples)):
        raise ValueError("Pronunciation phrase must contain 0.25–10 seconds of finite mono audio")
    peak = float(np.max(np.abs(samples)))
    if peak < 0.001 or float(np.std(samples)) < 0.0001:
        raise ValueError("Pronunciation sample contains no audible speech")
    # Trim only low-energy edges. Interior pauses remain part of pronunciation.
    energy = np.array(
        [np.sqrt(np.mean(part * part)) for part in np.array_split(samples, len(samples) // 160)]
    )
    active = np.flatnonzero(energy > max(0.0005, float(energy.max()) * 0.06))
    if not len(active):
        raise ValueError("Pronunciation sample contains no audible speech")
    step = len(samples) / len(energy)
    start = max(0, int((active[0] - 2) * step))
    end = min(len(samples), int((active[-1] + 3) * step))
    samples = samples[start:end]
    if len(samples) < 3200:
        raise ValueError("Pronunciation sample contains too little speech")
    return samples / peak


def acoustic_features(audio: object) -> AcousticFeatures:
    samples = _samples(audio)
    duration = len(samples) / 16000
    emphasized = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    starts = range(0, len(emphasized) - 399, 160)
    frames = np.stack([emphasized[start : start + 400] for start in starts])
    power = np.abs(np.fft.rfft(frames * np.hamming(400), n=512)) ** 2
    mel_edges = np.linspace(2595 * np.log10(1 + 70 / 700), 2595 * np.log10(1 + 7600 / 700), 28)
    frequencies = 700 * (10 ** (mel_edges / 2595) - 1)
    bins = np.floor(513 * frequencies / 16000).astype(int)
    filters = np.zeros((26, 257))
    for index, (left, center, right) in enumerate(zip(bins, bins[1:], bins[2:], strict=False)):
        filters[index, left:center] = np.arange(center - left) / max(1, center - left)
        filters[index, center:right] = np.arange(right - center, 0, -1) / max(1, right - center)
    mel_power = power @ filters.T
    floor = np.maximum(mel_power.max(axis=1, keepdims=True) * 1e-3, 1e-10)
    log_mel = np.log(np.maximum(mel_power, floor))
    dct = np.cos(np.pi / 26 * (np.arange(26) + 0.5)[:, None] * np.arange(1, 13))
    cepstra = log_mel @ dct
    # Per-frame normalization removes gain while retaining spectral articulation.
    cepstra /= np.maximum(np.linalg.norm(cepstra, axis=1, keepdims=True), 1e-8)
    if len(cepstra) > 80:
        indices = np.linspace(0, len(cepstra) - 1, 80)
        cepstra = np.stack(
            [np.interp(indices, np.arange(len(cepstra)), cepstra[:, i]) for i in range(12)], axis=1
        )
    return {"duration": duration, "features": np.round(cepstra, 6).tolist()}


def acoustic_distance(first: AcousticFeatures, second: AcousticFeatures) -> float:
    """Length-normalized DTW cosine distance; constrained to plausible speech rate."""
    ratio = first["duration"] / second["duration"]
    if not 0.5 <= ratio <= 2:
        return 1.0
    left, right = np.asarray(first["features"]), np.asarray(second["features"])
    costs = np.clip(1 - left @ right.T, 0, 2)
    rows, cols = costs.shape
    previous = np.full(cols + 1, np.inf)
    previous[0] = 0
    for i in range(1, rows + 1):
        current = np.full(cols + 1, np.inf)
        indices = np.flatnonzero(np.abs(i / rows - np.arange(1, cols + 1) / cols) <= 0.35) + 1
        if len(indices):
            row = costs[i - 1, indices - 1]
            prefix = np.cumsum(row)
            incoming = np.minimum(previous[indices], previous[indices - 1])
            # Exact left/diagonal/up recurrence, expressed as a prefix minimum.
            current[indices] = prefix + np.minimum.accumulate(incoming + row - prefix)
        previous = current
    return float(previous[-1] / max(rows, cols))
