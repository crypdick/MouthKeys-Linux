import numpy as np
import pytest

from mouthkeys.pronunciation import PronunciationStore, acoustic_distance, acoustic_features


def utterance(base=150, seconds=0.8, speed=1, noise=0):
    # Deterministic changing harmonics exercise temporal/spectral matching, not ASR.
    t = np.arange(round(16000 * seconds / speed)) / 16000 * speed
    envelope = np.sin(np.pi * np.minimum(t / seconds, 1)) ** 2
    waveform = (
        0.2
        * envelope
        * (np.sin(2 * np.pi * base * t) + 0.5 * np.sin(2 * np.pi * (base * 4 + 200 * t) * t))
    )
    return (waveform + noise * np.random.default_rng(42).normal(size=len(t))).astype(np.float32)


def test_cepstral_dtw_tracks_gain_noise_and_speaking_rate():
    first = acoustic_features(utterance())
    assert acoustic_distance(first, acoustic_features(utterance() * 0.4)) < 0.001
    close = acoustic_distance(first, acoustic_features(utterance(speed=1.1, noise=0.001)))
    different = acoustic_distance(first, acoustic_features(utterance(base=900)))
    assert close < 0.18
    assert different > close + 0.06


def test_enrollment_needs_two_consistent_takes_and_persists_only_features():
    assert PronunciationStore.enroll("MouthKeys", utterance())["ready"] is False
    assert PronunciationStore.match(utterance()) is None
    assert PronunciationStore.enroll("MouthKeys", utterance(speed=1.03))["ready"] is True
    matched = PronunciationStore.match(utterance(noise=0.0002))
    assert matched["label"] == "MouthKeys"
    assert matched["confidence"] > 0.82
    assert PronunciationStore.correct("mouth keys", utterance()) == "MouthKeys"
    assert (
        PronunciationStore.correct(
            "This long unrelated sentence must not be overwritten", utterance()
        )
        == "This long unrelated sentence must not be overwritten"
    )
    assert PronunciationStore.path().stat().st_mode & 0o777 == 0o600
    assert set(PronunciationStore.load()[0]["samples"][0]) == {"features", "duration"}
    PronunciationStore.delete("mouthkeys")
    assert PronunciationStore.load() == []


def test_competing_near_identical_phrases_are_not_auto_corrected():
    for label in ("MouthKeys", "MouseKeys"):
        PronunciationStore.enroll(label, utterance())
        PronunciationStore.enroll(label, utterance(speed=1.02))
    assert PronunciationStore.match(utterance()) is None
    assert PronunciationStore.correct("mouse keys", utterance()) == "mouse keys"


def test_wrong_silence_long_and_nonfinite_audio_rejected():
    for samples in (np.zeros(16000), np.ones(160001), np.full(16000, np.nan), np.ones(100)):
        with pytest.raises(ValueError):
            PronunciationStore.enroll("phrase", samples)
    PronunciationStore.enroll("phrase", utterance())
    with pytest.raises(ValueError, match="differs"):
        PronunciationStore.enroll("phrase", utterance(base=1800))


def test_corrupt_feature_shapes_are_rejected():
    PronunciationStore.enroll("phrase", utterance())
    profiles = PronunciationStore.load()
    profiles[0]["samples"][0]["features"] = [[float("nan")] * 12]
    with pytest.raises(ValueError):
        PronunciationStore.save(profiles)


def test_real_fixture_wav_enrollment_survives_gain_and_speaking_rate_change():
    import wave
    from pathlib import Path

    path = Path(__file__).parent / "resources" / "dictation_fixture.wav"
    with wave.open(str(path)) as source:
        samples = (
            np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32)
            / 32768
        )
    slower = np.interp(
        np.linspace(0, len(samples) - 1, round(len(samples) * 1.06)),
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)
    PronunciationStore.enroll("fixture phrase", path)
    PronunciationStore.enroll("fixture phrase", slower)
    assert PronunciationStore.match(samples * 0.7)["label"] == "fixture phrase"
    assert PronunciationStore.match(utterance(base=1800)) is None


@pytest.mark.parametrize("label", [None, "", "  ", "x\ny", "x" * 129])
def test_invalid_labels_rejected_before_enrollment(label):
    with pytest.raises(ValueError, match="labels"):
        PronunciationStore.enroll(label, utterance())
    assert not PronunciationStore.path().exists()
