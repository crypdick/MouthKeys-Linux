import struct
import wave

import numpy as np
import pytest

from mouthkeys.audio import check_audio, live_meter, read_audio_from, read_live_audio


def test_unfinished_wav_preview_reads_bounded_tail(tmp_path):
    path = tmp_path / "live.wav"
    samples = np.arange(25 * 16000, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    # The same unfinished RIFF/data lengths FFmpeg writes while recording.
    with path.open("r+b") as file:
        file.seek(4)
        file.write(struct.pack("<I", 0xFFFFFFFF))
        file.seek(40)
        file.write(struct.pack("<I", 0xFFFFFFFF))
    audio = read_live_audio(path)
    assert len(audio) == 20 * 16000
    np.testing.assert_array_equal(audio, samples[-20 * 16000 :].astype(np.float32) / 32768)
    meter = live_meter(path)
    assert meter["frame"] == len(samples)
    assert 0 <= meter["level"] <= 1


def test_meter_uses_fresh_20ms_tail_not_previous_speech(tmp_path):
    path = tmp_path / "live.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.full(16000, 12000, dtype=np.int16).tobytes())
        wav.writeframes(bytes(320 * 2))
    assert live_meter(path) == {"frame": 16320, "level": 0.0}


def test_meter_ignores_trailing_wav_metadata(tmp_path):
    path = tmp_path / "capture.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(640))
    junk = b"JUNK" + struct.pack("<I", 1024) + bytes(1024)
    with path.open("r+b") as file:
        file.seek(4)
        length = struct.unpack("<I", file.read(4))[0]
        file.seek(4)
        file.write(struct.pack("<I", length + len(junk)))
        file.seek(0, 2)
        file.write(junk)
    assert live_meter(path) == {"frame": 320, "level": 0.0}


def test_empty_capture_cannot_be_transcribed_and_reports_zero_meter(tmp_path):
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    assert live_meter(path) == {"frame": 0, "level": 0.0}
    with pytest.raises(ValueError, match="too short"):
        check_audio(path)
    with pytest.raises(ValueError, match="shrank"):
        read_audio_from(path, 1)


def test_live_readers_reject_non_capture_wav_format(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(640))
    with pytest.raises(ValueError, match="Expected mono 16 kHz PCM"):
        read_audio_from(path)
