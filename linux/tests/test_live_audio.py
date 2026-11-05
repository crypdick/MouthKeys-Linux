import struct
import wave

import numpy as np

from mouthkeys.audio import live_meter, read_live_audio


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
