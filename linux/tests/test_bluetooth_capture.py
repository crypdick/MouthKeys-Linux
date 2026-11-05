import json
from types import SimpleNamespace

import pytest

from mouthkeys import audio


def node(name, **props):
    return {"info": {"state": "running", "props": {"node.name": name, **props}}}


def test_virtual_bluetooth_input_waits_for_physical_capture(monkeypatch):
    source = "bluez_input.test"
    virtual = node(source, **{"device.id": 12, "bluez5.loopback": True})
    target = node("internal", **{"device.id": 12, "bluez5.loopback-target": True})
    unrelated = node("another", **{"device.id": 99, "bluez5.loopback-target": True})
    nodes = [virtual, unrelated]
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[0] == "pactl":
            return SimpleNamespace(
                stdout=json.dumps([{"name": source, "properties": {"bluez5.loopback": "true"}}])
            )
        return SimpleNamespace(stdout=json.dumps(nodes))

    monkeypatch.setattr(audio.subprocess, "run", run)
    assert not audio.bluetooth_capture_ready(source)
    nodes.append(target)
    target["info"]["state"] = "suspended"
    assert not audio.bluetooth_capture_ready(source)
    target["info"]["state"] = "running"
    assert audio.bluetooth_capture_ready(source)
    assert ["pw-dump", "Node"] in calls


def test_default_bluetooth_source_is_resolved(monkeypatch):
    def run(args, **kwargs):
        if args == ["pactl", "get-default-source"]:
            return SimpleNamespace(stdout="bluez_input.test\n")
        if args[0] == "pactl":
            return SimpleNamespace(
                stdout=json.dumps(
                    [{"name": "bluez_input.test", "properties": {"bluez5.loopback": True}}]
                )
            )
        return SimpleNamespace(
            stdout=json.dumps(
                [node("bluez_input.test", **{"device.id": 12, "bluez5.loopback": True})]
            )
        )

    monkeypatch.setattr(audio.subprocess, "run", run)
    assert not audio.bluetooth_capture_ready("default")


def test_non_bluetooth_input_needs_no_graph_probe(monkeypatch):
    monkeypatch.setattr(audio.subprocess, "run", lambda *a, **kw: pytest.fail("Graph probe"))
    assert audio.bluetooth_capture_ready("alsa_input.test")


def test_graph_failure_does_not_report_bluetooth_ready(monkeypatch):
    def run(*a, **kw):
        raise audio.subprocess.TimeoutExpired("pw-dump", 0.5)

    monkeypatch.setattr(audio.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Bluetooth"):
        audio.bluetooth_capture_ready("bluez_input.test")


@pytest.mark.parametrize("becomes_ready", [True, False])
def test_recorder_waits_for_bluetooth_even_when_virtual_samples_exist(
    monkeypatch, tmp_path, becomes_ready
):
    import os
    import wave

    from mouthkeys.config import Config

    path = tmp_path / "capture.wav"
    elapsed = 0
    processes = []

    class Process:
        def __init__(self, args, **kwargs):
            self.stdin = open(os.devnull, "wb")
            processes.append(self)
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(bytes(640))

        def poll(self):
            return None

        def wait(self, timeout):
            assert self.stdin.closed
            return 0

    def advance(seconds):
        nonlocal elapsed
        elapsed += seconds

    monkeypatch.setattr(audio.subprocess, "Popen", Process)
    monkeypatch.setattr(audio.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(audio.time, "sleep", advance)
    monkeypatch.setattr(
        audio, "bluetooth_capture_ready", lambda _: becomes_ready and elapsed >= 1.5
    )
    if becomes_ready:
        recorder = audio.Recorder(Config(audio_source="bluez_input.test"), path)
        assert elapsed >= 1.5
        recorder.stop()
    else:
        with pytest.raises(RuntimeError, match="become ready"):
            audio.Recorder(Config(audio_source="bluez_input.test"), path)
        assert elapsed <= 5.1
    assert processes[0].stdin.closed


def test_native_bluetooth_does_not_require_local_pipewire_graph(monkeypatch):
    def run(args, **kwargs):
        assert args == ["pactl", "-f", "json", "list", "sources"]
        return SimpleNamespace(stdout=json.dumps([{"name": "bluez_input.test", "properties": {}}]))

    monkeypatch.setattr(audio.subprocess, "run", run)
    assert audio.bluetooth_capture_ready("bluez_input.test")


def test_virtual_bluetooth_missing_graph_tool_fails_clearly(monkeypatch):
    def run(args, **kwargs):
        if args[0] == "pactl":
            return SimpleNamespace(
                stdout=json.dumps(
                    [{"name": "bluez_input.test", "properties": {"bluez5.loopback": True}}]
                )
            )
        raise FileNotFoundError("pw-dump")

    monkeypatch.setattr(audio.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Bluetooth"):
        audio.bluetooth_capture_ready("bluez_input.test")
