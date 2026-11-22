import io
import json
import wave
from types import SimpleNamespace

import pytest

from mouthkeys import capture_readiness, capture_recorder
from mouthkeys.config import Config


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

    monkeypatch.setattr(capture_readiness.subprocess, "run", run)
    assert not capture_readiness.bluetooth_capture_ready(source)
    nodes.append(target)
    target["info"]["state"] = "suspended"
    assert not capture_readiness.bluetooth_capture_ready(source)
    target["info"]["state"] = "running"
    assert capture_readiness.bluetooth_capture_ready(source)
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

    monkeypatch.setattr(capture_readiness.subprocess, "run", run)
    assert not capture_readiness.bluetooth_capture_ready("default")


def test_non_bluetooth_input_needs_no_graph_probe(monkeypatch):
    monkeypatch.setattr(
        capture_readiness.subprocess, "run", lambda *a, **kw: pytest.fail("Graph probe")
    )
    assert capture_readiness.bluetooth_capture_ready("alsa_input.test")


def test_graph_failure_does_not_report_bluetooth_ready(monkeypatch):
    def run(*a, **kw):
        raise capture_readiness.subprocess.TimeoutExpired("pw-dump", 0.5)

    monkeypatch.setattr(capture_readiness.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Bluetooth"):
        capture_readiness.bluetooth_capture_ready("bluez_input.test")


@pytest.mark.parametrize("becomes_ready", [True, False])
def test_recorder_waits_for_bluetooth_even_when_virtual_samples_exist(
    monkeypatch, tmp_path, becomes_ready
):

    path = tmp_path / "capture.wav"
    elapsed = 0
    processes = []

    class Process:
        def __init__(self, args, **kwargs):
            self.stdin = io.BytesIO()
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

    monkeypatch.setattr(capture_readiness.subprocess, "Popen", Process)
    monkeypatch.setattr(capture_recorder.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(capture_recorder.time, "sleep", advance)
    monkeypatch.setattr(
        capture_recorder, "bluetooth_capture_ready", lambda _: becomes_ready and elapsed >= 1.5
    )
    if becomes_ready:
        recorder = capture_recorder.Recorder(Config(audio_source="bluez_input.test"), path)
        assert elapsed >= 1.5
        recorder.stop()
    else:
        with pytest.raises(RuntimeError, match="become ready"):
            capture_recorder.Recorder(Config(audio_source="bluez_input.test"), path)
        assert elapsed <= 5.1
    assert processes[0].stdin.closed


def test_native_bluetooth_does_not_require_local_pipewire_graph(monkeypatch):
    def run(args, **kwargs):
        assert args == ["pactl", "-f", "json", "list", "sources"]
        return SimpleNamespace(stdout=json.dumps([{"name": "bluez_input.test", "properties": {}}]))

    monkeypatch.setattr(capture_readiness.subprocess, "run", run)
    assert capture_readiness.bluetooth_capture_ready("bluez_input.test")


def test_virtual_bluetooth_missing_graph_tool_fails_clearly(monkeypatch):
    def run(args, **kwargs):
        if args[0] == "pactl":
            return SimpleNamespace(
                stdout=json.dumps(
                    [{"name": "bluez_input.test", "properties": {"bluez5.loopback": True}}]
                )
            )
        raise FileNotFoundError("pw-dump")

    monkeypatch.setattr(capture_readiness.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Bluetooth"):
        capture_readiness.bluetooth_capture_ready("bluez_input.test")


def test_malformed_bluetooth_snapshot_gives_readiness_error(monkeypatch):
    monkeypatch.setattr(
        capture_readiness.subprocess,
        "run",
        lambda *_args, **_opts: SimpleNamespace(stdout='{"unexpected": 1}'),
    )
    with pytest.raises(RuntimeError, match="Cannot verify Bluetooth microphone"):
        capture_readiness.bluetooth_capture_ready("bluez_input.test")


def test_default_lookup_failure_preserves_normal_capture_readiness(monkeypatch):
    def run(args, **_options):
        assert args == ["pactl", "get-default-source"]
        raise FileNotFoundError("pactl")

    monkeypatch.setattr(capture_readiness.subprocess, "run", run)
    assert capture_readiness.bluetooth_capture_ready("default")


@pytest.mark.parametrize(
    ("graph", "ready"),
    [
        (None, False),
        ([None, {"info": None}, node("unrelated")], False),
        ([node("bluez_input.test")], True),
        ([node("bluez_input.test", **{"bluez5.loopback": True})], False),
    ],
)
def test_missing_or_changed_bluetooth_nodes_are_handled(monkeypatch, graph, ready):
    def run(args, **_options):
        if args[0] == "pactl":
            payload = (
                []
                if graph is None
                else [{"name": "bluez_input.test", "properties": {"bluez5.loopback": True}}]
            )
        else:
            assert graph is not None
            payload = graph
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(capture_readiness.subprocess, "run", run)
    assert capture_readiness.bluetooth_capture_ready("bluez_input.test") is ready


def test_invalid_properties_do_not_mark_physical_capture_ready(monkeypatch):
    def run(_args, **_options):
        return SimpleNamespace(stdout='[{"name":"bluez_input.test","properties":[]}]')

    monkeypatch.setattr(capture_readiness.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Cannot verify Bluetooth microphone"):
        capture_readiness.bluetooth_capture_ready("bluez_input.test")
