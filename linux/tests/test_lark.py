import json
import runpy
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from mouthkeys import lark, lark_monitor
from mouthkeys.config import Config
from mouthkeys.overlay import Overlay

DESCRIPTOR = bytes([0x06, 0x12, 0xFF, 0x85, 5, 0xB1, 2])
REPLY = bytes([5, 3, 0xBB, 0xDD, 0x1F, 0, 4, 1, 0, 87, 99])


def test_protocol_exact_status_frame_and_linked_percent_validation():
    assert len(lark.HEARTBEAT) == 64
    assert lark.HEARTBEAT[:8] == bytes([5, 3, 0xAA, 0xDD, 0x1F, 0, 0, 0xEF])
    value = lark.parse_reply(REPLY)
    assert value["mic1"] == {"linked": True, "percent": 87}
    assert value["mic2"] == {"linked": False, "percent": None}
    assert lark.parse_reply(REPLY[:9] + bytes([255, 99]))["mic1"]["percent"] is None
    with pytest.raises(ValueError, match="short"):
        lark.parse_reply(REPLY[:9])


def test_hid_descriptor_only_accepts_vendor_status_feature_not_keyboard():
    assert lark.carries_status_page(DESCRIPTOR)
    assert not lark.carries_status_page(bytes([0x06, 0x12, 0xFF, 0x85, 5, 0x81, 2]))
    assert not lark.carries_status_page(bytes([0x05, 1, 0x85, 5, 0xB1, 2]))


def test_sysfs_discovery_matches_both_usb_ids_and_status_interface(tmp_path):
    def create(name, product, descriptor):
        root = tmp_path / name / "device"
        root.mkdir(parents=True)
        (root / "uevent").write_text(f"HID_ID=0003:00003547:{product}\n")
        (root / "report_descriptor").write_bytes(descriptor)

    create("hidraw0", "00000407", b"keyboard")
    create("hidraw1", "00000408", DESCRIPTOR)
    create("hidraw2", "00000407", DESCRIPTOR)
    assert lark.find_receiver(tmp_path, tmp_path / "fake-dev") == tmp_path / "fake-dev/hidraw2"


def test_heartbeat_revalidates_open_device_before_any_status_write(monkeypatch):
    calls = []
    closed = []
    monkeypatch.setattr(lark.os, "open", lambda *args: 42)
    monkeypatch.setattr(lark.os, "close", closed.append)

    def ioctl(fd, request, data, _mutable):
        number = request & 255
        calls.append((number, bytes(data)))
        if number == 3:
            data[:] = struct.pack("IHH", 3, lark.VENDOR, lark.PRODUCT)
        elif number == 1:
            data[:] = struct.pack("I", len(DESCRIPTOR))
        elif number == 2:
            data[4 : 4 + len(DESCRIPTOR)] = DESCRIPTOR
        elif number == 7:
            data[: len(REPLY)] = REPLY
            return len(REPLY)
        return 0

    monkeypatch.setattr(lark.fcntl, "ioctl", ioctl)
    assert lark.heartbeat("fake-device")["mic1"]["percent"] == 87
    assert [number for number, _ in calls] == [3, 1, 2, 6, 7]
    assert calls[3][1] == lark.HEARTBEAT
    assert closed == [42]
    calls.clear()

    def wrong_device(fd, request, data, _mutable):
        calls.append(request & 255)
        data[:] = struct.pack("IHH", 3, 1, 2)

    monkeypatch.setattr(lark.fcntl, "ioctl", wrong_device)
    with pytest.raises(ValueError, match="changed"):
        lark.heartbeat("fake-device")
    assert calls == [3]


def test_disabled_monitor_never_touches_device_or_subprocess(monkeypatch):
    monkeypatch.setattr(
        lark_monitor.subprocess, "run", lambda *a, **kw: pytest.fail("Device touched")
    )
    assert lark_monitor.BatteryMonitor().poll(Config(), "default") is None


def test_battery_identifies_receiver_behind_remapped_source_without_name_guessing(monkeypatch):
    sources = [
        {"name": "custom_voice_filter", "owner_module": 17},
        {
            "name": "physical_usb",
            "properties": {"device.vendor.id": 0x3547, "device.product.id": "0407"},
        },
    ]
    modules = [
        {
            "index": 17,
            "name": "module-remap-source",
            "argument": "master=physical_usb source_name=custom_voice_filter",
        }
    ]
    monkeypatch.setattr(
        lark_monitor.subprocess,
        "run",
        lambda args, **kw: SimpleNamespace(
            stdout=json.dumps(modules if args[-1] == "modules" else sources)
        ),
    )
    assert lark_monitor.input_is_receiver("custom_voice_filter")
    sources[1]["properties"]["device.product.id"] = "0408"
    assert not lark_monitor.input_is_receiver("custom_voice_filter")


def test_compact_overlay_has_real_shorter_layout_without_windows():

    app = QApplication.instance() or QApplication([])
    panel = Overlay()
    panel.timer.stop()
    panel.apply_settings(Config(overlay_style="compact"))
    assert (panel.width(), panel.height()) == (424, 106)
    assert panel.buttons["copy"].y() == 60
    assert not panel.isVisible()
    panel.close()
    app.processEvents()


@pytest.mark.parametrize("transition", ["off", "back-to-same-source"])
def test_old_poll_cannot_publish_after_source_or_opt_in_changes(monkeypatch, transition):
    entered, release = threading.Event(), threading.Event()
    percentages = iter([88, 51])

    monkeypatch.setattr(lark_monitor, "input_is_receiver", lambda _: True)

    def probe(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "result": "ok",
                    "mic1": {"linked": True, "percent": next(percentages)},
                    "mic2": {"linked": False, "percent": None},
                }
            )
        )

    monkeypatch.setattr(lark_monitor.subprocess, "run", probe)
    monitor = lark_monitor.BatteryMonitor()
    enabled = Config(mic_battery="on")
    assert monitor.poll(enabled, "usb") is None
    worker = monitor.worker
    try:
        assert entered.wait(1)
        if transition == "off":
            assert monitor.poll(Config(), "usb") is None
        else:
            assert monitor.poll(enabled, "other") is None
        assert monitor.poll(enabled, "usb") is None
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert monitor.poll(enabled, "usb") is None
    monitor.worker.join(2)
    assert not monitor.worker.is_alive()
    assert monitor.poll(enabled, "usb")["mic1"]["percent"] == 51


def test_disabling_during_source_check_prevents_device_helper(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    helper_calls = []

    def identify(_source):
        entered.set()
        assert release.wait(2)
        return True

    def helper(*_args, **_options):
        helper_calls.append(True)
        return SimpleNamespace(stdout='{"result":"no-device"}')

    monkeypatch.setattr(lark_monitor, "input_is_receiver", identify)
    monkeypatch.setattr(lark_monitor.subprocess, "run", helper)
    monitor = lark_monitor.BatteryMonitor()
    monitor.poll(Config(mic_battery="on"), "usb")
    try:
        assert entered.wait(1)
        assert monitor.poll(Config(), "usb") is None
    finally:
        release.set()
        monitor.worker.join(2)
    assert not monitor.worker.is_alive()
    assert helper_calls == []


def test_monitor_rejects_malformed_helper_status(monkeypatch):
    monkeypatch.setattr(lark_monitor, "input_is_receiver", lambda _: True)
    monkeypatch.setattr(
        lark_monitor.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout='{"result":"ok"}')
    )
    monitor = lark_monitor.BatteryMonitor()
    cfg = Config(mic_battery="on")
    monitor.poll(cfg, "usb")
    monitor.worker.join(2)
    assert not monitor.worker.is_alive()
    assert monitor.poll(cfg, "usb")["result"] == "unavailable"


@pytest.mark.parametrize(
    ("descriptor", "expected"),
    [
        (b"\xfe", False),
        (b"\x06\x12", False),
        (b"\xb4" + DESCRIPTOR, False),
        (b"\xfe\x02\x87XY" + DESCRIPTOR, True),
        # Preserve the vendor page and report ID across a nested collection.
        (DESCRIPTOR[:-2] + b"\xa4\x05\x01\x85\x01\xb4" + DESCRIPTOR[-2:], True),
    ],
)
def test_descriptor_checks_complete_items_and_push_pop(descriptor, expected):
    assert lark.carries_status_page(descriptor) is expected


def test_receiver_scan_ignores_unreadable_and_malformed_entries(tmp_path):
    (tmp_path / "hidraw0").mkdir()
    root = tmp_path / "hidraw1" / "device"
    root.mkdir(parents=True)
    (root / "uevent").write_text("HID_ID=0003:bad\nignored\n")
    assert lark.find_receiver(tmp_path, tmp_path / "fake-dev") is None


@pytest.mark.parametrize(
    ("reply", "message"), [(REPLY[:4], "short"), (b"\x06" + REPLY[1:], "malformed")]
)
def test_rejects_short_header_and_non_status_reply(reply, message):
    with pytest.raises(ValueError, match=message):
        lark.parse_reply(reply)


@pytest.mark.parametrize(
    ("length", "descriptor", "size", "error", "requests"),
    [
        (0, DESCRIPTOR, 0, "Invalid HID report descriptor", [3, 1]),
        (4097, DESCRIPTOR, 0, "Invalid HID report descriptor", [3, 1]),
        (7, bytes(7), 0, "no status feature report", [3, 1, 2]),
        (7, DESCRIPTOR, 0, "", [3, 1, 2, 6, 7]),
        (7, DESCRIPTOR, 65, "", [3, 1, 2, 6, 7]),
    ],
)
def test_heartbeat_rejects_invalid_descriptor_and_handles_size_fallback(
    monkeypatch, length, descriptor, size, error, requests
):
    calls, closed = [], []
    monkeypatch.setattr(lark.os, "open", lambda *_: 42)
    monkeypatch.setattr(lark.os, "close", closed.append)

    def ioctl(_fd, request, data, _mutable):
        number = request & 255
        calls.append(number)
        if number == 3:
            data[:] = struct.pack("IHH", 3, lark.VENDOR, lark.PRODUCT)
        elif number == 1:
            data[:] = struct.pack("I", length)
        elif number == 2:
            data[4 : 4 + len(descriptor)] = descriptor
        elif number == 7:
            data[: len(REPLY)] = REPLY
            return size
        return 0

    monkeypatch.setattr(lark.fcntl, "ioctl", ioctl)
    if error:
        with pytest.raises(ValueError, match=error):
            lark.heartbeat("fake-device")
    else:
        assert lark.heartbeat("fake-device") == lark.parse_reply(REPLY)
    assert calls == requests
    assert closed == [42]


@pytest.mark.parametrize(
    ("outcome", "result"),
    [
        (None, "no-device"),
        (PermissionError(), "permission-denied"),
        (OSError("lost receiver"), "unavailable"),
        (ValueError("malformed reply"), "unavailable"),
        ("status", "ok"),
    ],
)
def test_helper_emits_structured_status(monkeypatch, capsys, outcome, result):
    monkeypatch.setattr(
        lark, "find_receiver", lambda: Path("fake-device") if outcome is not None else None
    )

    def read(_path):
        if isinstance(outcome, Exception):
            raise outcome
        return lark.parse_reply(REPLY)

    monkeypatch.setattr(lark, "heartbeat", read)
    lark.main()
    assert json.loads(capsys.readouterr().out)["result"] == result


def test_helper_entry_point_never_opens_hid_when_sysfs_is_empty(monkeypatch, capsys):
    monkeypatch.setattr(Path, "glob", lambda *_: ())
    runpy.run_path(lark.__file__, run_name="__main__")
    assert json.loads(capsys.readouterr().out) == {"result": "no-device"}


def test_default_resolution_follows_source_chain_and_caches_module_listing(monkeypatch):
    commands = []
    sources = [
        {
            "name": "default-input",
            "properties": {"device.vendor.id": "invalid", "device.master_device": "remap"},
        },
        {"name": "remap", "owner_module": 5},
        {"name": "two", "owner_module": "6"},
        {"name": "usb", "properties": {"device.vendor.id": "3547", "device.product.id": 0x0407}},
    ]
    modules = [
        {"index": "5", "name": "module-remap-source", "argument": "unrelated=x master=two"},
        {"index": 6, "name": "module-remap-source", "argument": "master=usb"},
    ]

    def run(args, **kwargs):
        assert kwargs["timeout"] == 1
        commands.append(args[-1])
        return SimpleNamespace(
            stdout={
                "get-default-source": "default-input\n",
                "sources": json.dumps(sources),
                "modules": json.dumps(modules),
            }[args[-1]]
        )

    monkeypatch.setattr(lark_monitor.subprocess, "run", run)
    assert lark_monitor.input_is_receiver("default")
    assert commands == ["get-default-source", "sources", "modules"]


def test_cycle_does_not_revalidate_or_open_receiver(monkeypatch):
    sources = [
        {"name": "start", "properties": {"device.master_device": "cycle"}},
        {"name": "cycle", "properties": {"device.master_device": "start"}},
    ]
    monkeypatch.setattr(
        lark_monitor.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(sources))
    )
    assert not lark_monitor.input_is_receiver("start")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("[]", "unavailable"),
        ('{"result":"unknown"}', "unavailable"),
        ('{"result":"no-device"}', "no-device"),
        ('{"result":"permission-denied"}', "permission-denied"),
        ('{"result":"unavailable","error":"disconnected"}', "unavailable"),
        ('{"result":"ok","mic1":{"linked":false,"percent":2}}', "unavailable"),
        ('{"result":"ok","mic1":{"linked":true,"percent":false}}', "unavailable"),
    ],
)
def test_monitor_defensively_parses_helper_protocol(monkeypatch, payload, expected):
    monkeypatch.setattr(lark_monitor, "input_is_receiver", lambda _: True)
    monkeypatch.setattr(
        lark_monitor.subprocess, "run", lambda *args, **options: SimpleNamespace(stdout=payload)
    )
    monitor = lark_monitor.BatteryMonitor()
    cfg = Config(mic_battery="on")
    monitor.poll(cfg, "usb")
    monitor.worker.join(2)
    assert not monitor.worker.is_alive()
    assert monitor.poll(cfg, "usb")["result"] == expected


@pytest.mark.parametrize(("records", "status"), [("[]", "not-receiver"), ("[null]", "unavailable")])
def test_monitor_does_not_invoke_helper_for_unknown_or_invalid_source(monkeypatch, records, status):
    def run(args, **kwargs):
        assert args == ["pactl", "-f", "json", "list", "sources"]
        return SimpleNamespace(stdout=records)

    monkeypatch.setattr(lark_monitor.subprocess, "run", run)
    monitor = lark_monitor.BatteryMonitor()
    cfg = Config(mic_battery="on")
    monitor.poll(cfg, "usb")
    monitor.worker.join(2)
    assert not monitor.worker.is_alive()
    assert monitor.poll(cfg, "usb")["result"] == status


@pytest.mark.parametrize(
    "modules",
    [
        [{"index": 5, "name": "another-module", "argument": "master=usb"}],
        [{"index": 5, "name": "module-remap-source", "argument": None}],
    ],
)
def test_unusable_remap_module_does_not_guess_receiver(monkeypatch, modules):
    sources = [
        {"name": "remap", "owner_module": 5},
        {"name": "usb", "properties": {"device.vendor.id": "3547", "device.product.id": "0407"}},
    ]

    def run(args, **_kwargs):
        return SimpleNamespace(stdout=json.dumps(modules if args[-1] == "modules" else sources))

    monkeypatch.setattr(lark_monitor.subprocess, "run", run)
    assert not lark_monitor.input_is_receiver("remap")
