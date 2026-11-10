import struct

import pytest

from mouthkeys import lark
from mouthkeys.config import Config

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

    def ioctl(fd, request, data, mutable):
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

    def wrong_device(fd, request, data, mutable):
        calls.append(request & 255)
        data[:] = struct.pack("IHH", 3, 1, 2)

    monkeypatch.setattr(lark.fcntl, "ioctl", wrong_device)
    with pytest.raises(ValueError, match="changed"):
        lark.heartbeat("fake-device")
    assert calls == [3]


def test_disabled_monitor_never_touches_device_or_subprocess(monkeypatch):
    monkeypatch.setattr(lark.subprocess, "run", lambda *a, **kw: pytest.fail("Device touched"))
    assert lark.BatteryMonitor().poll(Config(), "default") is None


def test_battery_identifies_receiver_behind_remapped_source_without_name_guessing(monkeypatch):
    import json
    from types import SimpleNamespace

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
        lark.subprocess,
        "run",
        lambda args, **kw: SimpleNamespace(
            stdout=json.dumps(modules if args[-1] == "modules" else sources)
        ),
    )
    assert lark.input_is_receiver("custom_voice_filter")
    sources[1]["properties"]["device.product.id"] = "0408"
    assert not lark.input_is_receiver("custom_voice_filter")


def test_compact_overlay_has_real_shorter_layout_without_windows():
    from PySide6.QtWidgets import QApplication

    from mouthkeys.overlay import Overlay

    app = QApplication.instance() or QApplication([])
    panel = Overlay()
    panel.timer.stop()
    panel.apply_settings(Config(overlay_style="compact"))
    assert (panel.width(), panel.height()) == (424, 106)
    assert panel.buttons["copy"].y() == 60
    assert not panel.isVisible()
    panel.close()
    app.processEvents()
