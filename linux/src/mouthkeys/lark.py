"""Opt-in Hollyland Lark A1 status heartbeat through Linux hidraw.

Protocol adapted from BenevolentFutures/MouthKeys d50ad008 LarkA1Protocol.
Protocol decoding credit: pedrozimmermannsoares/vorssaint-utils#2510.
Only the fixed read-status heartbeat exists: no firmware/settings commands.
Linux ioctl definitions follow include/uapi/linux/hidraw.h.
"""

import fcntl
import json
import os
import shlex
import struct
import subprocess
import threading
import time
from pathlib import Path

from .processes import python_command

VENDOR = 0x3547
PRODUCT = 0x0407
REPORT = 5
LENGTH = 64
HEARTBEAT = bytes([5, 3, 0xAA, 0xDD, 0x1F, 0, 0, 0xEF]) + bytes(56)


def carries_status_page(descriptor):
    """Parse HID short items, including global Push/Pop; never open consumer HID."""
    page = report = None
    stack = []
    offset = 0
    while offset < len(descriptor):
        prefix = descriptor[offset]
        offset += 1
        if prefix == 0xFE:
            if offset + 2 > len(descriptor):
                return False
            size = descriptor[offset]
            offset += 2 + size
            continue
        size = (0, 1, 2, 4)[prefix & 3]
        if offset + size > len(descriptor):
            return False
        value = int.from_bytes(descriptor[offset : offset + size], "little")
        offset += size
        item_type, tag = (prefix >> 2) & 3, prefix >> 4
        if item_type == 1:
            if tag == 0:
                page = value
            elif tag == 8:
                report = value
            elif tag == 10:
                stack.append((page, report))
            elif tag == 11:
                if not stack:
                    return False
                page, report = stack.pop()
        if item_type == 0 and tag == 11 and page == 0xFF12 and report == REPORT:
            return True  # Feature item on this vendor page/report, not keyboard input.
    return False


def find_receiver(sys_root=Path("/sys/class/hidraw"), dev_root=Path("/dev")):
    for item in sorted(sys_root.glob("hidraw*")):
        try:
            attributes = dict(
                line.split("=", 1)
                for line in (item / "device/uevent").read_text().splitlines()
                if "=" in line
            )
            _, vendor, product = attributes.get("HID_ID", "::").split(":")
            if int(vendor, 16) != VENDOR or int(product, 16) != PRODUCT:
                continue
            descriptor = (item / "device/report_descriptor").read_bytes()
            if carries_status_page(descriptor):
                return dev_root / item.name
        except (OSError, ValueError):
            continue
    return None


def parse_reply(reply):
    if len(reply) < 7:
        raise ValueError("short")
    if reply[:5] != bytes([REPORT, 3, 0xBB, 0xDD, 0x1F]):
        raise ValueError("malformed heartbeat reply")
    length = int.from_bytes(reply[5:7], "big")
    if length < 4 or len(reply) < 7 + length:
        raise ValueError("short heartbeat payload")
    result = []
    for index in (0, 1):
        linked, percent = reply[7 + index] == 1, reply[9 + index]
        result.append({"linked": linked, "percent": percent if linked and percent <= 100 else None})
    return {"result": "ok", "mic1": result[0], "mic2": result[1]}


def heartbeat(path):
    # Generic Linux _IOC: read|write, 64-byte feature, type H, request 6/7.
    feature = (3 << 30) | (LENGTH << 16) | (ord("H") << 8)
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        info = bytearray(8)
        fcntl.ioctl(fd, (2 << 30) | (8 << 16) | (ord("H") << 8) | 3, info, True)
        _, vendor, product = struct.unpack("IHH", info)
        if vendor != VENDOR or product != PRODUCT:
            raise ValueError("Receiver changed before open")
        descriptor_size = bytearray(4)
        fcntl.ioctl(fd, (2 << 30) | (4 << 16) | (ord("H") << 8) | 1, descriptor_size, True)
        length = struct.unpack("I", descriptor_size)[0]
        if not 0 < length <= 4096:
            raise ValueError("Invalid HID report descriptor")
        descriptor = bytearray(struct.pack("I", length)) + bytearray(4096)
        fcntl.ioctl(fd, (2 << 30) | (4100 << 16) | (ord("H") << 8) | 2, descriptor, True)
        if not carries_status_page(descriptor[4 : 4 + length]):
            raise ValueError("Receiver has no status feature report")
        fcntl.ioctl(fd, feature | 6, bytearray(HEARTBEAT), True)
        reply = bytearray([REPORT]) + bytearray(LENGTH - 1)
        size = fcntl.ioctl(fd, feature | 7, reply, True)
        return parse_reply(reply[: size if 0 < size <= LENGTH else LENGTH])
    finally:
        os.close(fd)


def input_is_receiver(source):
    def run(args):
        return subprocess.run(args, capture_output=True, text=True, check=True, timeout=1).stdout

    if source == "default":
        source = run(["pactl", "get-default-source"]).strip()
    sources = json.loads(run(["pactl", "-f", "json", "list", "sources"]))
    pending, visited = [source], set()
    modules = None
    while pending and len(visited) < 16:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        item = next((item for item in sources if item.get("name") == name), None)
        if item:
            props = item.get("properties", {})
            try:
                vendor = props.get("device.vendor.id", "0")
                product = props.get("device.product.id", "0")
                if (vendor if isinstance(vendor, int) else int(vendor, 16)) == VENDOR and (
                    product if isinstance(product, int) else int(product, 16)
                ) == PRODUCT:
                    return True
            except (ValueError, TypeError):
                pass
            master = props.get("device.master_device")
            if isinstance(master, str) and master:
                pending.append(master)
            # PulseAudio remapped sources name their master in the owning module.
            if item.get("owner_module") is not None:
                if modules is None:
                    modules = json.loads(run(["pactl", "-f", "json", "list", "modules"]))
                owner = next(
                    (
                        module
                        for module in modules
                        if str(module.get("index")) == str(item["owner_module"])
                    ),
                    {},
                )
                if owner.get("name") == "module-remap-source":
                    for argument in shlex.split(owner.get("argument", "")):
                        if argument.startswith("master="):
                            pending.append(argument[7:])
    return False


class BatteryMonitor:
    def __init__(self):
        self.value = None
        self.polled_at = float("-inf")
        self.worker = None
        self.source = None

    def poll(self, config, source):
        if config.mic_battery != "on":
            self.value = None
            return None
        if source != self.source:
            self.value = None
            self.polled_at = float("-inf")
        self.source = source
        if time.monotonic() - self.polled_at >= 15 and (
            self.worker is None or not self.worker.is_alive()
        ):
            self.polled_at = time.monotonic()
            self.worker = threading.Thread(target=self._poll, args=(source,), daemon=True)
            self.worker.start()
        return self.value

    def _poll(self, source):
        try:
            if not input_is_receiver(source):
                value = {"result": "not-receiver"}
            else:
                result = subprocess.run(
                    python_command("mouthkeys.lark"),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2,
                )
                value = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            value = {"result": "unavailable", "error": str(exc)}
        if self.source == source:
            self.value = value


def main():
    try:
        device = find_receiver()
        value = heartbeat(device) if device else {"result": "no-device"}
    except PermissionError:
        value = {"result": "permission-denied"}
    except (OSError, ValueError) as exc:
        value = {"result": "unavailable", "error": str(exc)}
    print(json.dumps(value))


if __name__ == "__main__":
    main()
