"""Opt-in Hollyland Lark A1 status heartbeat through Linux hidraw.

Protocol adapted from BenevolentFutures/MouthKeys d50ad008 LarkA1Protocol.
Protocol decoding credit: pedrozimmermannsoares/vorssaint-utils#2510.
Only the fixed read-status heartbeat exists: no firmware/settings commands.
Linux ioctl definitions follow include/uapi/linux/hidraw.h.
"""

import fcntl
import json
import os
import struct
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

VENDOR = 0x3547
PRODUCT = 0x0407
REPORT = 5
LENGTH = 64
HEARTBEAT = bytes([5, 3, 0xAA, 0xDD, 0x1F, 0, 0, 0xEF]) + bytes(56)


class MicBattery(TypedDict):
    linked: bool
    percent: int | None


class ReceiverBattery(TypedDict):
    result: Literal["ok"]
    mic1: MicBattery
    mic2: MicBattery


UnavailableReason = Literal["no-device", "permission-denied", "not-receiver", "unavailable"]


class UnavailableBattery(TypedDict):
    result: UnavailableReason
    error: NotRequired[str]


BatteryStatus = ReceiverBattery | UnavailableBattery


def carries_status_page(descriptor: bytes | bytearray) -> bool:
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


def find_receiver(
    sys_root: Path = Path("/sys/class/hidraw"), dev_root: Path = Path("/dev")
) -> Path | None:
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


def parse_reply(reply: bytes | bytearray) -> ReceiverBattery:
    if len(reply) < 7:
        raise ValueError("short")
    if reply[:5] != bytes([REPORT, 3, 0xBB, 0xDD, 0x1F]):
        raise ValueError("malformed heartbeat reply")
    length = int.from_bytes(reply[5:7], "big")
    if length < 4 or len(reply) < 7 + length:
        raise ValueError("short heartbeat payload")

    def microphone(index: int) -> MicBattery:
        linked, percent = reply[7 + index] == 1, reply[9 + index]
        return {"linked": linked, "percent": percent if linked and percent <= 100 else None}

    return {"result": "ok", "mic1": microphone(0), "mic2": microphone(1)}


def heartbeat(path: str | Path) -> ReceiverBattery:
    # Generic Linux _IOC: read|write, 64-byte feature, type H, request 6/7.
    feature = (3 << 30) | (LENGTH << 16) | (ord("H") << 8)
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        info = bytearray(8)
        fcntl.ioctl(fd, (2 << 30) | (8 << 16) | (ord("H") << 8) | 3, info, True)  # noqa: FBT003 - ioctl mutate flag is positional-only
        _, vendor, product = struct.unpack("IHH", info)
        if vendor != VENDOR or product != PRODUCT:
            raise ValueError("Receiver changed before open")
        descriptor_size = bytearray(4)
        fcntl.ioctl(fd, (2 << 30) | (4 << 16) | (ord("H") << 8) | 1, descriptor_size, True)  # noqa: FBT003 - ioctl mutate flag is positional-only
        length = struct.unpack("I", descriptor_size)[0]
        if not 0 < length <= 4096:
            raise ValueError("Invalid HID report descriptor")
        descriptor = bytearray(struct.pack("I", length)) + bytearray(4096)
        fcntl.ioctl(fd, (2 << 30) | (4100 << 16) | (ord("H") << 8) | 2, descriptor, True)  # noqa: FBT003 - ioctl mutate flag is positional-only
        if not carries_status_page(descriptor[4 : 4 + length]):
            raise ValueError("Receiver has no status feature report")
        fcntl.ioctl(fd, feature | 6, bytearray(HEARTBEAT), True)  # noqa: FBT003 - ioctl mutate flag is positional-only
        reply = bytearray([REPORT]) + bytearray(LENGTH - 1)
        size = fcntl.ioctl(fd, feature | 7, reply, True)  # noqa: FBT003 - ioctl mutate flag is positional-only
        return parse_reply(reply[: size if 0 < size <= LENGTH else LENGTH])
    finally:
        os.close(fd)


def main() -> None:
    try:
        device = find_receiver()
        value: BatteryStatus = heartbeat(device) if device else {"result": "no-device"}
    except PermissionError:
        value = {"result": "permission-denied"}
    except (OSError, ValueError) as exc:
        value = {"result": "unavailable", "error": str(exc)}
    print(json.dumps(value))  # noqa: T201


if __name__ == "__main__":
    main()
