"""Interpret service status without Qt or desktop notifications."""

from collections.abc import Mapping
from dataclasses import dataclass

from .config import Config

_RECORD_LABELS = {
    "starting": "Cancel microphone start",
    "capture_failed": "Retry microphone stop",
    "recording": "Stop recording",
    "switching": "Stop recording",
}
_SAVED_INSERTION = "Insertion requested — transcript saved in history"
Alert = tuple[str, str]


def status_text(data: Mapping[str, object], key: str, default: str = "") -> str:
    value = data.get(key)
    return value if isinstance(value, str) else default


@dataclass(frozen=True)
class TrayStatus:
    phase: str
    message: str
    can_record: bool
    microphone: str
    item_id: str
    has_preview: bool
    notice_kind: str
    notice_serial: str
    notice_message: str

    @property
    def record_label(self) -> str:
        return _RECORD_LABELS.get(self.phase, "Record")

    @classmethod
    def parse(cls, payload: object) -> "TrayStatus | None":
        if not isinstance(payload, dict) or not (phase := status_text(payload, "phase")):
            return None
        can_record = payload.get("can_record", phase == "idle")
        notice = payload.get("notice")
        if not isinstance(notice, dict):
            notice = {}
        return cls(
            phase=phase,
            message=status_text(payload, "message", "Unavailable"),
            can_record=can_record is True or phase in _RECORD_LABELS,
            microphone=status_text(payload, "microphone"),
            item_id=status_text(payload, "id"),
            has_preview=bool(status_text(payload, "preview")),
            notice_kind=status_text(notice, "kind"),
            notice_serial=status_text(notice, "serial"),
            notice_message=status_text(notice, "message"),
        )


def _paste_alert(config: Config, status: TrayStatus) -> Alert | None:
    if config.show_paste_alerts != "on" or not status.item_id or not status.has_preview:
        return None
    if status.message.startswith(("Insertion requested", "Paste requested")):
        message = "Destination acceptance cannot be verified."
        if status.message.startswith(_SAVED_INSERTION):
            message += " Transcript remains available in History."
        return "Paste requested", message
    if status.notice_kind == "paste":
        return "Manual paste needed", status.notice_message
    if status.message.startswith("Copied") and config.delivery != "clipboard":
        return (
            "Manual paste needed",
            "Automatic insertion was not requested. Transcript copied to clipboard.",
        )
    return None


@dataclass
class TrayAlerts:
    observed_status: bool = False
    last_microphone: str = ""
    last_notice: str = ""
    completed_id: str = ""

    def observe(self, config: Config, status: TrayStatus) -> list[Alert]:
        alerts: list[Alert] = []
        if (
            config.show_microphone_alerts == "on"
            and self.observed_status
            and status.microphone
            and self.last_microphone
            and status.microphone != self.last_microphone
        ):
            alerts.append(("Microphone changed", status.microphone))
        new_notice = status.notice_serial and status.notice_serial != self.last_notice
        if new_notice:
            if (
                config.show_ai_alerts == "on"
                and self.observed_status
                and status.notice_kind == "ai"
            ):
                alerts.append(("AI processing failed", status.notice_message))
            if config.show_microphone_alerts == "on" and any(
                word in status.notice_message.lower() for word in ("microphone", "source")
            ):
                alerts.append(("Microphone unavailable", status.notice_message))
        if status.phase == "idle" and status.item_id != self.completed_id:
            if self.observed_status and (alert := _paste_alert(config, status)) is not None:
                alerts.append(alert)
            self.completed_id = status.item_id
        self.last_microphone = status.microphone
        self.last_notice = status.notice_serial
        self.observed_status = True
        return alerts
