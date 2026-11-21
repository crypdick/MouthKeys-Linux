"""Frame-clock voice trace, following the native DatasheetTraceModel motion."""

import math


class Trace:
    count = 63
    sample = 1 / 12

    def __init__(self) -> None:
        self.begin(0)
        self.live = False

    def begin(self, now: float) -> None:
        self.current = [2.0] * self.count
        self.shown = self.current.copy()
        self.live = True
        self.last_frame = now
        self.window_start = now
        self.last_voice: float | None = None
        self.window_peak = self.pending_peak = 0.0
        self.quiet_floor: float | None = None
        self.loud_peak = 0.2
        self.fraction = 0.0
        self.stopped_at: float | None = None
        self.stop_from = self.shown.copy()

    @property
    def gate(self) -> float:
        return (self.quiet_floor or 0) + 0.4 * 15 / 55

    def ingest(self, level: float, now: float) -> None:
        if not self.live:
            return
        level = min(1.0, max(0.0, level))
        self.window_peak = max(self.window_peak, level)
        self.pending_peak = max(self.pending_peak, level)
        if level > self.gate:
            self.last_voice = now

    def advance(self, now: float) -> None:
        dt = min(0.25, max(0, now - self.last_frame))
        self.last_frame = max(self.last_frame, now)
        if not self.live:
            if self.stopped_at is not None:
                progress = min(1, max(0, (now - self.stopped_at) / 0.06))
                self.shown = [h + (2 - h) * progress for h in self.stop_from]
            return
        if now - self.window_start >= self.sample:
            level = self.window_peak
            window_dt = min(0.25, now - self.window_start)
            self.quiet_floor = (
                min(level, 0.2)
                if self.quiet_floor is None
                else min(level, self.quiet_floor + 0.03 * window_dt)
            )
            self.loud_peak = max(level, self.loud_peak - 0.02 * window_dt, self.gate + 0.2)
            self.window_peak = 0.0
            self.window_start = now
        # Do not turn an old sample into an indefinitely moving waveform.
        if self.last_voice is not None:
            voiced_dt = min(dt, max(0, self.last_voice + 0.25 - (now - dt)))
            self.fraction += voiced_dt / self.sample
            while self.fraction >= 1:
                amplitude = min(
                    1,
                    max(
                        0, (self.pending_peak - self.gate) / max(0.001, self.loud_peak - self.gate)
                    ),
                )
                self.current = [*self.current[1:], 2 + 28 * amplitude**1.15]
                self.shown = [*self.shown[1:], 2.0]
                self.pending_peak = 0.0
                self.fraction -= 1
        ease = 1 - math.exp(-dt / 0.045)  # ~135 ms to settle, as on macOS.
        self.shown = [a + (b - a) * ease for a, b in zip(self.shown, self.current, strict=True)]

    def stop(self, now: float) -> None:
        if self.live:
            self.advance(now)
            self.live = False
            self.stopped_at = now
            self.stop_from = self.shown.copy()
