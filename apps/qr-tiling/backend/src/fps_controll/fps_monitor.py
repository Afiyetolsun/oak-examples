import time
from collections import deque

import depthai as dai

from depthai_nodes.node import BaseHostNode


class FPSFeedback(dai.Buffer):
    actual_fps: int = 0


class FPSMonitor(BaseHostNode):
    """
    Monitors actual FPS of the given input stream
    """

    def __init__(self) -> None:
        super().__init__()

        self._report_interval_sec: float = 3.0

        self._timestamps: deque = deque()
        self._label: str = "After patcher"
        self._last_report_time: float = 0.0
        self._start_time: float = 0.0
        self._warmup_sec: float = 5.0

    def build(
        self,
        input_stream: dai.Node.Output,
        report_interval_sec: float = 3.0,
        label: str = "After patcher",
    ) -> "FPSMonitor":
        self.link_args(input_stream)
        self._report_interval_sec = report_interval_sec
        self._label = label
        self._start_time = time.monotonic()
        return self

    def process(self, msg: dai.Buffer) -> None:
        now = time.monotonic()
        self._timestamps.append(now)

        cutoff = now - self._report_interval_sec
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        if now - self._start_time < self._warmup_sec:
            return

        if now - self._last_report_time >= self._report_interval_sec:
            fps = self._calculate_fps()
            self._send_feedback(fps, msg)
            self._last_report_time = now

    def _calculate_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0

        time_span = self._timestamps[-1] - self._timestamps[0]
        if time_span <= 0:
            return 0.0

        return len(self._timestamps) / time_span

    def _send_feedback(self, fps: float, ref_msg: dai.Buffer) -> None:
        feedback = FPSFeedback()
        feedback.actual_fps = int(fps)
        feedback.setTimestamp(ref_msg.getTimestamp())
        feedback.setSequenceNum(ref_msg.getSequenceNum())
        self.out.send(feedback)
