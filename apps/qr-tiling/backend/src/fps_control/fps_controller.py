import time
import threading
from dataclasses import dataclass

import depthai as dai
import numpy as np


@dataclass(frozen=True)
class FPSControllerConfig:
    """
    Tuning parameters for the FPS controller.

    max_fps / min_fps:              Allowed FPS range.
    fps_tolerance:                  Allowed deviation before DROP/RISE decisions (+/-1 FPS noise).
    rise_step:                      How many FPS to add on each rise attempt.
    stable_feedbacks_before_rise:   Consecutive stable reports needed before attempting a rise.
    freeze_feedback_cycles:         Feedback cycles to ignore after a target change (settling time).
    fails_to_detect_ceiling:        Consecutive rise failures before declaring a ceiling FPS.
    ceiling_retry_cycles:           Stable feedbacks before retrying after ceiling FPS detected.
    """

    max_fps: int = 30
    min_fps: int = 5
    fps_tolerance: int = 1
    rise_step: int = 2
    stable_feedbacks_before_rise: int = 2
    freeze_feedback_cycles: int = 2
    fails_to_detect_ceiling: int = 3
    ceiling_retry_cycles: int = 15


class FPSController(dai.node.ThreadedHostNode):
    """
    Controls frame rate for two synchronized streams using a Script node on device.

    The Script node throttles both streams to a target FPS.
    A host-side feedback loop consumes measured actual FPS (from FPSMonitor):
    - if actual FPS drops below target FPS (with tolerance), immediately stabilize down to actual FPS
    - if actual FPS stays stable for N cycles, attempt to rise by 'rise_step'
      and detect a ceiling FPS after repeated failed rise attempts, retrying more slowly after.

    'set_target()' provides feed-forward control (e.g. on tile-count increase) by immediately
    updating the target FPS and freezing feedback briefly to allow the pipeline to settle.
    """

    SCRIPT_TEMPLATE = """
    msg = node.inputs['target_fps'].get()
    target_fps = msg.getData()[0]
    max_fps = {max_fps}
    frame_budget = 0

    while True:
        while node.inputs['target_fps'].has():
            msg = node.inputs['target_fps'].get()
            target_fps = msg.getData()[0]

        nn_frame = node.inputs['nn_frames'].get()
        display_frame = node.inputs['display_frames'].get()

        frame_budget += target_fps

        if frame_budget >= max_fps:
            frame_budget -= max_fps
            node.outputs['rgb_nn'].send(nn_frame)
            node.outputs['rgb_display'].send(display_frame)
    """

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = self.getParentPipeline()

        self._script = self._pipeline.create(dai.node.Script)

        self._fps_out = self.createOutput()
        self._feedback_input = self.createInput()

        self._config = FPSControllerConfig()

        self._output_fps: int = self._config.max_fps
        self._last_stable_fps: int = self._config.max_fps
        self._trying_to_rise: bool = False
        self._stable_count: int = 0
        self._frozen_count: int = 0
        self._rise_fail_count: int = 0
        self._ceiling_detected: bool = False

        self._lock = threading.Lock()

    def build(
        self,
        nn_frames: dai.Node.Output,
        display_frames: dai.Node.Output,
        config: FPSControllerConfig | None = None,
    ) -> "FPSController":
        if config is not None:
            self._config = config

        self._script.setScript(
            self.SCRIPT_TEMPLATE.format(max_fps=self._config.max_fps)
        )

        self._output_fps = self._config.max_fps
        self._last_stable_fps = self._config.max_fps

        nn_frames.link(self._script.inputs["nn_frames"])
        display_frames.link(self._script.inputs["display_frames"])
        self._fps_out.link(self._script.inputs["target_fps"])

        self._script.inputs["target_fps"].setBlocking(False)

        return self

    def set_target(self, fps: int) -> None:
        """
        Set target FPS externally (e.g., feed-forward from tile count change).

        Immediately sends the new target to the device script,
        resets the rise/stable state, and freezes feedback for a few
        cycles so the pipeline can settle before the loop resumes.
        """
        fps = max(self._config.min_fps, min(self._config.max_fps, int(fps)))

        with self._lock:
            old = self._output_fps
            self._update_script_config(fps)
            self._last_stable_fps = fps
            self._stable_count = 0
            self._trying_to_rise = False
            self._frozen_count = self._config.freeze_feedback_cycles
            self._rise_fail_count = 0
            self._ceiling_detected = False
        print(f"[FPS {time.monotonic():.2f}] SET_TARGET: {old} -> {fps} (feed-forward)")

    def run(self) -> None:
        with self._lock:
            self._update_script_config(self._output_fps)

        while self.isRunning():
            feedback = self._feedback_input.get()
            if hasattr(feedback, "actual_fps"):
                self._handle_feedback(feedback.actual_fps)

    def _handle_feedback(self, actual_fps: int) -> None:
        t = time.monotonic()
        with self._lock:
            if self._frozen_count > 0:
                self._frozen_count -= 1
                log = f"[FPS {t:.2f}] FROZEN: target={self._output_fps} actual={actual_fps} remaining={self._frozen_count}"
            elif self._trying_to_rise:
                log = f"[FPS {t:.2f}] RISE_CHECK: target={self._output_fps} actual={actual_fps}"
                self._handle_rise_result(actual_fps)
            elif actual_fps < self._output_fps - self._config.fps_tolerance:
                log = f"[FPS {t:.2f}] DROP: target={self._output_fps} actual={actual_fps} -> stabilize({actual_fps})"
                self._stabilize(actual_fps)
            else:
                stable_cycles_required = (
                    self._config.ceiling_retry_cycles
                    if self._ceiling_detected
                    else self._config.stable_feedbacks_before_rise
                )

                target_before = self._output_fps
                self._try_rise(stable_cycles_required)

                log = (
                    f"[FPS {t:.2f}] STABLE: target={target_before} actual={actual_fps} "
                    f"count={self._stable_count}/{stable_cycles_required}"
                )

        print(log)

    def _handle_rise_result(self, actual_fps: int) -> None:
        self._trying_to_rise = False

        if actual_fps >= self._output_fps - self._config.fps_tolerance:
            self._last_stable_fps = self._output_fps
            self._rise_fail_count = 0
            self._stable_count = 0
            self._ceiling_detected = False
        else:
            self._rise_fail_count += 1
            if (
                self._rise_fail_count >= self._config.fails_to_detect_ceiling
                and not self._ceiling_detected
            ):
                self._ceiling_detected = True
                print(
                    f"[FPS {time.monotonic():.2f}] CEILING: detected at {self._last_stable_fps} FPS after {self._rise_fail_count} failed rises"
                )
            self._stabilize(self._last_stable_fps, freeze=True)

    def _stabilize(self, fps: int, freeze: bool = False) -> None:
        fps = max(self._config.min_fps, fps)
        self._update_script_config(fps)
        self._last_stable_fps = fps
        self._stable_count = 0
        if freeze:
            self._frozen_count = self._config.freeze_feedback_cycles

    def _try_rise(self, stable_cycles_required: int) -> None:
        self._stable_count += 1

        if (
            self._stable_count >= stable_cycles_required
            and self._output_fps < self._config.max_fps
        ):
            new_fps = min(
                self._config.max_fps, self._output_fps + self._config.rise_step
            )
            self._update_script_config(new_fps)
            self._trying_to_rise = True

    def _update_script_config(self, fps: int) -> None:
        self._output_fps = fps
        buff = dai.Buffer()
        buff.setData(np.array([np.uint8(self._output_fps)]))
        self._fps_out.send(buff)

    @property
    def rgb_nn(self) -> dai.Node.Output:
        return self._script.outputs["rgb_nn"]

    @property
    def rgb_display(self) -> dai.Node.Output:
        return self._script.outputs["rgb_display"]

    @property
    def feedback(self) -> dai.Node.Input:
        return self._feedback_input

    @property
    def current_fps(self) -> int:
        with self._lock:
            return self._output_fps

    @property
    def max_fps(self) -> int:
        return self._config.max_fps
