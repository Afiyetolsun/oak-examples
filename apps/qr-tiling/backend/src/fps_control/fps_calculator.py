import time
import threading
from dataclasses import dataclass

import depthai as dai
import numpy as np


@dataclass(frozen=True)
class FPSCalculatorConfig:
    """
    Tuning parameters for the FPS calculator.

    max_fps / min_fps:              Allowed FPS range.
    fps_tolerance:                  Allowed deviation before DROP/RISE decisions (+/-1 FPS noise).
    rise_step:                      How many FPS to add on each rise attempt.
    stable_feedbacks_before_rise:   Consecutive stable reports needed before attempting a rise.
    freeze_feedback_cycles:         Feedback cycles to ignore after a target change (settling time).
    fails_to_detect_ceiling:        Consecutive rise failures before declaring a ceiling FPS.
    ceiling_retry_cycles:           Stable feedbacks before retrying after ceiling FPS detected.
    safety_margin:                  Multiplier for estimated max FPS (conservative estimate).
    tile_decrease_fps_boost:        FPS bump when tile count decreases.
    nn_node_name:                   Name of the NN node in the pipeline (for timing queries).
    """

    max_fps: int = 30
    min_fps: int = 5
    fps_tolerance: int = 1
    rise_step: int = 2
    stable_feedbacks_before_rise: int = 2
    freeze_feedback_cycles: int = 2
    fails_to_detect_ceiling: int = 3
    ceiling_retry_cycles: int = 15
    safety_margin: float = 0.8
    tile_decrease_fps_boost: int = 2
    nn_node_name: str = "NeuralNetwork"


class FPSCalculator(dai.node.ThreadedHostNode):
    """
    Decides the target FPS based on:
    - Feedback: actual FPS from FPSMonitor (rise/drop/stable/ceiling state machine)
    - Feed-forward: tile count changes (pipeline state NeuralNetwork node timing estimation)

    Outputs target FPS to FPSController.
    """

    def __init__(self) -> None:
        super().__init__()

        self._feedback_input = self.createInput()
        self._target_fps_out = self.createOutput()

        self._config = FPSCalculatorConfig()
        self._pipeline: dai.Pipeline | None = None

        self._output_fps: int = self._config.max_fps
        self._last_stable_fps: int = self._config.max_fps
        self._trying_to_rise: bool = False
        self._stable_count: int = 0
        self._frozen_count: int = 0
        self._rise_fail_count: int = 0
        self._ceiling_detected: bool = False

        self._old_tile_count: int = 0

        self._lock = threading.Lock()

    def build(
        self,
        input_feedback: dai.Node.Output,
        pipeline: dai.Pipeline,
        initial_tile_count: int,
        config: FPSCalculatorConfig | None = None,
    ) -> "FPSCalculator":
        input_feedback.link(self._feedback_input)
        self._pipeline = pipeline
        self._old_tile_count = initial_tile_count

        if config is not None:
            self._config = config

        self._output_fps = self._config.max_fps
        self._last_stable_fps = self._config.max_fps

        return self

    def run(self) -> None:
        self._send_target(self._output_fps)

        while self.isRunning():
            feedback = self._feedback_input.get()
            if hasattr(feedback, "actual_fps"):
                self._handle_feedback(feedback.actual_fps)

    def adjust_fps_from_tile_count(self, tile_count: int) -> None:
        """Called when the tile count changes. Adjusts target FPS accordingly."""
        with self._lock:
            if tile_count > self._old_tile_count:
                try:
                    est_fps = self._estimate_max_fps(tile_count)
                except Exception as e:
                    print(f"[TILES_INCREASE] {self._old_tile_count}\u2192{tile_count}, estimation failed: {e}")
                    return
                print(
                    f"[TILES_INCREASE] {self._old_tile_count}\u2192{tile_count}, est={est_fps}"
                )
                self._set_target(est_fps)
            else:
                print(
                    f"[TILES_DECREASE] {self._old_tile_count}\u2192{tile_count}, allowing rise"
                )
                self._set_target(
                    self._output_fps + self._config.tile_decrease_fps_boost
                )
            self._old_tile_count = tile_count

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
        self._update_target(fps)
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
            self._update_target(new_fps)
            self._trying_to_rise = True

    def _set_target(self, fps: int) -> None:
        """Set target FPS from feed-forward. Resets state machine, freezes feedback."""
        fps = max(self._config.min_fps, min(self._config.max_fps, int(fps)))

        old = self._output_fps
        self._update_target(fps)
        self._last_stable_fps = fps
        self._stable_count = 0
        self._trying_to_rise = False
        self._frozen_count = self._config.freeze_feedback_cycles
        self._rise_fail_count = 0
        self._ceiling_detected = False
        print(
            f"[FPS {time.monotonic():.2f}] SET_TARGET: {old} -> {fps} (feed-forward)"
        )

    def _estimate_max_fps(self, tile_count: int) -> int:
        """Estimate safe FPS using NN per-tile processing time from pipeline state."""
        pipeline_state = self._pipeline.getPipelineState().nodes().detailed()
        for node_id, ns in pipeline_state.nodeStates.items():
            if (
                self._pipeline.getNode(node_id).getName()
                == self._config.nn_node_name
            ):
                nn_tile_us = ns.mainLoopTiming.durationStats.medianMicrosRecent
                est_max_fps = 1_000_000 / (nn_tile_us * tile_count)
                est_fps_target = int(est_max_fps * self._config.safety_margin)
                print(
                    f"[PIPELINE_STATE] tiles={tile_count}, nn_us={nn_tile_us}, est_target={est_fps_target}"
                )
                return est_fps_target
        raise RuntimeError(
            f"NN node '{self._config.nn_node_name}' not found in pipeline state"
        )

    def _update_target(self, fps: int) -> None:
        self._output_fps = fps
        self._send_target(fps)

    def _send_target(self, fps: int) -> None:
        buff = dai.Buffer()
        buff.setData(np.array([np.uint8(fps)]))
        self._target_fps_out.send(buff)

    @property
    def out(self) -> dai.Node.Output:
        return self._target_fps_out

    @property
    def current_fps(self) -> int:
        with self._lock:
            return self._output_fps
