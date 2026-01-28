import depthai as dai
import numpy as np


class FPSController(dai.node.ThreadedHostNode):
    """
    Controls frame rate for two synchronized streams using a Script node on device.

    Automatically adjusts FPS based on pipeline throughput feedback:
    - Drops FPS when pipeline can't keep up
    - Gradually rises FPS when stable
    """

    SCRIPT_CONTENT = """
skip_count_msg = node.inputs['skip_count'].get()
skip_count = skip_count_msg.getData()[0]
frame_counter = 0

while True:
    while node.inputs['skip_count'].has():
        skip_count_msg = node.inputs['skip_count'].get()
        skip_count = skip_count_msg.getData()[0]
        frame_counter = 0

    nn_frame = node.inputs['nn_video'].get()
    preview_frame = node.inputs['preview'].get()

    while nn_frame.getSequenceNum() != preview_frame.getSequenceNum():
        if nn_frame.getSequenceNum() < preview_frame.getSequenceNum():
            nn_frame = node.inputs['nn_video'].get()
        else:
            preview_frame = node.inputs['preview'].get()

    frame_counter += 1

    if frame_counter > skip_count:
        frame_counter = 0
        node.outputs['nn_video_out'].send(nn_frame)
        node.outputs['preview_out'].send(preview_frame)
"""

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = self.getParentPipeline()

        self._script = self._pipeline.create(dai.node.Script)
        self._script.setScript(self.SCRIPT_CONTENT)

        self._skip_count_out = self.createOutput()
        self._feedback_input = self.createInput()

        self._target_fps: int = 30
        self._max_fps: int = 30
        self._min_fps: int = 5
        self._skip_count: int = 0

        self._drop_threshold: float = 0.9
        self._rise_step: int = 3
        self._rise_step_max: int = 30
        self._stable_feedbacks_needed: int = 10

        self._current_step: int = self._rise_step
        self._last_stable_fps: int = self._target_fps
        self._trying_to_rise: bool = False
        self._stable_count: int = 0

    def build(
            self,
            nn_video: dai.Node.Output,
            preview: dai.Node.Output,
            max_fps: int = 30,
            min_fps: int = 5,
            drop_threshold: float = 0.9,
            rise_step: int = 3,
            stable_feedbacks_needed: int = 3,
    ) -> "FPSController":
        self._max_fps = max_fps
        self._min_fps = min_fps
        self._target_fps = max_fps
        self._last_stable_fps = max_fps
        self._drop_threshold = drop_threshold
        self._rise_step = rise_step
        self._current_step = rise_step
        self._stable_feedbacks_needed = stable_feedbacks_needed

        nn_video.link(self._script.inputs["nn_video"])
        preview.link(self._script.inputs["preview"])
        self._skip_count_out.link(self._script.inputs["skip_count"])

        self._script.inputs["skip_count"].setBlocking(False)

        print(f"[FPSController] Built with max_fps={max_fps}, min_fps={min_fps}, "
              f"drop_threshold={drop_threshold}, rise_step={rise_step}, "
              f"stable_feedbacks_needed={stable_feedbacks_needed}")

        return self

    def run(self) -> None:
        self._send_skip_count()
        print(f"[FPSController] Started with target_fps={self._target_fps}")

        while self.isRunning():
            feedback = self._feedback_input.get()
            if hasattr(feedback, 'actual_fps'):
                self._handle_feedback(feedback.actual_fps)

    def _handle_feedback(self, actual_fps: int) -> None:
        actual_fps = max(actual_fps, self._min_fps)

        if self._trying_to_rise:
            self._handle_rise_result(actual_fps)
            return

        if actual_fps < self._target_fps * self._drop_threshold:
            self._stabilize(actual_fps, self._rise_step, 0)
        elif actual_fps >= self._target_fps:
            self._handle_stable_state()

    def _handle_rise_result(self, actual_fps: int) -> None:
        if actual_fps >= self._target_fps:
            next_step = min(self._current_step + self._rise_step, self._rise_step_max)
            self._stabilize(actual_fps, next_step, self._stable_feedbacks_needed)
        elif actual_fps > self._last_stable_fps:
            self._stabilize(actual_fps, self._rise_step, self._stable_feedbacks_needed)
        else:
            self._stabilize(self._last_stable_fps, self._rise_step, 0)

        self._trying_to_rise = False

    def _handle_stable_state(self) -> None:
        self._stable_count += 1

        if self._stable_count >= self._stable_feedbacks_needed and self._target_fps < self._max_fps:
            new_target = min(self._max_fps, self._target_fps + self._current_step)
            self._set_target_fps(new_target)
            self._trying_to_rise = True

    def _set_target_fps(self, fps: int) -> None:
        self._target_fps = fps
        self._skip_count = 0 if fps >= self._max_fps else int(self._max_fps / fps) - 1
        self._send_skip_count()

    def _send_skip_count(self) -> None:
        buff = dai.Buffer()
        buff.setData(np.array([np.uint8(self._skip_count)]))
        self._skip_count_out.send(buff)

    def _stabilize(self, fps: int, step: int, stable_count: int) -> None:
        print(f"[FPSController] Stabilize: {self._target_fps} -> {fps} (actual={fps})")
        fps = max(self._min_fps, fps)
        self._set_target_fps(fps)
        self._last_stable_fps = fps
        self._current_step = step
        self._stable_count = stable_count

    @property
    def nn_video_out(self) -> dai.Node.Output:
        return self._script.outputs["nn_video_out"]

    @property
    def preview_out(self) -> dai.Node.Output:
        return self._script.outputs["preview_out"]

    @property
    def feedback(self) -> dai.Node.Input:
        return self._feedback_input