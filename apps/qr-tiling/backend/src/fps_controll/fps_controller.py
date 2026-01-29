import depthai as dai
import numpy as np


class FPSController(dai.node.ThreadedHostNode):
    """
    Controls frame rate for two synchronized streams using a Script node on device.
    """

    SCRIPT_CONTENT = """
    msg = node.inputs['target_fps'].get()
    target_fps = msg.getData()[0]
    frame_budget = 0

    while True:
        while node.inputs['target_fps'].has():
            msg = node.inputs['target_fps'].get()
            target_fps = msg.getData()[0]

        nn_frame = node.inputs['nn_video'].get()
        preview_frame = node.inputs['preview'].get()

        frame_budget += target_fps

        if frame_budget >= 30:
            frame_budget -= 30
            node.outputs['nn_video_out'].send(nn_frame)
            node.outputs['preview_out'].send(preview_frame)
    """

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = self.getParentPipeline()

        self._script = self._pipeline.create(dai.node.Script)
        self._script.setScript(self.SCRIPT_CONTENT)

        self._fps_out = self.createOutput()
        self._feedback_input = self.createInput()

        self._output_fps: int = 30
        self._max_fps: int = 30

        self._rise_step: int = 2
        self._rise_step_max: int = 10
        self._feedbacks_to_rise: int = 3
        self._freeze_time: int = 3

        self._current_step: int = 2
        self._last_stable_fps: int = 30
        self._trying_to_rise: bool = False
        self._stable_count: int = 0
        self._frozen_count: int = 0

    def build(
        self,
        nn_video: dai.Node.Output,
        preview: dai.Node.Output,
        max_fps: int = 30,
        rise_step: int = 2,
        stable_feedbacks_needed: int = 3,
    ) -> "FPSController":
        self._max_fps = max_fps
        self._output_fps = max_fps
        self._last_stable_fps = max_fps
        self._rise_step = rise_step
        self._current_step = rise_step
        self._feedbacks_to_rise = stable_feedbacks_needed

        nn_video.link(self._script.inputs["nn_video"])
        preview.link(self._script.inputs["preview"])
        self._fps_out.link(self._script.inputs["target_fps"])

        self._script.inputs["target_fps"].setBlocking(False)

        return self

    def run(self) -> None:
        self._update_script_config(self._output_fps)

        while self.isRunning():
            feedback = self._feedback_input.get()
            if hasattr(feedback, "actual_fps"):
                self._handle_feedback(feedback.actual_fps)

    def _handle_feedback(self, actual_fps: int) -> None:
        if self._frozen_count > 0:
            self._frozen_count -= 1
            return

        if self._trying_to_rise:
            self._handle_rise_result(actual_fps)
            return

        if actual_fps < self._output_fps:
            self._stabilize(actual_fps)
        elif actual_fps >= self._output_fps:
            self._try_rise()

    def _handle_rise_result(self, actual_fps: int) -> None:
        self._trying_to_rise = False

        if actual_fps >= self._output_fps:
            self._last_stable_fps = self._output_fps
            self._current_step += self._rise_step
            new_fps = min(self._max_fps, self._output_fps + self._current_step)
            if new_fps > self._output_fps:
                self._update_script_config(new_fps)
                self._trying_to_rise = True
            else:
                self._stable_count = self._feedbacks_to_rise

        else:
            new_fps = (
                actual_fps - 1 if actual_fps > self._last_stable_fps else actual_fps
            )
            self._stabilize(new_fps, freeze=True)

    def _stabilize(self, fps: int, freeze: bool = False) -> None:
        self._update_script_config(fps)
        self._last_stable_fps = fps
        self._current_step = self._rise_step
        self._stable_count = 0
        if freeze:
            self._frozen_count = self._freeze_time

    def _try_rise(self) -> None:
        self._stable_count += 1

        if (
            self._stable_count >= self._feedbacks_to_rise
            and self._output_fps < self._max_fps
        ):
            new_fps = min(self._max_fps, self._output_fps + self._current_step)
            self._update_script_config(new_fps)
            self._trying_to_rise = True

    def _update_script_config(self, fps: int) -> None:
        self._output_fps = fps
        buff = dai.Buffer()
        buff.setData(np.array([np.uint8(self._output_fps)]))
        self._fps_out.send(buff)

    @property
    def nn_video_out(self) -> dai.Node.Output:
        return self._script.outputs["nn_video_out"]

    @property
    def preview_out(self) -> dai.Node.Output:
        return self._script.outputs["preview_out"]

    @property
    def feedback(self) -> dai.Node.Input:
        return self._feedback_input
