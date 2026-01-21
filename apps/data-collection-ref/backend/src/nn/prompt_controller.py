import numpy as np
import depthai as dai
from dataclasses import dataclass
from typing import Optional

from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsFilter
from core.neural_network.pipeline.annotation_node import AnnotationNode


@dataclass
class ModelState:
    current_classes: list[str] = None
    confidence_threshold: float = 0.0


class PromptController:
    """
    Handles sending prompts to the neural network and updating related components.

    Responsibilities:
      - Send prompt tensors into NN input queues
      - Update label filtering + label-name encoding on annotators
      - Update parser confidence threshold
      - Track model state for export/UI
    """

    def __init__(
        self,
        nn: ParsingNeuralNetwork,
        det_filter: ImgDetectionsFilter,
        annot_ext: AnnotationNode,
        annot_det: AnnotationNode,
        precision: str,
    ):
        self._nn = nn
        self._det_filter = det_filter
        self._annot_ext = annot_ext
        self._annot_det = annot_det
        self._precision = precision

        self.state = ModelState(current_classes=[])

        self._text_q: Optional[dai.InputQueue] = None
        self._img_q: Optional[dai.InputQueue] = None

        self._parser = self._nn.getParser(0)

    def _ensure_queues(self):
        """Input queues setup."""
        if self._text_q is not None and self._img_q is not None:
            return

        self._text_q = self._nn.inputs["texts"].createInputQueue()
        self._img_q = self._nn.inputs["image_prompts"].createInputQueue()

        self._nn.inputs["texts"].setReusePreviousMessage(True)
        self._nn.inputs["image_prompts"].setReusePreviousMessage(True)

    def _tensor_dtype(self) -> dai.TensorInfo.DataType:
        """Get tensor data type based on model precision."""
        return dai.TensorInfo.DataType.FP16 if self._precision == "fp16" else dai.TensorInfo.DataType.U8F

    @staticmethod
    def _make_nn_data(tensor_name: str, data: np.ndarray, dtype: dai.TensorInfo.DataType) -> dai.NNData:
        msg = dai.NNData()
        msg.addTensor(tensor_name, data, dataType=dtype)
        return msg

    def _update_labels(self, label_names: list[str], offset: int = 0) -> None:
        """Update label filtering and annotator encodings.
        @param label_names: List of class names to keep.
        @param offset: Label index offset (default: 0).
        """
        self._det_filter.setLabels(labels=list(range(offset, offset + len(label_names))), keep=True)

        encoding = {offset + k: v for k, v in enumerate(label_names)}
        self._annot_ext.set_label_encoding(encoding)
        self._annot_det.set_label_encoding(encoding)

    def apply_prompts(
        self,
        image_prompt: np.ndarray,
        text_prompt: np.ndarray,
        class_names: list[str],
        offset: int = 0,
    ) -> None:
        self._ensure_queues()
        dtype = self._tensor_dtype()

        # Tensor names must match those defined in the model YAML
        self._text_q.send(self._make_nn_data("text_prompts", text_prompt, dtype))
        self._img_q.send(self._make_nn_data("image_prompts", image_prompt, dtype))

        self._update_labels(class_names, offset)
        self.state.current_classes = list(class_names)

    def set_confidence_threshold(self, threshold: float) -> None:
        with self._lock:
            t = float(max(0.0, min(1.0, threshold)))
            self._parser.setConfidenceThreshold(t)
            self.state.confidence_threshold = t

    def get_model_state(self) -> ModelState:
        return self.state
