import numpy as np
import depthai as dai
from dataclasses import dataclass, field
from typing import List

from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsFilter

from nn.label_mapper_node import DetectionsLabelMapper
from prompting.encoders.textual_prompt_encoder import TextualPromptEncoder
from prompting.encoders.visual_prompt_encoder import VisualPromptEncoder


@dataclass
class ModelState:
    current_classes: list[str] = field(default_factory=list)
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
        text_encoder: TextualPromptEncoder,
        visual_encoder: VisualPromptEncoder,
        det_filter: ImgDetectionsFilter,
        det_label_mappers: List[DetectionsLabelMapper],
        precision: str,
    ):
        self._nn = nn
        self._text_encoder = text_encoder
        self._visual_encoder = visual_encoder
        self._det_filter = det_filter
        self._det_label_mappers = det_label_mappers
        self._precision = precision

        self.state = ModelState()

        # NN input queues
        self._text_q = self._nn.inputs["texts"].createInputQueue()
        self._img_q = self._nn.inputs["image_prompts"].createInputQueue()
        self._nn.inputs["texts"].setReusePreviousMessage(True)
        self._nn.inputs["image_prompts"].setReusePreviousMessage(True)

        self._parser = self._nn.getParser(0)

    def _tensor_dtype(self) -> dai.TensorInfo.DataType:
        """Get tensor data type based on model precision."""
        if self._precision == "fp16":
            return dai.TensorInfo.DataType.FP16
        return dai.TensorInfo.DataType.U8

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
        if offset < 0:
            raise ValueError("offset must be >= 0")

        self._det_filter.setLabels(labels=list(range(offset, offset + len(label_names))), keep=True)

        encoding = {offset + k: v for k, v in enumerate(label_names)}
        for label_mapper in self._det_label_mappers:
            label_mapper.set_label_encoding(encoding)

    def send_initial_prompts(self, class_names: list[str], text_offset: int, detection_threshold: float) -> None:
        """Send initial prompts at startup."""
        text_embeddings = self._text_encoder.extract_embeddings(class_names)
        dummy_image = self._visual_encoder.make_dummy()

        self.apply_prompts(
            image_prompt=dummy_image,
            text_prompt=text_embeddings,
            class_names=class_names,
            offset=text_offset,
        )
        self.set_confidence_threshold(detection_threshold)

    def apply_prompts(
        self,
        image_prompt: np.ndarray,
        text_prompt: np.ndarray,
        class_names: list[str],
        offset: int = 0,
    ) -> None:
        dtype = self._tensor_dtype()

        # Tensor names must match those defined in the model YAML
        self._text_q.send(self._make_nn_data("text_prompts", text_prompt, dtype))
        self._img_q.send(self._make_nn_data("image_prompts", image_prompt, dtype))

        self._update_labels(class_names, offset)
        self.state.current_classes = list(class_names)

    def set_confidence_threshold(self, threshold: float) -> None:
        t = float(max(0.0, min(1.0, threshold)))
        self._parser.setConfidenceThreshold(t)
        self.state.confidence_threshold = t

    def get_model_state(self) -> ModelState:
        return self.state
