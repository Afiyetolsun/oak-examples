import depthai as dai
from box import Box
from typing import Optional

from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsFilter, ImgDetectionsBridge
from config.config_data_classes import NeuralNetworkConfig
from nn.prompt_controller import PromptController
from nn.label_mapper_node import DetectionsLabelMapper
from prompting.encoders.textual_prompt_encoder import TextualPromptEncoder
from prompting.encoders.visual_prompt_encoder import VisualPromptEncoder


class NNDetectionNode(dai.node.ThreadedHostNode):
    """
    High-level node grouping the neural-network detection block.
    Handles object detection + filtering + annotation, and exposes a PromptController to
    update detection classes and confidence threshold at runtime.

    Internal pipeline:
        image_source
          -> ParsingNeuralNetwork
          -> ImgDetectionsFilter (filter by enabled label IDs)
          -> LabelMapperNode (add label names for visualization)
          -> ImgDetectionsBridge (convert to dai.ImgDetections)
          -> LabelMapperNode (re-add label names after bridge)

    Exposes:
      - detections_extended: ImgDetectionsExtended with label names (for visualizer)
      - detections: dai.ImgDetections with label names (for snapping)
      - controller: PromptController for dynamic prompt updates (classes, confidence threshold)
    """
    def __init__(self) -> None:
        super().__init__()

        self._nn: ParsingNeuralNetwork = self.createSubnode(ParsingNeuralNetwork)
        self._det_filter: ImgDetectionsFilter = self.createSubnode(ImgDetectionsFilter)
        self._bridge: ImgDetectionsBridge = self.createSubnode(ImgDetectionsBridge)
        self._det_label_mapper_extended: DetectionsLabelMapper = self.createSubnode(DetectionsLabelMapper)
        self._det_label_mapper: DetectionsLabelMapper = self.createSubnode(DetectionsLabelMapper)

        # Internal controller
        self.controller: Optional[PromptController] = None

        # Prompt encoders
        self.text_encoder: Optional[TextualPromptEncoder] = None
        self.visual_encoder: Optional[VisualPromptEncoder] = None

        # Outputs
        self.detections_extended: Optional[dai.Node.Output] = None
        self.detections: Optional[dai.Node.Output] = None

    def build(
        self,
        image_source: dai.Node.Output,
        cfg_nn: NeuralNetworkConfig,
        cfg_prompts: Box,
    ) -> "NNDetectionNode":
        """
        @param image_source: BGR image frames from camera.
        @param cfg: Neural network configuration.
        """
        # NN config
        self._nn.setNNArchive(cfg_nn.model.archive)
        self._nn.setBackend(cfg_nn.backend_type)
        self._nn.setBackendProperties({
            "runtime": cfg_nn.runtime,
            "performance_profile": cfg_nn.performance_profile
        })
        self._nn.setNumInferenceThreads(cfg_nn.num_inference_threads)
        self._nn.getParser(0).setConfidenceThreshold(0.0)

        image_source.link(self._nn.inputs["images"])

        # Detection filter
        self._det_filter.build(self._nn.out)

        # Annotation for visualization (ImgDetectionsExtended)
        self._det_label_mapper_extended.build(self._det_filter.out)
        self.detections_extended = self._det_label_mapper_extended.out

        # Bridge to convert ImgDetectionsExtended -> dai.ImgDetections
        self._bridge.build(self._det_filter.out)

        # Re-annotate after bridge (until ImgDetectionsBridge fix - it doesn't copy label names)
        self._det_label_mapper.build(self._bridge.out)
        self.detections = self._det_label_mapper.out

        # Prompt encoders
        self.text_encoder = TextualPromptEncoder(cfg_prompts)
        self.visual_encoder = VisualPromptEncoder(cfg_prompts)

        # Controller
        self.controller = PromptController(
            nn=self._nn,
            text_encoder=self.text_encoder,
            visual_encoder=self.visual_encoder,
            det_filter=self._det_filter,
            det_label_mappers=[self._det_label_mapper_extended, self._det_label_mapper],
            precision=cfg_nn.model.precision,
        )
        self.controller.send_initial_prompts(
            class_names=cfg_prompts.class_names,
            detection_threshold=cfg_prompts.detection_threshold,
        )

        return self

    def run(self) -> None:
        # High-level node: no host-side processing here. Processing happens in the composed subnodes.
        pass
