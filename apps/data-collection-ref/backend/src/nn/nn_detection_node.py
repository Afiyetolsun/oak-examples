import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsFilter, ImgDetectionsBridge
from config.config_data_classes import NeuralNetworkConfig
from nn.prompt_controller import PromptController
from nn.annotation_node import AnnotationNode


class NNDetectionNode(dai.node.ThreadedHostNode):
    """
    High-level node grouping the neural-network detection block.
    Handles object detection + filtering + annotation + tracking.

    Internal pipeline:
        image_source
          -> ParsingNeuralNetwork
          -> ImgDetectionsFilter (filter by labels)
          -> AnnotationNode (add label names for visualization)
          -> ImgDetectionsBridge (convert to dai.ImgDetections)
          -> AnnotationNode (re-add label names after bridge)

    Exposes:
      - detections_extended: ImgDetectionsExtended with labels (for visualizer)
      - detections: dai.ImgDetections with labels (for snapping)
      - controller: PromptController for dynamic prompt updates
    """
    def __init__(self) -> None:
        super().__init__()

        self._nn: ParsingNeuralNetwork = self.createSubnode(ParsingNeuralNetwork)
        self._det_filter: ImgDetectionsFilter = self.createSubnode(ImgDetectionsFilter)
        self._bridge: ImgDetectionsBridge = self.createSubnode(ImgDetectionsBridge)
        self._annotation_extended: AnnotationNode = self.createSubnode(AnnotationNode)
        self._annotation: AnnotationNode = self.createSubnode(AnnotationNode)

        # Internal controller
        self.controller: PromptController = None

        # Outputs
        self.detections_extended: dai.Node.Output = None
        self.detections: dai.Node.Output = None

    def build(
        self,
        image_source: dai.Node.Output,
        cfg: NeuralNetworkConfig,
    ) -> "NNDetectionNode":
        """
        @param image_source: BGR image frames from camera.
        @param cfg: Neural network configuration.
        """
        # NN config
        backend = cfg.nn_yaml.nn_backend
        self._nn.setNNArchive(cfg.model.archive)
        self._nn.setBackend(backend.type)
        self._nn.setBackendProperties({
            "runtime": backend.runtime,
            "performance_profile": backend.performance_profile,
        })
        self._nn.setNumInferenceThreads(backend.inference_threads)
        self._nn.getParser(0).setConfidenceThreshold(0.0)

        image_source.link(self._nn.inputs["images"])

        # Detection filter
        self._det_filter.build(self._nn.out)

        # Annotation for visualization (ImgDetectionsExtended)
        self._annotation_extended.build(self._det_filter.out)
        self.detections_extended = self._annotation_extended.out

        # Bridge to convert ImgDetectionsExtended -> dai.ImgDetections
        self._bridge.build(self._det_filter.out)

        # Re-annotate after bridge (until ImgDetectionsBridge fix - it doesn't copy label names)
        self._annotation.build(self._bridge.out)
        self.detections = self._annotation.out

        # Controller
        self.controller = PromptController(
            self._nn,
            self._det_filter,
            self._annotation_extended,
            self._annotation,
            cfg.model.precision,
        )

        return self

    def run(self) -> None:
        # High-level node: no host-side processing, subnodes run on device.
        pass
