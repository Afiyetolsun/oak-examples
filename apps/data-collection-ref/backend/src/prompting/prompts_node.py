from typing import Optional

import depthai as dai
from box import Box

from nn.prompt_controller import PromptController
from prompting.frame_cache_node import FrameCacheNode
from prompting.encoders.textual_prompt_encoder import TextualPromptEncoder
from prompting.encoders.visual_prompt_encoder import VisualPromptEncoder
from prompting.services import (
    ClassUpdateService,
    ThresholdUpdateService,
    ImageUploadService,
    BBoxPromptService,
)


class PromptsNode(dai.node.ThreadedHostNode):
    """
    High-level node grouping the prompt-handling block.
    Handles prompt encoding (text + image/bbox) and exposes RemoteConnection services
    for updating prompts at runtime via FE.

    Internal structure:
        image_source
          -> FrameCacheNode (caches frames for bbox prompts)

    Components:
      - TextualPromptEncoder: encodes class names into text embeddings
      - VisualPromptEncoder: encodes image/bbox prompts
      - PromptController: pushes prompt tensors into NN input queues and updates label filtering

    Exposes:
        - ClassUpdateService: update detection classes via text prompts
        - ThresholdUpdateService: update confidence threshold
        - ImageUploadService: upload reference image prompt
        - BBoxPromptService: select a region prompt using a bounding box
    """

    def __init__(self) -> None:
        super().__init__()

        self._frame_cache: FrameCacheNode = self.createSubnode(FrameCacheNode)
        self._text_encoder: Optional[TextualPromptEncoder] = None
        self._visual_encoder: Optional[VisualPromptEncoder] = None
        self._services: list = []

    def build(
        self,
        image_source: dai.Node.Output,
        controller: PromptController,
        cfg: Box,
    ) -> "PromptsNode":
        """
        Build the prompts node.

        @param image_source: BGR frame output for caching.
        @param controller: PromptController from NNDetectionNode.
        @param cfg: Prompts configuration (paths, precision, class_names, etc.)
        """
        self._frame_cache.build(image_source)
        self._text_encoder = TextualPromptEncoder(cfg)
        self._visual_encoder = VisualPromptEncoder(cfg)
        self._services = [
            ClassUpdateService(controller, self._text_encoder, self._visual_encoder),
            ThresholdUpdateService(controller),
            ImageUploadService(controller, self._visual_encoder, self._text_encoder),
            BBoxPromptService(controller, self._visual_encoder, self._text_encoder, self._frame_cache),
        ]
        self._send_initial_prompts(controller, cfg)
        return self

    def _send_initial_prompts(self, controller: PromptController, cfg: Box) -> None:
        """
        Send initial class prompts to the model.
        """
        text_embeddings = self._text_encoder.extract_embeddings(cfg.class_names)
        dummy_image = self._visual_encoder.make_dummy()

        controller.apply_prompts(
            image_prompt=dummy_image,
            text_prompt=text_embeddings,
            class_names=cfg.class_names,
            offset=cfg.text_offset,
        )

        controller.set_confidence_threshold(cfg.detection_threshold)

    def register_services(self, visualizer: dai.RemoteConnection) -> None:
        """
        Register all prompt services with the visualizer.
        @param visualizer: RemoteConnection to register services with.
        """
        for service in self._services:
            visualizer.registerService(service.name, service.handle)

    def run(self) -> None:
        # High-level node: no host-side processing here. Processing happens in the composed subnodes.
        pass
