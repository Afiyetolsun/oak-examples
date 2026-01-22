from typing import TYPE_CHECKING

import depthai as dai
from box import Box

from prompting.frame_cache_node import FrameCacheNode
from prompting.encoders.textual_prompt_encoder import TextualPromptEncoder
from prompting.encoders.visual_prompt_encoder import VisualPromptEncoder
from prompting.services import (
    ClassUpdateService,
    ThresholdUpdateService,
    ImageUploadService,
    BBoxPromptService,
)

if TYPE_CHECKING:
    from nn.prompt_controller import PromptController


class PromptsNode(dai.node.ThreadedHostNode):
    """
    High-level node for prompt handling and encoding.

    Internal structure:
        image_source
          -> FrameCacheNode (caches frames for bbox prompts)

    Components:
        - TextualPromptEncoder: CLIP text encoder for class names
        - VisualPromptEncoder: SAM visual encoder for image/bbox prompts

    Services exposed:
        - ClassUpdateService: Update detection classes via text
        - ThresholdUpdateService: Adjust confidence threshold
        - ImageUploadService: Upload reference image
        - BBoxPromptService: Select region via bounding box

    Usage:
        prompts_node = pipeline.create(PromptsNode).build(...)
        prompts_node.register_services(visualizer)
    """

    def __init__(self) -> None:
        super().__init__()

        # Subnode
        self._frame_cache: FrameCacheNode = self.createSubnode(FrameCacheNode)

        # Encoders (not subnodes, just helper objects)
        self._text_encoder: TextualPromptEncoder = None
        self._visual_encoder: VisualPromptEncoder = None

        # Services
        self._services: list = []

    def build(
        self,
        image_source: dai.Node.Output,
        controller: "PromptController",
        cfg: Box,
    ) -> "PromptsNode":
        """
        Build the prompts node.

        @param image_source: BGR frame output for caching.
        @param controller: PromptController from NNDetectionNode.
        @param cfg: Prompts configuration (paths, precision, class_names, etc.)
        """
        # Build frame cache
        self._frame_cache.build(image_source)

        # Initialize encoders
        self._text_encoder = TextualPromptEncoder(cfg)
        self._visual_encoder = VisualPromptEncoder(cfg)

        # Create services
        self._services = [
            ClassUpdateService(controller, self._text_encoder),
            ThresholdUpdateService(controller),
            ImageUploadService(controller, self._visual_encoder),
            BBoxPromptService(controller, self._visual_encoder, self._frame_cache),
        ]

        # Send initial prompts
        self._send_initial_prompts(controller, cfg)

        return self

    def _send_initial_prompts(self, controller: "PromptController", cfg: Box) -> None:
        """
        Send initial class prompts to the model.

        This ensures the model is ready with default classes on startup.
        """
        # Encode initial classes
        text_embeddings = self._text_encoder.extract_embeddings(cfg.class_names)
        dummy_image = self._visual_encoder.make_dummy()

        # Send to model
        controller.apply_prompts(
            image_prompt=dummy_image,
            text_prompt=text_embeddings,
            class_names=cfg.class_names,
            offset=cfg.text_offset,
        )

        # Set initial threshold
        controller.set_confidence_threshold(cfg.detection_threshold)

    def register_services(self, visualizer: dai.RemoteConnection) -> None:
        """
        Register all prompt services with the visualizer.

        @param visualizer: RemoteConnection to register services with.
        """
        for service in self._services:
            visualizer.registerService(service.name, service.handle)

    def run(self) -> None:
        # High-level node: FrameCacheNode handles processing.
        pass
