"""
Frontend prompt services - handles all prompt-related API requests.

All services follow the same pattern:
1. Validate payload with Pydantic
2. Process with encoder (extract embeddings)
3. Send prompts to controller
4. Return response
"""
import base64
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from prompts.encoders.textual_encoder import TextualEncoder
    from prompts.encoders.visual_encoder import VisualEncoder
    from prompts.frame_cache_node import FrameCacheNode
    from nn.prompt_controller import PromptController


# =============================================================================
# Service Names
# =============================================================================

class ServiceName(str, Enum):
    """Unique identifiers for prompt services."""
    CLASS_UPDATE = "Class Update Service"
    THRESHOLD_UPDATE = "Threshold Update Service"
    IMAGE_UPLOAD = "Image Upload Service"
    BBOX_PROMPT = "BBox Prompt Service"


# =============================================================================
# Payloads (Pydantic models for validation)
# =============================================================================

class ClassUpdatePayload(BaseModel):
    """Payload for updating detection classes."""
    classes: list[str] = Field(..., min_length=1, description="List of class names")


class ThresholdUpdatePayload(BaseModel):
    """Payload for updating NN confidence threshold."""
    threshold: float = Field(..., ge=0.0, le=1.0)


class ImageUploadPayload(BaseModel):
    """Payload for uploading an image from the frontend."""
    filename: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    data: str = Field(..., description="Base64-encoded image data")


class BBoxPromptPayload(BaseModel):
    """Payload for bounding box region selection."""
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate [0-1]")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate [0-1]")
    width: float = Field(..., gt=0.0, le=1.0, description="Normalized width [0-1]")
    height: float = Field(..., gt=0.0, le=1.0, description="Normalized height [0-1]")


# =============================================================================
# Services
# =============================================================================

class ClassUpdateService:
    """
    Handles text-based class updates.

    Takes a list of class names, encodes them using CLIP text encoder,
    and sends the embeddings to the detection model.
    """

    name = ServiceName.CLASS_UPDATE

    def __init__(self, controller: "PromptController", encoder: "TextualEncoder"):
        self._controller = controller
        self._encoder = encoder
        self._class_names: list[str] = []

    def handle(self, payload: dict) -> dict:
        """Process class update request."""
        try:
            validated = ClassUpdatePayload.model_validate(payload)
        except ValidationError as e:
            return {"ok": False, "error": e.errors()}

        self._class_names = validated.classes

        # Encode text prompts
        text_embeddings = self._encoder.extract_embeddings(self._class_names)
        dummy = self._encoder.make_dummy()

        # Send to model (text goes as text_prompt, dummy as image_prompt)
        self._controller.apply_prompts(
            image_prompt=dummy,
            text_prompt=text_embeddings,
            class_names=self._class_names,
            offset=self._encoder.offset,
        )

        return {"ok": True, "classes": self._class_names}


class ThresholdUpdateService:
    """
    Handles confidence threshold updates.

    Simple service that adjusts the detection confidence threshold.
    """

    name = ServiceName.THRESHOLD_UPDATE

    def __init__(self, controller: "PromptController"):
        self._controller = controller

    def handle(self, payload: dict) -> dict:
        """Process threshold update request."""
        try:
            validated = ThresholdUpdatePayload.model_validate(payload)
        except ValidationError as e:
            return {"ok": False, "error": e.errors()}

        clamped = max(0.0, min(1.0, validated.threshold))
        self._controller.set_confidence_threshold(clamped)

        return {"ok": True, "threshold": clamped}


class ImageUploadService:
    """
    Handles image upload for visual prompting.

    Takes a base64-encoded image, extracts visual embeddings using SAM encoder,
    and sends them to the detection model.
    """

    name = ServiceName.IMAGE_UPLOAD

    def __init__(self, controller: "PromptController", encoder: "VisualEncoder"):
        self._controller = controller
        self._encoder = encoder
        self._class_names: list[str] = []

    def handle(self, payload: dict) -> dict:
        """Process image upload request."""
        try:
            validated = ImageUploadPayload.model_validate(payload)
        except ValidationError as e:
            return {"ok": False, "error": e.errors()}

        # Use filename (without extension) as class name
        self._class_names = [validated.filename.split(".")[0]]

        # Decode image
        image = self._decode_image(validated.data)

        # Extract visual embeddings
        image_embeddings = self._encoder.extract_embeddings(image)
        dummy = self._encoder.make_dummy()

        # Send to model (image goes as image_prompt, dummy as text_prompt)
        self._controller.apply_prompts(
            image_prompt=image_embeddings,
            text_prompt=dummy,
            class_names=self._class_names,
            offset=self._encoder.offset,
        )

        return {"ok": True, "class": self._class_names}

    def _decode_image(self, data_uri: str) -> np.ndarray:
        """Convert base64-encoded image to OpenCV array."""
        if "," in data_uri:
            _, base64_data = data_uri.split(",", 1)
        else:
            base64_data = data_uri
        np_arr = np.frombuffer(base64.b64decode(base64_data), np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


class BBoxPromptService:
    """
    Handles bounding box region selection for visual prompting.

    Takes normalized bbox coordinates, creates a mask from the current frame,
    extracts visual embeddings for that region, and sends them to the model.
    """

    name = ServiceName.BBOX_PROMPT

    def __init__(
        self,
        controller: "PromptController",
        encoder: "VisualEncoder",
        frame_cache: "FrameCacheNode",
    ):
        self._controller = controller
        self._encoder = encoder
        self._frame_cache = frame_cache
        self._class_names: list[str] = ["Bounding Box Object"]

    def handle(self, payload: dict) -> dict:
        """Process bbox prompt request."""
        try:
            validated = BBoxPromptPayload.model_validate(payload)
        except ValidationError as e:
            return {"ok": False, "error": e.errors()}

        # Get current frame
        image = self._frame_cache.get_last_frame()
        if image is None:
            return {"ok": False, "error": "No frame available"}

        # Create mask from bbox
        mask = self._make_mask(image, validated)

        # Extract visual embeddings with mask
        image_embeddings = self._encoder.extract_embeddings(image, mask)
        dummy = self._encoder.make_dummy()

        # Send to model
        self._controller.apply_prompts(
            image_prompt=image_embeddings,
            text_prompt=dummy,
            class_names=self._class_names,
            offset=self._encoder.offset,
        )

        return {"ok": True, "classes": self._class_names}

    def _make_mask(self, image: np.ndarray, bbox: BBoxPromptPayload) -> np.ndarray:
        """Build a binary mask corresponding to the provided bounding box."""
        H, W = image.shape[:2]

        x0 = int(bbox.x * W)
        y0 = int(bbox.y * H)
        x1 = int((bbox.x + bbox.width) * W)
        y1 = int((bbox.y + bbox.height) * H)

        # Ensure proper ordering
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid bbox coordinates: {(x0, y0, x1, y1)}")

        mask = np.zeros((H, W), dtype=np.float32)
        mask[y0:y1, x0:x1] = 1.0
        return mask
