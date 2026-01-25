import base64
import cv2
import numpy as np
from pydantic import ValidationError

from nn.prompt_controller import PromptController
from prompting.encoders.textual_prompt_encoder import TextualPromptEncoder
from prompting.encoders.visual_prompt_encoder import VisualPromptEncoder
from prompting.frame_cache_node import FrameCacheNode
from prompting.payloads import (
    ClassUpdatePayload,
    ThresholdUpdatePayload,
    ImageUploadPayload,
    BBoxPromptPayload,
)


def _decode_image(data_uri: str) -> np.ndarray | None:
    """Decode a base64 image (supports raw base64 or data URI)."""
    try:
        base64_data = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
        np_arr = np.frombuffer(base64.b64decode(base64_data), np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _make_bbox_mask(image: np.ndarray, bbox: BBoxPromptPayload) -> np.ndarray | None:
    """Create a float mask in [0,1] for the bbox."""
    H, W = image.shape[:2]

    x0 = int(bbox.x * W)
    y0 = int(bbox.y * H)
    x1 = int((bbox.x + bbox.width) * W)
    y1 = int((bbox.y + bbox.height) * H)

    x0 = max(0, min(W, x0))
    x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0))
    y1 = max(0, min(H, y1))

    if x1 <= x0 or y1 <= y0:
        return None

    mask = np.zeros((H, W), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


class ClassUpdateService:
    """Update detection classes via text prompts."""

    name = "Class Update Service"

    def __init__(
        self,
        controller: PromptController,
        text_encoder: TextualPromptEncoder,
        visual_encoder: VisualPromptEncoder,
    ):
        self._controller = controller
        self._text = text_encoder
        self._visual = visual_encoder

    def handle(self, payload: dict) -> dict:
        try:
            validated = ClassUpdatePayload.model_validate(payload)
        except ValidationError as e:
            return {"ok": False, "error": e.errors()}

        classes = validated.classes
        text_embeddings = self._text.extract_embeddings(classes)
        dummy_image = self._visual.make_dummy()

        self._controller.apply_prompts(
            image_prompt=dummy_image,
            text_prompt=text_embeddings,
            class_names=classes,
            offset=self._text.offset,
        )

        return {"ok": True, "classes": classes}


class ThresholdUpdateService:
    """Update detection confidence threshold."""

    name = "Threshold Update Service"

    def __init__(self, controller: PromptController):
        self._controller = controller

    def handle(self, payload: dict) -> dict:
        try:
            validated = ThresholdUpdatePayload.model_validate(payload)
        except ValidationError as e:
            return {"ok": False, "error": e.errors()}

        self._controller.set_confidence_threshold(validated.threshold)
        return {"ok": True, "threshold": float(validated.threshold)}


class ImageUploadService:
    """Upload reference image for visual prompting."""

    name = "Image Upload Service"

    def __init__(
        self,
        controller: PromptController,
        visual_encoder: VisualPromptEncoder,
        text_encoder: TextualPromptEncoder,
    ):
        self._controller = controller
        self._visual = visual_encoder
        self._text = text_encoder

    def handle(self, payload: dict) -> dict:
        try:
            validated = ImageUploadPayload.model_validate(payload)
        except ValidationError as e:
            return {"ok": False, "error": e.errors()}

        class_name = validated.filename.rsplit(".", 1)[0]

        image = _decode_image(validated.data)
        if image is None:
            return {"ok": False, "error": "Invalid image data"}

        image_embeddings = self._visual.extract_embeddings(image)
        dummy_text = self._text.make_dummy()

        self._controller.apply_prompts(
            image_prompt=image_embeddings,
            text_prompt=dummy_text,
            class_names=[class_name],
            offset=self._visual.offset,
        )

        return {"ok": True, "classes": class_name}


class BBoxPromptService:
    """Select region via bounding box for visual prompting."""

    name = "BBox Prompt Service"

    def __init__(
        self,
        controller: PromptController,
        visual_encoder: VisualPromptEncoder,
        text_encoder: TextualPromptEncoder,
        frame_cache: FrameCacheNode,
    ):
        self._controller = controller
        self._visual = visual_encoder
        self._text = text_encoder
        self._frame_cache = frame_cache

    def handle(self, payload: dict) -> dict:
        try:
            validated = BBoxPromptPayload.model_validate(payload)
        except ValidationError as e:
            return {"ok": False, "error": e.errors()}

        image = self._frame_cache.get_last_frame()
        if image is None:
            return {"ok": False, "error": "No frame available"}

        mask = _make_bbox_mask(image, validated)
        if mask is None:
            return {"ok": False, "error": "Invalid bbox"}

        image_embeddings = self._visual.extract_embeddings(image, mask)
        dummy_text = self._text.make_dummy()

        class_names = ["Selected Region"]
        self._controller.apply_prompts(
            image_prompt=image_embeddings,
            text_prompt=dummy_text,
            class_names=class_names,
            offset=self._visual.offset,
        )

        return {"ok": True, "classes": class_names}
