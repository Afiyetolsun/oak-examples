import depthai as dai
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode

from depthai_nodes.node import BaseHostNode
from qr_scan.qr_detections import QRDetection, QRDetections


class QRScanner(BaseHostNode):
    """Decodes QR codes from detected bounding boxes."""

    def __init__(self) -> None:
        super().__init__()
        self._decode_enabled = True
        self._last_decoded: str = ""

        self._out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )

    def build(
        self,
        preview: dai.Node.Output,
        detections: dai.Node.Output,
        decode_enabled: bool = True,
    ) -> "QRScanner":
        self.link_args(preview, detections)
        self._decode_enabled = decode_enabled

        self.inputs["preview"].setBlocking(False)
        self.inputs["preview"].setMaxSize(2)
        self.inputs["detections"].setBlocking(False)
        self.inputs["detections"].setMaxSize(2)

        return self

    def process(self, preview: dai.ImgFrame, detections: dai.Buffer) -> None:
        frame = preview.getCvFrame()
        assert isinstance(detections, dai.ImgDetections)

        qr_detections = QRDetections()
        for det in detections.detections:
            qr_detection = QRDetection()
            qr_detection.confidence = det.confidence
            qr_detection.xmin = det.xmin
            qr_detection.xmax = det.xmax
            qr_detection.ymin = det.ymin
            qr_detection.ymax = det.ymax

            if self._decode_enabled:
                bbox = self._denormalize_bbox(frame, det)
                decoded_text = self._decode_qr(frame, bbox)
                if decoded_text:
                    qr_detection.label = decoded_text

            qr_detections.detections.append(qr_detection)

        qr_detections.setSequenceNum(detections.getSequenceNum())
        qr_detections.setTimestamp(detections.getTimestamp())

        self._out.send(qr_detections)

    def _decode_qr(self, frame: np.ndarray, bbox: np.ndarray) -> str:
        """Decode QR code in the given bounding box."""
        if bbox[1] >= bbox[3] or bbox[0] >= bbox[2]:
            return ""

        bbox = self._expand_bbox(bbox, frame, percentage=5)
        img = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]

        data = pyzbar_decode(img)
        if data:
            text = data[0].data.decode("utf-8")
            if text != self._last_decoded:
                print(f"Decoded QR: {text}")
                self._last_decoded = text
            return text
        return ""

    @staticmethod
    def _denormalize_bbox(frame: np.ndarray, det) -> np.ndarray:
        """Convert normalized detection bbox to pixel coordinates."""
        bbox = (det.xmin, det.ymin, det.xmax, det.ymax)
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    @staticmethod
    def _expand_bbox(
        bbox: np.ndarray, frame: np.ndarray, percentage: float
    ) -> np.ndarray:
        """Expand the bounding box by a percentage."""
        bbox = bbox.copy()
        h_expand = (bbox[3] - bbox[1]) * (percentage / 100)
        w_expand = (bbox[2] - bbox[0]) * (percentage / 100)
        bbox[0] = max(0, bbox[0] - w_expand)
        bbox[1] = max(0, bbox[1] - h_expand)
        bbox[2] = min(frame.shape[1], bbox[2] + w_expand)
        bbox[3] = min(frame.shape[0], bbox[3] + h_expand)
        return bbox.astype(int)

    @property
    def out(self) -> dai.Node.Output:
        return self._out
