from pydantic import BaseModel

from base_service import BaseService
from qr_scan.host_qr_scanner import QRScanner


class QRConfigPayload(BaseModel):
    state: bool = False


class QRConfigService(BaseService[QRConfigPayload]):
    NAME = "QR Config Service"
    PAYLOAD_MODEL = QRConfigPayload

    def __init__(self, scanner: QRScanner):
        self._scanner = scanner

    def handle_typed(self, payload: QRConfigPayload) -> dict:
        print(payload.state)
        self._scanner.set_decode(payload.state)
        print(self._scanner.decode_enabled)
        return {"ok": True}
