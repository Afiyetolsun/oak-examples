from base_service import BaseService
from qr_scan.host_qr_scanner import QRScanner
from tiling.dynamic_tiling import DynamicTiling


class CurrentParamsService(BaseService[None]):
    NAME = "Get Current Params Service"
    PAYLOAD_MODEL = None

    def __init__(
        self,
        dynamic_tiling: DynamicTiling,
        qr_scanner: QRScanner,
    ):
        self._dynamic_tiling = dynamic_tiling
        self._scanner = qr_scanner

    def handle_typed(self, payload: None = None) -> dict:
        return {
            "tiling": self._dynamic_tiling.current_params,
            "scanner": self._scanner.decode_enabled,
        }
