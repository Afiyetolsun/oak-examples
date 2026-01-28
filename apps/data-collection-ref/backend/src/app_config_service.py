from typing import Any, Dict, Callable
from nn.prompt_controller import ModelState


class GetAppConfigService:
    """
    Read-only service that exports current configuration state to the frontend.

    Aggregates state from:
    - ModelState (classes, confidence threshold)
    - get_snapping_config() callable (snapping configuration export)
    """
    def __init__(
        self,
        model_state: ModelState,
        get_snap_conditions_config: Callable[[], Dict[str, Any]],
    ):
        self._model_state = model_state
        self._get_snap_conditions_config = get_snap_conditions_config

    def handle(self, payload: Any = None) -> Dict[str, Any]:
        return {
            "ok": True,
            "data": {
                "classes": list(self._model_state.current_classes),
                "confidence_threshold": float(self._model_state.confidence_threshold),
                "snapping": self._get_snap_conditions_config(),
            },
        }
