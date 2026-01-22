from typing import Any, Dict

from nn.prompt_controller import ModelState
from snapping.conditions import Condition, ConditionKey


class ExportService:
    """
    Read-only service that exports current configuration state to the frontend.

    Aggregates state from:
    - ModelState (classes, confidence threshold)
    - Conditions (snapping configuration)
    """

    name = "Export Service"

    def __init__(
        self,
        model_state: ModelState,
        conditions: Dict[ConditionKey, Condition],
    ):
        self._model_state = model_state
        self._conditions = conditions

    def _export_conditions_config(self) -> Dict[str, Dict]:
        """Export current configuration of all conditions."""
        configs: Dict[str, Dict] = {}

        for key, condition in self._conditions.items():
            cfg = condition.export_config()
            # Convert cooldown from seconds to minutes for frontend
            if "cooldown" in cfg:
                cfg["cooldown"] = round(cfg["cooldown"] / 60.0, 1)
            configs[key.value] = cfg

        return configs

    def handle(self, payload: Any = None) -> Dict[str, Any]:
        """Return current configuration state."""
        return {
            "classes": self._model_state.current_classes,
            "confidence_threshold": self._model_state.confidence_threshold,
            "snapping": {
                "running": any(cond.enabled for cond in self._conditions.values()),
                **self._export_conditions_config(),
            },
        }
