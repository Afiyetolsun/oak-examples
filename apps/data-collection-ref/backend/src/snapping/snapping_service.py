from typing import Dict, Any

from pydantic import RootModel, ValidationError

from snapping.conditions import Condition, ConditionKey, ConditionConfig


class SnapPayload(RootModel[Dict[ConditionKey, ConditionConfig]]):
    """Payload for updating multiple conditions at once."""
    pass


class SnappingService:
    """
    Handles updates to snapping conditions from the frontend.

    Receives configuration updates and applies them to the conditions.
    """

    name = "Snap Collection Service"

    def __init__(self, conditions: Dict[ConditionKey, Condition]):
        self._conditions = conditions

    def handle(self, payload: dict) -> Dict[str, Any]:
        """Process snapping configuration update."""
        try:
            validated = SnapPayload.model_validate(payload)
        except ValidationError as e:
            return {"ok": False, "error": e.errors()}

        any_active = False
        for key, params in validated.root.items():
            condition = self._conditions.get(key)
            if condition is None:
                continue
            condition.apply_config(params)
            any_active = any_active or condition.enabled

        return {"ok": True, "active": any_active}
