from __future__ import annotations

import json
from typing import Any, Dict

from .domain import ChatRequest


class ValidationFailure(ValueError):
    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type


def validate_response(request: ChatRequest, payload: Dict[str, Any]) -> None:
    choices = payload.get("choices")
    if not choices or not isinstance(choices, list):
        raise ValidationFailure("provider returned no choices", "empty_response")
    message = choices[0].get("message") or {}
    content = message.get("content")
    tool_calls = message.get("tool_calls") or []
    if content in (None, "") and not tool_calls:
        raise ValidationFailure("provider returned an empty answer", "empty_response")

    options = request.route
    if request.response_format and (options is None or options.validate_json):
        try:
            json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValidationFailure("response is not valid JSON", "invalid_json") from exc

    for call in tool_calls:
        arguments = ((call.get("function") or {}).get("arguments"))
        try:
            json.loads(arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValidationFailure("tool arguments are not valid JSON", "invalid_tool_arguments") from exc
