from __future__ import annotations

import json
import re
from typing import Any

from .domain import ChatRequest, RequestFeatures, RouteOptions

COMPLEX_MARKERS = re.compile(
    r"\b(debug|root cause|prove|reason|architecture|race condition|distributed|optimi[sz]e|"
    r"trade[- ]?off|investigate|multi[- ]?step|analy[sz]e)\b",
    re.IGNORECASE,
)
SIMPLE_MARKERS = re.compile(r"\b(rewrite|classify|translate|format|shorten|extract)\b", re.IGNORECASE)
CODE_MARKERS = re.compile(r"```|\b(traceback|exception|function|class |SQL|Python|JavaScript)\b", re.I)
CLASSIFICATION_MARKERS = re.compile(
    r"\b(classify|categori[sz]e|sentiment|which (?:category|label)|label (?:this|the|it))\b", re.I
)
MATH_MARKERS = re.compile(
    r"\b(calculate|compute|how (?:many|much)|total|sum|product|percent(?:age)?|average|solve|"
    r"remain(?:s|ing)?|left over)\b",
    re.I,
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def analyze(request: ChatRequest) -> RequestFeatures:
    options = request.route or RouteOptions()
    text = "\n".join(_content_text(message.content) for message in request.messages)
    # A conservative byte/4 estimate is deterministic and avoids a tokenizer dependency.
    input_tokens = max(1, (len(text.encode("utf-8")) + 3) // 4)
    output_tokens = request.max_tokens or min(2048, max(128, input_tokens // 2))
    operations = len(re.findall(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+", text))
    complexity = 0.25
    complexity += min(0.25, input_tokens / 12000)
    complexity += min(0.20, len(COMPLEX_MARKERS.findall(text)) * 0.06)
    complexity += min(0.15, operations * 0.025)
    complexity += 0.10 if CODE_MARKERS.search(text) else 0
    complexity -= 0.10 if SIMPLE_MARKERS.search(text) and input_tokens < 1000 else 0
    complexity = options.complexity if options.complexity is not None else complexity

    task_type = options.task_type or "general"
    if options.task_type is None:
        if request.tools:
            task_type = "tool_calling"
        elif CODE_MARKERS.search(text):
            task_type = "coding"
        elif re.search(r"\b(json|schema|fields?|extract)\b", text, re.I):
            task_type = "extraction"
        elif CLASSIFICATION_MARKERS.search(text):
            task_type = "classification"
        elif MATH_MARKERS.search(text) and re.search(r"\d", text):
            task_type = "math"
        elif re.search(r"\b(rewrite|edit|format|summari[sz]e)\b", text, re.I):
            task_type = "transformation"

    requires_json = bool(request.response_format)
    requires_vision = any(
        isinstance(message.content, list)
        and any(isinstance(part, dict) and part.get("type") in {"image_url", "input_image"} for part in message.content)
        for message in request.messages
    )
    return RequestFeatures(
        estimated_input_tokens=input_tokens,
        expected_output_tokens=output_tokens,
        required_tokens=input_tokens + output_tokens,
        task_type=task_type,
        complexity=round(max(0, min(1, complexity)), 3),
        requires_tools=bool(request.tools),
        requires_json=requires_json,
        requires_vision=requires_vision,
        privacy=options.privacy,
    )
