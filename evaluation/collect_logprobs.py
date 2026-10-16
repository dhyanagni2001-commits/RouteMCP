#!/usr/bin/env python3
"""Answer every task with each model and record token log-probabilities.

Used to choose the cascade's default `cascade_min_logprob`. Reads OPENAI_API_KEY
from the environment. Usage:

    python evaluation/collect_logprobs.py evaluation/results/openai-logprobs.json \
        gpt-4.1-nano,gpt-4.1-mini,gpt-4.1
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict

import httpx

from routemcp.evaluation import EvaluationCase, validate_answer

MODELS = {
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
}
PRICES = {"gpt-4.1-nano": (0.10, 0.40), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.0, 8.0)}
FIELDS = ("model", "messages", "tools", "tool_choice", "response_format", "max_tokens", "temperature")


async def answer(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, case: EvaluationCase, model: str
) -> Dict[str, Any]:
    request = case.request(MODELS[model]).model_dump(exclude_none=True)
    body = {key: value for key, value in request.items() if key in FIELDS}
    body["logprobs"] = True
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    async with semaphore:
        for attempt in range(5):
            response = await client.post(
                "https://api.openai.com/v1/chat/completions", json=body, headers=headers
            )
            if response.status_code != 429 and response.status_code < 500:
                break
            await asyncio.sleep(2**attempt)
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    content = choice["message"].get("content") or ""
    logprobs = [token["logprob"] for token in (choice.get("logprobs") or {}).get("content") or []]
    usage = payload["usage"]
    input_price, output_price = PRICES[model]
    return {
        "case_id": case.id,
        "task_type": case.task_type,
        "model": model,
        "passed": validate_answer(content, case.expected, payload),
        "cost_usd": (usage["prompt_tokens"] * input_price + usage["completion_tokens"] * output_price)
        / 1_000_000,
        "min_logprob": min(logprobs) if logprobs else None,
        "mean_logprob": sum(logprobs) / len(logprobs) if logprobs else None,
        "tool_call": bool(choice["message"].get("tool_calls")),
        "answer": content[:300],
    }


async def main(output: str, models: str) -> None:
    with open("evaluation/tasks.jsonl", encoding="utf-8") as handle:
        cases = [EvaluationCase.model_validate(json.loads(line)) for line in handle]
    semaphore = asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=60) as client:
        rows = await asyncio.gather(
            *[answer(client, semaphore, case, model) for case in cases for model in models.split(",")]
        )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1)
    print(f"{len(rows)} rows; cost ${sum(row['cost_usd'] for row in rows):.4f}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
