
from dataclasses import dataclass
import json
import math
import time
from typing import Any

import httpx

from patchrook_agent.config import validate_sampling


@dataclass
class ModelResponse:
    message: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    prompt_duration_ns: int | None
    completion_duration_ns: int | None
    load_duration_ns: int | None
    total_duration_ns: int
    done_reason: str = "stop"
    prompt_eval_cached_count: int | None = None
    prompt_tokens_reported: bool = True
    completion_tokens_reported: bool = True

    def __post_init__(self):
        # Optional telemetry must not invalidate an otherwise usable response.
        # Explicit zero is measured; absent or malformed values remain unknown.
        self.prompt_tokens_reported = (self.prompt_tokens_reported is True
                                       and type(self.prompt_tokens) is int and self.prompt_tokens >= 0)
        self.completion_tokens_reported = (self.completion_tokens_reported is True
                                           and type(self.completion_tokens) is int and self.completion_tokens >= 0)
        for name in ("prompt_duration_ns", "completion_duration_ns", "load_duration_ns",
                     "prompt_eval_cached_count"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                setattr(self, name, None)
        if (self.prompt_eval_cached_count is not None
                and (not self.prompt_tokens_reported or self.prompt_eval_cached_count > self.prompt_tokens)):
            self.prompt_eval_cached_count = None


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        context_size: int = 8192,
        request_timeout_seconds: float = 1800.0,
        max_output_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float | None = 0.8,
        top_k: int | None = 20,
        min_p: float | None = 0.0,
    ):
        if (type(context_size) is not int or context_size <= 0 or context_size > 16384
                or type(max_output_tokens) is not int or not 0 < max_output_tokens < context_size):
            raise ValueError("Context must be at most 16384 and exceed the positive output token limit")
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise ValueError("Request timeout must be positive and finite")
        validate_sampling(temperature, top_p, top_k, min_p)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.context_size = context_size
        self.request_timeout_seconds = request_timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p

    @property
    def sampling_settings(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in ("temperature", "top_p", "top_k", "min_p")
                if getattr(self, name) is not None}

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        think: bool = False,
        timeout_seconds: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "think": think,
            "stream": False,
            "truncate": False,
            "shift": False,
            "options": {
                "num_ctx": self.context_size,
                **self.sampling_settings,
                "num_predict": self.max_output_tokens,
            },
        }

        if response_format is not None:
            if not isinstance(response_format, dict) or not response_format:
                raise ValueError("response_format must be a nonempty JSON schema")
            if tools:
                raise ValueError("Schema actions and native tool calling cannot be combined")
            payload["format"] = response_format
        elif tools is not None:
            payload["tools"] = tools

        timeout = self.request_timeout_seconds
        if timeout_seconds is not None:
            timeout = min(timeout, timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Request timeout must be positive and finite")
        # HTTPX's read timeout is per chunk. Also check elapsed time while
        # collecting the non-streaming JSON response to enforce a task budget.
        started = time.monotonic()
        body = bytearray()
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                for chunk in response.iter_bytes():
                    if time.monotonic() - started >= timeout:
                        raise httpx.ReadTimeout("Ollama request exceeded its time budget")
                    body.extend(chunk)
                    if len(body) > 4_194_304:
                        raise ValueError("Ollama response exceeded the size limit")
                if not response.is_success:
                    # Attach the bounded error body to the exception so the
                    # controller can identify a server-side context rejection.
                    httpx.Response(response.status_code, headers=response.headers,
                                   content=bytes(body), request=response.request).raise_for_status()
        data = json.loads(body)
        prompt_reported = type(data.get("prompt_eval_count")) is int and data["prompt_eval_count"] >= 0
        completion_reported = type(data.get("eval_count")) is int and data["eval_count"] >= 0

        return ModelResponse(
            message=data["message"],
            prompt_tokens=data["prompt_eval_count"] if prompt_reported else 0,
            completion_tokens=data["eval_count"] if completion_reported else 0,
            prompt_duration_ns=data.get("prompt_eval_duration"),
            completion_duration_ns=data.get("eval_duration"),
            load_duration_ns=data.get("load_duration"),
            total_duration_ns=data.get("total_duration", 0),
            done_reason=data.get("done_reason", "stop"),
            prompt_eval_cached_count=data.get("prompt_eval_cached_count"),
            prompt_tokens_reported=prompt_reported,
            completion_tokens_reported=completion_reported,
        )
