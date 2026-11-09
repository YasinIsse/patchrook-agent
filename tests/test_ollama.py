import json

import httpx
import pytest

from patchrook_agent.models.ollama import OllamaClient


def transport_client(monkeypatch, handler):
    client_type = httpx.Client
    settings = []

    def create_client(**kwargs):
        settings.append(kwargs)
        return client_type(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", create_client)
    return settings


def test_chat_keeps_tool_history_thinking_and_metrics(monkeypatch):
    captured = []
    message = {"role": "assistant", "content": "Finished", "thinking": "checked"}

    def handler(request):
        captured.append(json.loads(request.content))
        assert str(request.url) == "http://localhost:11434/api/chat"
        return httpx.Response(200, json={
            "message": message, "prompt_eval_count": 123, "eval_count": 45,
            "prompt_eval_duration": 1, "eval_duration": 2, "load_duration": 3,
            "total_duration": 4, "done_reason": "length",
        })

    settings = transport_client(monkeypatch, handler)
    history = [{"role": "assistant", "content": "", "tool_calls": [{"function": {
        "name": "list_files", "arguments": {},
    }}]}, {"role": "tool", "tool_name": "list_files", "content": "[]"}]
    tools = [{"type": "function", "function": {"name": "list_files"}}]
    result = OllamaClient(base_url="http://localhost:11434/").chat(
        history, tools=tools, think=True, timeout_seconds=2,
    )
    assert captured[0]["messages"] == history
    assert captured[0]["tools"] == tools
    assert captured[0]["think"] is True and captured[0]["stream"] is False
    assert captured[0]["truncate"] is False and captured[0]["shift"] is False
    assert captured[0]["model"] == "qwen3:8b"
    assert captured[0]["options"] == {"num_ctx": 8192, "temperature": 0.7, "num_predict": 512,
                                        "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    assert settings[0]["timeout"] == 2
    assert result.message == message and result.done_reason == "length"
    assert result.prompt_tokens == 123 and result.completion_tokens == 45
    assert result.prompt_duration_ns == 1 and result.completion_duration_ns == 2
    assert result.load_duration_ns == 3 and result.total_duration_ns == 4


def test_minimal_response_defaults_usage(monkeypatch):
    transport_client(monkeypatch, lambda request: httpx.Response(200, json={
        "message": {"role": "assistant", "content": "Hello"},
    }))
    response = OllamaClient().chat([{"role": "user", "content": "Hello"}])
    assert response.prompt_tokens == 0 and response.total_duration_ns == 0
    assert response.prompt_duration_ns is None and response.completion_duration_ns is None
    assert response.load_duration_ns is None and response.prompt_eval_cached_count is None
    assert not response.prompt_tokens_reported and not response.completion_tokens_reported


def test_custom_sampling_and_legacy_unspecified_options_reach_ollama(monkeypatch):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Done"}})

    transport_client(monkeypatch, handler)
    OllamaClient(temperature=0.4, top_p=0.95, top_k=12, min_p=0.05).chat([])
    assert captured[-1]["options"] == {"num_ctx": 8192, "num_predict": 512,
                                       "temperature": 0.4, "top_p": 0.95, "top_k": 12, "min_p": 0.05}
    OllamaClient(temperature=0, top_p=None, top_k=None, min_p=None).chat([])
    assert captured[-1]["options"] == {"num_ctx": 8192, "num_predict": 512, "temperature": 0}


@pytest.mark.parametrize("status", [404, 500])
def test_http_errors_propagate_for_controller_to_handle(monkeypatch, status):
    transport_client(monkeypatch, lambda request: httpx.Response(status, json={"error": "unavailable"}))
    with pytest.raises(httpx.HTTPStatusError):
        OllamaClient().chat([])


def test_timeout_propagates_for_controller_to_handle(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    transport_client(monkeypatch, handler)
    with pytest.raises(httpx.ReadTimeout):
        OllamaClient().chat([], timeout_seconds=0.1)


def test_streamed_context_error_reaches_controller_without_crashing(monkeypatch, tmp_path):
    from patchrook_agent.agent import Agent

    class ErrorBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"error":"input exceeds context length"}'

    transport_client(monkeypatch, lambda request: httpx.Response(400, stream=ErrorBody()))
    state = Agent(OllamaClient(), workspace=tmp_path).run("Inspect the repository")
    assert state.status == "incomplete" and state.stop_reason == "context_limit"


@pytest.mark.parametrize("kwargs", [
    {"context_size": 0}, {"max_output_tokens": 0}, {"request_timeout_seconds": float("nan")},
    {"temperature": -0.1}, {"temperature": float("nan")}, {"temperature": True},
    {"top_p": 1.1}, {"top_p": float("inf")}, {"top_p": False},
    {"top_k": -1}, {"top_k": 2.5}, {"top_k": True},
    {"min_p": -0.1}, {"min_p": float("nan")}, {"min_p": "0"},
])
def test_invalid_model_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        OllamaClient(**kwargs)
