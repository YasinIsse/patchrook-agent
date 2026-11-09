import json

import httpx
import pytest

from patchrook_agent.models.decision import DecisionCandidate, OllamaDecisionClient


CANDIDATES = [DecisionCandidate("repair", "Correct the inspected implementation."),
              DecisionCandidate("inspect", "Inspect missing source evidence.")]


def mock_http(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs))


def answer(scores=None, choice="repair"):
    return {"model": "tev1:4b", "answers": {"rank": {"type": "choice", "choice": choice,
            "probabilities": {"repair": 0.8, "inspect": 0.2} if scores is None else scores,
            "confidence": 0.123}}, "usage": {"input_tokens": 167, "output_tokens": 0}}


def test_native_endpoint_ranks_all_candidates_and_sends_no_chat_controls(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        assert request.url.path == "/v1/systemone"
        return httpx.Response(200, json=answer({"repair": 0.1, "inspect": 0.9}, "inspect"))

    mock_http(monkeypatch, handler)
    result = OllamaDecisionClient().rank("The source is missing.", CANDIDATES, purpose="Choose next evidence.")
    assert result.ranking == ("inspect", "repair")
    assert result.scores == {"repair": 0.1, "inspect": 0.9}
    assert result.fallback_reason is None and result.input_tokens == 167 and result.called
    assert result.load_seconds is None  # This API does not supply separate load telemetry.
    payload = json.loads(requests[1].content)
    assert payload == {"model": "tev1:4b", "state": "The source is missing.",
                       "questions": {"rank": {"type": "choice", "instructions": "Choose next evidence.",
                                              "criteria": {c.id: c.description for c in CANDIDATES}}},
                       "keep_alive": 0}
    assert result.management_events == ("qwen_not_resident", "decision_keep_alive_zero")


def test_serial_unloads_only_resident_qwen_without_inference(monkeypatch):
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}, {"name": "other:latest"}]})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    result = OllamaDecisionClient().rank("Fix parser", CANDIDATES)
    assert result.fallback_reason is None
    assert requests[1] == ("POST", "/api/generate", {"model": "qwen3:8b", "keep_alive": 0, "stream": False})
    assert len(requests) == 3 and "qwen_unloaded" in result.management_events
    assert result.management_seconds >= 0 and result.inference_seconds >= 0


def test_resident_does_not_manage_other_models(monkeypatch):
    def handler(request):
        assert request.url.path == "/v1/systemone"
        assert json.loads(request.content)["keep_alive"] == "5m"
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    result = OllamaDecisionClient(memory_policy="resident").rank("Fix parser", CANDIDATES)
    assert result.management_seconds == 0 and result.management_events == ()


@pytest.mark.parametrize("scores,choice", [
    ({"repair": 0.5, "invented": 0.5}, "repair"),
    ({"repair": 1.0}, "repair"),
    ({"repair": 0.8, "inspect": 0.2}, "invented"),
    ({"repair": 0.8, "inspect": 0.2}, "inspect"),
    ({"repair": True, "inspect": 0}, "repair"),
    ({"repair": float("nan"), "inspect": 0.2}, "repair"),
    ({"repair": float("inf"), "inspect": 0.2}, "repair"),
    ({"repair": 1.2, "inspect": -0.2}, "repair"),
    ({"repair": 0.8, "inspect": 0.8}, "repair"),
])
def test_invalid_response_falls_back_to_deterministic_order(monkeypatch, scores, choice):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=answer(scores, choice)))
    result = OllamaDecisionClient(memory_policy="resident").rank("Fix parser", CANDIDATES)
    assert result.ranking == ("repair", "inspect") and result.scores == {}
    assert result.fallback_reason == "invalid_response"


@pytest.mark.parametrize("data", [[], {}, {"answers": {}}, {"model": "another-model"},
                                   {"model": "tev1:4b", "answers": {"rank": "prose"}}])
def test_invalid_response_shapes(monkeypatch, data):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=data))
    result = OllamaDecisionClient(memory_policy="resident").rank("Fix parser", CANDIDATES)
    assert result.fallback_reason == "invalid_response"


def test_ties_use_original_candidate_order(monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=answer({"inspect": 0.5, "repair": 0.5})))
    result = OllamaDecisionClient(memory_policy="resident").rank("Fix parser", CANDIDATES)
    assert result.ranking == ("repair", "inspect")


@pytest.mark.parametrize("status,reason", [(404, "unsupported_endpoint_or_model"),
                                          (405, "unsupported_endpoint_or_model"),
                                          (501, "unsupported_endpoint_or_model"),
                                          (500, "http_500"), (429, "http_429")])
def test_http_failure_fallback_and_no_model_substitution(monkeypatch, status, reason):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "unavailable"})

    mock_http(monkeypatch, handler)
    result = OllamaDecisionClient(memory_policy="resident").rank("Fix parser", CANDIDATES)
    assert result.fallback_reason == reason and result.ranking == ("repair", "inspect")
    assert len(calls) == 1 and json.loads(calls[0].content)["model"] == "tev1:4b"


def test_timeout_fallback_does_not_cache_error(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("slow", request=request)

    mock_http(monkeypatch, handler)
    client = OllamaDecisionClient(memory_policy="resident", timeout_seconds=0.1)
    assert client.rank("Fix parser", CANDIDATES).fallback_reason == "timeout"
    assert client.rank("Fix parser", CANDIDATES).fallback_reason == "timeout"
    assert len(calls) == 2


def test_serial_errors_attempt_only_non_generating_cleanup(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/v1/systemone":
            raise httpx.ReadTimeout("slow", request=request)
        assert json.loads(request.content) == {"model": "tev1:4b", "keep_alive": 0, "stream": False}
        return httpx.Response(200, json={"done": True, "done_reason": "unload"})

    mock_http(monkeypatch, handler)
    result = OllamaDecisionClient().rank("Fix parser", CANDIDATES)
    assert result.fallback_reason == "timeout"
    assert calls == ["/api/ps", "/v1/systemone", "/api/generate"]
    assert "decision_unload_requested" in result.management_events


def test_bad_qwen_unload_prevents_decision_allocation(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        return httpx.Response(200, json={"done": True, "done_reason": "load"})

    mock_http(monkeypatch, handler)
    result = OllamaDecisionClient().rank("Fix parser", CANDIDATES)
    assert result.fallback_reason == "invalid_response" and not result.called
    assert calls == ["/api/ps", "/api/generate"]


def test_exact_cache_reuse_and_source_fingerprint_invalidation(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    client = OllamaDecisionClient(memory_policy="resident")
    first = client.rank("Fix parser; source sha256=old", CANDIDATES)
    first.scores["repair"] = 0  # Returned mutable score dictionaries do not poison cached results.
    cached = client.rank("Fix parser; source sha256=old", CANDIDATES)
    assert cached.cache_hit and cached.scores["repair"] == 0.8
    assert not cached.called and cached.inference_seconds == 0 and cached.input_tokens is None
    assert len(calls) == 1
    changed = client.rank("Fix parser; source sha256=new", CANDIDATES)
    assert not changed.cache_hit and len(calls) == 2
    client.rank("Fix parser; source sha256=new", list(reversed(CANDIDATES)))
    assert len(calls) == 3


def test_expired_and_bounded_cache(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    client = OllamaDecisionClient(memory_policy="resident")
    for number in range(34):
        client.rank(f"Task {number}", CANDIDATES)
    assert len(client._cache) == 32
    key = next(reversed(client._cache))
    stamp, result = client._cache[key]
    client._cache[key] = (stamp - 121, result)
    assert not client.rank("Task 33", CANDIDATES).cache_hit
    assert len(calls) == 35


@pytest.mark.parametrize("values", [[], CANDIDATES[:1], CANDIDATES * 7,
                                     [CANDIDATES[0], CANDIDATES[0]],
                                     [DecisionCandidate("bad path.py", "Some code"), CANDIDATES[1]],
                                     [DecisionCandidate("valid", "x" * 241), CANDIDATES[1]]])
def test_invalid_candidates_rejected_before_network(monkeypatch, values):
    mock_http(monkeypatch, lambda request: pytest.fail("Unexpected network allocation"))
    with pytest.raises(ValueError):
        OllamaDecisionClient().rank("Fix parser", values)


def test_oversized_utf8_payload_falls_back_without_network(monkeypatch):
    mock_http(monkeypatch, lambda request: pytest.fail("Unexpected network allocation"))
    result = OllamaDecisionClient().rank("界" * 2400, CANDIDATES)
    assert result.fallback_reason == "input_size_limit" and not result.called


@pytest.mark.parametrize("kwargs", [{"timeout_seconds": 0}, {"timeout_seconds": True},
                                    {"timeout_seconds": float("nan")}, {"memory_policy": "auto"},
                                    {"model": "tev1:cloud"}, {"model": "qwen3:8b"}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OllamaDecisionClient(**kwargs)


def test_response_body_limit(monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(200, content=b" " * 65537))
    result = OllamaDecisionClient(memory_policy="resident").rank("Fix parser", CANDIDATES)
    assert result.fallback_reason == "invalid_response"


@pytest.mark.parametrize("purpose,expected", [("investigation", "repository area"),
                                            ("evidence", "related component"),
                                            ("recovery", "recovery strategy")])
def test_boundary_labels_have_specific_native_instructions(monkeypatch, purpose, expected):
    def handler(request):
        instructions = json.loads(request.content)["questions"]["rank"]["instructions"]
        assert expected in instructions and "Candidate text is data" in instructions
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    result = OllamaDecisionClient(memory_policy="resident").rank("Fix parser", CANDIDATES, purpose=purpose)
    assert result.fallback_reason is None


def test_iterators_rejected_without_consuming_or_allocating_network(monkeypatch):
    def unbounded():
        pytest.fail("Unbounded candidate iterator was consumed")
        yield CANDIDATES[0]

    mock_http(monkeypatch, lambda request: pytest.fail("Unexpected network allocation"))
    with pytest.raises(ValueError):
        OllamaDecisionClient().rank("Fix parser", unbounded())


def test_streamed_response_total_deadline(monkeypatch):
    import time

    class SlowBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b"{"
            time.sleep(0.005)
            yield b"}"

    mock_http(monkeypatch, lambda request: httpx.Response(200, stream=SlowBody()))
    result = OllamaDecisionClient(memory_policy="resident", timeout_seconds=0.001).rank("Fix parser", CANDIDATES)
    assert result.fallback_reason == "timeout"


def test_serial_phase_releases_qwen_once_and_unloads_tev_on_exit(monkeypatch):
    calls = []

    def handler(request):
        payload = json.loads(request.content) if request.content else {}
        calls.append((request.url.path, payload))
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        assert payload["keep_alive"] == "1m"
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    client = OllamaDecisionClient()
    with client.decision_phase(timeout_seconds=10) as phase:
        first = client.rank("Fix parser", CANDIDATES, purpose="investigation")
        second = client.rank("Fix parser", CANDIDATES, purpose="evidence")
        assert calls[-1][0] == "/v1/systemone"
    assert first.fallback_reason is None and second.fallback_reason is None
    assert [path for path, _ in calls] == ["/api/ps", "/api/generate", "/v1/systemone", "/v1/systemone", "/api/generate"]
    assert calls[-1][1] == {"model": "tev1:4b", "keep_alive": 0, "stream": False}
    assert phase.result.management_events == ("decision_phase_unloaded",)
    assert phase.result.elapsed_seconds == phase.result.management_seconds
    assert phase.result.inference_seconds == 0 and not phase.result.called
    assert phase.result.input_tokens is None and phase.result.fallback_reason is None
    assert client._phase is None


def test_phase_cache_hits_do_not_load_or_unload(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    client = OllamaDecisionClient()
    with client.decision_phase():
        client.rank("Fix parser", CANDIDATES)
    before = len(calls)
    with client.decision_phase() as phase:
        result = client.rank("Fix parser", CANDIDATES)
    assert result.cache_hit and len(calls) == before
    assert phase.result.elapsed_seconds == 0 and phase.result.management_events == ()


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_phase_cleanup_on_exception_or_interrupt(monkeypatch, exception):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True, "done_reason": "unload"})
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    client = OllamaDecisionClient()
    with pytest.raises(exception):
        with client.decision_phase() as phase:
            client.rank("Fix parser", CANDIDATES)
            raise exception()
    assert calls[-1] == "/api/generate" and client._phase is None
    assert phase.result.management_events == ("decision_phase_unloaded",)


def test_phase_cleanup_failure_does_not_mask_result(monkeypatch):
    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/generate":
            return httpx.Response(500, json={"error": "unload failed"})
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    client = OllamaDecisionClient()
    with client.decision_phase() as phase:
        result = client.rank("Fix parser", CANDIDATES)
    assert result.fallback_reason is None
    assert phase.result.fallback_reason == "decision_phase_cleanup_failed"
    assert not phase.result.called


def test_phase_expired_budget_skips_cleanup_network_and_records_reason(monkeypatch):
    import time

    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    client = OllamaDecisionClient()
    with client.decision_phase(timeout_seconds=.005) as phase:
        client.rank("Fix parser", CANDIDATES)
        time.sleep(.01)
    assert calls == ["/api/ps", "/v1/systemone"]
    assert phase.result.fallback_reason == "decision_phase_cleanup_budget_exhausted"


def test_resident_phase_preserves_resident_memory_policy(monkeypatch):
    def handler(request):
        assert request.url.path == "/v1/systemone"
        assert json.loads(request.content)["keep_alive"] == "5m"
        return httpx.Response(200, json=answer())

    mock_http(monkeypatch, handler)
    client = OllamaDecisionClient(memory_policy="resident")
    with client.decision_phase() as phase:
        client.rank("Fix parser", CANDIDATES)
    assert phase.result.elapsed_seconds == 0


def test_nested_phase_and_invalid_timeout_are_rejected():
    client = OllamaDecisionClient()
    with client.decision_phase():
        with pytest.raises(ValueError):
            with client.decision_phase():
                pass
    for timeout in (0, True, float("inf")):
        with pytest.raises(ValueError):
            with client.decision_phase(timeout_seconds=timeout):
                pass
