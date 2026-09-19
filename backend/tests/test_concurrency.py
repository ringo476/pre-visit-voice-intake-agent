"""Proves the get_llm() cache race is actually closed, not just guarded by
code that looks right. Launches many real OS threads at the same protocol
simultaneously — the exact scenario (multiple patients' first turns for the
same protocol landing on different asyncio.to_thread threads at once) that
previously let two threads both build a separate client and one silently
overwrite the other's wasted work."""

import os
import threading
import time

import app.agent.graph as graph_module
from app.protocol.registry import get_protocol


class _SlowCountingFakeModel:
    """Stands in for ChatGoogleGenerativeAI. Sleeps during construction to
    deliberately widen the race window a real network-backed client
    constructor would also occupy, and counts how many times it was ever
    actually built."""

    build_count = 0
    lock = threading.Lock()

    def __init__(self, model, google_api_key):
        time.sleep(0.05)
        with _SlowCountingFakeModel.lock:
            _SlowCountingFakeModel.build_count += 1

    def bind_tools(self, tools):
        return self


def test_concurrent_first_calls_for_the_same_protocol_build_the_client_exactly_once(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", _SlowCountingFakeModel)
    graph_module._cached_llms.clear()
    _SlowCountingFakeModel.build_count = 0

    protocol = get_protocol("respiratory-intake")
    results = []

    def worker():
        results.append(graph_module.get_llm(protocol))

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _SlowCountingFakeModel.build_count == 1, "the client should be constructed exactly once, not once per racing thread"
    assert len({id(r) for r in results}) == 1, "every thread must get back the exact same cached client instance"


def test_different_protocols_still_get_independently_cached_clients(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", _SlowCountingFakeModel)
    graph_module._cached_llms.clear()
    _SlowCountingFakeModel.build_count = 0

    a = graph_module.get_llm(get_protocol("respiratory-intake"))
    b = graph_module.get_llm(get_protocol("musculoskeletal-leg-injury"))

    assert _SlowCountingFakeModel.build_count == 2
    assert a is not b
