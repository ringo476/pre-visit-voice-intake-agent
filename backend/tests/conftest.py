import shutil
from pathlib import Path

import pytest

_SESSIONS_DIR = Path(__file__).parent.parent / ".chroma_data" / "sessions"


def pytest_configure(config):
    """Session-scoped Chroma stores created by tests use random UUIDs and
    are never cleaned up by test code (unlike main.py's real WS lifecycle,
    which calls clear_session on disconnect) — wipe them once per test run
    so repeated runs don't accumulate disk usage indefinitely."""
    shutil.rmtree(_SESSIONS_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def fresh_graph_state():
    """The agent graph is compiled once and keeps each session's saved state in a
    checkpointer under the session id. Many tests reuse ids like "s1", so every test
    starts with a fresh checkpointer and cannot see another test's saved state. (Imported
    here, not at the top, so the test modules have already set up their environment.)"""
    from app.agent.graph import configure_checkpointer

    configure_checkpointer()
    yield
