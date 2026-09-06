import shutil
from pathlib import Path

_SESSIONS_DIR = Path(__file__).parent.parent / ".chroma_data" / "sessions"


def pytest_configure(config):
    """Session-scoped Chroma stores created by tests use random UUIDs and
    are never cleaned up by test code (unlike main.py's real WS lifecycle,
    which calls clear_session on disconnect) — wipe them once per test run
    so repeated runs don't accumulate disk usage indefinitely."""
    shutil.rmtree(_SESSIONS_DIR, ignore_errors=True)
