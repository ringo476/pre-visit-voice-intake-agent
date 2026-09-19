"""A small, dependency-free retry helper for the external API calls this
project depends on (Gemini, Google Cloud STT/TTS). Without this, a single
transient network blip — the exact kind of thing that happens routinely
against any real network API — surfaces as a hard failure straight to the
patient mid-conversation, with no attempt to recover from something that
would very likely succeed a moment later."""

import time
from typing import Callable, TypeVar

from app.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    what: str = "external call",
) -> T:
    """Calls fn() with exponential backoff (0.5s, 1s, 2s, ...) on failure.
    Re-raises the last exception if every attempt fails — this never hides
    a genuine, persistent failure, it only absorbs a transient one. Kept
    deliberately simple (no dependency, no jitter, no per-exception-type
    policy) since this project's scale doesn't need more than that; a
    real high-traffic deployment would want a proper circuit breaker on
    top of this, not instead of it."""
    last_error: Exception = RuntimeError("call_with_retry: fn was never actually called")
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - deliberately broad: any transient failure should retry
            last_error = e
            if attempt == max_attempts:
                logger.warning(f"{what} failed on final attempt {attempt}/{max_attempts}: {e}")
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            logger.warning(f"{what} failed on attempt {attempt}/{max_attempts}, retrying in {delay}s: {e}")
            time.sleep(delay)
    raise last_error
