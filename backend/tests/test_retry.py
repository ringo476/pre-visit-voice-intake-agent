import pytest

from app.retry import call_with_retry


def test_succeeds_immediately_without_retrying():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert call_with_retry(fn, max_attempts=3, base_delay_seconds=0) == "ok"
    assert len(calls) == 1


def test_recovers_after_transient_failures():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient network blip")
        return "recovered"

    assert call_with_retry(fn, max_attempts=5, base_delay_seconds=0) == "recovered"
    assert len(calls) == 3


def test_reraises_the_real_error_after_exhausting_all_attempts():
    def always_fails():
        raise ConnectionError("persistent failure")

    with pytest.raises(ConnectionError, match="persistent failure"):
        call_with_retry(always_fails, max_attempts=3, base_delay_seconds=0)
