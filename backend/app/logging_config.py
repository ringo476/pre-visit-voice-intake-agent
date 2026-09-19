"""Structured logging setup. Replaces bare print() statements in the live
server path — print() has no severity level, can't be filtered, carries no
machine-readable context (which session, which turn), and nothing forwards
it anywhere; if the safety engine or the reasoning loop started silently
failing in production, nobody would know until a patient or clinician
noticed something was wrong. This doesn't stand up a monitoring platform —
it emits one JSON object per line to stdout, the standard shape any real
log aggregator (CloudWatch, Datadog, a plain `docker logs | jq`) expects to
ingest, and it's the piece those tools actually need to plug into."""

import json
import logging
import sys
from datetime import datetime, timezone

_RESERVED = frozenset(logging.LogRecord(None, 0, "", 0, "", (), None).__dict__.keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Anything passed via logger.info(..., extra={"session_id": ...})
        # rides along as its own field, not string-formatted into the
        # message — so a log shipper can filter/aggregate by session_id,
        # turn_generation, etc. without parsing free text.
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured — main.py's startup can call this more than once safely
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
