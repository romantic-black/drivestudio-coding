from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Debug logging configuration
_DEBUG_LOG_PATH = "/root/drivestudio-coding/.cursor/debug.log"


def _debug_log(location: str, message: str, data: dict, hypothesis_id: Optional[str] = None, run_id: str = "initial") -> None:
    """
    Write a debug entry as NDJSON for offline inspection.

    Args:
        location: identifier of the call site (usually a function name)
        message: human readable message
        data: arbitrary JSON-serializable payload
        hypothesis_id: optional hypothesis identifier
        run_id: optional run id (defaults to "initial")
    """
    try:
        entry = {
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "sessionId": "debug-session",
            "runId": run_id,
        }
        if hypothesis_id:
            entry["hypothesisId"] = hypothesis_id
        Path(_DEBUG_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug("Failed to write debug log: %s", exc)

__all__ = ["_debug_log", "_DEBUG_LOG_PATH"]
