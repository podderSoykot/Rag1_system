"""API security helpers: rate limiting, safe errors, trusted pickle, upload limits."""

import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock

from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

GENERIC_ERROR = "An internal error occurred. Please try again later."


class RateLimiter:
    """Simple sliding-window rate limiter keyed by client + bucket."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, bucket: str, client_key: str, limit: int, window_s: int = 60) -> None:
        if limit <= 0:
            return
        key = f"{bucket}:{client_key}"
        now = time.time()
        cutoff = now - window_s

        with self._lock:
            events = self._events[key]
            while events and events[0] < cutoff:
                events.popleft()
            if len(events) >= limit:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for {bucket}. Try again in a minute.",
                )
            events.append(now)


rate_limiter = RateLimiter()


def client_key(host: str | None) -> str:
    return host or "unknown"


def api_error(exc: Exception, *, context: str) -> HTTPException:
    logger.exception("%s failed", context, exc_info=exc)
    return HTTPException(status_code=500, detail=GENERIC_ERROR)


async def save_upload_limited(
    upload: UploadFile,
    dest: Path,
    max_bytes: int,
) -> int:
    """Stream upload to disk; enforce size limit and basic PDF header check."""
    total = 0
    header_checked = False

    try:
        with dest.open("wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                if not header_checked:
                    if not chunk.startswith(b"%PDF"):
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid PDF file (missing %PDF header).",
                        )
                    header_checked = True
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {max_bytes // (1024 * 1024)} MB.",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise api_error(exc, context="upload") from exc

    if total == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    return total
