"""Health check endpoint returning JSON state snapshot."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from src.utils.time_utils import utc_now


@dataclass
class HealthState:
    """Mutable shared state for the health endpoint."""

    status: str = "starting"  # starting | running | paused | stopped
    last_candle_ts: str = ""
    last_signal: str = "none"
    last_order_id: str = ""
    circuit_breaker_active: bool = False
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    uptime_start: str = field(default_factory=lambda: utc_now().isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise state to a plain dict suitable for JSON response."""
        d = asdict(self)
        d["checked_at"] = utc_now().isoformat()
        return d

    def update(self, **kwargs: Any) -> None:
        """Partial update of state fields (not thread-safe; use module-level update_state for concurrent access)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Module-level singleton used by the HTTP handler
_state: HealthState = HealthState()
_lock: threading.Lock = threading.Lock()


def get_state() -> HealthState:
    """Return the module-level health state object."""
    return _state


def update_state(**kwargs: Any) -> None:
    """Update the module-level health state (thread-safe)."""
    with _lock:
        _state.update(**kwargs)


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP request handler serving /health as JSON."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in ("/health", "/health/"):
            self.send_error(404, "Not Found")
            return

        with _lock:
            payload = json.dumps(_state.to_dict(), default=str).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN002
        """Suppress default stderr logging."""


class HealthServer:
    """Background HTTP server exposing the /health endpoint.

    Example::

        server = HealthServer(host="0.0.0.0", port=8080)
        server.start()
        # … bot loop …
        server.stop()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the health server in a daemon thread."""
        self._server = HTTPServer((self._host, self._port), _HealthHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Shut down the health server gracefully."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
