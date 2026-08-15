"""Health endpoint with compatibility helpers used by tests and runtime."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from src.utils.time_utils import utc_now


@dataclass
class HealthState:
    """Mutable shared state for the compatibility health endpoint."""

    status: str = "starting"
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
        data = asdict(self)
        data["checked_at"] = utc_now().isoformat()
        return data

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


_state: HealthState = HealthState()
_lock = threading.Lock()



def get_state() -> HealthState:
    return _state



def update_state(**kwargs: Any) -> None:
    with _lock:
        _state.update(**kwargs)


class _HealthHandler(BaseHTTPRequestHandler):
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

    def log_message(self, _fmt: str, *_args: Any) -> None:
        return None


class HealthServer:
    """Background HTTP server exposing the shared module-level health state."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._server = HTTPServer((self._host, self._port), _HealthHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
