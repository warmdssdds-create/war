"""Alerting utilities including Telegram notifications and a health server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import requests


class TelegramAlerter:
    """Best-effort Telegram notifier; never hard-fails."""

    def __init__(self, token: str | None, chat_id: str | None) -> None:
        self.token = token
        self.chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, message: str) -> bool:
        """Send alert if configured, otherwise no-op success."""
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            resp = requests.post(url, json={"chat_id": self.chat_id, "text": message}, timeout=10)
            return resp.ok
        except requests.RequestException:
            return False



def format_trade_alert(symbol: str, side: str, qty: float, entry: float, sl: float, tp: float, score: float) -> str:
    """Format a concise trade-entry alert."""
    return (
        f"📈 Trade Entry\n"
        f"Symbol: {symbol}\nSide: {side}\nQty: {qty:.4f}\n"
        f"Entry: {entry:.4f}\nSL: {sl:.4f}\nTP: {tp:.4f}\n"
        f"Confluence: {score:.2f}"
    )



def format_pnl_alert(symbol: str, pnl: float, win_rate: float, total_trades: int) -> str:
    """Format a PnL summary alert."""
    emoji = "✅" if pnl >= 0 else "⚠️"
    return (
        f"{emoji} Trade Summary\n"
        f"Symbol: {symbol}\nPnL: {pnl:.2f} USDT\n"
        f"Win rate: {win_rate:.1%}\nTotal trades: {total_trades}"
    )


class HealthServer:
    """Simple HTTP health endpoint serving a mutable JSON payload."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self._status: dict[str, Any] = {"status": "starting"}
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def set_status(self, key: str, value: Any) -> None:
        """Update a single health-status key."""
        with self._lock:
            self._status[key] = value

    def get_status(self) -> dict[str, Any]:
        """Return the current health payload."""
        with self._lock:
            return dict(self._status)

    def start(self) -> None:
        """Start the health server in a daemon thread."""
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path not in {"/health", "/health/"}:
                    self.send_error(404, "Not Found")
                    return
                payload = json.dumps(parent.get_status(), default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _fmt: str, *_args: Any) -> None:
                return None

        self._server = HTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Shutdown the health server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
