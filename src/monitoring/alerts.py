"""Optional alert integrations."""

from __future__ import annotations

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
