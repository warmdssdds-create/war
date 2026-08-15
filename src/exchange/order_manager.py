"""Order helpers and idempotent link-id generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderIntent:
    """Serializable shape for stable order id generation."""

    symbol: str
    side: str
    qty: float
    signal_ts: int


def generate_order_link_id(intent: OrderIntent, namespace: str = "bot") -> str:
    """Create deterministic 32-char idempotent orderLinkId."""
    raw = f"{namespace}|{intent.symbol}|{intent.side}|{intent.qty:.8f}|{intent.signal_ts}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest[:32]
