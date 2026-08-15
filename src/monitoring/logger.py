"""Logging configuration and convenience wrappers for structured trade events."""

from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml


class StructuredFormatter(logging.Formatter):
    """Format records as compact JSON with an ``extra`` container."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "extra": {
                key: value
                for key, value in record.__dict__.items()
                if key
                not in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                }
            },
        }
        if record.exc_info:
            payload["extra"]["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class BotLogger:
    """Thin convenience wrapper over ``logging.Logger``."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def trade_entry(self, symbol: str, side: str, qty: float, price: float, sl: float, tp: float) -> None:
        self.logger.info(
            "trade_entry",
            extra={"symbol": symbol, "side": side, "qty": qty, "price": price, "sl": sl, "tp": tp},
        )

    def trade_exit(self, symbol: str, side: str, qty: float, pnl: float) -> None:
        self.logger.info("trade_exit", extra={"symbol": symbol, "side": side, "qty": qty, "pnl": pnl})

    def signal_generated(self, symbol: str, regime: str, score: float, side: str | None) -> None:
        self.logger.info("signal_generated", extra={"symbol": symbol, "regime": regime, "score": score, "side": side})

    def circuit_breaker_tripped(self, reason: str) -> None:
        self.logger.warning("circuit_breaker_tripped", extra={"reason": reason})

    def error_with_context(self, error: Exception, context_dict: dict[str, Any]) -> None:
        self.logger.error("error", extra={"error": str(error), "context": context_dict}, exc_info=error)



def setup_logging(config_path: str = "config/logging.yaml") -> logging.Logger:
    """Configure logging from YAML with safe fallback."""
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            config: dict[str, Any] = yaml.safe_load(handle) or {}
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger("bot")



def get_logger(name: str) -> logging.Logger:
    """Return a standard logger instance."""
    return logging.getLogger(name)
