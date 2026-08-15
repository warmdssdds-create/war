"""Logging setup for the bot."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml


def setup_logging(config_path: str = "config/logging.yaml") -> logging.Logger:
    """Configure logging from YAML with safe fallback."""
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            config: dict[str, Any] = yaml.safe_load(handle) or {}
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    return logging.getLogger("bot")
