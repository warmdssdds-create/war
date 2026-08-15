"""Input validation helpers used across the codebase."""

from __future__ import annotations

from typing import Any


class ValidationError(ValueError):
    """Raised when a validation check fails."""


def validate_positive(value: float, name: str) -> float:
    """Ensure *value* is strictly positive.

    Args:
        value: The numeric value to check.
        name: Human-readable field name used in the error message.

    Returns:
        The original *value* if valid.

    Raises:
        ValidationError: When *value* is not strictly positive.
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value!r}")
    return value


def validate_fraction(value: float, name: str) -> float:
    """Ensure *value* is in the open interval (0, 1).

    Args:
        value: The numeric fraction to check.
        name: Human-readable field name used in the error message.

    Returns:
        The original *value* if valid.

    Raises:
        ValidationError: When *value* is outside (0, 1).
    """
    if not 0 < value < 1:
        raise ValidationError(f"{name} must be in (0, 1), got {value!r}")
    return value


def validate_side(side: str) -> str:
    """Ensure *side* is a valid order direction.

    Args:
        side: Order side string.

    Returns:
        The normalised side string (``'Buy'`` or ``'Sell'``).

    Raises:
        ValidationError: When *side* is not recognised.
    """
    normalised = side.capitalize()
    if normalised not in ("Buy", "Sell"):
        raise ValidationError(f"side must be 'Buy' or 'Sell', got {side!r}")
    return normalised


def validate_symbol(symbol: str) -> str:
    """Ensure *symbol* is a non-empty, uppercase string.

    Args:
        symbol: Trading pair symbol (e.g. ``'ETCUSDT'``).

    Returns:
        The original *symbol* if valid.

    Raises:
        ValidationError: When *symbol* is empty or contains lowercase letters.
    """
    if not symbol or symbol != symbol.upper():
        raise ValidationError(f"symbol must be non-empty and uppercase, got {symbol!r}")
    return symbol


def validate_config_keys(cfg: dict[str, Any], required_keys: list[str], section: str = "config") -> None:
    """Ensure *cfg* contains all *required_keys*.

    Args:
        cfg: Configuration dictionary to validate.
        required_keys: Keys that must be present.
        section: Human-readable section name used in the error message.

    Raises:
        ValidationError: When any required key is missing.
    """
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValidationError(f"{section} is missing required keys: {missing}")


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the closed interval [*lo*, *hi*].

    Args:
        value: Value to clamp.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        Clamped value.

    Raises:
        ValidationError: When *lo* > *hi*.
    """
    if lo > hi:
        raise ValidationError(f"clamp bounds invalid: lo={lo} > hi={hi}")
    return max(lo, min(value, hi))
