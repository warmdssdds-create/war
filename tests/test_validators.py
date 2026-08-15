"""Tests for src/utils/validators.py."""

from __future__ import annotations

import pytest

from src.utils.validators import (
    ValidationError,
    clamp,
    validate_config_keys,
    validate_fraction,
    validate_positive,
    validate_side,
    validate_symbol,
)


def test_validate_positive_accepts_positive() -> None:
    assert validate_positive(1.0, "qty") == 1.0


def test_validate_positive_rejects_zero() -> None:
    with pytest.raises(ValidationError, match="positive"):
        validate_positive(0.0, "qty")


def test_validate_positive_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        validate_positive(-3.5, "price")


def test_validate_fraction_accepts_valid() -> None:
    assert validate_fraction(0.5, "risk") == 0.5


def test_validate_fraction_rejects_zero() -> None:
    with pytest.raises(ValidationError, match="\\(0, 1\\)"):
        validate_fraction(0.0, "risk")


def test_validate_fraction_rejects_one() -> None:
    with pytest.raises(ValidationError):
        validate_fraction(1.0, "risk")


def test_validate_side_buy() -> None:
    assert validate_side("Buy") == "Buy"


def test_validate_side_sell() -> None:
    assert validate_side("sell") == "Sell"


def test_validate_side_invalid() -> None:
    with pytest.raises(ValidationError, match="Buy.*Sell"):
        validate_side("long")


def test_validate_symbol_valid() -> None:
    assert validate_symbol("ETCUSDT") == "ETCUSDT"


def test_validate_symbol_lowercase() -> None:
    with pytest.raises(ValidationError):
        validate_symbol("etcusdt")


def test_validate_symbol_empty() -> None:
    with pytest.raises(ValidationError):
        validate_symbol("")


def test_validate_config_keys_passes() -> None:
    validate_config_keys({"a": 1, "b": 2}, ["a", "b"])


def test_validate_config_keys_missing() -> None:
    with pytest.raises(ValidationError, match="missing"):
        validate_config_keys({"a": 1}, ["a", "b"], section="strategy")


def test_clamp_within_bounds() -> None:
    assert clamp(5.0, 0.0, 10.0) == 5.0


def test_clamp_below_lo() -> None:
    assert clamp(-1.0, 0.0, 10.0) == 0.0


def test_clamp_above_hi() -> None:
    assert clamp(15.0, 0.0, 10.0) == 10.0


def test_clamp_invalid_bounds() -> None:
    with pytest.raises(ValidationError, match="bounds invalid"):
        clamp(5.0, 10.0, 0.0)
