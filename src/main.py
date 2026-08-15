"""Live runner for Bybit ETCUSDT strategy (testnet-first)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.data.fetcher import CandleFetcher
from src.data.storage import ParquetCache
from src.execution.engine import ExecutionConfig, ExecutionEngine
from src.exchange.bybit_rest import BybitRestClient
from src.indicators.compute import compute_indicators
from src.monitoring.alerts import TelegramAlerter
from src.monitoring.logger import setup_logging
from src.risk.circuit_breaker import CircuitBreaker
from src.strategy.signals import evaluate_signal


def _load_yaml(path: str, fallback: dict[str, Any]) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return fallback
    with p.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    merged = fallback.copy()
    merged.update(data)
    return merged


def run() -> None:
    """Start bot loop with safe defaults and defensive guards."""
    load_dotenv()
    logger = setup_logging("config/logging.yaml")

    strategy_cfg = _load_yaml(
        "config/strategy.yaml",
        {
            "symbol": "ETCUSDT",
            "category": "linear",
            "confluence_threshold": 0.65,
            "weights": {"trending": {}, "ranging": {}},
            "spread": {"max_bps": 12},
            "execution": {"reconcile_interval_sec": 30},
            "use_refinement": False,
        },
    )
    risk_cfg = _load_yaml(
        "config/risk.yaml",
        {
            "mode": "testnet",
            "account": {"capital_base_usdt": 1000},
            "position_sizing": {
                "risk_per_trade": 0.005,
                "min_qty": 0.1,
                "qty_step": 0.1,
                "max_notional_usdt": 200,
            },
            "stops": {"atr_multiple": 1.6, "reward_risk": 1.8},
            "circuit_breakers": {
                "daily_loss_limit": 0.03,
                "weekly_loss_limit": 0.06,
                "max_consecutive_losses": 3,
                "pause_minutes": 120,
            },
        },
    )

    testnet = str(risk_cfg.get("mode", "testnet")).lower() == "testnet"
    rest = BybitRestClient(testnet=testnet)
    fetcher = CandleFetcher(rest=rest, cache=ParquetCache("data/cache"))

    engine = ExecutionEngine(
        rest=rest,
        cfg=ExecutionConfig(
            category=strategy_cfg["category"],
            symbol=strategy_cfg["symbol"],
            max_spread_bps=float(strategy_cfg["spread"]["max_bps"]),
        ),
        logger=logger,
    )

    breaker = CircuitBreaker(
        daily_loss_limit=float(risk_cfg["circuit_breakers"]["daily_loss_limit"]),
        weekly_loss_limit=float(risk_cfg["circuit_breakers"]["weekly_loss_limit"]),
        max_consecutive_losses=int(risk_cfg["circuit_breakers"]["max_consecutive_losses"]),
        pause_minutes=int(risk_cfg["circuit_breakers"]["pause_minutes"]),
        starting_equity=float(risk_cfg["account"]["capital_base_usdt"]),
    )

    alerter = TelegramAlerter(
        token=os.getenv("TELEGRAM_BOT_TOKEN"),
        chat_id=os.getenv("TELEGRAM_CHAT_ID"),
    )
    logger.info("Bot started in %s mode for %s", "testnet" if testnet else "mainnet", strategy_cfg["symbol"])

    last_reconcile = 0.0
    while True:
        if breaker.is_paused():
            logger.warning("Circuit breaker pause active")
            time.sleep(5)
            continue

        candles = fetcher.backfill_required(strategy_cfg["category"], strategy_cfg["symbol"])
        df15 = compute_indicators(candles["15"])
        df5 = compute_indicators(candles["5"]) if strategy_cfg.get("use_refinement", False) else None
        last_start = df15.iloc[-1]["start"] if not df15.empty else None
        confirmed = False
        if last_start is not None:
            confirmed = (last_start.to_pydatetime().timestamp() + 15 * 60) <= time.time()

        signal = evaluate_signal(
            frame_15m=df15,
            weights=strategy_cfg["weights"],
            threshold=float(strategy_cfg["confluence_threshold"]),
            confirmed=confirmed,
            frame_5m=df5,
            use_refinement=bool(strategy_cfg.get("use_refinement", False)),
        )

        if signal.side:
            last = df15.iloc[-1]
            response = engine.place_entry(
                side=signal.side,
                signal_ts=int(last["start"].timestamp()),
                entry=float(last["close"]),
                capital=float(risk_cfg["account"]["capital_base_usdt"]),
                atr=float(last["atr14"]),
                structure_stop=float(last["low"] if signal.side == "Buy" else last["high"]),
                risk_cfg=risk_cfg,
            )
            if response:
                logger.info("Order submitted: %s", response)
                alerter.send(f"Order submitted: {signal.side} {strategy_cfg['symbol']}")

        now = time.time()
        if now - last_reconcile >= float(strategy_cfg["execution"]["reconcile_interval_sec"]):
            engine.reconcile()
            last_reconcile = now

        time.sleep(15)


if __name__ == "__main__":
    run()
