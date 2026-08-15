"""Live runner for the Bybit ETCUSDT strategy with startup orchestration."""

from __future__ import annotations

import argparse
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.data.fetcher import CandleFetcher, LiveCandleAssembler, backfill_with_pagination, merge_candles, validate_candles
from src.data.storage import ParquetCache
from src.execution.engine import ExecutionConfig, ExecutionEngine, get_execution_state
from src.exchange.bybit_rest import BybitRestClient
from src.exchange.bybit_ws import BybitWsClient, ReconnectManager
from src.exchange.order_manager import OrderManager
from src.indicators.compute import compute_indicators
from src.monitoring.alerts import HealthServer, TelegramAlerter, format_pnl_alert, format_trade_alert
from src.monitoring.logger import BotLogger, setup_logging
from src.risk.circuit_breaker import CircuitBreaker, VolatilityStopTightener
from src.strategy.regime_filter import RegimeHistory
from src.strategy.signals import SignalHistory, evaluate_exit_signal, evaluate_signal
from src.utils.time_utils import format_ts, time_to_next_candle, utc_now

try:  # pragma: no cover - optional dependency
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:  # pragma: no cover - optional dependency
    class BackgroundScheduler:  # type: ignore[override]
        def __init__(self, timezone: str = "UTC") -> None:
            self.jobs: list[tuple[Callable[..., Any], int]] = []
            self._running = False
            self._thread: threading.Thread | None = None
            self._timezone = timezone

        def add_job(self, func: Callable[..., Any], trigger: str, minutes: int | None = None, hours: int | None = None, **_kwargs: Any) -> None:
            interval = int((minutes or 0) * 60 + (hours or 0) * 3600)
            interval = max(interval, 60)
            self.jobs.append((func, interval))

        def start(self) -> None:
            self._running = True

            def loop() -> None:
                last_run: dict[int, float] = {}
                while self._running:
                    now = time.time()
                    for idx, (func, interval) in enumerate(self.jobs):
                        if now - last_run.get(idx, 0.0) >= interval:
                            try:
                                func()
                            except Exception:
                                pass
                            last_run[idx] = now
                    time.sleep(1)

            self._thread = threading.Thread(target=loop, daemon=True)
            self._thread.start()

        def shutdown(self, wait: bool = True) -> None:
            self._running = False
            if wait and self._thread is not None:
                self._thread.join(timeout=3)


class TradingBotApp:
    """Coordinates REST, websocket, scheduling, and strategy evaluation."""

    def __init__(self, env: str = "testnet") -> None:
        self.env = env
        self.testnet = env == "testnet"
        self.shutdown_requested = False
        self.health = HealthServer(port=8080)
        self.scheduler = BackgroundScheduler(timezone="UTC")
        self.regime_history = RegimeHistory(maxlen=200)
        self.signal_history = SignalHistory(maxlen=20)
        self.stop_tightener = VolatilityStopTightener()
        self.reconnector = ReconnectManager()
        self.live_candles = LiveCandleAssembler(symbol="ETCUSDT")
        self.logger = BotLogger(setup_logging("config/logging.yaml"))
        self.raw_logger = setup_logging("config/logging.yaml")
        self.strategy_cfg: dict[str, Any] = {}
        self.risk_cfg: dict[str, Any] = {}
        self.rest: BybitRestClient | None = None
        self.ws: BybitWsClient | None = None
        self.cache = ParquetCache("data/cache")
        self.fetcher: CandleFetcher | None = None
        self.order_manager = OrderManager()
        self.engine: ExecutionEngine | None = None
        self.circuit_breaker: CircuitBreaker | None = None
        self.alerter: TelegramAlerter | None = None
        self.market_frames: dict[str, pd.DataFrame] = {}

    def _load_yaml(self, path: str, fallback: dict[str, Any]) -> dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists():
            return fallback
        data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        merged = fallback.copy()
        merged.update(data)
        return merged

    def load_configs(self) -> None:
        """Load runtime configuration from YAML files and environment."""
        self.strategy_cfg = self._load_yaml(
            "config/strategy.yaml",
            {
                "symbol": "ETCUSDT",
                "category": "linear",
                "confluence_threshold": 0.65,
                "weights": {"trending": {}, "ranging": {}},
                "spread": {"max_bps": 12},
                "execution": {"reconcile_interval_sec": 30},
                "use_refinement": True,
                "intervals": ["60", "15", "5"],
            },
        )
        self.risk_cfg = self._load_yaml(
            "config/risk.yaml",
            {
                "account": {"capital_base_usdt": 1000},
                "position_sizing": {"risk_per_trade": 0.005, "min_qty": 0.1, "qty_step": 0.1, "max_notional_usdt": 200},
                "stops": {"atr_multiple": 1.6, "reward_risk": 1.8},
                "circuit_breakers": {"daily_loss_limit": 0.03, "weekly_loss_limit": 0.06, "max_consecutive_losses": 3, "pause_minutes": 120},
            },
        )
        self.health.set_status("config_loaded", True)

    def setup_clients(self) -> None:
        """Initialize REST, WebSocket, alerting, circuit breaker, and engine."""
        self.rest = BybitRestClient(testnet=self.testnet)
        self.ws = BybitWsClient(testnet=self.testnet)
        self.fetcher = CandleFetcher(rest=self.rest, cache=self.cache)
        self.alerter = TelegramAlerter(token=os.getenv("TELEGRAM_BOT_TOKEN"), chat_id=os.getenv("TELEGRAM_CHAT_ID"))
        self.circuit_breaker = CircuitBreaker(
            daily_loss_limit=float(self.risk_cfg["circuit_breakers"]["daily_loss_limit"]),
            weekly_loss_limit=float(self.risk_cfg["circuit_breakers"]["weekly_loss_limit"]),
            max_consecutive_losses=int(self.risk_cfg["circuit_breakers"]["max_consecutive_losses"]),
            pause_minutes=int(self.risk_cfg["circuit_breakers"]["pause_minutes"]),
            starting_equity=float(self.risk_cfg["account"]["capital_base_usdt"]),
        )
        self.engine = ExecutionEngine(
            rest=self.rest,
            cfg=ExecutionConfig(
                category=self.strategy_cfg["category"],
                symbol=self.strategy_cfg["symbol"],
                max_spread_bps=float(self.strategy_cfg["spread"]["max_bps"]),
            ),
            logger=self.raw_logger,
            order_manager=self.order_manager,
        )
        self.health.set_status("status", "initialized")
        self.health.set_status("env", self.env)

    def backfill_historical_data(self) -> None:
        """Backfill all required intervals and validate cache integrity."""
        assert self.fetcher is not None and self.rest is not None
        symbol = self.strategy_cfg["symbol"]
        category = self.strategy_cfg["category"]
        intervals = self.strategy_cfg.get("intervals", ["60", "15", "5"])
        for interval in intervals:
            frame = backfill_with_pagination(self.rest, category, symbol, interval, total_candles=1200 if interval == "5" else 1000)
            if frame.empty:
                frame = self.fetcher.get_candles(category, symbol, interval, limit=1000, force_refresh=True)
            valid, issues = validate_candles(frame)
            if not valid:
                self.raw_logger.warning("Historical candle issues for %s: %s", interval, issues)
            self.cache.save(symbol, interval, frame)
            self.market_frames[interval] = compute_indicators(frame)
        self.health.set_status("historical_backfill", True)

    def start_websocket_streams(self) -> None:
        """Subscribe to public and private websocket feeds."""
        assert self.ws is not None
        symbol = self.strategy_cfg["symbol"]
        self.ws.subscribe_klines(symbol=symbol, interval=15, callback=self._on_kline_message)
        self.ws.subscribe_klines(symbol=symbol, interval=5, callback=self._on_kline_message)
        self.ws.subscribe_private(order_cb=self.handle_order_update, position_cb=self.handle_position_update)
        self.health.set_status("websocket", "subscribed")

    def start_health_server(self) -> None:
        """Start the health endpoint."""
        self.health.start()
        self.health.set_status("status", "running")
        self.health.set_status("started_at", utc_now().isoformat())

    def register_signal_handlers(self) -> None:
        """Install SIGINT and SIGTERM handlers for graceful shutdown."""

        def _handler(signum: int, _frame: Any) -> None:
            self.raw_logger.info("Received signal %s", signum)
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def setup_scheduler(self) -> None:
        """Register periodic operational jobs."""
        self.scheduler.add_job(self.reconcile_state, trigger="interval", hours=1)
        self.scheduler.add_job(self.reset_daily_stats_if_needed, trigger="interval", hours=1)
        self.scheduler.add_job(self.heartbeat, trigger="interval", minutes=15)
        self.scheduler.add_job(self.check_websocket_staleness, trigger="interval", minutes=1)
        self.scheduler.start()
        self.health.set_status("scheduler", "started")

    def startup(self) -> None:
        """Full startup sequence requested by the task."""
        load_dotenv()
        self.load_configs()
        self.setup_clients()
        self.backfill_historical_data()
        self.start_websocket_streams()
        self.start_health_server()
        self.register_signal_handlers()
        self.setup_scheduler()
        self.raw_logger.info("Bot started in %s mode for %s", self.env, self.strategy_cfg["symbol"])

    def _on_kline_message(self, message: dict[str, Any]) -> None:
        """Handle raw websocket kline events and dispatch candle closes."""
        self.live_candles.update(message)
        topic = str(message.get("topic", ""))
        interval = topic.split(".")[1] if "." in topic else str(message.get("interval", ""))
        confirmed = self.live_candles.get_confirmed(interval)
        partial = self.live_candles.get_partial(interval)
        if partial is not None:
            self.health.set_status(f"last_partial_{interval}", partial["start"])
        if confirmed is None:
            return
        frame = pd.DataFrame([confirmed])
        frame["start"] = pd.to_datetime(frame["start"], unit="ms", utc=True)
        existing = self.market_frames.get(interval, pd.DataFrame(columns=frame.columns))
        merged = merge_candles(existing, frame)
        self.market_frames[interval] = compute_indicators(merged.tail(1500).reset_index(drop=True))
        self.health.set_status(f"last_confirmed_{interval}", int(confirmed["start"]))
        if str(interval) == "15":
            self.handle_candle_close(confirmed)

    def handle_candle_close(self, candle_15m: dict[str, Any]) -> None:
        """Main event handler called on each confirmed 15m candle."""
        if self.circuit_breaker is None or self.engine is None:
            return
        if self.circuit_breaker.is_paused():
            self.health.set_status("paused_until", format_ts(self.circuit_breaker.paused_until))
            return
        frame_15m = self.market_frames.get("15")
        if frame_15m is None or frame_15m.empty:
            return
        frame_5m = self.market_frames.get("5")
        signal_result = evaluate_signal(
            frame_15m=frame_15m,
            weights=self.strategy_cfg.get("weights", {}),
            threshold=float(self.strategy_cfg["confluence_threshold"]),
            confirmed=True,
            frame_5m=frame_5m,
            use_refinement=bool(self.strategy_cfg.get("use_refinement", False)),
        )
        self.signal_history.add(signal_result.side, signal_result.score, signal_result.regime, signal_result.reason)
        self.regime_history.add(signal_result.regime)
        self.logger.signal_generated(self.strategy_cfg["symbol"], signal_result.regime.value, signal_result.score, signal_result.side)
        self.health.set_status("last_signal", signal_result.reason)
        self.health.set_status("last_signal_score", signal_result.score)
        self.stop_tightener.set_spike(signal_result.regime.value == "VOLATILITY_SPIKE")

        state = get_execution_state(self.rest, self.strategy_cfg["category"], self.strategy_cfg["symbol"]) if self.rest is not None else None
        if state is not None and state.has_position and state.current_sl is not None:
            exit_result = evaluate_exit_signal(frame_15m, state.position_side or "Buy", state.entry_price, state.current_sl, signal_result.regime)
            new_stop = self.stop_tightener.apply(exit_result.new_stop or state.current_sl, state.entry_price, float(frame_15m.iloc[-1]["atr14"]), state.position_side or "Buy")
            if exit_result.should_exit and self.rest is not None:
                self.raw_logger.warning("Emergency exit signal: %s", exit_result.reason)
                from src.execution.engine import emergency_close

                emergency_close(self.rest, self.strategy_cfg["category"], self.strategy_cfg["symbol"])
            elif new_stop != state.current_sl:
                self.engine.place_exit_adjustment(position_idx=0, new_sl=new_stop, new_tp=state.current_tp)

        if signal_result.side is None:
            return
        last = frame_15m.iloc[-1]
        response = self.engine.place_entry(
            side=signal_result.side,
            signal_ts=int(pd.Timestamp(last["start"]).timestamp()),
            entry=float(last["close"]),
            capital=float(self.risk_cfg["account"]["capital_base_usdt"]),
            atr=float(last["atr14"]),
            structure_stop=float(last["low"] if signal_result.side == "Buy" else last["high"]),
            risk_cfg=self.risk_cfg,
        )
        if response:
            self.health.set_status("last_order_response", str(response))
            if self.alerter is not None:
                self.alerter.send(
                    format_trade_alert(
                        symbol=self.strategy_cfg["symbol"],
                        side=signal_result.side,
                        qty=float(self.order_manager.get_active_entries()[-1].qty) if self.order_manager.get_active_entries() else 0.0,
                        entry=float(last["close"]),
                        sl=float(self.order_manager.get_active_entries()[-1].sl) if self.order_manager.get_active_entries() else 0.0,
                        tp=float(self.order_manager.get_active_entries()[-1].tp) if self.order_manager.get_active_entries() else 0.0,
                        score=signal_result.score,
                    )
                )

    def handle_position_update(self, position_data: dict[str, Any]) -> None:
        """Handle websocket position updates."""
        self.health.set_status("last_position_update", utc_now().isoformat())
        self.health.set_status("position_payload", str(position_data)[:500])

    def handle_order_update(self, order_data: dict[str, Any]) -> None:
        """Handle websocket order updates and update tracked order state."""
        self.order_manager.update_from_ws_message(order_data)
        self.health.set_status("last_order_update", utc_now().isoformat())
        open_orders = [record.link_id for record in self.order_manager.get_active_entries()]
        self.health.set_status("open_orders", open_orders)

    def reconcile_state(self) -> None:
        """Reconcile exchange state and update health payload."""
        if self.engine is None or self.circuit_breaker is None:
            return
        snapshot = self.engine.reconcile()
        self.health.set_status("reconciled_at", utc_now().isoformat())
        self.health.set_status("exchange_state", str(snapshot)[:1000])
        status = self.circuit_breaker.status()
        self.health.set_status("circuit_breaker", status.__dict__)

    def reset_daily_stats_if_needed(self) -> None:
        """Reset daily stats at midnight UTC."""
        if self.circuit_breaker is None:
            return
        now = utc_now()
        if now.hour == 0 and now.minute < 5:
            self.circuit_breaker.reset_daily()
            self.health.set_status("daily_reset_at", now.isoformat())

    def heartbeat(self) -> None:
        """Send periodic heartbeat updates."""
        if self.circuit_breaker is None:
            return
        status = self.circuit_breaker.status()
        self.health.set_status("heartbeat_at", utc_now().isoformat())
        self.health.set_status("win_rate", self.signal_history.win_rate())
        if self.alerter is not None:
            self.alerter.send(format_pnl_alert(self.strategy_cfg["symbol"], status.daily_pnl, status.win_rate, self.circuit_breaker.trade_stats.total))

    def check_websocket_staleness(self) -> None:
        """Reconnect websockets when private/public streams go stale."""
        if self.ws is None:
            return
        if self.reconnector.should_reconnect(self.ws.last_message_ts, stale_thresh=60):
            self.raw_logger.warning("WebSocket stale, reconnecting")
            self.reconnector.reconnect(self.ws)
            self.start_websocket_streams()
            self.health.set_status("ws_reconnected_at", utc_now().isoformat())

    def run_forever(self) -> None:
        """Idle event loop; real work occurs through websocket callbacks and scheduler."""
        self.startup()
        try:
            while not self.shutdown_requested:
                self.health.set_status("time_to_next_15m", time_to_next_candle(15))
                time.sleep(1)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown scheduler, health server, and mark the app stopped."""
        self.scheduler.shutdown(wait=False)
        self.health.set_status("status", "stopped")
        self.health.stop()
        self.raw_logger.info("Bot stopped")



def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the ETCUSDT Bybit bot")
    parser.add_argument("--env", choices=["testnet", "mainnet"], default="testnet")
    args = parser.parse_args()
    app = TradingBotApp(env=args.env)
    app.run_forever()


if __name__ == "__main__":
    main()
