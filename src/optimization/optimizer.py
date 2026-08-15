"""Walk-forward strategy optimizer built around Optuna with safe fallbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest import BacktestConfig, BacktestResult, EventDrivenBacktest, run_walk_forward
from src.strategy.confluence import RANGING_WEIGHTS, TRENDING_WEIGHTS

try:  # pragma: no cover - optional dependency
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore[assignment]


@dataclass
class OptimizationConfig:
    """Configuration for walk-forward parameter search."""

    n_trials: int = 100
    n_splits: int = 5
    train_pct: float = 0.7
    metric_weights: dict[str, float] = field(
        default_factory=lambda: {
            "sharpe": 0.3,
            "profit_factor": 0.25,
            "win_rate": 0.2,
            "expectancy": 0.15,
            "max_drawdown": -0.1,
        }
    )


@dataclass
class StrategyParams:
    """Parameter bundle exposed to the optimizer."""

    confluence_threshold: float
    ema_short: int
    ema_mid: int
    ema_long: int
    adx_threshold: float
    rsi_low: float
    rsi_high: float
    atr_multiple: float
    reward_risk: float



def combined_objective(result: BacktestResult) -> float:
    """Compute the default combined optimization score."""
    return (
        (result.sharpe_ratio * 0.3)
        + (result.profit_factor * 0.25)
        + (result.win_rate * 0.2)
        + (result.expectancy * 0.15)
        + (-result.max_drawdown * 0.1)
    )



def overfitting_guard(train_results: list[BacktestResult], test_results: list[BacktestResult], threshold: float = 0.5) -> bool:
    """Return True when out-of-sample performance is not too degraded."""
    if not train_results or not test_results:
        return False
    train_score = sum(combined_objective(item) for item in train_results) / len(train_results)
    test_score = sum(combined_objective(item) for item in test_results) / len(test_results)
    if train_score <= 0:
        return False
    return (test_score / train_score) > threshold


class WalkForwardOptimizer:
    """Optimize strategy parameters using walk-forward backtests."""

    def __init__(self, candles_15m: pd.DataFrame, candles_5m: pd.DataFrame | None, config: OptimizationConfig) -> None:
        self.candles_15m = candles_15m
        self.candles_5m = candles_5m
        self.config = config
        self._best_params: StrategyParams | None = None
        self._best_score: float | None = None
        self._history: list[tuple[StrategyParams, float]] = []
        self.study = optuna.create_study(direction="maximize") if optuna is not None else None

    def _suggest_params(self, trial: Any) -> StrategyParams:
        """Suggest a parameter set from an Optuna trial or fallback iterator."""
        if optuna is None:
            idx = int(getattr(trial, "number", len(self._history)))
            return StrategyParams(
                confluence_threshold=0.55 + (idx % 5) * 0.05,
                ema_short=13 + (idx % 4) * 2,
                ema_mid=34 + (idx % 4) * 5,
                ema_long=144 + (idx % 3) * 20,
                adx_threshold=20 + (idx % 4) * 2.5,
                rsi_low=25 + (idx % 5) * 2,
                rsi_high=75 - (idx % 5) * 2,
                atr_multiple=1.2 + (idx % 4) * 0.2,
                reward_risk=1.4 + (idx % 4) * 0.2,
            )
        return StrategyParams(
            confluence_threshold=trial.suggest_float("confluence_threshold", 0.5, 0.8),
            ema_short=trial.suggest_int("ema_short", 8, 21),
            ema_mid=trial.suggest_int("ema_mid", 21, 55),
            ema_long=trial.suggest_int("ema_long", 89, 200),
            adx_threshold=trial.suggest_float("adx_threshold", 18.0, 32.0),
            rsi_low=trial.suggest_float("rsi_low", 20.0, 40.0),
            rsi_high=trial.suggest_float("rsi_high", 60.0, 80.0),
            atr_multiple=trial.suggest_float("atr_multiple", 1.0, 2.5),
            reward_risk=trial.suggest_float("reward_risk", 1.2, 3.0),
        )

    def _build_weights(self, params: StrategyParams) -> dict[str, dict[str, float]]:
        """Translate parameters into a weight configuration."""
        trending = TRENDING_WEIGHTS.copy()
        ranging = RANGING_WEIGHTS.copy()
        trending["adx_strength"] = min(max(params.adx_threshold / 100.0, 0.1), 0.35)
        trending["rsi_bias"] = min(max((50 - params.rsi_low) / 100.0, 0.05), 0.2)
        ranging["rsi_reversion"] = min(max((params.rsi_high - params.rsi_low) / 100.0, 0.1), 0.3)
        return {"trending": trending, "ranging": ranging}

    def objective(self, trial: Any) -> float:
        """Suggest params, run walk-forward tests, and return a score."""
        params = self._suggest_params(trial)
        weights = self._build_weights(params)
        config = BacktestConfig(symbol="ETCUSDT", initial_capital=10_000.0, risk_per_trade=0.01)
        results = run_walk_forward(self.candles_15m, config, weights, params.confluence_threshold, n_splits=self.config.n_splits)
        if not results:
            score = float("-inf")
        else:
            score = sum(combined_objective(item) for item in results) / len(results)
        self._history.append((params, score))
        if self._best_score is None or score > self._best_score:
            self._best_score = score
            self._best_params = params
        return score

    def optimize(self, n_trials: int | None = None) -> StrategyParams:
        """Run optimization and return the best parameter set."""
        total_trials = n_trials or self.config.n_trials
        if optuna is None:
            class DummyTrial:
                def __init__(self, number: int) -> None:
                    self.number = number

            for idx in range(total_trials):
                self.objective(DummyTrial(idx))
            assert self._best_params is not None
            return self._best_params
        assert self.study is not None
        self.study.optimize(self.objective, n_trials=total_trials)
        best = self.study.best_trial.params
        self._best_params = StrategyParams(**best)
        return self._best_params

    def get_best_params(self) -> StrategyParams:
        """Return the best parameters discovered so far."""
        if self._best_params is None:
            return self.optimize(self.config.n_trials)
        return self._best_params

    def plot_optimization_history(self) -> str | None:
        """Save an optimization-history plot when Optuna visualization is available."""
        if optuna is None or self.study is None:
            return None
        try:
            fig = optuna.visualization.plot_optimization_history(self.study)
            output = Path("optimization_history.html")
            fig.write_html(str(output))
            return str(output)
        except Exception:
            return None



def run_optimization(candles_path: str, output_path: str) -> dict[str, Any]:
    """Load candles, run optimization, and persist the best parameter set."""
    candles = pd.read_csv(candles_path)
    if "start" in candles.columns:
        candles["start"] = pd.to_datetime(candles["start"], utc=True)
    optimizer = WalkForwardOptimizer(candles, None, OptimizationConfig())
    best = optimizer.optimize()
    payload = best.__dict__.copy()
    Path(output_path).write_text(pd.Series(payload).to_json(indent=2), encoding="utf-8")
    return payload
