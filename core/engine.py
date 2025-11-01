"""Main trading engine for AI Trading Beast."""
from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from loguru import logger

from .data import load_time_series
from .features import generate_features
from .model import DirectionClassifier
from .risk import RiskManager
from .utils import AppConfig, load_config

try:
    from stable_baselines3 import DQN, PPO
except Exception:  # pragma: no cover
    DQN = PPO = None  # type: ignore


@dataclasses.dataclass
class PositionState:
    symbol: str
    side: str
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None


class TradingEngine:
    """Coordinates data, models, and execution."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.risk_manager = RiskManager(self.config.risk)
        self.lightgbm_classifier: Optional[DirectionClassifier] = None
        self.rl_policy = None
        self.positions: Dict[str, PositionState] = {}
        self.paused = False
        self.rl_window = int(self.config.rl.get("env_window", 48))

    def load_model(self) -> None:
        model_type = self.config.ml.get("model_type", "lightgbm")
        logger.info("AI Trading Beast loading model", model_type=model_type)
        if model_type == "lightgbm":
            threshold = float(self.config.ml.get("probability_threshold", 0.55))
            classifier = DirectionClassifier(probability_threshold=threshold)
            path = self.config.ml.get("model_path")
            if path and Path(path).exists():
                classifier.load(path)
            else:
                logger.warning("AI Trading Beast LightGBM model not found; please train before live trading")
            self.lightgbm_classifier = classifier
        elif model_type == "rl":
            algo = self.config.rl.get("algo", "ppo").lower()
            model_path = self.config.rl.get("model_path")
            if not model_path or not Path(model_path).exists():
                raise FileNotFoundError("RL model path missing; train via core/rl_train.py")
            if PPO is None or DQN is None:
                raise RuntimeError("stable-baselines3 is required for RL trading")
            if algo == "ppo":
                self.rl_policy = PPO.load(model_path)
            elif algo == "dqn":
                self.rl_policy = DQN.load(model_path)
            else:
                raise ValueError(f"Unsupported RL algo: {algo}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def run(self) -> None:
        heartbeat = int(self.config.execution.get("heartbeat_seconds", 60))
        logger.info("AI Trading Beast engine starting", heartbeat=heartbeat)
        self.load_model()
        while True:
            if not self.paused:
                await self.process_markets()
            await asyncio.sleep(heartbeat)

    async def process_markets(self) -> None:
        timeframe = self.config.general.get("base_timeframe", "1h")
        horizon = int(self.config.general.get("prediction_horizon_bars", 3))
        for market in self.config.markets:
            if not market.get("trade", True):
                continue
            symbol = market["symbol"]
            venue = market.get("venue", "ccxt")
            exchange = market.get("exchange", "binance")
            try:
                raw = load_time_series(symbol, timeframe, venue=venue, exchange=exchange, limit=500)
            except Exception as exc:
                logger.error("AI Trading Beast failed to load data", symbol=symbol, error=str(exc))
                continue
            features = generate_features(raw)
            if features.empty:
                continue
            latest = features.iloc[-1]
            decision = self.decide(symbol, features, latest, horizon)
            if decision:
                logger.info("AI Trading Beast trade signal", symbol=symbol, decision=decision)
                # Integrate with execution layer here (CCXT/OANDA)

    def decide(self, symbol: str, features: pd.DataFrame, latest: pd.Series, horizon: int) -> Optional[Dict[str, float]]:
        model_type = self.config.ml.get("model_type", "lightgbm")
        price = float(latest["close"])
        if model_type == "lightgbm" and self.lightgbm_classifier:
            if latest["ema_12"] <= latest["ema_26"]:
                return None
            should_trade = self.lightgbm_classifier.should_trade(latest.drop("label", errors="ignore"))
            if should_trade:
                size = self.risk_manager.position_size(equity=10000, df=features, latest_price=price)
                if size.quantity <= 0:
                    return None
                return {
                    "action": "buy",
                    "size": size.quantity,
                    "price": price,
                    "stop_loss": size.stop_loss,
                    "take_profit": size.take_profit,
                    "trailing_stop": size.trailing_stop,
                }
            return None
        if model_type == "rl" and self.rl_policy is not None:
            window_df = features.tail(self.rl_window)
            if len(window_df) < self.rl_window:
                return None
            obs = window_df.to_numpy(dtype=float).flatten()
            obs = np.concatenate([obs, np.array([0.0, 1.0, 1.0], dtype=float)])
            action, _ = self.rl_policy.predict(obs, deterministic=True)
            return {"action": int(action), "price": price}
        return None

    def pause(self) -> None:
        self.paused = True
        logger.warning("AI Trading Beast engine paused")

    def resume(self) -> None:
        self.paused = False
        logger.info("AI Trading Beast engine resumed")

    def status(self) -> Dict[str, any]:
        return {
            "paused": self.paused,
            "positions": {symbol: dataclasses.asdict(position) for symbol, position in self.positions.items()},
        }


__all__ = ["TradingEngine", "PositionState"]
