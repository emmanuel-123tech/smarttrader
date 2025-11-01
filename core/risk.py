"""Risk management utilities for AI Trading Beast."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .features import atr


@dataclass
class PositionSizingResult:
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: float | None


class RiskManager:
    def __init__(self, config: Dict[str, float]):
        self.config = config
        self.daily_loss = 0.0

    def reset_daily(self) -> None:
        self.daily_loss = 0.0

    def register_loss(self, loss_pct: float) -> None:
        self.daily_loss += loss_pct

    def breached_daily_loss(self) -> bool:
        max_loss = float(self.config.get("max_daily_loss_pct", 10.0))
        return self.daily_loss <= -abs(max_loss)

    def position_size(self, equity: float, df: pd.DataFrame, latest_price: float) -> PositionSizingResult:
        risk_pct = float(self.config.get("risk_per_trade_pct", 1.0)) / 100.0
        atr_mult = float(self.config.get("sl_atr_mult", 1.5))
        tp_rr = float(self.config.get("tp_rr", 2.0))
        trailing_mult = float(self.config.get("trailing_atr_mult", 0))

        atr_series = df.get("atr_14")
        if atr_series is None or atr_series.empty:
            raise ValueError("ATR feature missing; ensure generate_features() was applied")
        atr_value = float(atr_series.iloc[-1])
        stop_loss_distance = atr_value * atr_mult
        if stop_loss_distance <= 0:
            return PositionSizingResult(
                quantity=0.0,
                stop_loss=latest_price,
                take_profit=latest_price,
                trailing_stop=None,
            )
        quantity = max((equity * risk_pct) / stop_loss_distance, 0)
        stop_loss = latest_price - stop_loss_distance
        take_profit = latest_price + stop_loss_distance * tp_rr
        trailing_stop = latest_price - atr_value * trailing_mult if trailing_mult > 0 else None
        return PositionSizingResult(quantity=quantity, stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop)


__all__ = ["RiskManager", "PositionSizingResult"]
