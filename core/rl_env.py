"""Gymnasium environment wrapper for RL trading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass
class EnvConfig:
    window: int
    slippage_bps: float
    commission_bps: float
    reward_type: str = "equity"


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: pd.DataFrame, config: EnvConfig):
        super().__init__()
        self.data = data
        self.config = config
        self.pointer = config.window
        self.position = 0.0  # -1 short, 0 flat, 1 long
        self.cash = 1.0
        self.equity = 1.0
        self._last_price = float(self.data["close"].iloc[self.pointer - 1])
        feature_count = data.shape[1]
        self._obs_shape = feature_count * config.window + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_shape,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)  # hold, open long, close long, open short, close short

    def _get_observation(self) -> np.ndarray:
        window = self.data.iloc[self.pointer - self.config.window : self.pointer]
        obs = window.to_numpy(dtype=np.float32).flatten()
        obs = np.concatenate([obs, np.array([self.position, self.cash, self.equity], dtype=np.float32)])
        return obs

    def step(self, action: int):  # type: ignore[override]
        done = False
        info: Dict[str, float] = {}
        price = float(self.data["close"].iloc[self.pointer])
        # Mark-to-market PnL
        pnl = (price - self._last_price) * self.position
        self.cash += pnl
        self._last_price = price
        reward = 0.0
        if action == 1:
            reward += self._update_position(target=1.0, price=price)
        elif action == 2:
            reward += self._update_position(target=0.0, price=price)
        elif action == 3:
            reward += self._update_position(target=-1.0, price=price)
        elif action == 4:
            reward += self._update_position(target=0.0, price=price)
        self.equity = self.cash + self.position * price
        if self.config.reward_type == "equity":
            reward += self.equity
        elif self.config.reward_type == "pnl":
            reward += pnl
        self.pointer += 1
        if self.pointer >= len(self.data):
            done = True
        obs = self._get_observation() if not done else np.zeros(self._obs_shape, dtype=np.float32)
        return obs, reward, done, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        self.pointer = self.config.window
        self.position = 0.0
        self.cash = 1.0
        self.equity = 1.0
        self._last_price = float(self.data["close"].iloc[self.pointer - 1])
        return self._get_observation(), {}

    def _update_position(self, target: float, price: float) -> float:
        if target == self.position:
            return 0.0
        transaction_cost = self._apply_costs(price) * abs(target - self.position)
        self.cash -= transaction_cost
        reward_adjustment = -transaction_cost
        self.position = target
        return reward_adjustment

    def _apply_costs(self, price: float) -> float:
        commission = self.config.commission_bps / 10000.0 * price
        slippage = self.config.slippage_bps / 10000.0 * price
        return commission + slippage


__all__ = ["TradingEnv", "EnvConfig"]
