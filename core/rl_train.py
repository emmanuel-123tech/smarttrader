"""RL training utilities for AI Trading Beast."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger

try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("stable-baselines3 is required for RL training") from exc

from .data import MarketDataLoader, MarketDataRequest
from .features import generate_features
from .rl_env import EnvConfig, TradingEnv
from .utils import ensure_dirs, load_config


def prepare_data(symbol: str, timeframe: str, limit: int = 2000) -> pd.DataFrame:
    config = load_config()
    market = None
    for entry in config.markets:
        if entry["symbol"] == symbol:
            market = entry
            break
    if not market:
        raise ValueError(f"Symbol {symbol} not found in config markets")
    loader = MarketDataLoader()
    request = MarketDataRequest(symbol=symbol, exchange=market.get("exchange", "binance"), timeframe=timeframe, limit=limit, venue=market.get("venue", "ccxt"))
    raw = loader.fetch(request)
    features = generate_features(raw)
    return features


def split_train_validation(data: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(data) * split_ratio)
    return data.iloc[:split], data.iloc[split:]


def train_rl(symbol: str) -> Path:
    config = load_config()
    rl_cfg = config.rl
    timeframe = config.general.get("base_timeframe", "1h")
    data = prepare_data(symbol, timeframe)
    train, validation = split_train_validation(data)
    env_config = EnvConfig(
        window=int(rl_cfg.get("env_window", 48)),
        slippage_bps=float(rl_cfg.get("slippage_bps", 1)),
        commission_bps=float(rl_cfg.get("commission_bps", 2)),
        reward_type=rl_cfg.get("reward", "equity"),
    )
    train_env = TradingEnv(train, env_config)
    eval_env = TradingEnv(validation, env_config)

    algo = rl_cfg.get("algo", "ppo").lower()
    total_timesteps = int(rl_cfg.get("total_timesteps", 100000))
    logger.info("AI Trading Beast training RL", algo=algo, timesteps=total_timesteps)
    if algo == "ppo":
        model = PPO("MlpPolicy", train_env, verbose=1)
    elif algo == "dqn":
        model = DQN("MlpPolicy", train_env, verbose=1)
    else:
        raise ValueError(f"Unsupported RL algo: {algo}")
    eval_callback = EvalCallback(eval_env, best_model_save_path="models/rl/tmp", log_path="models/rl/tmp", eval_freq=5000)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    ensure_dirs("models/rl", f"models/rl/{algo}")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    model_path = Path(f"models/rl/{algo}/{symbol.replace('/', '_')}_{timestamp}.zip")
    model.save(model_path)
    logger.info("AI Trading Beast RL model saved", path=str(model_path))
    return model_path


__all__ = ["train_rl"]


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train RL policy for AI Trading Beast")
    parser.add_argument("symbol", help="Symbol to train (e.g. BTC/USDT)")
    args = parser.parse_args()
    train_rl(args.symbol)


if __name__ == "__main__":
    _cli()
