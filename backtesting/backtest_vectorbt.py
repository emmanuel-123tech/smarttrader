"""VectorBT backtesting for AI Trading Beast."""
from __future__ import annotations

import vectorbt as vbt

from core.data import load_time_series
from core.features import generate_features
from core.model import DirectionClassifier
from core.utils import load_config


def run_backtest(symbol: str) -> vbt.Portfolio:
    config = load_config()
    timeframe = config.general.get("base_timeframe", "1h")
    raw = load_time_series(symbol, timeframe)
    data = generate_features(raw)
    clf = DirectionClassifier(probability_threshold=float(config.ml.get("probability_threshold", 0.55)))
    clf.load(config.ml.get("model_path", "models/lightgbm_default.txt"))
    features = data.drop(columns=["close"], errors="ignore")
    proba = clf.predict_proba(features)
    entries = (proba >= clf.threshold) & (data["ema_12"] > data["ema_26"])
    exits = ~entries
    portfolio = vbt.Portfolio.from_signals(
        close=data["close"],
        entries=entries,
        exits=exits,
        fees=config.execution.get("slippage_bps", 0) / 10000,
        slippage=config.execution.get("slippage_bps", 0) / 10000,
    )
    return portfolio


if __name__ == "__main__":
    cfg = load_config()
    for market in cfg.markets:
        if market.get("trade", True):
            symbol = market["symbol"]
            portfolio = run_backtest(symbol)
            stats = portfolio.stats()
            print(f"Backtest summary for {symbol}\n{stats}")
