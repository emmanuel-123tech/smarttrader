"""Market data loaders for AI Trading Beast."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

try:
    import ccxt
except ImportError:  # pragma: no cover
    ccxt = None  # type: ignore

import requests

from .utils import load_config, load_env


@dataclass
class MarketDataRequest:
    symbol: str
    exchange: str
    timeframe: str
    limit: int = 500
    venue: str = "ccxt"


class MarketDataLoader:
    """Fetch historical bars from exchanges."""

    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path or "config.yaml")

    def fetch(self, market: MarketDataRequest) -> pd.DataFrame:
        if market.venue == "ccxt":
            return self._fetch_ccxt(market)
        if market.venue == "oanda":
            return self._fetch_oanda(market)
        raise ValueError(f"Unsupported venue: {market.venue}")

    def _fetch_ccxt(self, market: MarketDataRequest) -> pd.DataFrame:
        if ccxt is None:
            raise RuntimeError("ccxt package not installed")
        exchange_name = market.exchange or self.config.general.get("exchange")
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({"enableRateLimit": True})
        logger.info(
            "AI Trading Beast fetching CCXT data",
            symbol=market.symbol,
            timeframe=market.timeframe,
        )
        ohlcv = exchange.fetch_ohlcv(market.symbol, timeframe=market.timeframe, limit=market.limit)
        frame = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame.set_index("timestamp", inplace=True)
        return frame

    def _fetch_oanda(self, market: MarketDataRequest) -> pd.DataFrame:
        env_vars = load_env()
        env = env_vars.get("OANDA_ENV", "practice")
        base_url = "https://api-fxpractice.oanda.com" if env != "live" else "https://api-fxtrade.oanda.com"
        params = {
            "granularity": market.timeframe.upper(),
            "price": "M",
            "count": str(market.limit),
        }
        headers = {
            "Authorization": f"Bearer {env_vars.get('OANDA_API_TOKEN', '')}",
        }
        instrument = market.symbol.replace("_", "-")
        url = f"{base_url}/v3/instruments/{instrument}/candles"
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        candles = response.json().get("candles", [])
        data = [
            {
                "timestamp": pd.to_datetime(candle["time"], utc=True),
                "open": float(candle["mid"]["o"]),
                "high": float(candle["mid"]["h"]),
                "low": float(candle["mid"]["l"]),
                "close": float(candle["mid"]["c"]),
                "volume": float(candle.get("volume", 0)),
            }
            for candle in candles
            if candle.get("complete")
        ]
        frame = pd.DataFrame(data)
        frame.set_index("timestamp", inplace=True)
        return frame


def load_time_series(symbol: str, timeframe: str, venue: str = "ccxt", exchange: str | None = None, limit: int = 500) -> pd.DataFrame:
    loader = MarketDataLoader()
    request = MarketDataRequest(symbol=symbol, timeframe=timeframe, limit=limit, venue=venue, exchange=exchange or "binance")
    return loader.fetch(request)


__all__ = ["MarketDataLoader", "MarketDataRequest", "load_time_series"]
