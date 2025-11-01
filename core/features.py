"""Feature engineering for AI Trading Beast."""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=window, min_periods=window).mean()
    roll_down = down.rolling(window=window, min_periods=window).mean()
    rs = roll_up / roll_down.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def realized_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change().rolling(window=window).std() * (window ** 0.5)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    rsi_series = rsi(df["close"], 14)
    macd_df = macd(df["close"], 12, 26, 9)
    df = df.join(macd_df)
    df["rsi_14"] = rsi_series
    df["returns_1"] = df["close"].pct_change(1)
    df["returns_3"] = df["close"].pct_change(3)
    df["volatility_20"] = realized_volatility(df["close"], 20)
    df["atr_14"] = atr(df, 14)
    df.dropna(inplace=True)
    return df


__all__ = ["generate_features", "rsi", "macd", "ema", "atr", "realized_volatility"]
