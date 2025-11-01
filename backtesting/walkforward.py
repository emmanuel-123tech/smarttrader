"""Walk-forward evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from core.model import DirectionClassifier


@dataclass
class WalkForwardResult:
    window_start: int
    window_end: int
    metrics: dict


def walk_forward(df: pd.DataFrame, horizon: int, train_size: int, test_size: int) -> List[WalkForwardResult]:
    results: List[WalkForwardResult] = []
    start = 0
    while start + train_size + test_size <= len(df):
        train_df = df.iloc[start : start + train_size]
        test_df = df.iloc[start + train_size : start + train_size + test_size]
        clf = DirectionClassifier()
        clf.fit(train_df, horizon=horizon)
        preds = clf.predict_proba(test_df.drop(columns=["close"], errors="ignore"))
        metrics = {
            "mean_probability": float(preds.mean()),
            "std_probability": float(preds.std()),
        }
        results.append(
            WalkForwardResult(
                window_start=start,
                window_end=start + train_size + test_size,
                metrics=metrics,
            )
        )
        start += test_size
    return results


__all__ = ["walk_forward", "WalkForwardResult"]
