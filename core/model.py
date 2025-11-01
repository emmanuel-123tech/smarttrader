"""ML models for AI Trading Beast."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .utils import ensure_dirs


@dataclass
class TrainResult:
    model_path: Path
    metrics: dict


class DirectionClassifier:
    """Binary direction classifier using LightGBM."""

    def __init__(self, probability_threshold: float = 0.55):
        if lgb is None:
            raise RuntimeError("LightGBM is required for DirectionClassifier")
        self.model: Optional[lgb.Booster] = None
        self.threshold = probability_threshold

    def _build_dataset(self, df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        df["future_return"] = df["close"].pct_change(periods=horizon).shift(-horizon)
        df["label"] = (df["future_return"] > 0).astype(int)
        features = df.drop(columns=["future_return", "label"]).dropna()
        features = features.loc[~features.index.duplicated(keep="last")]
        labels = df.loc[features.index, "label"]
        return features, labels

    def fit(self, df: pd.DataFrame, horizon: int = 3, test_size: float = 0.2, random_state: int = 42) -> TrainResult:
        features, labels = self._build_dataset(df, horizon)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        train_dataset = lgb.Dataset(X_train, label=y_train)
        valid_dataset = lgb.Dataset(X_test, label=y_test)
        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "verbosity": -1,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        }
        self.model = lgb.train(params, train_dataset, valid_sets=[valid_dataset], num_boost_round=400, early_stopping_rounds=50)
        preds = self.model.predict(X_test)
        pred_labels = (preds >= 0.5).astype(int)
        report = classification_report(y_test, pred_labels, output_dict=True)
        ensure_dirs("models")
        model_path = Path("models") / "lightgbm_default.txt"
        self.model.save_model(str(model_path))
        logger.info("AI Trading Beast trained LightGBM", metrics=report)
        return TrainResult(model_path=model_path, metrics=report)

    def load(self, path: str | Path) -> None:
        if lgb is None:
            raise RuntimeError("LightGBM is required for DirectionClassifier")
        self.model = lgb.Booster(model_file=str(path))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.predict(features)

    def should_trade(self, feature_row: pd.Series) -> bool:
        proba = self.predict_proba(feature_row.to_frame().T)[0]
        return float(proba) >= self.threshold


__all__ = ["DirectionClassifier", "TrainResult"]
