"""Utility helpers for AI Trading Beast."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger
from dotenv import dotenv_values


CONFIG_PATH = Path("config.yaml")


def setup_logging() -> None:
    """Configure loguru to write to stdout and rotating file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(log_dir / "ai_trading_beast.log", rotation="1 week")
    logger.info("AI Trading Beast logging initialized")


@dataclass
class AppConfig:
    raw: Dict[str, Any]

    @property
    def general(self) -> Dict[str, Any]:
        return self.raw.get("general", {})

    @property
    def risk(self) -> Dict[str, Any]:
        return self.raw.get("risk", {})

    @property
    def markets(self) -> Any:
        return self.raw.get("markets", [])

    @property
    def ml(self) -> Dict[str, Any]:
        return self.raw.get("ml", {})

    @property
    def rl(self) -> Dict[str, Any]:
        return self.raw.get("rl", {})

    @property
    def execution(self) -> Dict[str, Any]:
        return self.raw.get("execution", {})


_ENV_CACHE: Optional[Dict[str, str]] = None


def load_env(path: Path | str = Path(".env")) -> Dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        env_path = Path(path)
        env_file = dotenv_values(str(env_path)) if env_path.exists() else {}
        merged = {**env_file, **os.environ}
        _ENV_CACHE = {key: value for key, value in merged.items() if value is not None}
    return _ENV_CACHE


def load_config(path: Path | str = CONFIG_PATH) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(raw=raw)


def timestamp() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def dumps_json(data: Any) -> str:
    return json.dumps(data, default=str, indent=2, sort_keys=True)


def ensure_dirs(*paths: Path | str) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


__all__ = [
    "AppConfig",
    "load_config",
    "load_env",
    "setup_logging",
    "timestamp",
    "dumps_json",
    "ensure_dirs",
]
