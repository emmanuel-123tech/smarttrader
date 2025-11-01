"""OANDA REST client wrapper."""
from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class OandaCredentials:
    account_id: str
    api_token: str
    environment: str = "practice"


class OandaClient:
    def __init__(self, credentials: OandaCredentials):
        self.credentials = credentials
        self.base_url = "https://api-fxpractice.oanda.com" if credentials.environment != "live" else "https://api-fxtrade.oanda.com"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {credentials.api_token}", "Content-Type": "application/json"})
        logger.info("AI Trading Beast OANDA client initialized", env=credentials.environment)

    def place_order(self, instrument: str, units: float, order_type: str = "MARKET", price: Optional[float] = None, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/accounts/{self.credentials.account_id}/orders"
        order_body: Dict[str, Any] = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": order_type,
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        if price:
            order_body["order"]["price"] = str(price)
        if stop_loss:
            order_body["order"]["stopLossOnFill"] = {"price": str(stop_loss)}
        if take_profit:
            order_body["order"]["takeProfitOnFill"] = {"price": str(take_profit)}
        response = self.session.post(url, json=order_body, timeout=30)
        response.raise_for_status()
        data = response.json()
        logger.info("AI Trading Beast OANDA order response", response=data)
        return data

    def get_account_summary(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/accounts/{self.credentials.account_id}/summary"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self.session.close()


__all__ = ["OandaClient", "OandaCredentials"]
