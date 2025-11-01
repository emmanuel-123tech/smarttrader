"""CCXT exchange client wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from loguru import logger

try:
    import ccxt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("ccxt is required for CCXT client") from exc


@dataclass
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reduce_only: bool = False


class CCXTClient:
    def __init__(self, exchange: str, api_key: str, api_secret: str, password: Optional[str] = None, testnet: bool = False):
        exchange_class = getattr(ccxt, exchange)
        params = {"apiKey": api_key, "secret": api_secret}
        if password:
            params["password"] = password
        self.exchange = exchange_class(params)
        if testnet and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)
        logger.info("AI Trading Beast CCXT client initialized", exchange=exchange, testnet=testnet)

    def place_order(self, request: OrderRequest) -> Dict[str, Any]:
        logger.info("AI Trading Beast placing order", order=request)
        params: Dict[str, Any] = {}
        if request.stop_loss or request.take_profit:
            params["stopLossPrice"] = request.stop_loss
            params["takeProfitPrice"] = request.take_profit
        if request.reduce_only:
            params["reduceOnly"] = True
        order = self.exchange.create_order(
            symbol=request.symbol,
            type=request.order_type,
            side=request.side,
            amount=request.quantity,
            price=request.price,
            params=params,
        )
        logger.info("AI Trading Beast order response", response=order)
        return order

    def fetch_balance(self) -> Dict[str, Any]:
        return self.exchange.fetch_balance()

    def close(self) -> None:
        self.exchange.close()


__all__ = ["CCXTClient", "OrderRequest"]
