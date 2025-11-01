"""Entrypoint for AI Trading Beast."""
from __future__ import annotations

import asyncio

from core.utils import setup_logging
from telegram_bot.bot import TradingBot


async def main() -> None:
    setup_logging()
    bot = TradingBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
